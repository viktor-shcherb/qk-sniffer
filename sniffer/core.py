from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from queue import Queue
from threading import Thread
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import torch

from saver.dataset import CaptureBatch, DatasetSaver
from .samplers import LogUniformSampler, Sampler


@dataclass(slots=True)
class SnifferConfig:
    model_name: str
    data_root: Union[str, Path] = "data"
    readme_path: Union[str, Path] = "README.md"
    capture_queries: bool = True
    capture_keys: bool = True
    layers: Optional[Set[int]] = None
    heads: Optional[Set[int]] = None
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    sampler_factory: Optional[Callable[[], Sampler]] = None
    queue_size: int = 8
    max_rows_per_batch: Optional[int] = None
    write_batch_size: int = 2048
    min_bucket_size: int = 128


class Sniffer:
    def __init__(self, config: SnifferConfig):
        self.config = config
        queue_size = max(1, int(config.queue_size))
        self.saver = DatasetSaver(
            root=config.data_root,
            readme_path=config.readme_path,
            write_batch_size=max(1, int(config.write_batch_size)),
        )
        self.saver.register_model_metadata(self.config.model_name, self.config.metadata)
        self._example_ids: Optional[Sequence[int]] = None
        self._sequence_lengths: Optional[List[int]] = None
        requested_min_bucket = int(config.min_bucket_size)
        if requested_min_bucket < 1:
            raise ValueError("min_bucket_size must be at least 1.")
        if requested_min_bucket == 1:
            self._min_bucket_power = 0
        else:
            self._min_bucket_power = int(math.ceil(math.log2(requested_min_bucket)))
        self._min_bucket_size = 1 << self._min_bucket_power
        self.sampler: Sampler = (
            config.sampler_factory()
            if config.sampler_factory is not None
            else LogUniformSampler(min_bucket_size=self._min_bucket_size)
        )
        self._writer = _CaptureWorker(self.saver, queue_size=queue_size)
        self._max_rows_per_batch: Optional[int] = (
            int(config.max_rows_per_batch) if config.max_rows_per_batch and config.max_rows_per_batch > 0 else None
        )

    def capture(
        self,
        layer_idx: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        positions: torch.Tensor,
        sliding_window: Optional[int],
    ) -> None:
        if self.config.layers is not None and layer_idx not in self.config.layers:
            return
        if not self.config.capture_queries and not self.config.capture_keys:
            return

        with torch.no_grad():
            batch_size, num_heads, seq_len, _ = query_states.shape
            device = query_states.device
            positions = positions.to(device=device, dtype=torch.int64)
            example_ids = self._prepare_example_ids(batch_size, seq_len, device)
            positions_fp = positions.to(torch.float32)
            raw_buckets = torch.floor(torch.log2(positions_fp + 1.0))
            bucket_floor = float(self._min_bucket_power)
            clamped = torch.clamp(raw_buckets, min=bucket_floor)
            buckets = clamped.to(torch.int64)
            valid_lengths = self._resolve_sequence_lengths(batch_size, seq_len)

            if self.config.capture_queries:
                self._capture_tensor(
                    layer_idx=layer_idx,
                    vector_kind="q",
                    states=query_states.detach(),
                    example_ids=example_ids,
                    positions=positions,
                    buckets=buckets,
                    valid_lengths=valid_lengths,
                    sliding_window=sliding_window,
                )
            if self.config.capture_keys:
                self._capture_tensor(
                    layer_idx=layer_idx,
                    vector_kind="k",
                    states=key_states.detach(),
                    example_ids=example_ids,
                    positions=positions,
                    buckets=buckets,
                    valid_lengths=valid_lengths,
                    sliding_window=sliding_window,
                )

    def close(self) -> None:
        self._writer.close()
        self.saver.close()

    def __enter__(self) -> "Sniffer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def set_example_ids(self, example_ids: Sequence[int]) -> None:
        self._example_ids = list(example_ids)
        self._sequence_lengths = None

    def set_sequence_lengths(self, sequence_lengths: Sequence[int]) -> None:
        self._sequence_lengths = [max(0, int(length)) for length in sequence_lengths]

    def _resolve_sequence_lengths(self, batch_size: int, seq_len: int) -> List[int]:
        if self._sequence_lengths is None:
            return [seq_len] * batch_size
        if len(self._sequence_lengths) != batch_size:
            raise ValueError(
                f"Expected {batch_size} sequence lengths, got {len(self._sequence_lengths)}. "
                "Call set_active_sequence_lengths with values for each batch item."
            )
        return [min(seq_len, length) for length in self._sequence_lengths]

    def _capture_tensor(
        self,
        *,
        layer_idx: int,
        vector_kind: str,
        states: torch.Tensor,
        example_ids: torch.Tensor,
        positions: torch.Tensor,
        buckets: torch.Tensor,
        valid_lengths: Sequence[int],
        sliding_window: Optional[int],
    ) -> None:
        batch_size, num_heads, _, _ = states.shape
        for head_idx in self._head_indices(num_heads):
            head_states = states[:, head_idx]
            for batch_idx in range(batch_size):
                valid_length = valid_lengths[batch_idx]
                if valid_length <= 0:
                    continue
                example_slice = example_ids[batch_idx, :valid_length]
                example_id = int(example_slice[0].item())
                position_slice = positions[batch_idx, :valid_length]
                bucket_slice = buckets[batch_idx, :valid_length]
                mask = self.sampler.sample_positions(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    vector_kind=vector_kind,
                    example_id=example_id,
                    positions=position_slice,
                    buckets=bucket_slice,
                )
                if mask.ndim != 1:
                    raise ValueError("Sampler must return a 1D mask.")
                if mask.shape[0] != valid_length:
                    raise ValueError(
                        f"Sampler returned mask of shape {mask.shape} for sequence length {valid_length}."
                    )
                if mask.dtype != torch.bool or mask.device != states.device:
                    mask = mask.to(device=states.device, dtype=torch.bool)
                if not mask.any():
                    continue
                vectors = head_states[batch_idx, :valid_length][mask]
                positions_kept = position_slice[mask]
                buckets_kept = bucket_slice[mask]
                examples_kept = example_slice[mask]
                (
                    vectors,
                    examples_kept,
                    positions_kept,
                    buckets_kept,
                ) = self._maybe_downsample(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    vector_kind=vector_kind,
                    example_id=example_id,
                    vectors=vectors,
                    example_ids=examples_kept,
                    positions=positions_kept,
                    buckets=buckets_kept,
                )
                batch_payload = self._build_capture_batch(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    vector_kind=vector_kind,
                    vectors=vectors,
                    example_ids=examples_kept,
                    positions=positions_kept,
                    buckets=buckets_kept,
                    sliding_window=sliding_window,
                )
                if batch_payload is not None:
                    self._writer.submit(batch_payload)

    def _head_indices(self, total_heads: int) -> Sequence[int]:
        if self.config.heads is None:
            return range(total_heads)
        invalid = [head for head in self.config.heads if head < 0 or head >= total_heads]
        if invalid:
            raise ValueError(f"Invalid head indices {invalid}; layer exposes {total_heads} heads.")
        return sorted(self.config.heads)

    def _maybe_downsample(
        self,
        *,
        layer_idx: int,
        head_idx: int,
        vector_kind: str,
        example_id: int,
        vectors: torch.Tensor,
        example_ids: torch.Tensor,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        limit = self._max_rows_per_batch
        row_count = vectors.shape[0]
        if limit is None or row_count <= limit:
            return vectors, example_ids, positions, buckets
        generator = torch.Generator(device=vectors.device)
        generator.manual_seed(
            self._subsample_seed(layer_idx=layer_idx, head_idx=head_idx, vector_kind=vector_kind, example_id=example_id)
        )
        perm = torch.randperm(row_count, device=vectors.device, generator=generator)
        keep = perm[:limit]
        return vectors[keep], example_ids[keep], positions[keep], buckets[keep]

    def _build_capture_batch(
        self,
        *,
        layer_idx: int,
        head_idx: int,
        vector_kind: str,
        vectors: torch.Tensor,
        example_ids: torch.Tensor,
        positions: torch.Tensor,
        buckets: torch.Tensor,
        sliding_window: Optional[int],
    ) -> Optional[CaptureBatch]:
        row_count = vectors.shape[0]
        if row_count == 0:
            return None
        return CaptureBatch(
            model_name=self.config.model_name,
            layer_idx=layer_idx,
            head_idx=head_idx,
            vector_kind=vector_kind,  # type: ignore[arg-type]
            buckets=buckets.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy(),
            example_ids=example_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy(),
            positions=positions.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy(),
            vectors=vectors.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy(),
            sliding_window=sliding_window,
        )

    def _subsample_seed(self, *, layer_idx: int, head_idx: int, vector_kind: str, example_id: int) -> int:
        seed = (
            (layer_idx & 0xFFFF)
            ^ ((head_idx & 0xFFFF) << 16)
            ^ ((ord(vector_kind[0]) & 0xFF) << 32)
            ^ ((example_id & 0xFFFFFFFF) << 40)
        )
        return seed & 0xFFFFFFFFFFFFFFFF

    def _prepare_example_ids(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._example_ids is None:
            base = torch.arange(batch_size, device=device, dtype=torch.int64)
        else:
            if len(self._example_ids) != batch_size:
                raise ValueError(
                    f"Expected {batch_size} example ids, got {len(self._example_ids)}. "
                    "Call set_example_ids with the correct batch size."
                )
            base = torch.tensor(self._example_ids, device=device, dtype=torch.int64)
        return base.unsqueeze(1).expand(-1, seq_len)


_CAPTURE_SENTINEL: object = object()


class _CaptureWorker:
    def __init__(self, saver: DatasetSaver, queue_size: int):
        self._saver = saver
        self._queue: "Queue[Optional[CaptureBatch]]" = Queue(maxsize=queue_size)
        self._thread = Thread(target=self._run, daemon=True)
        self._closed = False
        self._exception: Optional[BaseException] = None
        self._thread.start()

    def submit(self, batch: CaptureBatch) -> None:
        if batch.vectors.size == 0:
            return
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot submit captures after the worker is closed.")
        self._queue.put(batch)

    def close(self) -> None:
        if self._closed:
            self._queue.join()
            self._raise_if_failed()
            return
        self._queue.put(_CAPTURE_SENTINEL)
        self._queue.join()
        self._thread.join()
        self._closed = True
        self._raise_if_failed()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _CAPTURE_SENTINEL:
                self._queue.task_done()
                break
            try:
                if self._exception is None:
                    self._saver.add_batch(item)
            except Exception as exc:  # pragma: no cover - propagated to main thread
                if self._exception is None:
                    self._exception = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Capture worker failed; see the chained exception.") from self._exception


_ACTIVE_SNIFFER: Optional[Sniffer] = None


def get_active_sniffer() -> Optional[Sniffer]:
    return _ACTIVE_SNIFFER


def set_active_example_ids(example_ids: Sequence[int]) -> None:
    sniffer = get_active_sniffer()
    if sniffer is None:
        raise RuntimeError("No active sniffer session to attach example ids.")
    sniffer.set_example_ids(example_ids)


def set_active_sequence_lengths(sequence_lengths: Sequence[int]) -> None:
    sniffer = get_active_sniffer()
    if sniffer is None:
        raise RuntimeError("No active sniffer session to attach sequence lengths.")
    sniffer.set_sequence_lengths(sequence_lengths)


@contextmanager
def use_sniffer(sniffer: Sniffer):
    global _ACTIVE_SNIFFER
    if _ACTIVE_SNIFFER is not None:
        raise RuntimeError("Another sniffer session is already active.")
    previous = _ACTIVE_SNIFFER
    _ACTIVE_SNIFFER = sniffer
    try:
        yield sniffer
    finally:
        _ACTIVE_SNIFFER = previous


@contextmanager
def activate_sniffer(config: SnifferConfig):
    sniffer = Sniffer(config)
    with use_sniffer(sniffer):
        try:
            yield sniffer
        finally:
            sniffer.close()
