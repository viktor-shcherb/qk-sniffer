from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from queue import Full, Queue
from threading import Thread
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union, Literal

import numpy as np
import torch

from saver.dataset import CaptureBatch, DatasetSaver
from .samplers import LogUniformSampler, UniformSampler, Sampler


def _debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[sniff][debug] {message}", flush=True)


@dataclass(slots=True)
class SnifferConfig:
    model_name: str
    data_root: Union[str, Path] = "data"
    readme_path: Union[str, Path] = "README.md"
    capture_queries: bool = True
    capture_keys: bool = True
    layers: Optional[Set[int]] = None
    heads: Optional[Set[int]] = None
    sampled_query_heads: Optional[Dict[int, Set[int]]] = None
    sampled_key_heads: Optional[Dict[int, Set[int]]] = None
    metadata: Dict[str, Union[str, int, float]] = field(default_factory=dict)
    sampler_factory: Optional[Callable[[], Sampler]] = None
    queue_size: int = 32
    max_rows_per_batch: Optional[int] = None
    write_batch_size: int = 4096
    min_bucket_size: int = 128
    capture_pre_rope: bool = False
    capture_token_strings: bool = False
    full_attention_only: bool = False
    debug_logging: bool = False


@dataclass(slots=True)
class _PendingCapture:
    layer_idx: int
    vector_kind: Literal["q", "k"]
    sliding_window: Optional[int]
    vectors: List[torch.Tensor] = field(default_factory=list)
    head_indices: List[torch.Tensor] = field(default_factory=list)
    example_ids: List[torch.Tensor] = field(default_factory=list)
    positions: List[torch.Tensor] = field(default_factory=list)
    buckets: List[torch.Tensor] = field(default_factory=list)
    token_strings: Optional[List[str]] = None


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return
    if device.type == "mps":
        mps_backend = getattr(torch, "mps", None)
        if mps_backend is not None and hasattr(mps_backend, "synchronize"):
            mps_backend.synchronize()


class Sniffer:
    def __init__(self, config: SnifferConfig):
        self.config = config
        queue_size = max(1, int(config.queue_size))
        self.saver = DatasetSaver(
            root=config.data_root,
            readme_path=config.readme_path,
            write_batch_size=max(1, int(config.write_batch_size)),
        )
        metadata = dict(self.config.metadata)
        self._example_ids: Optional[Sequence[int]] = None
        self._sequence_lengths: Optional[List[int]] = None
        self._token_strings: Optional[List[List[str]]] = None
        requested_min_bucket = int(config.min_bucket_size)
        if requested_min_bucket < 1:
            raise ValueError("min_bucket_size must be at least 1.")
        self.sampler = (
            config.sampler_factory()
            if config.sampler_factory is not None
            else LogUniformSampler(min_bucket_size=requested_min_bucket)
        )
        self._bucket_kind = getattr(self.sampler, "bucket_kind", "log").lower()
        if self._bucket_kind not in {"log", "uniform", "all"}:
            raise ValueError("Sampler must declare bucket_kind 'log', 'uniform', or 'all'.")
        if self._bucket_kind == "log":
            if requested_min_bucket == 1:
                self._min_bucket_power = 0
            else:
                self._min_bucket_power = int(math.ceil(math.log2(requested_min_bucket)))
            self._log_bucket_floor = float(self._min_bucket_power)
            self._effective_min_bucket_size = 1 << self._min_bucket_power
            metadata.setdefault("sampling_min_bucket_size", self._effective_min_bucket_size)
        elif self._bucket_kind in {"uniform", "all"}:
            self._uniform_bucket_size = requested_min_bucket
            metadata.setdefault("sampling_bucket_size", self._uniform_bucket_size)
        metadata.setdefault("sampling_strategy", "all" if self._bucket_kind == "all" else self._bucket_kind)
        self.saver.register_model_metadata(self.config.model_name, metadata)
        self._writer = _CaptureWorker(
            self.saver,
            queue_size=queue_size,
            debug_logging=bool(config.debug_logging),
        )
        self._max_rows_per_batch: Optional[int] = (
            int(config.max_rows_per_batch) if config.max_rows_per_batch and config.max_rows_per_batch > 0 else None
        )
        self._pending_captures: Dict[str, _PendingCapture] = {}
        self._capture_invocations = 0
        self._capture_time_s = 0.0
        self._flush_invocations = 0
        self._flush_time_s = 0.0
        self._captured_payloads = 0
        self._captured_rows = 0
        self._submit_wait_s = 0.0

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
        if self.config.full_attention_only and sliding_window is not None:
            return

        capture_start = perf_counter()
        with torch.no_grad():
            batch_size, _, seq_len, _ = query_states.shape
            device = query_states.device
            positions = positions.to(device=device, dtype=torch.int64)
            example_ids = self._resolve_example_ids(batch_size)
            token_strings = self._prepare_token_strings(batch_size, seq_len)
            if self._bucket_kind == "log":
                positions_fp = positions.to(torch.float32)
                raw_buckets = torch.floor(torch.log2(positions_fp + 1.0))
                bucket_floor = float(self._min_bucket_power)
                clamped = torch.clamp(raw_buckets, min=bucket_floor)
                buckets = clamped.to(torch.int64)
            elif self._bucket_kind in {"uniform", "all"}:
                positions_fp = positions.to(torch.float32)
                bucket_size = float(self._uniform_bucket_size)
                buckets = torch.floor(positions_fp / bucket_size).to(torch.int64)
            else:
                buckets = torch.zeros_like(positions, dtype=torch.int64)
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
                    token_strings=token_strings,
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
                    token_strings=token_strings,
                )
        self._capture_invocations += 1
        self._capture_time_s += perf_counter() - capture_start

    def close(self) -> None:
        self.flush_batch()
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

    def set_token_strings(self, token_strings: Sequence[Sequence[str]]) -> None:
        self._token_strings = [list(tokens) for tokens in token_strings]

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
        example_ids: Sequence[int],
        positions: torch.Tensor,
        buckets: torch.Tensor,
        valid_lengths: Sequence[int],
        sliding_window: Optional[int],
        token_strings: Optional[Sequence[Sequence[str]]],
    ) -> None:
        batch_size, num_heads, _, _ = states.shape
        head_indices_list = self._head_indices(
            layer_idx=layer_idx,
            total_heads=num_heads,
            vector_kind=vector_kind,
        )
        if not head_indices_list:
            return

        device = states.device
        n_heads = len(head_indices_list)
        all_heads = n_heads == num_heads
        head_idx_tensor = torch.tensor(list(head_indices_list), device=device, dtype=torch.int64)

        all_vectors: List[torch.Tensor] = []
        all_head_ids: List[torch.Tensor] = []
        all_examples: List[torch.Tensor] = []
        all_positions: List[torch.Tensor] = []
        all_buckets: List[torch.Tensor] = []
        all_tokens: Optional[List[str]] = [] if token_strings is not None else None

        for batch_idx in range(batch_size):
            valid_length = valid_lengths[batch_idx]
            if valid_length <= 0:
                continue
            example_id = int(example_ids[batch_idx])
            pos_slice = positions[batch_idx, :valid_length]
            bkt_slice = buckets[batch_idx, :valid_length]

            # Generate ONE shared mask for all heads: (valid_length,)
            mask = self.sampler.sample_positions_batch(
                layer_idx=layer_idx,
                head_indices=head_indices_list,
                vector_kind=vector_kind,
                example_id=example_id,
                positions=pos_slice,
                buckets=bkt_slice,
            )
            if mask.ndim != 1:
                raise ValueError("Sampler batch method must return a 1D mask.")
            if mask.shape[0] != valid_length:
                raise ValueError(
                    f"Sampler returned mask of shape {mask.shape} for sequence length {valid_length}."
                )
            if mask.dtype != torch.bool or mask.device != device:
                mask = mask.to(device=device, dtype=torch.bool)

            # Find selected position indices (shared across all heads)
            selected_pos = mask.nonzero(as_tuple=False).squeeze(1)  # (K,)
            if selected_pos.ndim == 0:
                selected_pos = selected_pos.unsqueeze(0)
            K = selected_pos.shape[0]
            if K == 0:
                continue

            # Per-batch-item downsample (shared across heads)
            if self._max_rows_per_batch is not None and K > self._max_rows_per_batch:
                limit = self._max_rows_per_batch
                gen = torch.Generator(device=device)
                gen.manual_seed(self._subsample_seed(
                    layer_idx=layer_idx, head_idx=int(head_idx_tensor[0].item()),
                    vector_kind=vector_kind, example_id=example_id,
                ))
                perm = torch.randperm(K, device=device, generator=gen)
                selected_pos = selected_pos[perm[:limit]]
                K = limit

            # Gather vectors: first select K positions (small), then select heads.
            # states[batch_idx, :, selected_pos] is (total_heads, K, dim) — e.g. 4MB
            # vs pre-gathering all heads at all positions — e.g. 512MB.
            at_pos = states[batch_idx, :, selected_pos]  # (total_heads, K, dim)
            vectors = at_pos if all_heads else at_pos[head_idx_tensor]  # (n_heads, K, dim)
            vectors_flat = vectors.reshape(n_heads * K, -1)

            # Build head indices: [h0, h0, ..., h1, h1, ...]
            head_ids = head_idx_tensor.unsqueeze(1).expand(n_heads, K).reshape(-1)

            # Replicate positions/buckets/example_ids for each head
            positions_flat = pos_slice[selected_pos].unsqueeze(0).expand(n_heads, K).reshape(-1)
            buckets_flat = bkt_slice[selected_pos].unsqueeze(0).expand(n_heads, K).reshape(-1)
            examples_flat = torch.full(
                (n_heads * K,), fill_value=example_id, device=device, dtype=torch.int64,
            )

            all_vectors.append(vectors_flat)
            all_head_ids.append(head_ids)
            all_examples.append(examples_flat)
            all_positions.append(positions_flat)
            all_buckets.append(buckets_flat)

            if token_strings is not None and all_tokens is not None:
                token_slice = list(token_strings[batch_idx])[:valid_length]
                sel_pos_cpu = selected_pos.cpu().tolist()
                tokens_at_pos = [token_slice[p] for p in sel_pos_cpu]
                # Replicate for each head (same tokens at same positions)
                for _ in range(n_heads):
                    all_tokens.extend(tokens_at_pos)

        if not all_vectors:
            return

        vectors_joined = torch.cat(all_vectors) if len(all_vectors) > 1 else all_vectors[0]
        head_ids_joined = torch.cat(all_head_ids) if len(all_head_ids) > 1 else all_head_ids[0]
        examples_joined = torch.cat(all_examples) if len(all_examples) > 1 else all_examples[0]
        positions_joined = torch.cat(all_positions) if len(all_positions) > 1 else all_positions[0]
        buckets_joined = torch.cat(all_buckets) if len(all_buckets) > 1 else all_buckets[0]

        # Store in pending capture keyed by (layer, kind) instead of (layer, head, kind)
        key = f"l{layer_idx:02d}{vector_kind}"
        pending = self._pending_captures.get(key)
        has_tokens = token_strings is not None
        if pending is None:
            pending = _PendingCapture(
                layer_idx=layer_idx,
                vector_kind=vector_kind,  # type: ignore[arg-type]
                sliding_window=sliding_window,
                token_strings=[] if has_tokens else None,
            )
            self._pending_captures[key] = pending
        elif pending.sliding_window != sliding_window:
            raise ValueError(
                f"Inconsistent sliding_window for {key}: {pending.sliding_window} vs {sliding_window}."
            )
        if (pending.token_strings is None) != (not has_tokens):
            raise ValueError(f"Inconsistent token string capture for {key} within the same batch.")
        pending.vectors.append(vectors_joined)
        pending.head_indices.append(head_ids_joined)
        pending.example_ids.append(examples_joined)
        pending.positions.append(positions_joined)
        pending.buckets.append(buckets_joined)
        if pending.token_strings is not None:
            pending.token_strings.extend(all_tokens or [])

    def flush_batch(self) -> None:
        if not self._pending_captures:
            return
        flush_start = perf_counter()
        seen_devices: Set[torch.device] = set()
        for pending in self._pending_captures.values():
            if not pending.vectors:
                continue
            device = pending.vectors[0].device
            if device in seen_devices:
                continue
            _synchronize_device(device)
            seen_devices.add(device)

        for pending in self._pending_captures.values():
            if not pending.vectors:
                continue
            vectors = torch.cat(pending.vectors, dim=0) if len(pending.vectors) > 1 else pending.vectors[0]
            head_indices = (
                torch.cat(pending.head_indices, dim=0) if len(pending.head_indices) > 1 else pending.head_indices[0]
            )
            example_ids = (
                torch.cat(pending.example_ids, dim=0) if len(pending.example_ids) > 1 else pending.example_ids[0]
            )
            positions = torch.cat(pending.positions, dim=0) if len(pending.positions) > 1 else pending.positions[0]
            buckets = torch.cat(pending.buckets, dim=0) if len(pending.buckets) > 1 else pending.buckets[0]

            # Batch GPU→CPU transfer (few large copies instead of many small ones)
            vectors_np = vectors.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            head_indices_np = head_indices.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
            example_ids_np = example_ids.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
            positions_np = positions.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()
            buckets_np = buckets.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy()

            # Split by head on CPU
            unique_heads = np.unique(head_indices_np)
            for head_idx in unique_heads:
                mask = head_indices_np == int(head_idx)
                h_vectors = vectors_np[mask]
                row_count = h_vectors.shape[0]
                if row_count == 0:
                    continue
                h_token_strings: Optional[List[str]] = None
                if pending.token_strings is not None:
                    indices = np.where(mask)[0]
                    h_token_strings = [pending.token_strings[i] for i in indices]
                batch_payload = CaptureBatch(
                    model_name=self.config.model_name,
                    layer_idx=pending.layer_idx,
                    head_idx=int(head_idx),
                    vector_kind=pending.vector_kind,
                    buckets=buckets_np[mask],
                    example_ids=example_ids_np[mask],
                    positions=positions_np[mask],
                    vectors=h_vectors,
                    sliding_window=pending.sliding_window,
                    token_strings=h_token_strings,
                )
                submit_wait_s = self._writer.submit(batch_payload)
                self._captured_payloads += 1
                self._captured_rows += row_count
                self._submit_wait_s += submit_wait_s
        self._pending_captures.clear()
        self._flush_invocations += 1
        self._flush_time_s += perf_counter() - flush_start

    def _head_indices(
        self,
        *,
        layer_idx: int,
        total_heads: int,
        vector_kind: Literal["q", "k"],
    ) -> Sequence[int]:
        explicit_heads = self.config.heads
        if explicit_heads is not None:
            invalid = sorted(head for head in explicit_heads if head < 0 or head >= total_heads)
            if invalid:
                raise ValueError(
                    f"Invalid head indices {invalid} for {vector_kind} in layer {layer_idx}; "
                    f"layer exposes {total_heads} heads."
                )

        sampled_map = self.config.sampled_query_heads if vector_kind == "q" else self.config.sampled_key_heads
        sampled_heads: Optional[Set[int]] = None
        if sampled_map is not None:
            sampled_heads = sampled_map.get(layer_idx, set())
            if not sampled_heads:
                return ()
            invalid_sampled = sorted(head for head in sampled_heads if head < 0 or head >= total_heads)
            if invalid_sampled:
                raise ValueError(
                    f"Invalid sampled {vector_kind} head indices {invalid_sampled} for layer {layer_idx}; "
                    f"layer exposes {total_heads} heads."
                )

        if explicit_heads is None and sampled_heads is None:
            return range(total_heads)

        allowed = set(range(total_heads))
        if explicit_heads is not None:
            allowed &= explicit_heads
        if sampled_heads is not None:
            allowed &= sampled_heads
        return sorted(allowed)


    def _subsample_seed(self, *, layer_idx: int, head_idx: int, vector_kind: str, example_id: int) -> int:
        seed = (
            (layer_idx & 0xFFFF)
            ^ ((head_idx & 0xFFFF) << 16)
            ^ ((ord(vector_kind[0]) & 0xFF) << 32)
            ^ ((example_id & 0xFFFFFFFF) << 40)
        )
        return seed & 0xFFFFFFFFFFFFFFFF

    def _resolve_example_ids(self, batch_size: int) -> List[int]:
        if self._example_ids is None:
            return list(range(batch_size))
        if len(self._example_ids) != batch_size:
            raise ValueError(
                f"Expected {batch_size} example ids, got {len(self._example_ids)}. "
                "Call set_example_ids with the correct batch size."
            )
        return [int(value) for value in self._example_ids]

    def _prepare_token_strings(self, batch_size: int, seq_len: int) -> Optional[List[List[str]]]:
        if not self.config.capture_token_strings:
            return None
        if self._token_strings is None:
            raise ValueError("Token string capture enabled but token strings were not provided.")
        if len(self._token_strings) != batch_size:
            raise ValueError(
                f"Expected {batch_size} token string rows, got {len(self._token_strings)}. "
                "Call set_active_token_strings with values for each batch item."
            )
        prepared: List[List[str]] = []
        for idx, tokens in enumerate(self._token_strings):
            if len(tokens) < seq_len:
                raise ValueError(
                    f"Token string length {len(tokens)} shorter than sequence length {seq_len} for batch index {idx}."
                )
            prepared.append(list(tokens[:seq_len]))
        return prepared

    def consume_debug_stats(self) -> Dict[str, Union[int, float]]:
        writer_batches, writer_rows, queue_depth = self._writer.stats_snapshot()
        snapshot: Dict[str, Union[int, float]] = {
            "capture_invocations": self._capture_invocations,
            "capture_time_s": self._capture_time_s,
            "flush_invocations": self._flush_invocations,
            "flush_time_s": self._flush_time_s,
            "captured_payloads": self._captured_payloads,
            "captured_rows": self._captured_rows,
            "submit_wait_s": self._submit_wait_s,
            "pending_configs": len(self._pending_captures),
            "writer_batches_total": writer_batches,
            "writer_rows_total": writer_rows,
            "writer_queue_depth": queue_depth,
        }
        self._capture_invocations = 0
        self._capture_time_s = 0.0
        self._flush_invocations = 0
        self._flush_time_s = 0.0
        self._captured_payloads = 0
        self._captured_rows = 0
        self._submit_wait_s = 0.0
        return snapshot


_CAPTURE_SENTINEL: object = object()


class _CaptureWorker:
    def __init__(self, saver: DatasetSaver, queue_size: int, *, debug_logging: bool = False):
        self._saver = saver
        self._queue: "Queue[Optional[CaptureBatch]]" = Queue(maxsize=queue_size)
        self._thread = Thread(target=self._run, daemon=True)
        self._closed = False
        self._exception: Optional[BaseException] = None
        self._debug_logging = bool(debug_logging)
        self._processed_batches = 0
        self._processed_rows = 0
        self._thread.start()

    def submit(self, batch: CaptureBatch) -> float:
        if batch.vectors.size == 0:
            return 0.0
        self._raise_if_failed()
        if self._closed:
            raise RuntimeError("Cannot submit captures after the worker is closed.")
        put_start = perf_counter()
        if not self._debug_logging:
            self._queue.put(batch)
            return perf_counter() - put_start
        while True:
            try:
                self._queue.put(batch, timeout=5.0)
                return perf_counter() - put_start
            except Full:
                self._raise_if_failed()
                _debug_log(
                    True,
                    (
                        "capture queue is full "
                        f"({self._queue.qsize()}/{self._queue.maxsize}); waiting for writer thread."
                    ),
                )

    def close(self) -> None:
        if self._closed:
            self._queue.join()
            self._raise_if_failed()
            return
        _debug_log(self._debug_logging, "capture worker close requested")
        self._queue.put(_CAPTURE_SENTINEL)
        self._queue.join()
        self._thread.join()
        self._closed = True
        _debug_log(
            self._debug_logging,
            f"capture worker closed: batches={self._processed_batches}, rows={self._processed_rows}",
        )
        self._raise_if_failed()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is _CAPTURE_SENTINEL:
                self._queue.task_done()
                break
            try:
                if self._exception is None:
                    write_start = perf_counter()
                    self._saver.add_batch(item)
                    self._processed_batches += 1
                    self._processed_rows += int(item.vectors.shape[0])
                    if self._debug_logging and (
                        self._processed_batches <= 3 or self._processed_batches % 2000 == 0
                    ):
                        _debug_log(
                            True,
                            (
                                f"writer processed batch {self._processed_batches} "
                                f"(rows={item.vectors.shape[0]}, queue={self._queue.qsize()}, "
                                f"write_s={perf_counter() - write_start:.3f})"
                            ),
                        )
            except Exception as exc:  # pragma: no cover - propagated to main thread
                if self._exception is None:
                    self._exception = exc
            finally:
                self._queue.task_done()

    def _raise_if_failed(self) -> None:
        if self._exception is not None:
            raise RuntimeError("Capture worker failed; see the chained exception.") from self._exception

    def stats_snapshot(self) -> Tuple[int, int, int]:
        return self._processed_batches, self._processed_rows, self._queue.qsize()


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


def set_active_token_strings(token_strings: Sequence[Sequence[str]]) -> None:
    sniffer = get_active_sniffer()
    if sniffer is None:
        raise RuntimeError("No active sniffer session to attach token strings.")
    sniffer.set_token_strings(token_strings)


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
