from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Set, Union, Literal

import numpy as np
import torch

from saver.dataset import DatasetSaver
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
    max_rows_per_batch: Optional[int] = None
    write_batch_size: int = 4096
    min_bucket_size: int = 128
    capture_pre_rope: bool = False
    capture_token_strings: bool = False
    full_attention_only: bool = False
    debug_logging: bool = False


@dataclass(slots=True)
class _PendingChunk:
    """One batch item's captured data for a (layer, kind) pair.

    Vectors are (n_heads, K, dim) on the model device; positions/buckets
    are shared across heads and stored as CPU numpy arrays of length K.
    """
    vectors: torch.Tensor           # (n_heads, K, dim) on device
    positions: np.ndarray           # (K,) int64 CPU
    buckets: np.ndarray             # (K,) int64 CPU
    example_id: int
    token_strings: Optional[List[str]]  # K strings or None


@dataclass(slots=True)
class _PendingCapture:
    layer_idx: int
    vector_kind: Literal["q", "k"]
    sliding_window: Optional[int]
    head_indices: List[int]
    chunks: List[_PendingChunk] = field(default_factory=list)


@dataclass(slots=True)
class _AccumulatedCapture:
    """Accumulated data for a (layer, kind) pair across all batches.

    Vectors are (n_heads, K_i, dim) per chunk.  Positions, buckets, and
    example ids are shared across heads — stored once per chunk.
    """
    layer_idx: int
    vector_kind: Literal["q", "k"]
    sliding_window: Optional[int]
    head_indices: List[int]
    vectors: List[np.ndarray] = field(default_factory=list)         # each (n_heads, K_i, dim)
    positions: List[np.ndarray] = field(default_factory=list)       # each (K_i,)
    buckets: List[np.ndarray] = field(default_factory=list)         # each (K_i,)
    example_ids: List[int] = field(default_factory=list)            # one int per chunk
    token_strings: Optional[List[List[str]]] = None                 # each K_i strings
    total_rows: int = 0


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
        self._accumulated: Dict[str, _AccumulatedCapture] = {}
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
            # Positions and buckets on CPU: sampling + nonzero happen on CPU
            # (instant) instead of GPU (forces CUDA sync per call).
            positions_cpu = positions.to(device="cpu", dtype=torch.int64)
            example_ids = self._resolve_example_ids(batch_size)
            token_strings = self._prepare_token_strings(batch_size, seq_len)
            if self._bucket_kind == "log":
                positions_fp = positions_cpu.to(torch.float32)
                raw_buckets = torch.floor(torch.log2(positions_fp + 1.0))
                bucket_floor = float(self._min_bucket_power)
                clamped = torch.clamp(raw_buckets, min=bucket_floor)
                buckets_cpu = clamped.to(torch.int64)
            elif self._bucket_kind in {"uniform", "all"}:
                positions_fp = positions_cpu.to(torch.float32)
                bucket_size = float(self._uniform_bucket_size)
                buckets_cpu = torch.floor(positions_fp / bucket_size).to(torch.int64)
            else:
                buckets_cpu = torch.zeros_like(positions_cpu, dtype=torch.int64)
            valid_lengths = self._resolve_sequence_lengths(batch_size, seq_len)

            if self.config.capture_queries:
                self._capture_tensor(
                    layer_idx=layer_idx,
                    vector_kind="q",
                    states=query_states.detach(),
                    example_ids=example_ids,
                    positions=positions_cpu,
                    buckets=buckets_cpu,
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
                    positions=positions_cpu,
                    buckets=buckets_cpu,
                    valid_lengths=valid_lengths,
                    sliding_window=sliding_window,
                    token_strings=token_strings,
                )
        self._capture_invocations += 1
        self._capture_time_s += perf_counter() - capture_start

    def close(self) -> None:
        self.flush_batch()
        for acc in self._accumulated.values():
            if not acc.vectors:
                continue
            n_chunks = len(acc.vectors)
            # Concatenate along positions axis: (n_heads, total_K, dim)
            all_vectors = np.concatenate(acc.vectors, axis=1) if n_chunks > 1 else acc.vectors[0]
            # Shared across heads — one copy
            all_positions = np.concatenate(acc.positions) if n_chunks > 1 else acc.positions[0]
            all_buckets = np.concatenate(acc.buckets) if n_chunks > 1 else acc.buckets[0]
            all_example_ids = np.concatenate([
                np.full(pos.shape[0], eid, dtype=np.int64)
                for pos, eid in zip(acc.positions, acc.example_ids)
            ])
            all_token_strings: Optional[List[str]] = None
            if acc.token_strings is not None:
                all_token_strings = []
                for ts in acc.token_strings:
                    all_token_strings.extend(ts)
            # Write per head — only vectors differ; metadata is shared
            for i, head_idx in enumerate(acc.head_indices):
                config_name = f"l{acc.layer_idx:02d}h{head_idx:02d}{acc.vector_kind}"
                self.saver.write_config_data(
                    config_name,
                    vectors=all_vectors[i],  # (total_K, dim)
                    buckets=all_buckets,
                    example_ids=all_example_ids,
                    positions=all_positions,
                    sliding_window=acc.sliding_window,
                    token_strings=all_token_strings,
                )
        self._accumulated.clear()
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
        """Capture vectors for selected positions across all active heads.

        Vectors are kept as (n_heads, K, dim) per batch item — positions and
        buckets are shared across heads and stored once.  Only the small index
        tensor is sent to the GPU for async vector gathering.
        """
        batch_size, num_heads, _, _ = states.shape
        head_indices_list = self._head_indices(
            layer_idx=layer_idx,
            total_heads=num_heads,
            vector_kind=vector_kind,
        )
        if not head_indices_list:
            return

        device = states.device
        all_heads = len(head_indices_list) == num_heads
        head_idx_gpu = (
            torch.tensor(list(head_indices_list), dtype=torch.int64, device=device)
            if not all_heads else None
        )

        key = f"l{layer_idx:02d}{vector_kind}"
        pending = self._pending_captures.get(key)
        if pending is None:
            pending = _PendingCapture(
                layer_idx=layer_idx,
                vector_kind=vector_kind,  # type: ignore[arg-type]
                sliding_window=sliding_window,
                head_indices=list(head_indices_list),
            )
            self._pending_captures[key] = pending
        elif pending.sliding_window != sliding_window:
            raise ValueError(
                f"Inconsistent sliding_window for {key}: {pending.sliding_window} vs {sliding_window}."
            )

        for batch_idx in range(batch_size):
            valid_length = valid_lengths[batch_idx]
            if valid_length <= 0:
                continue
            example_id = int(example_ids[batch_idx])
            pos_slice = positions[batch_idx, :valid_length]
            bkt_slice = buckets[batch_idx, :valid_length]

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
            if mask.dtype != torch.bool:
                mask = mask.to(dtype=torch.bool)

            selected_pos = mask.nonzero(as_tuple=False).squeeze(1)
            if selected_pos.ndim == 0:
                selected_pos = selected_pos.unsqueeze(0)
            K = selected_pos.shape[0]
            if K == 0:
                continue

            if self._max_rows_per_batch is not None and K > self._max_rows_per_batch:
                limit = self._max_rows_per_batch
                gen = torch.Generator()
                gen.manual_seed(self._subsample_seed(
                    layer_idx=layer_idx, head_idx=int(head_indices_list[0]),
                    vector_kind=vector_kind, example_id=example_id,
                ))
                perm = torch.randperm(K, generator=gen)
                selected_pos = selected_pos[perm[:limit]]
                K = limit

            selected_pos_gpu = selected_pos.to(device=device, non_blocking=True)
            at_pos = states[batch_idx, :, selected_pos_gpu]  # (total_heads, K, dim)
            vectors = at_pos if all_heads else at_pos[head_idx_gpu]  # (n_heads, K, dim)

            # Shared metadata stays on CPU as numpy
            sel_positions_np = pos_slice[selected_pos].numpy()
            sel_buckets_np = bkt_slice[selected_pos].numpy()

            chunk_tokens: Optional[List[str]] = None
            if token_strings is not None:
                token_slice = list(token_strings[batch_idx])[:valid_length]
                sel_pos_list = selected_pos.tolist()
                chunk_tokens = [token_slice[p] for p in sel_pos_list]

            pending.chunks.append(_PendingChunk(
                vectors=vectors,
                positions=sel_positions_np,
                buckets=sel_buckets_np,
                example_id=example_id,
                token_strings=chunk_tokens,
            ))

    def flush_batch(self) -> None:
        if not self._pending_captures:
            return
        flush_start = perf_counter()
        # Synchronize GPU once per device before CPU transfer
        seen_devices: Set[torch.device] = set()
        for pending in self._pending_captures.values():
            if not pending.chunks:
                continue
            device = pending.chunks[0].vectors.device
            if device not in seen_devices:
                _synchronize_device(device)
                seen_devices.add(device)

        for pending in self._pending_captures.values():
            if not pending.chunks:
                continue
            acc_key = f"l{pending.layer_idx:02d}{pending.vector_kind}"
            acc = self._accumulated.get(acc_key)
            if acc is None:
                acc = _AccumulatedCapture(
                    layer_idx=pending.layer_idx,
                    vector_kind=pending.vector_kind,
                    sliding_window=pending.sliding_window,
                    head_indices=list(pending.head_indices),
                )
                self._accumulated[acc_key] = acc

            n_heads = len(pending.head_indices)
            for chunk in pending.chunks:
                # GPU→CPU transfer: one copy of (n_heads, K, dim)
                vectors_np = chunk.vectors.detach().to(
                    device="cpu", dtype=torch.float32,
                ).contiguous().numpy()
                K = chunk.positions.shape[0]
                acc.vectors.append(vectors_np)
                acc.positions.append(chunk.positions)
                acc.buckets.append(chunk.buckets)
                acc.example_ids.append(chunk.example_id)
                if chunk.token_strings is not None:
                    if acc.token_strings is None:
                        acc.token_strings = []
                    acc.token_strings.append(chunk.token_strings)
                acc.total_rows += K * n_heads
                self._captured_payloads += 1
                self._captured_rows += K * n_heads

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
        snapshot: Dict[str, Union[int, float]] = {
            "capture_invocations": self._capture_invocations,
            "capture_time_s": self._capture_time_s,
            "flush_invocations": self._flush_invocations,
            "flush_time_s": self._flush_time_s,
            "captured_payloads": self._captured_payloads,
            "captured_rows": self._captured_rows,
            "pending_configs": len(self._pending_captures),
            "accumulated_configs": len(self._accumulated),
        }
        self._capture_invocations = 0
        self._capture_time_s = 0.0
        self._flush_invocations = 0
        self._flush_time_s = 0.0
        self._captured_payloads = 0
        self._captured_rows = 0
        return snapshot


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
