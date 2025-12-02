from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Union

import numpy as np
import torch

from saver.dataset import CaptureRow, DatasetSaver
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


class Sniffer:
    def __init__(self, config: SnifferConfig):
        self.config = config
        self.saver = DatasetSaver(root=config.data_root, readme_path=config.readme_path)
        self.saver.register_model_metadata(self.config.model_name, self.config.metadata)
        self._example_ids: Optional[Sequence[int]] = None
        self.sampler: Sampler = (
            config.sampler_factory()
            if config.sampler_factory is not None
            else LogUniformSampler()
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

        with torch.no_grad():
            batch_size, num_heads, seq_len, _ = query_states.shape
            device = query_states.device
            example_ids = self._prepare_example_ids(batch_size, seq_len, device)
            buckets = torch.floor(torch.log2(positions.to(torch.float32) + 1)).to(torch.int64)

            rows: list[CaptureRow] = []
            if self.config.capture_queries:
                rows.extend(
                    self._rows_from_tensor(
                        layer_idx=layer_idx,
                        vector_kind="q",
                        tensor=query_states,
                        example_ids=example_ids,
                        positions=positions,
                        buckets=buckets,
                        sliding_window=sliding_window,
                    )
                )
            if self.config.capture_keys:
                rows.extend(
                    self._rows_from_tensor(
                        layer_idx=layer_idx,
                        vector_kind="k",
                        tensor=key_states,
                        example_ids=example_ids,
                        positions=positions,
                        buckets=buckets,
                        sliding_window=sliding_window,
                    )
                )

            if rows:
                self.saver.add_many(rows)

    def close(self) -> None:
        self.saver.close()

    def __enter__(self) -> "Sniffer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def set_example_ids(self, example_ids: Sequence[int]) -> None:
        self._example_ids = list(example_ids)

    def _rows_from_tensor(
        self,
        layer_idx: int,
        vector_kind: str,
        tensor: torch.Tensor,
        example_ids: torch.Tensor,
        positions: torch.Tensor,
        buckets: torch.Tensor,
        sliding_window: Optional[int],
    ) -> list[CaptureRow]:
        heads_filter = self.config.heads
        tensor_cpu = tensor.detach().to("cpu", dtype=torch.float32).numpy()
        example_cpu = example_ids.detach().to("cpu").numpy()
        position_cpu = positions.detach().to("cpu").numpy()
        bucket_cpu = buckets.detach().to("cpu").numpy()

        batch_size, num_heads, seq_len, head_dim = tensor_cpu.shape
        rows: list[CaptureRow] = []
        for head_idx in range(num_heads):
            if heads_filter is not None and head_idx not in heads_filter:
                continue
            for batch_idx in range(batch_size):
                example_id = int(example_cpu[batch_idx, 0])
                pos_slice = position_cpu[batch_idx]
                bucket_slice = bucket_cpu[batch_idx]
                mask = self.sampler.sample_positions(
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    vector_kind=vector_kind,
                    example_id=example_id,
                    positions=pos_slice,
                    buckets=bucket_slice,
                )
                if not mask.any():
                    continue
                vectors = tensor_cpu[batch_idx, head_idx]
                for pos_idx, keep in enumerate(mask):
                    if not keep:
                        continue
                    rows.append(
                        CaptureRow(
                            model_name=self.config.model_name,
                            layer_idx=layer_idx,
                            head_idx=head_idx,
                            vector_kind=vector_kind,  # type: ignore[arg-type]
                            bucket=int(bucket_slice[pos_idx]),
                            example_id=example_id,
                            position=int(pos_slice[pos_idx]),
                            vector=np.copy(vectors[pos_idx]),
                            sliding_window=sliding_window,
                        )
                    )
        return rows

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


_ACTIVE_SNIFFER: Optional[Sniffer] = None


def get_active_sniffer() -> Optional[Sniffer]:
    return _ACTIVE_SNIFFER


def set_active_example_ids(example_ids: Sequence[int]) -> None:
    sniffer = get_active_sniffer()
    if sniffer is None:
        raise RuntimeError("No active sniffer session to attach example ids.")
    sniffer.set_example_ids(example_ids)


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
