from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math

import torch


class Sampler(ABC):
    @abstractmethod
    def sample_positions(
        self,
        *,
        layer_idx: int,
        head_idx: int,
        vector_kind: str,
        example_id: int,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        ...


def _seed(example_id: int, layer_idx: int, head_idx: int, vector_kind: str) -> int:
    seed = (
        (example_id & 0xFFFFFFFF)
        ^ ((layer_idx & 0xFFFFFFFF) << 8)
        ^ ((head_idx & 0xFFFFFFFF) << 16)
        ^ (ord(vector_kind[0]) << 24)
    )
    seed ^= 0x9E3779B97F4A7C15
    return seed & 0xFFFFFFFFFFFFFFFF


@dataclass(slots=True)
class LogUniformSampler(Sampler):
    base_rate: float = 1.0
    min_bucket_size: int = 128

    def __post_init__(self) -> None:
        min_bucket_size = int(self.min_bucket_size)
        if min_bucket_size < 1:
            raise ValueError("min_bucket_size must be at least 1.")
        if min_bucket_size == 1:
            self._min_bucket_power = 0
        else:
            self._min_bucket_power = int(math.ceil(math.log2(min_bucket_size)))
        self._min_bucket_size = 1 << self._min_bucket_power
        self.min_bucket_size = self._min_bucket_size
        self._min_bucket_floor = float(self._min_bucket_power)

    def sample_positions(
        self,
        *,
        layer_idx: int,
        head_idx: int,
        vector_kind: str,
        example_id: int,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        if buckets.ndim != 1:
            raise ValueError("Buckets must be a 1D tensor.")
        # positions are only used for interface compatibility; sampler only depends on bucket widths
        _ = positions
        device = buckets.device
        bucket_tensor = buckets.to(device=device, dtype=torch.float32)
        bucket_tensor = torch.clamp(bucket_tensor, min=self._min_bucket_floor)
        # bucket i spans 2^i positions; clamp enforces minimum bucket width
        bucket_sizes = torch.pow(2.0, bucket_tensor)
        denominator = torch.clamp(bucket_sizes, min=1.0)
        probabilities = torch.minimum(
            torch.ones_like(denominator),
            torch.tensor(self.base_rate, device=device, dtype=torch.float32) / denominator,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(_seed(example_id, layer_idx, head_idx, vector_kind))
        random_values = torch.rand(probabilities.shape, generator=generator, device=device, dtype=torch.float32)
        return random_values < probabilities
