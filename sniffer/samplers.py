from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        # bucket i spans [2^i, 2^{i+1}), theoretical size is 2^i
        bucket_sizes = torch.pow(2.0, torch.clamp(bucket_tensor, min=0.0))
        denominator = torch.clamp(bucket_sizes, min=1.0)
        probabilities = torch.minimum(
            torch.ones_like(denominator),
            torch.tensor(self.base_rate, device=device, dtype=torch.float32) / denominator,
        )
        generator = torch.Generator(device=device)
        generator.manual_seed(_seed(example_id, layer_idx, head_idx, vector_kind))
        random_values = torch.rand(probabilities.shape, generator=generator, device=device, dtype=torch.float32)
        return random_values < probabilities
