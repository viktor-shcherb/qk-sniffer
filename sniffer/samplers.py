from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Optional, Sequence

import torch

_UINT64_MASK = 0xFFFFFFFFFFFFFFFF


def _mix64(value: int) -> int:
    """Avalanche a 64-bit integer (SplitMix64 finalizer)."""
    value &= _UINT64_MASK
    value ^= value >> 30
    value = (value * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value ^= value >> 27
    value = (value * 0x94D049BB133111EB) & _UINT64_MASK
    value ^= value >> 31
    return value


def _combine_seed(seed: int, value: int) -> int:
    seed ^= value
    seed = (seed * 0x9E3779B97F4A7C15) & _UINT64_MASK
    seed ^= seed >> 32
    return seed & _UINT64_MASK


class Sampler(ABC):
    _cached_generator: Optional[torch.Generator] = None
    _cached_generator_device: Optional[torch.device] = None

    def _get_generator(self, device: torch.device, seed: int) -> torch.Generator:
        """Return a reusable Generator for *device*, seeded with *seed*."""
        if self._cached_generator is None or self._cached_generator_device != device:
            self._cached_generator = torch.Generator(device=device)
            self._cached_generator_device = device
        self._cached_generator.manual_seed(seed)
        return self._cached_generator

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

    def sample_positions_batch(
        self,
        *,
        layer_idx: int,
        head_indices: Sequence[int],
        vector_kind: str,
        example_id: int,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        """Generate a single shared mask for all heads. Returns (seq_len,) bool tensor.

        Default implementation calls sample_positions for the first head.
        Subclasses can override for vectorized generation.
        """
        return self.sample_positions(
            layer_idx=layer_idx,
            head_idx=head_indices[0],
            vector_kind=vector_kind,
            example_id=example_id,
            positions=positions,
            buckets=buckets,
        )


def _seed(example_id: int, layer_idx: int, head_idx: int, vector_kind: str) -> int:
    """Generate a deterministic 64-bit seed for RNGs without bit collisions."""

    components = (
        _mix64(int(example_id)),
        _mix64(int(layer_idx)),
        _mix64(int(head_idx)),
        _mix64(ord(vector_kind[0]) & 0xFF),
    )
    seed = 0x243F6A8885A308D3  # non-zero initializer from ChaCha constants
    for value in components:
        seed = _combine_seed(seed, value)
    return seed & _UINT64_MASK


def _batch_seed(example_id: int, layer_idx: int, vector_kind: str) -> int:
    """Generate a deterministic 64-bit seed shared across heads for batch RNGs."""
    components = (
        _mix64(int(example_id)),
        _mix64(int(layer_idx)),
        _mix64(ord(vector_kind[0]) & 0xFF),
    )
    seed = 0x243F6A8885A308D3
    for value in components:
        seed = _combine_seed(seed, value)
    return seed & _UINT64_MASK


@dataclass(slots=True)
class LogUniformSampler(Sampler):
    base_rate: float = 1.0
    min_bucket_size: int = 128
    bucket_kind: str = "log"

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

    def sample_positions_batch(
        self,
        *,
        layer_idx: int,
        head_indices: Sequence[int],
        vector_kind: str,
        example_id: int,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        if buckets.ndim != 1:
            raise ValueError("Buckets must be a 1D tensor.")
        _ = (positions, head_indices)
        device = buckets.device
        bucket_tensor = buckets.to(device=device, dtype=torch.float32)
        bucket_tensor = torch.clamp(bucket_tensor, min=self._min_bucket_floor)
        bucket_sizes = torch.pow(2.0, bucket_tensor)
        denominator = torch.clamp(bucket_sizes, min=1.0)
        probabilities = torch.minimum(
            torch.ones_like(denominator),
            torch.tensor(self.base_rate, device=device, dtype=torch.float32) / denominator,
        )
        generator = self._get_generator(device, _batch_seed(example_id, layer_idx, vector_kind))
        random_values = torch.rand(probabilities.shape, generator=generator, device=device, dtype=torch.float32)
        return random_values < probabilities


@dataclass(slots=True)
class UniformSampler(Sampler):
    base_rate: float = 1.0
    bucket_size: int = 128
    bucket_kind: str = "uniform"

    def __post_init__(self) -> None:
        if self.bucket_size < 1:
            raise ValueError("bucket_size must be at least 1.")

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
        if positions.ndim != 1:
            raise ValueError("Positions must be a 1D tensor.")
        device = positions.device
        denominator = float(self.bucket_size)
        probability = min(1.0, float(self.base_rate) / denominator)
        probs = torch.full_like(positions, fill_value=probability, dtype=torch.float32, device=device)
        generator = torch.Generator(device=device)
        generator.manual_seed(_seed(example_id, layer_idx, head_idx, vector_kind))
        random_values = torch.rand(probs.shape, generator=generator, device=device, dtype=torch.float32)
        return random_values < probs

    def sample_positions_batch(
        self,
        *,
        layer_idx: int,
        head_indices: Sequence[int],
        vector_kind: str,
        example_id: int,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        if positions.ndim != 1:
            raise ValueError("Positions must be a 1D tensor.")
        _ = (head_indices, buckets)
        seq_len = positions.shape[0]
        device = positions.device
        probability = min(1.0, float(self.base_rate) / float(self.bucket_size))
        generator = self._get_generator(device, _batch_seed(example_id, layer_idx, vector_kind))
        random_values = torch.rand(seq_len, generator=generator, device=device, dtype=torch.float32)
        return random_values < probability


@dataclass(slots=True)
class AllSampler(Sampler):
    bucket_kind: str = "all"

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
        _ = (layer_idx, head_idx, vector_kind, example_id, buckets)
        return torch.ones_like(positions, dtype=torch.bool)

    def sample_positions_batch(
        self,
        *,
        layer_idx: int,
        head_indices: Sequence[int],
        vector_kind: str,
        example_id: int,
        positions: torch.Tensor,
        buckets: torch.Tensor,
    ) -> torch.Tensor:
        _ = (layer_idx, head_indices, vector_kind, example_id, buckets)
        return torch.ones_like(positions, dtype=torch.bool)
