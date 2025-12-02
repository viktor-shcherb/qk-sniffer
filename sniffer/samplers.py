from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


class Sampler(ABC):
    @abstractmethod
    def sample_positions(
        self,
        *,
        layer_idx: int,
        head_idx: int,
        vector_kind: str,
        example_id: int,
        positions: np.ndarray,
        buckets: np.ndarray,
    ) -> np.ndarray:
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
        positions: np.ndarray,
        buckets: np.ndarray,
    ) -> np.ndarray:
        bucket_array = np.asarray(buckets, dtype=np.int64)
        # bucket i spans [2^i, 2^{i+1}), so its theoretical size is 2^i
        bucket_sizes = np.power(2.0, np.maximum(bucket_array, 0))
        probabilities = np.minimum(1.0, self.base_rate / np.maximum(bucket_sizes, 1.0))
        rng = np.random.default_rng(_seed(example_id, layer_idx, head_idx, vector_kind))
        random_values = rng.random(size=probabilities.shape)
        return random_values < probabilities
