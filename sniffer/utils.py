from __future__ import annotations

import torch


def compute_positions(
    batch_size: int,
    seq_len: int,
    device: torch.device,
    cache_position: torch.Tensor | None,
) -> torch.Tensor:
    if cache_position is None:
        base = torch.arange(seq_len, device=device, dtype=torch.int64)
        return base.unsqueeze(0).expand(batch_size, -1)

    cache = cache_position.detach()
    if cache.ndim == 0:
        cache = cache.unsqueeze(0)
    if cache.ndim == 1:
        cache = cache.unsqueeze(0)
    if cache.ndim != 2:
        cache = cache.view(1, -1)

    if cache.shape[0] == 1 and batch_size > 1:
        cache = cache.expand(batch_size, -1)
    elif cache.shape[0] > batch_size:
        cache = cache[:batch_size]
    elif cache.shape[0] < batch_size:
        repeats = batch_size - cache.shape[0]
        cache = torch.cat([cache, cache[-1:].expand(repeats, -1)], dim=0)

    if cache.shape[1] < seq_len:
        pad = seq_len - cache.shape[1]
        increments = torch.arange(1, pad + 1, device=cache.device, dtype=torch.int64)
        extra = cache[:, -1:] + increments
        cache = torch.cat([cache, extra], dim=1)
    elif cache.shape[1] > seq_len:
        cache = cache[:, :seq_len]

    return cache.to(device=device, dtype=torch.int64)
