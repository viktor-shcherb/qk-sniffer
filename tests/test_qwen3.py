from __future__ import annotations

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

import models.qwen3.modeling_qwen3 as modeling_qwen3


def test_qwen3_attention_invokes_sniffer_capture(monkeypatch):
    calls = []

    class DummySniffer:
        def capture(self, *, layer_idx, query_states, key_states, positions, sliding_window):
            calls.append(
                {
                    "layer_idx": layer_idx,
                    "query_shape": tuple(query_states.shape),
                    "key_shape": tuple(key_states.shape),
                    "positions_shape": tuple(positions.shape),
                    "sliding_window": sliding_window,
                }
            )

    dummy = DummySniffer()
    monkeypatch.setattr(modeling_qwen3, "get_active_sniffer", lambda: dummy)

    def fake_positions(*, batch_size, seq_len, device, cache_position):
        base = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        return base

    monkeypatch.setattr(modeling_qwen3, "compute_positions", fake_positions)

    config = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        sliding_window=64,
    )
    config._attn_implementation = "eager"
    attention = modeling_qwen3.Qwen3Attention(config, layer_idx=0)
    attention.sliding_window = 64
    hidden_states = torch.randn(1, 3, config.hidden_size)
    cos = torch.ones(1, 3, attention.head_dim)
    sin = torch.zeros(1, 3, attention.head_dim)

    attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
    )

    assert len(calls) == 1
    entry = calls[0]
    assert entry["layer_idx"] == 0
    assert entry["query_shape"] == (1, config.num_attention_heads, hidden_states.shape[1], attention.head_dim)
    assert entry["key_shape"] == entry["query_shape"]
    assert entry["positions_shape"] == (1, hidden_states.shape[1])
    assert entry["sliding_window"] == 64
