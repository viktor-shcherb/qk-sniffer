from __future__ import annotations

import torch
from transformers.models.glm.configuration_glm import GlmConfig

import models.glm.modeling_glm as modeling_glm


def test_glm_attention_invokes_sniffer_capture(monkeypatch):
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
    monkeypatch.setattr(modeling_glm, "get_active_sniffer", lambda: dummy)

    def fake_positions(*, batch_size, seq_len, device, cache_position):
        base = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        return base

    monkeypatch.setattr(modeling_glm, "compute_positions", fake_positions)

    config = GlmConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        head_dim=8,
    )
    config._attn_implementation = "eager"

    attention = modeling_glm.GlmAttention(config, layer_idx=0)
    hidden_states = torch.randn(1, 3, config.hidden_size)
    cos = torch.ones(1, hidden_states.shape[1], attention.head_dim)
    sin = torch.zeros(1, hidden_states.shape[1], attention.head_dim)

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
    assert entry["key_shape"] == (1, config.num_key_value_heads, hidden_states.shape[1], attention.head_dim)
    assert entry["positions_shape"] == (1, hidden_states.shape[1])
    assert entry["sliding_window"] is None
