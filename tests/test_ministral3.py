from __future__ import annotations

import pyarrow.parquet as pq
import torch
from transformers.models.ministral3.configuration_ministral3 import Ministral3Config

import models.ministral3.modeling_ministral3 as modeling_ministral3
from sniffer import (
    AllSampler,
    Sniffer,
    SnifferConfig,
    activate_sniffer,
    compute_positions,
)


ROPE_PARAMS = {
    "rope_type": "default",
    "rope_theta": 10000.0,
    "llama_4_scaling_beta": 0.5,
    "original_max_position_embeddings": 128,
}


def _make_config(num_hidden_layers=1, **overrides):
    defaults = dict(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=num_hidden_layers,
        rope_parameters=ROPE_PARAMS,
    )
    defaults.update(overrides)
    cfg = Ministral3Config(**defaults)
    cfg._attn_implementation = "eager"
    return cfg


def test_ministral3_attention_invokes_sniffer_capture(monkeypatch):
    calls = []

    class DummySniffer:
        class _Config:
            capture_pre_rope = False

        config = _Config()

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
    monkeypatch.setattr(modeling_ministral3, "get_active_sniffer", lambda: dummy)

    def fake_positions(*, batch_size, seq_len, device, cache_position):
        base = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        return base

    monkeypatch.setattr(modeling_ministral3, "compute_positions", fake_positions)

    config = _make_config()

    attention = modeling_ministral3.Ministral3Attention(config, layer_idx=0)
    seq_len = 3
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    cos = torch.ones(1, seq_len, attention.head_dim)
    sin = torch.zeros(1, seq_len, attention.head_dim)
    cache_position = torch.arange(seq_len)

    attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=cache_position,
    )

    assert len(calls) == 1
    entry = calls[0]
    assert entry["layer_idx"] == 0
    assert entry["query_shape"] == (1, config.num_attention_heads, seq_len, attention.head_dim)
    assert entry["key_shape"] == (1, config.num_key_value_heads, seq_len, attention.head_dim)
    assert entry["positions_shape"] == (1, seq_len)
    assert entry["sliding_window"] is None


def test_ministral3_layers_produce_captures(tmp_path):
    """Run a tiny Ministral3 through the real sniffer and verify parquet output."""
    num_layers = 2
    config = _make_config(num_hidden_layers=num_layers)

    layers = [modeling_ministral3.Ministral3Attention(config, layer_idx=i) for i in range(num_layers)]

    sniffer_config = SnifferConfig(
        model_name="ministral3-test",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AllSampler(),
        min_bucket_size=1,
    )

    seq_len = 5
    batch = 1
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    cos = torch.ones(batch, seq_len, config.head_dim)
    sin = torch.zeros(batch, seq_len, config.head_dim)
    cache_position = torch.arange(seq_len)

    with activate_sniffer(sniffer_config):
        for layer in layers:
            layer(
                hidden_states=hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=None,
                past_key_values=None,
                cache_position=cache_position,
            )

    base = tmp_path / "data"
    parquets = list(base.glob("l*h*/data.parquet"))
    assert len(parquets) > 0, "No parquet files written — no heads were sniffed"

    # Verify at least one file has the expected number of rows
    table = pq.read_table(parquets[0])
    assert table.num_rows == seq_len
    assert "vector" in table.schema.names
    assert "position" in table.schema.names
    assert "sliding_window" in table.schema.names
