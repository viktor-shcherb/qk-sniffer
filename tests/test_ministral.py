from __future__ import annotations

import pyarrow.parquet as pq
import torch
from transformers.models.ministral.configuration_ministral import MinistralConfig

import models.ministral.modeling_ministral as modeling_ministral
from sniffer import (
    AllSampler,
    Sniffer,
    SnifferConfig,
    activate_sniffer,
    compute_positions,
)


def test_ministral_attention_invokes_sniffer_capture(monkeypatch):
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
    monkeypatch.setattr(modeling_ministral, "get_active_sniffer", lambda: dummy)

    def fake_positions(*, batch_size, seq_len, device, cache_position):
        base = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        return base

    monkeypatch.setattr(modeling_ministral, "compute_positions", fake_positions)

    config = MinistralConfig(
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

    attention = modeling_ministral.MinistralAttention(config, layer_idx=0)
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


def test_ministral_mixed_layers_produce_captures(tmp_path):
    """Run a tiny Ministral with mixed layer types through the real sniffer
    and verify that at least some heads are captured to parquet."""
    num_layers = 4
    layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]

    config = MinistralConfig(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=num_layers,
        layer_types=layer_types,
        sliding_window=64,
    )
    config._attn_implementation = "eager"

    layers = [modeling_ministral.MinistralAttention(config, layer_idx=i) for i in range(num_layers)]

    sniffer_config = SnifferConfig(
        model_name="ministral-test",
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

    with activate_sniffer(sniffer_config):
        for i, layer in enumerate(layers):
            layer(
                hidden_states=hidden_states,
                position_embeddings=(cos, sin),
                attention_mask=None,
                past_key_values=None,
                cache_position=None,
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
