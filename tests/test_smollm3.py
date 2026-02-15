from __future__ import annotations

import pyarrow.parquet as pq
import torch
from transformers.models.smollm3.configuration_smollm3 import SmolLM3Config

import models.smollm3.modeling_smollm3 as modeling_smollm3
from sniffer import (
    AllSampler,
    SnifferConfig,
    activate_sniffer,
)


def _make_config(num_hidden_layers=1, **overrides):
    defaults = dict(
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        num_hidden_layers=num_hidden_layers,
    )
    defaults.update(overrides)
    cfg = SmolLM3Config(**defaults)
    cfg._attn_implementation = "eager"
    return cfg


def _dummy_sniffer_and_recorder(monkeypatch, *, capture_pre_rope=False):
    calls = []

    class DummySniffer:
        class _Config:
            pass
        _Config.capture_pre_rope = capture_pre_rope
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
    monkeypatch.setattr(modeling_smollm3, "get_active_sniffer", lambda: dummy)

    def fake_positions(*, batch_size, seq_len, device, cache_position):
        return torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)

    monkeypatch.setattr(modeling_smollm3, "compute_positions", fake_positions)
    return calls


def test_smollm3_attention_invokes_sniffer_capture(monkeypatch):
    calls = _dummy_sniffer_and_recorder(monkeypatch)

    config = _make_config(no_rope_layers=[1])
    attention = modeling_smollm3.SmolLM3Attention(config, layer_idx=0)

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


def test_smollm3_no_rope_layer_captures(monkeypatch):
    """Layer with use_rope=0 (NoPE) should still invoke sniffer capture."""
    calls = _dummy_sniffer_and_recorder(monkeypatch)

    config = _make_config(no_rope_layers=[0])  # layer 0 has no RoPE
    attention = modeling_smollm3.SmolLM3Attention(config, layer_idx=0)
    assert not attention.use_rope

    seq_len = 4
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


def test_smollm3_capture_pre_rope(monkeypatch):
    """When capture_pre_rope=True, capture should happen before RoPE is applied."""
    calls = _dummy_sniffer_and_recorder(monkeypatch, capture_pre_rope=True)

    config = _make_config(no_rope_layers=[1])
    attention = modeling_smollm3.SmolLM3Attention(config, layer_idx=0)

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


def test_smollm3_sliding_window_passed_to_sniffer(monkeypatch):
    """Layers with sliding attention should pass sliding_window to sniffer."""
    calls = _dummy_sniffer_and_recorder(monkeypatch)

    config = _make_config(
        no_rope_layers=[0],  # NoPE layer → sliding_attention when use_sliding_window=True
        use_sliding_window=True,
        sliding_window=64,
        layer_types=["sliding_attention"],
    )
    attention = modeling_smollm3.SmolLM3Attention(config, layer_idx=0)
    assert attention.sliding_window == 64

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
    assert calls[0]["sliding_window"] == 64


def test_smollm3_no_capture_without_sniffer(monkeypatch):
    """Without an active sniffer, forward should still work normally."""
    monkeypatch.setattr(modeling_smollm3, "get_active_sniffer", lambda: None)

    config = _make_config(no_rope_layers=[1])
    attention = modeling_smollm3.SmolLM3Attention(config, layer_idx=0)

    seq_len = 3
    hidden_states = torch.randn(1, seq_len, config.hidden_size)
    cos = torch.ones(1, seq_len, attention.head_dim)
    sin = torch.zeros(1, seq_len, attention.head_dim)

    output, _ = attention(
        hidden_states=hidden_states,
        position_embeddings=(cos, sin),
        attention_mask=None,
        past_key_values=None,
        cache_position=None,
    )

    assert output.shape == hidden_states.shape


def test_smollm3_layers_produce_captures(tmp_path):
    """Run a tiny SmolLM3 through the real sniffer and verify parquet output."""
    num_layers = 4
    # Layers 0,1,2 have RoPE (1), layer 3 is NoPE (0) — matches default interval=4
    config = _make_config(
        num_hidden_layers=num_layers,
        no_rope_layers=[1, 1, 1, 0],
    )

    layers = [modeling_smollm3.SmolLM3Attention(config, layer_idx=i) for i in range(num_layers)]

    sniffer_config = SnifferConfig(
        model_name="smollm3-test",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AllSampler(),
        min_bucket_size=1,
    )

    seq_len = 5
    batch = 1
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    cos = torch.ones(batch, seq_len, config.hidden_size // config.num_attention_heads)
    sin = torch.zeros(batch, seq_len, config.hidden_size // config.num_attention_heads)
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

    table = pq.read_table(parquets[0])
    assert table.num_rows == seq_len
    assert "vector" in table.schema.names
    assert "position" in table.schema.names
    assert "sliding_window" in table.schema.names
