from __future__ import annotations

import re
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
import yaml

from sniffer import (
    LogUniformSampler,
    UniformSampler,
    Sampler,
    Sniffer,
    SnifferConfig,
    activate_sniffer,
    compute_positions,
    get_active_sniffer,
    set_active_example_ids,
    set_active_sequence_lengths,
)
from sniffer.samplers import _seed


class AlwaysSampler(Sampler):
    bucket_kind = "log"

    def sample_positions(self, **kwargs):
        positions: torch.Tensor = kwargs["positions"]
        return torch.ones_like(positions, dtype=torch.bool)


class BucketZeroSampler(Sampler):
    bucket_kind = "log"

    def sample_positions(self, **kwargs):
        buckets: torch.Tensor = kwargs["buckets"]
        return buckets == 0


class UniformBucketSampler(Sampler):
    bucket_kind = "uniform"

    def sample_positions(self, **kwargs):
        positions: torch.Tensor = kwargs["positions"]
        return torch.ones_like(positions, dtype=torch.bool)


def test_sniffer_captures_qk(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)

    query_states = torch.ones(1, 2, 3, 4)
    key_states = torch.ones(1, 2, 3, 4) * 2
    positions = compute_positions(batch_size=1, seq_len=3, device=query_states.device, cache_position=None)
    sniffer.capture(
        layer_idx=0,
        query_states=query_states,
        key_states=key_states,
        positions=positions,
        sliding_window=None,
    )
    sniffer.close()

    base = tmp_path / "data" / "meta_llama3_8b"
    q_path = base / "l00h00q" / "data.parquet"
    k_path = base / "l00h00k" / "data.parquet"

    assert q_path.exists()
    assert k_path.exists()

    q_table = pq.read_table(q_path)
    k_table = pq.read_table(k_path)

    # batch=1, seq=3 => 3 entries per head per vector kind
    assert q_table.num_rows == 3
    assert k_table.num_rows == 3
    assert q_table.column("bucket").to_pylist()[0] == 0


def test_sniffer_respects_cache_position(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        capture_keys=False,
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)

    query_states = torch.ones(1, 1, 2, 4)
    cache_position = torch.tensor([10], dtype=torch.int64)
    positions = compute_positions(batch_size=1, seq_len=2, device=query_states.device, cache_position=cache_position)
    sniffer.capture(
        layer_idx=1,
        query_states=query_states,
        key_states=query_states,
        positions=positions,
        sliding_window=128,
    )
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l01h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("position").to_pylist() == [10, 11]
    assert table.column("sliding_window").to_pylist() == [128, 128]


def test_sniffer_custom_example_ids(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    sniffer.set_example_ids([42])
    query_states = torch.ones(1, 1, 1, 4)
    positions = compute_positions(batch_size=1, seq_len=1, device=query_states.device, cache_position=None)
    sniffer.capture(
        layer_idx=0,
        query_states=query_states,
        key_states=query_states,
        positions=positions,
        sliding_window=None,
    )
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("example_id").to_pylist() == [42]


def test_sniffer_respects_sequence_lengths(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    sniffer.set_sequence_lengths([2])

    states = torch.randn(1, 1, 4, 4)
    positions = compute_positions(batch_size=1, seq_len=4, device=states.device, cache_position=None)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.num_rows == 2
    assert table.column("position").to_pylist() == [0, 1]


def test_sniffer_max_rows_per_batch(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        max_rows_per_batch=1,
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    states = torch.randn(1, 1, 4, 4)
    positions = compute_positions(batch_size=1, seq_len=4, device=states.device, cache_position=None)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.num_rows == 1


def test_sniffer_respects_layer_and_head_filters(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        layers={1},
        heads={0},
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    states = torch.randn(1, 2, 2, 4)
    positions = compute_positions(batch_size=1, seq_len=2, device=states.device, cache_position=None)

    # layer 0 should be ignored entirely
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.capture(layer_idx=1, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    base = tmp_path / "data" / "meta_llama3_8b"
    assert not (base / "l00h00q").exists()
    assert (base / "l01h00q").exists()
    assert not (base / "l01h01q").exists()


def test_sniffer_toggle_queries_vs_keys(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        capture_queries=False,
        capture_keys=True,
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    states = torch.randn(1, 1, 1, 4)
    positions = compute_positions(batch_size=1, seq_len=1, device=states.device, cache_position=None)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    base = tmp_path / "data" / "meta_llama3_8b"
    assert not (base / "l00h00q").exists()
    assert (base / "l00h00k").exists()


def test_sniffer_example_id_mismatch_raises(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    sniffer.set_example_ids([1, 2])
    states = torch.randn(1, 1, 1, 4)
    positions = compute_positions(batch_size=1, seq_len=1, device=states.device, cache_position=None)
    with pytest.raises(ValueError):
        sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)


def test_set_active_example_ids_updates_current_session(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    with activate_sniffer(config) as sniffer:
        assert get_active_sniffer() is sniffer
        set_active_example_ids([99])
        states = torch.ones(1, 1, 1, 4)
        positions = compute_positions(batch_size=1, seq_len=1, device=states.device, cache_position=None)
        sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("example_id").to_pylist() == [99]


def test_set_active_sequence_lengths_updates_current_session(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=1,
    )
    with activate_sniffer(config) as sniffer:
        set_active_sequence_lengths([1])
        states = torch.ones(1, 1, 3, 4)
        positions = compute_positions(batch_size=1, seq_len=3, device=states.device, cache_position=None)
        sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("position").to_pylist() == [0]


def test_custom_sampler_drops_non_zero_buckets(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: BucketZeroSampler(),
        min_bucket_size=1,
    )
    sniffer = Sniffer(config)
    states = torch.ones(1, 1, 3, 4)
    positions = torch.tensor([[0, 256, 400]], dtype=torch.int64, device=states.device)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    base = tmp_path / "data" / "meta_llama3_8b"
    q_path = base / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.num_rows == 1  # only bucket 0 kept


def test_sniffer_default_min_bucket_size_groups_positions(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
    )
    sniffer = Sniffer(config)
    states = torch.ones(1, 1, 5, 4)
    positions = torch.tensor([[0, 1, 127, 128, 384]], dtype=torch.int64, device=states.device)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("bucket").to_pylist() == [7, 7, 7, 7, 8]


def test_sniffer_custom_min_bucket_size(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=4,
    )
    sniffer = Sniffer(config)
    states = torch.ones(1, 1, 4, 4)
    positions = torch.tensor([[0, 3, 4, 12]], dtype=torch.int64, device=states.device)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("bucket").to_pylist() == [2, 2, 2, 3]


def test_sniffer_min_bucket_size_rounds_up(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: AlwaysSampler(),
        min_bucket_size=130,
    )
    sniffer = Sniffer(config)
    states = torch.ones(1, 1, 3, 4)
    positions = torch.tensor([[0, 200, 511]], dtype=torch.int64, device=states.device)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("bucket").to_pylist() == [8, 8, 9]


def test_log_uniform_sampler_deterministic():
    sampler = LogUniformSampler(base_rate=1.0, min_bucket_size=1)
    buckets = torch.tensor([0, 1, 1, 2], dtype=torch.int64)
    positions = torch.arange(4, dtype=torch.int64)
    mask1 = sampler.sample_positions(
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        example_id=7,
        positions=positions,
        buckets=buckets,
    )
    mask2 = sampler.sample_positions(
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        example_id=7,
        positions=positions,
        buckets=buckets,
    )
    assert torch.equal(mask1, mask2)


def test_log_uniform_sampler_bucket_size_scaling():
    sampler = LogUniformSampler(base_rate=4.0, min_bucket_size=1)
    buckets = torch.tensor([0, 1, 1, 2, 2, 2, 2], dtype=torch.int64)
    positions = torch.arange(len(buckets), dtype=torch.int64)
    mask = sampler.sample_positions(
        layer_idx=0,
        head_idx=0,
        vector_kind="k",
        example_id=3,
        positions=positions,
        buckets=buckets,
    )
    # base_rate equals bucket_size up to bucket 2, so every entry kept deterministically
    assert mask.all().item()


def test_sampler_seed_avoids_periodic_head_collisions():
    a = _seed(example_id=0, layer_idx=0, head_idx=1, vector_kind="q")
    b = _seed(example_id=1 << 16, layer_idx=0, head_idx=0, vector_kind="q")
    assert a != b


def test_log_uniform_sampler_rejects_invalid_min_bucket_size():
    with pytest.raises(ValueError, match="min_bucket_size"):
        LogUniformSampler(base_rate=1.0, min_bucket_size=0)


def test_sniffer_uniform_bucket_strategy(tmp_path):
    config = SnifferConfig(
        model_name="meta/llama3-8b",
        data_root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        sampler_factory=lambda: UniformBucketSampler(),
        min_bucket_size=50,
    )
    sniffer = Sniffer(config)
    states = torch.ones(1, 1, 5, 4)
    positions = torch.tensor([[0, 10, 49, 50, 120]], dtype=torch.int64, device=states.device)
    sniffer.capture(layer_idx=0, query_states=states, key_states=states, positions=positions, sliding_window=None)
    sniffer.close()

    q_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(q_path)
    assert table.column("bucket").to_pylist() == [0, 0, 0, 1, 2]


def test_uniform_sampler_probability_clamps():
    sampler = UniformSampler(base_rate=64.0, bucket_size=32)
    positions = torch.arange(6, dtype=torch.int64)
    buckets = torch.zeros_like(positions)
    mask = sampler.sample_positions(
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        example_id=42,
        positions=positions,
        buckets=buckets,
    )
    assert mask.all().item()


def test_compute_positions_matches_reference():
    device = torch.device("cpu")
    cache_position = torch.tensor([5, 6, 7], dtype=torch.int64, device=device)
    positions = compute_positions(batch_size=2, seq_len=3, device=device, cache_position=cache_position)
    assert positions.shape == (2, 3)
    assert torch.equal(positions[0], cache_position)
    assert torch.equal(positions[1], cache_position)


def test_compute_positions_with_none_cache():
    device = torch.device("cpu")
    positions = compute_positions(batch_size=3, seq_len=4, device=device, cache_position=None)
    expected = torch.arange(4, device=device, dtype=torch.int64).unsqueeze(0).expand(3, -1)
    assert torch.equal(positions, expected)


def test_sniffer_multiple_models_share_dataset(tmp_path):
    data_root = tmp_path / "data"
    readme_path = tmp_path / "README.md"

    def run_capture(model_name: str, fill_value: float) -> None:
        config = SnifferConfig(
            model_name=model_name,
            data_root=data_root,
            readme_path=readme_path,
            sampler_factory=lambda: AlwaysSampler(),
            min_bucket_size=1,
        )
        sniffer = Sniffer(config)
        states = torch.full((1, 1, 1, 4), fill_value, dtype=torch.float32)
        positions = compute_positions(batch_size=1, seq_len=1, device=states.device, cache_position=None)
        sniffer.capture(
            layer_idx=0,
            query_states=states,
            key_states=states,
            positions=positions,
            sliding_window=None,
        )
        sniffer.close()

    run_capture("meta/llama3-8b", 1.0)
    run_capture("HuggingFaceTB/SmolLM2-360M", 2.0)

    def sanitized_split(name: str) -> str:
        return re.sub(r"\W", "_", name)

    for model_name in ("meta/llama3-8b", "HuggingFaceTB/SmolLM2-360M"):
        expected = data_root / sanitized_split(model_name) / "l00h00q" / "data.parquet"
        assert expected.exists()

    front_matter, _ = _read_front_matter(readme_path)
    recorded_models = {entry["name"] for entry in front_matter["models"]}
    assert "meta/llama3-8b" in recorded_models
    assert "HuggingFaceTB/SmolLM2-360M" in recorded_models

    configs = {entry["config_name"]: entry for entry in front_matter["configs"]}
    all_entry = configs["all"]
    splits = {item["split"] for item in all_entry["data_files"]}
    assert sanitized_split("meta/llama3-8b") in splits
    assert sanitized_split("HuggingFaceTB/SmolLM2-360M") in splits


def _read_front_matter(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---")
    _, front_raw, body = content.split("---", 2)
    return yaml.safe_load(front_raw.strip()), body
