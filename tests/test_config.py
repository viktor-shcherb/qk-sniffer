from pathlib import Path

import sniff


def test_load_config_instantiates_sampler(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  path: foo/bar
  split: train
model:
  name: foo/model
tokenizer:
  name: foo/model
inference:
  batch_size: 1
capture:
  capture_queries: true
  capture_keys: true
  sampler:
    type: log_uniform
    base_rate: 0.25
output:
  data_root: data
""",
        encoding="utf-8",
    )

    config = sniff.load_config(config_path)
    assert isinstance(config.capture.sampler, sniff.SamplerSettings)
    assert config.capture.sampler.base_rate == 0.25


def test_load_config_instantiates_head_sampling(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  path: foo/bar
  split: train
model:
  name: foo/model
tokenizer:
  name: foo/model
inference:
  batch_size: 1
capture:
  capture_queries: true
  capture_keys: true
  full_attention_only: true
  head_sampling:
    count: 300
    seed: 17
output:
  data_root: data
""",
        encoding="utf-8",
    )

    config = sniff.load_config(config_path)
    assert isinstance(config.capture.head_sampling, sniff.HeadSamplingSettings)
    assert config.capture.head_sampling.count == 300
    assert config.capture.head_sampling.seed == 17
    assert config.capture.full_attention_only is True


def test_resolve_readme_path_relative():
    settings = sniff.OutputSettings(data_root="data/sniffed-qk", readme_path="README.md")
    resolved = sniff.resolve_readme_path(settings)
    assert resolved == Path("data/sniffed-qk/README.md")


def test_resolve_readme_path_absolute(tmp_path):
    readme = tmp_path / "README.md"
    settings = sniff.OutputSettings(data_root="data/sniffed-qk", readme_path=str(readme))
    resolved = sniff.resolve_readme_path(settings)
    assert resolved == Path("data/sniffed-qk/README.md")


def test_load_config_ignores_removed_distributed_fields(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  path: foo/bar
  split: train
model:
  name: foo/model
tokenizer:
  name: foo/model
inference:
  batch_size: 4
  autocast_dtype: bfloat16
  distributed: true
  backend: nccl
capture:
  capture_queries: true
  capture_keys: true
output:
  data_root: data
""",
        encoding="utf-8",
    )

    config = sniff.load_config(config_path)
    assert config.inference.batch_size == 4
    assert not hasattr(config.inference, "distributed")


def test_load_config_parses_debug_inference_settings(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
dataset:
  path: foo/bar
  split: train
model:
  name: foo/model
tokenizer:
  name: foo/model
inference:
  batch_size: 2
  debug_logging: true
  debug_log_every_n_batches: 5
  mps_cleanup_every_batches: 7
capture:
  capture_queries: true
  capture_keys: true
output:
  data_root: data
""",
        encoding="utf-8",
    )

    config = sniff.load_config(config_path)
    assert config.inference.debug_logging is True
    assert config.inference.debug_log_every_n_batches == 5
    assert config.inference.mps_cleanup_every_batches == 7
