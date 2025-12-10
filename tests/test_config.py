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


def test_resolve_readme_path_relative():
    settings = sniff.OutputSettings(data_root="data/sniffed-qk", readme_path="README.md")
    resolved = sniff.resolve_readme_path(settings)
    assert resolved == Path("data/sniffed-qk/README.md")


def test_resolve_readme_path_absolute(tmp_path):
    readme = tmp_path / "README.md"
    settings = sniff.OutputSettings(data_root="data/sniffed-qk", readme_path=str(readme))
    resolved = sniff.resolve_readme_path(settings)
    assert resolved == readme
