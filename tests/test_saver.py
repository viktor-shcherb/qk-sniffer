from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import yaml
from datasets import load_dataset

from saver.dataset import CaptureBatch, CaptureRow, DatasetSaver


def test_dataset_saver_writes_parquet_and_readme(tmp_path):
    saver = DatasetSaver(
        root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        dataset_name="viktoroo/sniffed-qk-test",
    )

    saver.add_many(
        [
            _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1, 0.2]),
            _row("meta/llama3-8b", layer=0, head=0, kind="k", bucket=1, example_id=0, position=0, vector=[0.3, 0.4]),
            _row("google/gemma3-9b", layer=1, head=2, kind="q", bucket=2, example_id=1, position=5, vector=[0.5, 0.6]),
        ]
    )
    saver.close()

    parquet_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    assert table.num_rows == 1
    assert table.schema.names == ["bucket", "example_id", "position", "vector", "sliding_window"]
    assert table.column("bucket").to_pylist() == [0]
    assert table.column("position").to_pylist() == [0]
    vector_values = table.column("vector").to_pylist()
    assert len(vector_values) == 1
    assert vector_values[0][0] == pytest.approx(0.1)
    assert vector_values[0][1] == pytest.approx(0.2)
    assert table.column("sliding_window").to_pylist() == [None]

    front_matter, body = _read_readme(tmp_path / "README.md")
    config_names = {entry["config_name"] for entry in front_matter["configs"]}
    assert {"all", "layer00", "l00h00q", "all_q", "layer00_q", "all_k"}.issubset(config_names)
    all_entry = next(entry for entry in front_matter["configs"] if entry["config_name"] == "all")
    assert all_entry.get("default") is True

    specific_entry = next(entry for entry in front_matter["configs"] if entry["config_name"] == "l00h00q")
    assert specific_entry["data_files"][0]["split"] == "meta_llama3_8b"
    assert "l00h00q" in specific_entry["data_files"][0]["path"]
    assert "- [meta/llama3-8b](https://huggingface.co/meta/llama3-8b)" in body


def test_dataset_saver_skips_duplicates_same_session(tmp_path):
    saver = DatasetSaver(root=tmp_path / "data", readme_path=tmp_path / "README.md")
    row = _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1])
    saver.add(row)
    saver.add(row)
    saver.close()

    parquet_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows == 1


def test_dataset_saver_skips_duplicates_across_sessions(tmp_path):
    root = tmp_path / "data"
    readme = tmp_path / "README.md"
    row = _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1])

    saver1 = DatasetSaver(root=root, readme_path=readme)
    saver1.add(row)
    saver1.close()

    saver2 = DatasetSaver(root=root, readme_path=readme)
    saver2.add(row)
    saver2.close()

    parquet_path = root / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows == 1


def test_dataset_saver_add_batch_deduplicates_and_flushes(tmp_path):
    saver = DatasetSaver(root=tmp_path / "data", readme_path=tmp_path / "README.md", write_batch_size=16)
    batch = CaptureBatch(
        model_name="meta/llama3-8b",
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        buckets=np.array([0, 0, 1], dtype=np.int64),
        example_ids=np.array([7, 7, 8], dtype=np.int64),
        positions=np.array([0, 0, 1], dtype=np.int64),
        vectors=np.array([[0.1], [0.2], [0.3]], dtype=np.float32),
        sliding_window=64,
    )
    saver.add_batch(batch)
    saver.close()

    parquet_path = tmp_path / "data" / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    # duplicate (7, 0) dropped
    assert table.num_rows == 2
    assert table.column("sliding_window").to_pylist() == [64, 64]


def test_dataset_saver_includes_model_stats_in_readme(tmp_path):
    saver = DatasetSaver(root=tmp_path / "data", readme_path=tmp_path / "README.md")
    saver.register_model_metadata(
        "meta/llama3-8b",
        {
            "source_dataset": "dummy/dataset",
            "dataset_split": "train",
            "sampling_strategy": "log",
            "sampling_min_bucket_size": 128,
        },
    )
    row = _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=2, example_id=0, position=0, vector=[0.1, 0.2])
    saver.add(row)
    saver.close()
    readme = (tmp_path / "README.md").read_text()
    assert "dummy/dataset" in readme
    assert "b2=1" in readme
    assert "split: `meta_llama3_8b`" in readme
    assert "sampling: log" in readme
    assert "layers: 1" in readme
    assert "query heads: 1" in readme
    assert "key heads: 0" in readme


def test_relative_readme_path_is_saved_inside_data_root(tmp_path):
    data_root = tmp_path / "data"
    saver = DatasetSaver(root=data_root, readme_path="README.md")
    saver.add(_row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]))
    saver.close()

    assert (data_root / "README.md").exists()
    assert not (tmp_path / "README.md").exists()


def test_dataset_saver_preserves_model_stats_across_runs(tmp_path):
    root = tmp_path / "data"
    readme = tmp_path / "README.md"

    first = DatasetSaver(root=root, readme_path=readme)
    first.register_model_metadata(
        "meta/llama3-8b",
        {"source_dataset": "dummy/dataset", "dataset_split": "train"},
    )
    first.add(_row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=2, example_id=0, position=0, vector=[0.1]))
    first.close()

    second = DatasetSaver(root=root, readme_path=readme)
    second.register_model_metadata("google/gemma3-9b", {"source_dataset": "other/dataset", "dataset_split": "train"})
    second.add(_row("google/gemma3-9b", layer=1, head=1, kind="k", bucket=5, example_id=1, position=3, vector=[0.5]))
    second.close()

    readme_text = readme.read_text()
    assert "dummy/dataset" in readme_text, "metadata from the first run should remain"
    assert "b2=1" in readme_text, "bucket counts from the first run should remain"


def test_readme_preserves_custom_fields_and_description(tmp_path):
    readme_path = tmp_path / "README.md"
    readme_path.write_text(
        """---
configs:
  - config_name: legacy
    data_files: []
custom_field: preserved
---
# sniffed-qk
Custom introduction that should survive.

## Available Models
<!-- MODELS_START -->
- old
<!-- MODELS_END -->

Additional paragraph outside tracked sections.
""",
        encoding="utf-8",
    )

    saver = DatasetSaver(root=tmp_path / "data", readme_path=readme_path)
    saver.add(_row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]))
    saver.close()

    front_matter, body = _read_readme(readme_path)
    assert front_matter.get("custom_field") == "preserved"
    assert any(entry["config_name"] == "all" for entry in front_matter["configs"])
    assert "Custom introduction that should survive." in body


def test_hf_datasets_e2e_load(tmp_path):
    saver = DatasetSaver(
        root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        dataset_name="viktoroo/sniffed-qk-test",
    )

    saver.add_many(
        [
            _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]),
            _row("meta/llama3-8b", layer=0, head=0, kind="k", bucket=0, example_id=0, position=0, vector=[0.2]),
            _row("google/gemma3-9b", layer=0, head=1, kind="q", bucket=1, example_id=1, position=2, vector=[0.3]),
        ]
    )
    saver.close()

    front_matter, _ = _read_readme(tmp_path / "README.md")
    configs = {entry["config_name"]: entry for entry in front_matter["configs"]}

    q_entry = configs["l00h00q"]
    q_data_files = _abs_paths(q_entry["data_files"])
    q_dataset = load_dataset("parquet", data_files=q_data_files, split="meta_llama3_8b")
    assert q_dataset.num_rows == 1
    assert q_dataset[0]["bucket"] == 0

    layer_entry = configs["layer00"]
    layer_data_files = _abs_paths(layer_entry["data_files"])
    layer_dataset = load_dataset("parquet", data_files=layer_data_files, split="meta_llama3_8b")
    assert layer_dataset.num_rows == 2  # q + k samples

    all_q_entry = configs["all_q"]
    all_q_files = _abs_paths(all_q_entry["data_files"])
    gemma_dataset = load_dataset("parquet", data_files=all_q_files, split="google_gemma3_9b")
    assert gemma_dataset.num_rows == 1
    assert gemma_dataset[0]["position"] == 2


def _row(
    model: str,
    layer: int,
    head: int,
    kind: str,
    bucket: int,
    example_id: int,
    position: int,
    vector,
) -> CaptureRow:
    return CaptureRow(
        model_name=model,
        layer_idx=layer,
        head_idx=head,
        vector_kind=kind,  # type: ignore[arg-type]
        bucket=bucket,
        example_id=example_id,
        position=position,
        vector=vector,
        sliding_window=None,
    )


def _read_readme(path: Path) -> tuple[dict, str]:
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---")
    _, front_raw, body = content.split("---", 2)
    front = yaml.safe_load(front_raw.strip())
    return front, body


def _abs_paths(data_files: list[dict]) -> dict:
    result = {}
    for item in data_files:
        path = Path(item["path"])
        result[item["split"]] = str(path.resolve())
    return result
