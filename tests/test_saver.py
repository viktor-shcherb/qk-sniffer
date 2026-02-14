from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from datasets import load_dataset

from saver.dataset import CaptureBatch, CaptureRow, DatasetSaver
from saver.readme import DatasetReadme


def test_dataset_saver_writes_parquet_and_root_readme(tmp_path):
    saver = DatasetSaver(
        root=tmp_path / "data",
        readme_path=tmp_path / "ignored/README.md",
        dataset_name="viktoroo/sniffed-qk-test",
    )

    saver.add_many(
        [
            _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1, 0.2]),
            _row("meta/llama3-8b", layer=0, head=0, kind="k", bucket=1, example_id=0, position=0, vector=[0.3, 0.4]),
        ]
    )
    saver.close()

    parquet_path = tmp_path / "data" / "l00h00q" / "data.parquet"
    assert parquet_path.exists()

    table = pq.read_table(parquet_path)
    assert table.num_rows == 1
    assert table.schema.names == ["bucket", "example_id", "position", "vector", "sliding_window"]
    assert table.column("bucket").to_pylist() == [0]
    vector_values = table.column("vector").to_pylist()
    assert vector_values[0][0] == pytest.approx(0.1)
    assert vector_values[0][1] == pytest.approx(0.2)

    readme_path = tmp_path / "data" / "README.md"
    assert readme_path.exists()
    readme = readme_path.read_text(encoding="utf-8")
    assert "# sniffed-qk (model specialization)" in readme
    assert "`model`: `meta/llama3-8b`" in readme
    assert "`l00h00q`" in readme
    assert "`l00h00k`" in readme


def test_main_branch_readme_template_uses_real_newlines_and_repo_name(tmp_path):
    text = DatasetReadme(tmp_path / "README.md", dataset_name="dummy/sniffed-qk").main_branch_text()
    assert "\\n" not in text
    assert "repo = \"dummy/sniffed-qk\"" in text
    assert "This is the default branch for the sniffed-qk dataset." in text


def test_dataset_saver_writes_token_strings(tmp_path):
    saver = DatasetSaver(
        root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
    )
    row = CaptureRow(
        model_name="meta/llama3-8b",
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        bucket=0,
        example_id=0,
        position=0,
        vector=[0.1],
        sliding_window=None,
        token_str="tok",
    )
    saver.add(row)
    saver.close()

    parquet_path = tmp_path / "data" / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    assert "token_str" in table.schema.names
    assert table.column("token_str").to_pylist() == ["tok"]


def test_dataset_saver_rejects_multiple_models_in_one_branch(tmp_path):
    saver = DatasetSaver(root=tmp_path / "data", readme_path=tmp_path / "README.md")
    saver.add(_row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]))
    with pytest.raises(ValueError, match="one model per branch"):
        saver.add(_row("google/gemma3-9b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=1, vector=[0.2]))
    saver.close()


def test_dataset_saver_keeps_duplicates_same_session(tmp_path):
    saver = DatasetSaver(root=tmp_path / "data", readme_path=tmp_path / "README.md")
    row = _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1])
    saver.add(row)
    saver.add(row)
    saver.close()

    parquet_path = tmp_path / "data" / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows == 2


def test_dataset_saver_overwrites_config_file_across_sessions(tmp_path):
    root = tmp_path / "data"
    readme = tmp_path / "README.md"
    row = _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1])

    saver1 = DatasetSaver(root=root, readme_path=readme)
    saver1.add(row)
    saver1.close()

    saver2 = DatasetSaver(root=root, readme_path=readme)
    saver2.add(row)
    saver2.close()

    parquet_path = root / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows == 1


def test_dataset_saver_add_batch_flushes_without_deduplication(tmp_path):
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

    parquet_path = tmp_path / "data" / "l00h00q" / "data.parquet"
    table = pq.read_table(parquet_path)
    assert table.num_rows == 3
    assert table.column("sliding_window").to_pylist() == [64, 64, 64]


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

    readme = (tmp_path / "data" / "README.md").read_text(encoding="utf-8")
    assert "dummy/dataset" in readme
    assert "`sampling_strategy`: `log`" in readme
    assert "`sampling_min_bucket_size`: `128`" in readme
    assert "b2=1" in readme


def test_readme_is_always_written_to_dataset_root(tmp_path):
    data_root = tmp_path / "data"
    saver = DatasetSaver(root=data_root, readme_path=tmp_path / "outside" / "README.md")
    saver.add(_row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]))
    saver.close()

    assert (data_root / "README.md").exists()
    assert not (tmp_path / "outside" / "README.md").exists()


def test_dataset_saver_mirrors_readme_paths(tmp_path):
    repo_root = tmp_path / "repo"
    final_root = repo_root / "final"
    final_root.mkdir(parents=True)
    mirror_readme = repo_root / "README.copy.md"

    saver = DatasetSaver(
        root=final_root,
        readme_path=repo_root / "ignored.md",
        mirror_readme_paths=[mirror_readme],
        dataset_name="dummy/repo",
    )
    saver.add(_row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]))
    saver.close()

    primary_readme = final_root / "README.md"
    assert primary_readme.exists()
    assert mirror_readme.exists()
    assert primary_readme.read_text(encoding="utf-8") == mirror_readme.read_text(encoding="utf-8")


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
    second.register_model_metadata("meta/llama3-8b", {"dataset_split": "train"})
    second.add(_row("meta/llama3-8b", layer=1, head=1, kind="k", bucket=5, example_id=1, position=3, vector=[0.5]))
    second.close()

    readme_text = (root / "README.md").read_text(encoding="utf-8")
    assert "dummy/dataset" in readme_text
    assert "b2=1" in readme_text
    assert "b5=1" in readme_text


def test_hf_datasets_e2e_load_from_new_layout(tmp_path):
    saver = DatasetSaver(
        root=tmp_path / "data",
        readme_path=tmp_path / "README.md",
        dataset_name="viktoroo/sniffed-qk-test",
    )

    saver.add_many(
        [
            _row("meta/llama3-8b", layer=0, head=0, kind="q", bucket=0, example_id=0, position=0, vector=[0.1]),
            _row("meta/llama3-8b", layer=0, head=0, kind="k", bucket=0, example_id=0, position=0, vector=[0.2]),
            _row("meta/llama3-8b", layer=0, head=1, kind="q", bucket=1, example_id=1, position=2, vector=[0.3]),
        ]
    )
    saver.close()

    q_dataset = load_dataset(
        "parquet",
        data_files=str(tmp_path / "data" / "l00h00q" / "*.parquet"),
        split="train",
        cache_dir=str(tmp_path / "hf-cache"),
    )
    assert q_dataset.num_rows == 1
    assert q_dataset[0]["bucket"] == 0

    layer_dataset = load_dataset(
        "parquet",
        data_files=str(tmp_path / "data" / "l00h00*" / "*.parquet"),
        split="train",
        cache_dir=str(tmp_path / "hf-cache"),
    )
    assert layer_dataset.num_rows == 2  # q + k samples for l00h00


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
