from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

import sniff
from saver.dataset import CaptureRow, DatasetSaver


MODEL_NAME = "meta/llama3-8b"
OTHER_MODEL_NAME = "org/other-model"


def _capture_row(*, position: int, bucket: int, example_id: int) -> CaptureRow:
    return CaptureRow(
        model_name=MODEL_NAME,
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        bucket=bucket,
        example_id=example_id,
        position=position,
        vector=[0.1, 0.2],
        sliding_window=None,
    )


def _write_rows(root: Path, rows: list[CaptureRow]) -> None:
    saver = DatasetSaver(root=root, readme_path=root / "README.md", dataset_name="local/test")
    saver.register_model_metadata(
        MODEL_NAME,
        {
            "source_dataset": "dummy/dataset",
            "dataset_name": "",
            "dataset_split": "train",
        },
    )
    saver.add_many(rows)
    saver.close()


def test_finalize_capture_refreshes_before_push(tmp_path, monkeypatch):
    remote_v1 = tmp_path / "remote_v1"
    remote_v2 = tmp_path / "remote_v2"
    base_root = tmp_path / "local"
    final_root = base_root / "final"
    staging_root = base_root / "staging"
    remote_v1.mkdir()
    remote_v2.mkdir()
    staging_root.mkdir(parents=True)

    # Initial remote dataset and staged captures.
    _write_rows(remote_v1, [_capture_row(position=1, bucket=5, example_id=1)])
    _write_rows(staging_root, [_capture_row(position=2, bucket=6, example_id=2)])

    # Remote dataset changes while inference is running (new version adds another row).
    _write_rows(
        remote_v2,
        [
            _capture_row(position=1, bucket=5, example_id=1),
            _capture_row(position=99, bucket=6, example_id=9),
        ],
    )

    remote_v2_data = remote_v2 / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    assert remote_v2_data.exists()
    assert sorted(pq.read_table(remote_v2_data).column("position").to_pylist()) == [1, 99]

    current_remote = {"path": remote_v1}

    def fake_pull(settings):
        dest = Path(settings.data_root)
        src = current_remote["path"]
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    # Remote update happens while capture is running.
    current_remote["path"] = remote_v2

    monkeypatch.setattr(sniff, "pull_remote_dataset", fake_pull)

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name=MODEL_NAME),
        tokenizer=sniff.TokenizerSettings(name=MODEL_NAME),
        inference=sniff.InferenceSettings(),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(data_root=str(base_root), readme_path="README.md"),
    )

    final_output = replace(config.output, data_root=str(final_root))

    # Initial sync before capture begins.
    fake_pull(final_output)

    sniff.finalize_capture(config, staging_root, final_output)

    data_path = final_root / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    assert data_path.exists()
    table = pq.read_table(data_path)
    positions = sorted(table.column("position").to_pylist())
    assert positions == [1, 2, 99]


def test_finalize_capture_skips_corrupt_remote(tmp_path, monkeypatch):
    remote_repo = tmp_path / "remote_repo"
    base_root = tmp_path / "local"
    final_root = base_root / "final"
    staging_root = base_root / "staging"
    remote_repo.mkdir()
    staging_root.mkdir(parents=True)

    split_dir = remote_repo / "meta_llama3_8b" / "l00h00q"
    split_dir.mkdir(parents=True)
    (split_dir / "data.parquet").write_text("not parquet", encoding="utf-8")

    _write_rows(staging_root, [_capture_row(position=7, bucket=3, example_id=42)])

    def fake_pull(settings):
        dest = Path(settings.data_root)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(remote_repo, dest)

    monkeypatch.setattr(sniff, "pull_remote_dataset", fake_pull)

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name=MODEL_NAME),
        tokenizer=sniff.TokenizerSettings(name=MODEL_NAME),
        inference=sniff.InferenceSettings(),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(data_root=str(base_root), readme_path="README.md"),
    )

    final_output = replace(config.output, data_root=str(final_root))

    sniff.finalize_capture(config, staging_root, final_output)

    data_path = final_root / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    table = pq.read_table(data_path)
    assert table.column("position").to_pylist() == [7]


def test_finalize_capture_preserves_other_splits_and_readme(tmp_path, monkeypatch):
    remote_repo = tmp_path / "remote_repo"
    remote_repo.mkdir()
    base_root = tmp_path / "local"
    final_root = base_root / "final"
    staging_root = base_root / "staging"
    staging_root.mkdir(parents=True)

    other_row = CaptureRow(
        model_name=OTHER_MODEL_NAME,
        layer_idx=0,
        head_idx=0,
        vector_kind="q",
        bucket=1,
        example_id=111,
        position=5,
        vector=[0.5, 0.6],
        sliding_window=None,
    )
    remote_saver = DatasetSaver(
        root=remote_repo,
        readme_path=remote_repo / "README.md",
        dataset_name="local/test",
    )
    remote_saver.register_model_metadata(
        OTHER_MODEL_NAME,
        {
            "source_dataset": "dummy/remote",
            "dataset_name": "",
            "dataset_split": "train",
        },
    )
    remote_saver.add_many([other_row])
    remote_saver.close()

    _write_rows(staging_root, [_capture_row(position=13, bucket=2, example_id=9)])

    def fake_pull(settings):
        dest = Path(settings.data_root)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(remote_repo, dest)

    monkeypatch.setattr(sniff, "pull_remote_dataset", fake_pull)

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name=MODEL_NAME),
        tokenizer=sniff.TokenizerSettings(name=MODEL_NAME),
        inference=sniff.InferenceSettings(),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(data_root=str(base_root), readme_path="README.md"),
    )

    final_output = replace(config.output, data_root=str(final_root))
    sniff.finalize_capture(config, staging_root, final_output)

    other_split = sniff._sanitize_split_name(OTHER_MODEL_NAME)
    other_data = final_root / other_split / "l00h00q" / "data.parquet"
    assert other_data.exists()
    table = pq.read_table(other_data)
    assert table.column("example_id").to_pylist() == [111]

    readme = (final_root / "README.md").read_text(encoding="utf-8")
    parts = readme.split("---", 2)
    assert len(parts) >= 3
    front = yaml.safe_load(parts[1]) or {}
    model_names = {entry.get("name") for entry in front.get("models", [])}
    assert MODEL_NAME in model_names
    assert OTHER_MODEL_NAME in model_names
