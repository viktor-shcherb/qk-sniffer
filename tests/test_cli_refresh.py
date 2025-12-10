from __future__ import annotations

import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest

import sniff
from saver.dataset import CaptureRow, DatasetSaver


MODEL_NAME = "meta/llama3-8b"


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
    staging_root = tmp_path / "staging"
    output_root = tmp_path / "output"
    remote_v1.mkdir()
    remote_v2.mkdir()
    staging_root.mkdir()

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
        src = current_remote["path"]
        if output_root.exists():
            shutil.rmtree(output_root)
        shutil.copytree(src, output_root)

    # Initial sync before capture begins.
    fake_pull(None)
    # Remote update happens while capture is running.
    current_remote["path"] = remote_v2

    monkeypatch.setattr(sniff, "pull_remote_dataset", fake_pull)

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name=MODEL_NAME),
        tokenizer=sniff.TokenizerSettings(name=MODEL_NAME),
        inference=sniff.InferenceSettings(),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(data_root=str(output_root), readme_path="README.md"),
    )

    sniff.finalize_capture(config, staging_root)

    data_path = output_root / "meta_llama3_8b" / "l00h00q" / "data.parquet"
    assert data_path.exists()
    table = pq.read_table(data_path)
    positions = sorted(table.column("position").to_pylist())
    assert positions == [1, 2, 99]
