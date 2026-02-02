from __future__ import annotations

from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch

import sniff


class _FakeDataset:
    def __init__(self, texts: list[str]):
        self._texts = list(texts)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {"text": self._texts[key]}
        raise TypeError("Fake dataset only supports slicing")


class _FakeDatasetWithIds:
    def __init__(self, texts: list[str], ids: list[str]):
        self._texts = list(texts)
        self._ids = list(ids)

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return {"text": self._texts[key], "id": self._ids[key]}
        raise TypeError("Fake dataset only supports slicing")


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        batch = len(texts)
        seq_len = 3
        input_ids = torch.zeros((batch, seq_len), dtype=torch.int64)
        attention_mask = torch.ones((batch, seq_len), dtype=torch.int64)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _FakeModel:
    def __init__(self):
        self.hf_device_map = {"model": "cpu"}

    def eval(self):
        return self

    def __call__(self, **inputs):
        return inputs


class _DummyProgress:
    def __init__(self, *args, **kwargs):
        self.total = kwargs.get("total")
        self.updated = 0

    def update(self, amount):
        self.updated += amount

    def close(self):
        return None


def test_run_inference_syncs_dataset_once(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    events: list[tuple[str, Path]] = []
    captured_configs = []

    monkeypatch.setattr(sniff, "patch_modeling_modules", lambda *_, **__: None)

    def fake_pull(settings):
        events.append(("pull", Path(settings.data_root)))

    def fake_push(settings):
        events.append(("push", Path(settings.data_root)))

    monkeypatch.setattr(sniff, "pull_remote_dataset", fake_pull)
    monkeypatch.setattr(sniff, "push_remote_dataset", fake_push)
    monkeypatch.setattr(sniff, "load_hf_dataset", lambda *_: _FakeDataset(["hello", "world", "!!!"]))
    monkeypatch.setattr(sniff, "prepare_tokenizer", lambda *_, **__: _FakeTokenizer())
    monkeypatch.setattr(sniff, "prepare_model_config", lambda *_: None)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(sniff, "AutoModelForCausalLM", _FakeAutoModel)

    @contextmanager
    def fake_activate_sniffer(sniffer_config):
        captured_configs.append(sniffer_config)
        yield object()

    monkeypatch.setattr(sniff, "activate_sniffer", fake_activate_sniffer)
    monkeypatch.setattr(sniff, "set_active_example_ids", lambda *_: None)
    monkeypatch.setattr(sniff, "set_active_sequence_lengths", lambda *_: None)
    monkeypatch.setattr(sniff, "tqdm", lambda *args, **kwargs: _DummyProgress(*args, **kwargs))

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name="fake-model", dtype="float16"),
        tokenizer=sniff.TokenizerSettings(name="fake-tokenizer"),
        inference=sniff.InferenceSettings(batch_size=2, autocast_dtype="float16"),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(
            data_root=str(data_root),
            readme_path="README.md",
            hf_repo_id="dummy/sniffed-qk",
        ),
    )

    sniff.run_inference(config)

    assert events == [("pull", data_root), ("push", data_root)]
    assert captured_configs, "activate_sniffer should be called"
    sniffer_config = captured_configs[0]
    assert Path(sniffer_config.data_root) == data_root
    assert Path(sniffer_config.readme_path) == data_root / "README.md"
    assert sniffer_config.metadata["source_dataset"] == config.dataset.path
    assert not (data_root / "staging").exists()


def test_run_inference_passes_capture_pre_rope(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    captured_configs = []

    monkeypatch.setattr(sniff, "patch_modeling_modules", lambda *_, **__: None)
    monkeypatch.setattr(sniff, "pull_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "push_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "load_hf_dataset", lambda *_: _FakeDataset(["hello"]))
    monkeypatch.setattr(sniff, "prepare_tokenizer", lambda *_, **__: _FakeTokenizer())
    monkeypatch.setattr(sniff, "prepare_model_config", lambda *_: None)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(sniff, "AutoModelForCausalLM", _FakeAutoModel)

    @contextmanager
    def fake_activate_sniffer(sniffer_config):
        captured_configs.append(sniffer_config)
        yield object()

    monkeypatch.setattr(sniff, "activate_sniffer", fake_activate_sniffer)
    monkeypatch.setattr(sniff, "set_active_example_ids", lambda *_: None)
    monkeypatch.setattr(sniff, "set_active_sequence_lengths", lambda *_: None)
    monkeypatch.setattr(sniff, "tqdm", lambda *args, **kwargs: _DummyProgress(*args, **kwargs))

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name="fake-model", dtype="float16"),
        tokenizer=sniff.TokenizerSettings(name="fake-tokenizer"),
        inference=sniff.InferenceSettings(batch_size=1, autocast_dtype="float16"),
        capture=sniff.CaptureSettings(capture_pre_rope=True),
        output=sniff.OutputSettings(
            data_root=str(data_root),
            readme_path="README.md",
            hf_repo_id=None,
        ),
    )

    sniff.run_inference(config)

    assert captured_configs
    assert captured_configs[0].capture_pre_rope is True


def _hash_id(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    unsigned = int.from_bytes(digest, "little", signed=False)
    if unsigned > (1 << 63) - 1:
        return unsigned - (1 << 64)
    return unsigned


def test_run_inference_hashes_string_ids(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    captured_ids = []

    monkeypatch.setattr(sniff, "patch_modeling_modules", lambda *_, **__: None)
    monkeypatch.setattr(sniff, "pull_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "push_remote_dataset", lambda *_: None)
    monkeypatch.setattr(
        sniff,
        "load_hf_dataset",
        lambda *_: _FakeDatasetWithIds(["hello", "world"], ["id-a", "id-b"]),
    )
    monkeypatch.setattr(sniff, "prepare_tokenizer", lambda *_, **__: _FakeTokenizer())
    monkeypatch.setattr(sniff, "prepare_model_config", lambda *_: None)

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModel()

    monkeypatch.setattr(sniff, "AutoModelForCausalLM", _FakeAutoModel)

    @contextmanager
    def fake_activate_sniffer(sniffer_config):
        yield object()

    monkeypatch.setattr(sniff, "activate_sniffer", fake_activate_sniffer)
    monkeypatch.setattr(sniff, "set_active_sequence_lengths", lambda *_: None)
    monkeypatch.setattr(sniff, "tqdm", lambda *args, **kwargs: _DummyProgress(*args, **kwargs))

    def capture_ids(ids):
        captured_ids.append(list(ids))

    monkeypatch.setattr(sniff, "set_active_example_ids", capture_ids)

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text", id_column="id"),
        model=sniff.ModelSettings(name="fake-model", dtype="float16"),
        tokenizer=sniff.TokenizerSettings(name="fake-tokenizer"),
        inference=sniff.InferenceSettings(batch_size=2, autocast_dtype="float16"),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(
            data_root=str(data_root),
            readme_path="README.md",
            hf_repo_id=None,
        ),
    )

    sniff.run_inference(config)

    assert captured_ids
    assert captured_ids[0] == [_hash_id("id-a"), _hash_id("id-b")]
