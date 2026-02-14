from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

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

    def shard(self, num_shards: int, index: int):
        return _FakeDataset([text for idx, text in enumerate(self._texts) if idx % num_shards == index])


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

    def shard(self, num_shards: int, index: int):
        shard_texts = [text for idx, text in enumerate(self._texts) if idx % num_shards == index]
        shard_ids = [row_id for idx, row_id in enumerate(self._ids) if idx % num_shards == index]
        return _FakeDatasetWithIds(shard_texts, shard_ids)


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


def test_resolve_primary_device_skips_non_torch_hf_device_map_entries():
    class _Model:
        hf_device_map = {"offload": "disk", "decoder": "cuda:0"}

    resolved = sniff.resolve_primary_device(_Model())
    assert resolved == torch.device("cuda:0")


def test_resolve_primary_device_uses_parameter_device_without_hf_map():
    model = torch.nn.Linear(4, 4)
    resolved = sniff.resolve_primary_device(model)
    assert resolved == torch.device("cpu")


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


def test_run_inference_non_cuda_auto_device_map_falls_back_to_none(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    captured_loader_kwargs = {}

    monkeypatch.setattr(sniff, "patch_modeling_modules", lambda *_, **__: None)
    monkeypatch.setattr(sniff, "pull_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "push_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "load_hf_dataset", lambda *_: _FakeDataset(["hello"]))
    monkeypatch.setattr(sniff, "prepare_tokenizer", lambda *_, **__: _FakeTokenizer())
    monkeypatch.setattr(sniff, "prepare_model_config", lambda *_: None)
    monkeypatch.setattr(sniff.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sniff, "_mps_is_available", lambda: False)

    def fake_load_model_from_pretrained(*, settings, model_config, torch_dtype, device_map):
        captured_loader_kwargs["device_map"] = device_map
        return _FakeModel()

    monkeypatch.setattr(sniff, "_load_model_from_pretrained", fake_load_model_from_pretrained)

    @contextmanager
    def fake_activate_sniffer(sniffer_config):
        _ = sniffer_config
        yield object()

    monkeypatch.setattr(sniff, "activate_sniffer", fake_activate_sniffer)
    monkeypatch.setattr(sniff, "set_active_example_ids", lambda *_: None)
    monkeypatch.setattr(sniff, "set_active_sequence_lengths", lambda *_: None)
    monkeypatch.setattr(sniff, "tqdm", lambda *args, **kwargs: _DummyProgress(*args, **kwargs))

    config = sniff.SniffConfig(
        dataset=sniff.DatasetSettings(path="dummy", split="train", text_column="text"),
        model=sniff.ModelSettings(name="fake-model", dtype="float16", device_map="auto"),
        tokenizer=sniff.TokenizerSettings(name="fake-tokenizer"),
        inference=sniff.InferenceSettings(batch_size=1, autocast_dtype="float16"),
        capture=sniff.CaptureSettings(),
        output=sniff.OutputSettings(
            data_root=str(data_root),
            readme_path="README.md",
            hf_repo_id=None,
        ),
    )

    sniff.run_inference(config)

    assert captured_loader_kwargs["device_map"] is None


def test_maybe_place_model_moves_to_mps_and_casts_dtype(monkeypatch):
    class _TrackModel:
        def __init__(self):
            self.calls = []

        def to(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self

    model = _TrackModel()
    monkeypatch.setattr(sniff.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(sniff, "_mps_is_available", lambda: True)
    placed = sniff._maybe_place_model(
        model,
        requested_device_map=None,
        requested_dtype=torch.bfloat16,
    )

    assert placed is model
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert kwargs["device"] == torch.device("mps")
    assert kwargs["dtype"] == torch.float16


def test_maybe_place_model_moves_to_cuda(monkeypatch):
    class _TrackModel:
        def __init__(self):
            self.calls = []

        def to(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self

    model = _TrackModel()
    monkeypatch.setattr(sniff.torch.cuda, "is_available", lambda: True)
    placed = sniff._maybe_place_model(
        model,
        requested_device_map=None,
        requested_dtype=torch.float16,
    )

    assert placed is model
    assert len(model.calls) == 1
    _, kwargs = model.calls[0]
    assert kwargs["device"] == torch.device("cuda")
    assert kwargs["dtype"] == torch.float16


def test_prepare_model_config_supports_mistral3_text_model_type_alias(tmp_path, monkeypatch):
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    payload = Mistral3Config().to_dict()
    payload["text_config"]["model_type"] = "ministral3"
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(payload), encoding="utf-8")

    class _FailingAutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise KeyError("ministral3")

    monkeypatch.setattr(sniff, "AutoConfig", _FailingAutoConfig)
    monkeypatch.setattr(sniff, "hf_hub_download", lambda **kwargs: str(config_file))

    resolved = sniff.prepare_model_config(sniff.ModelSettings(name="dummy/model"), True)
    assert resolved.model_type == "mistral3"
    if "ministral3" in CONFIG_MAPPING:
        assert resolved.text_config.model_type == "ministral3"
    elif "ministral" in CONFIG_MAPPING:
        assert resolved.text_config.model_type == "ministral"
    else:
        assert resolved.text_config.model_type == "mistral"


def test_load_model_from_pretrained_falls_back_to_mistral3_loader(monkeypatch):
    captured = {}
    sentinel_model = object()

    class _FailingAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("Unrecognized configuration class Mistral3Config for AutoModelForCausalLM.")

    class _CompatConfig:
        model_type = "mistral3"

    monkeypatch.setattr(sniff, "AutoModelForCausalLM", _FailingAutoModel)
    monkeypatch.setattr(sniff, "prepare_model_config", lambda settings, force=False: _CompatConfig())
    monkeypatch.setattr(
        sniff,
        "_load_auto_model_for_image_text_to_text",
        lambda **kwargs: (_ for _ in ()).throw(ValueError("image-text loader unavailable")),
    )

    def fake_mistral3_loader(*, settings, model_config, torch_dtype, device_map):
        captured["settings"] = settings
        captured["model_config"] = model_config
        captured["torch_dtype"] = torch_dtype
        captured["device_map"] = device_map
        return sentinel_model

    monkeypatch.setattr(sniff, "_load_mistral3_for_conditional_generation_model", fake_mistral3_loader)

    settings = sniff.ModelSettings(name="mistralai/Ministral-3-3B-Base-2512", dtype="bfloat16")
    model = sniff._load_model_from_pretrained(
        settings=settings,
        model_config=None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    assert model is sentinel_model
    assert captured["settings"] is settings
    assert captured["model_config"].model_type == "mistral3"
    assert captured["torch_dtype"] == torch.bfloat16
    assert captured["device_map"] == "auto"


def test_load_model_from_pretrained_uses_auto_image_text_to_text_for_mistral3(monkeypatch):
    captured = {}
    sentinel_model = object()

    class _CompatConfig:
        model_type = "mistral3"

    def fake_auto_image_loader(*, settings, model_config, torch_dtype, device_map):
        captured["settings"] = settings
        captured["model_config"] = model_config
        captured["torch_dtype"] = torch_dtype
        captured["device_map"] = device_map
        return sentinel_model

    monkeypatch.setattr(sniff, "_load_auto_model_for_image_text_to_text", fake_auto_image_loader)
    monkeypatch.setattr(
        sniff,
        "_load_mistral3_for_conditional_generation_model",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("direct mistral3 loader should not be used")),
    )

    settings = sniff.ModelSettings(name="mistralai/Ministral-3-3B-Base-2512", dtype="bfloat16")
    model = sniff._load_model_from_pretrained(
        settings=settings,
        model_config=_CompatConfig(),
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    assert model is sentinel_model
    assert captured["settings"] is settings
    assert captured["model_config"].model_type == "mistral3"
    assert captured["torch_dtype"] == torch.bfloat16
    assert captured["device_map"] == "auto"


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


def test_run_inference_samples_query_heads_and_maps_key_heads(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    captured_configs = []

    monkeypatch.setattr(sniff, "patch_modeling_modules", lambda *_, **__: None)
    monkeypatch.setattr(sniff, "pull_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "push_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "load_hf_dataset", lambda *_: _FakeDataset(["hello"]))
    monkeypatch.setattr(sniff, "prepare_tokenizer", lambda *_, **__: _FakeTokenizer())
    monkeypatch.setattr(sniff, "prepare_model_config", lambda *_: None)

    class _FakeConfig:
        num_hidden_layers = 3
        num_attention_heads = 4
        num_key_value_heads = 2

    class _FakeModelWithConfig(_FakeModel):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModelWithConfig()

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
        capture=sniff.CaptureSettings(
            head_sampling=sniff.HeadSamplingSettings(count=3, seed=9),
        ),
        output=sniff.OutputSettings(
            data_root=str(data_root),
            readme_path="README.md",
            hf_repo_id=None,
        ),
    )

    sniff.run_inference(config)

    assert captured_configs
    sniffer_config = captured_configs[0]
    assert sniffer_config.heads is None
    assert sniffer_config.sampled_query_heads
    assert sniffer_config.sampled_key_heads

    selected_queries = [
        (layer_idx, head_idx)
        for layer_idx, heads in sniffer_config.sampled_query_heads.items()
        for head_idx in heads
    ]
    assert len(selected_queries) == 3
    for layer_idx, query_head in selected_queries:
        expected_key_head = query_head // 2
        assert expected_key_head in sniffer_config.sampled_key_heads[layer_idx]


def test_run_inference_sampling_excludes_sliding_layers_when_full_attention_only(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    captured_configs = []

    monkeypatch.setattr(sniff, "patch_modeling_modules", lambda *_, **__: None)
    monkeypatch.setattr(sniff, "pull_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "push_remote_dataset", lambda *_: None)
    monkeypatch.setattr(sniff, "load_hf_dataset", lambda *_: _FakeDataset(["hello"]))
    monkeypatch.setattr(sniff, "prepare_tokenizer", lambda *_, **__: _FakeTokenizer())
    monkeypatch.setattr(sniff, "prepare_model_config", lambda *_: None)

    class _FakeConfig:
        num_hidden_layers = 4
        num_attention_heads = 4
        num_key_value_heads = 2
        layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]

    class _FakeModelWithConfig(_FakeModel):
        def __init__(self):
            super().__init__()
            self.config = _FakeConfig()

    class _FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeModelWithConfig()

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
        capture=sniff.CaptureSettings(
            full_attention_only=True,
            head_sampling=sniff.HeadSamplingSettings(count=8, seed=9),
        ),
        output=sniff.OutputSettings(
            data_root=str(data_root),
            readme_path="README.md",
            hf_repo_id=None,
        ),
    )

    sniff.run_inference(config)

    assert captured_configs
    sniffer_config = captured_configs[0]
    assert sniffer_config.full_attention_only is True
    assert sniffer_config.sampled_query_heads
    assert sniffer_config.sampled_key_heads
    assert set(sniffer_config.sampled_query_heads.keys()) == {1, 3}
    assert 0 not in sniffer_config.sampled_query_heads
    assert 2 not in sniffer_config.sampled_query_heads
    assert sniffer_config.metadata["attention_scope"] == "full_only"


def test_push_remote_dataset_uses_branch_revision(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    data_root.mkdir(parents=True)

    events = []

    class _FakeApi:
        def repo_info(self, repo_id, repo_type="dataset", revision=None):
            events.append(("repo_info", repo_id, repo_type, revision))
            return SimpleNamespace(sha="deadbeef")

        def create_repo(self, *args, **kwargs):
            events.append(("create_repo", args, kwargs))

        def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
            events.append(("list_repo_files", repo_id, repo_type, revision))
            return []

        def upload_file(self, **kwargs):
            events.append(("upload_file", kwargs))
            return SimpleNamespace()

        def create_branch(self, repo_id, repo_type="dataset", branch=None, exist_ok=False):
            events.append(("create_branch", repo_id, repo_type, branch, exist_ok))

        def upload_folder(self, **kwargs):
            events.append(("upload_folder", kwargs))
            return SimpleNamespace()

    monkeypatch.setattr(sniff, "HfApi", lambda: _FakeApi())

    settings = sniff.OutputSettings(
        data_root=str(data_root),
        hf_repo_id="dummy/sniffed-qk",
        hf_branch="meta-llama3-8b-train",
    )
    sniff.push_remote_dataset(settings)

    assert ("create_branch", "dummy/sniffed-qk", "dataset", "meta-llama3-8b-train", True) in events
    readme_call = next(item for item in events if item[0] == "upload_file")
    assert readme_call[1]["path_in_repo"] == "README.md"
    assert readme_call[1]["revision"] == "main"
    upload_call = next(item for item in events if item[0] == "upload_folder")
    assert upload_call[1]["revision"] == "meta-llama3-8b-train"
    assert upload_call[1]["delete_patterns"] == ["*", "**/*"]


def test_push_remote_dataset_skips_main_readme_seed_when_main_has_files(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    data_root.mkdir(parents=True)

    events = []

    class _FakeApi:
        def repo_info(self, repo_id, repo_type="dataset", revision=None):
            events.append(("repo_info", repo_id, repo_type, revision))
            return SimpleNamespace(sha="deadbeef")

        def list_repo_files(self, repo_id, repo_type="dataset", revision=None):
            events.append(("list_repo_files", repo_id, repo_type, revision))
            return ["README.md"]

        def create_branch(self, repo_id, repo_type="dataset", branch=None, exist_ok=False):
            events.append(("create_branch", repo_id, repo_type, branch, exist_ok))

        def upload_file(self, **kwargs):
            events.append(("upload_file", kwargs))
            return SimpleNamespace()

        def upload_folder(self, **kwargs):
            events.append(("upload_folder", kwargs))
            return SimpleNamespace()

    monkeypatch.setattr(sniff, "HfApi", lambda: _FakeApi())

    settings = sniff.OutputSettings(
        data_root=str(data_root),
        hf_repo_id="dummy/sniffed-qk",
        hf_branch="meta-llama3-8b-train",
    )
    sniff.push_remote_dataset(settings)

    assert not any(item[0] == "upload_file" for item in events)


def test_pull_remote_dataset_resets_local_root(tmp_path, monkeypatch):
    data_root = tmp_path / "captures"
    data_root.mkdir(parents=True)
    (data_root / "stale.txt").write_text("stale", encoding="utf-8")

    settings = sniff.OutputSettings(
        data_root=str(data_root),
        hf_repo_id="dummy/sniffed-qk",
        hf_branch="meta-llama3-8b-train",
    )
    sniff.pull_remote_dataset(settings)

    assert data_root.exists()
    assert not (data_root / "stale.txt").exists()
