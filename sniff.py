from __future__ import annotations

import argparse
import importlib
import json
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm.auto import tqdm

import torch
import yaml
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from sniffer import (
    AllSampler,
    LogUniformSampler,
    SnifferConfig,
    UniformSampler,
    activate_sniffer,
    set_active_example_ids,
    set_active_sequence_lengths,
    set_active_token_strings,
)

load_dotenv()


@dataclass
class SamplerSettings:
    type: str = "log_uniform"
    base_rate: float = 1.0


@dataclass
class CaptureSettings:
    capture_queries: bool = True
    capture_keys: bool = True
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None
    sampler: SamplerSettings = field(default_factory=SamplerSettings)
    max_rows_per_batch: Optional[int] = None
    queue_size: int = 8
    min_bucket_size: int = 128
    capture_pre_rope: bool = False
    capture_token_strings: bool = False


@dataclass
class DatasetSettings:
    path: str
    name: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    id_column: Optional[str] = None
    max_samples: Optional[int] = None
    streaming: bool = False


@dataclass
class ModelSettings:
    name: str
    revision: Optional[str] = None
    dtype: str = "float16"
    device_map: Optional[Any] = "auto"
    trust_remote_code: bool = False
    config_overrides: Optional[Dict[str, Any]] = None


@dataclass
class TokenizerSettings:
    name: Optional[str] = None
    max_length: int = 4096
    padding: str = "longest"
    trust_remote_code: bool = False
    padding_side: str = "right"


@dataclass
class InferenceSettings:
    batch_size: int = 1
    autocast_dtype: str = "float16"


@dataclass
class OutputSettings:
    data_root: str = "data"
    readme_path: str = "README.md"
    hf_repo_id: Optional[str] = None
    private: bool = False
    write_batch_size: int = 2048


@dataclass
class SniffConfig:
    dataset: DatasetSettings
    model: ModelSettings
    tokenizer: TokenizerSettings
    inference: InferenceSettings
    capture: CaptureSettings
    output: OutputSettings


DTYPE_ALIASES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}

_CONFIG_NAME_RE = re.compile(r"^l(\d{2})h(\d{2})([qk])$")
_SPLIT_SANITIZE_RE = re.compile(r"\W")
_SNAPSHOT_STATE_FILENAME = ".sniff_snapshot.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run QK sniffing on a dataset.")
    parser.add_argument("--config", type=Path, required=True, help="Path to YAML configuration file.")
    return parser.parse_args()


def load_config(config_path: Path) -> SniffConfig:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    capture_raw = dict(raw["capture"])
    sampler_raw = capture_raw.get("sampler")
    if isinstance(sampler_raw, dict):
        capture_raw["sampler"] = SamplerSettings(**sampler_raw)
    elif sampler_raw is None:
        capture_raw["sampler"] = SamplerSettings()
    return SniffConfig(
        dataset=DatasetSettings(**raw["dataset"]),
        model=ModelSettings(**raw["model"]),
        tokenizer=TokenizerSettings(**raw["tokenizer"]),
        inference=InferenceSettings(**raw["inference"]),
        capture=CaptureSettings(**capture_raw),
        output=OutputSettings(**raw["output"]),
    )


def resolve_readme_path(settings: OutputSettings) -> Path:
    readme_raw = settings.readme_path or "README.md"
    readme_path = Path(readme_raw)
    if readme_path.is_absolute():
        return readme_path
    return Path(settings.data_root) / readme_path


def _absolute_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded
    return (Path.cwd() / expanded).resolve()


def load_hf_dataset(settings: DatasetSettings) -> Dataset | IterableDataset:
    if settings.streaming:
        dataset = load_dataset(settings.path, settings.name, split=settings.split, streaming=True)
        if settings.max_samples is not None:
            dataset = dataset.take(settings.max_samples)
        return dataset
    dataset = load_dataset(settings.path, settings.name, split=settings.split)
    if settings.max_samples is not None:
        dataset = dataset.select(range(min(settings.max_samples, len(dataset))))
    return dataset


def prepare_tokenizer(settings: TokenizerSettings, model_name: str):
    tokenizer_name = settings.name or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=settings.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = settings.padding_side
    return tokenizer


def _apply_model_config_overrides(model_config, overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict):
            existing = getattr(model_config, key, None)
            if isinstance(existing, dict):
                merged = {**existing, **value}
                setattr(model_config, key, merged)
                continue
        setattr(model_config, key, value)


def prepare_model_config(settings: ModelSettings):
    overrides = settings.config_overrides
    if not overrides:
        return None
    model_config = AutoConfig.from_pretrained(
        settings.name,
        revision=settings.revision,
        trust_remote_code=settings.trust_remote_code,
    )
    _apply_model_config_overrides(model_config, overrides)
    return model_config


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return DTYPE_ALIASES[key]


def resolve_primary_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_entry = next(iter(model.hf_device_map.values()))
        if isinstance(first_entry, (list, tuple)):
            first_entry = first_entry[0]
        if isinstance(first_entry, int):
            return torch.device(f"cuda:{first_entry}")
        return torch.device(first_entry)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_sampler_factory(settings: SamplerSettings, *, min_bucket_size: int):
    sampler_type = settings.type.lower()

    if sampler_type == "log_uniform":

        def factory():
            return LogUniformSampler(base_rate=settings.base_rate, min_bucket_size=min_bucket_size)

        return factory

    if sampler_type == "uniform":

        def factory():
            return UniformSampler(base_rate=settings.base_rate, bucket_size=max(1, int(min_bucket_size)))

        return factory

    if sampler_type == "all":

        def factory():
            return AllSampler()

        return factory

    raise ValueError(f"Unsupported sampler type '{settings.type}'.")


def batch_iter(dataset: Dataset | IterableDataset, settings: DatasetSettings, batch_size: int) -> Iterable[Dict[str, Any]]:
    text_col = settings.text_column
    id_col = settings.id_column
    if isinstance(dataset, IterableDataset):
        texts: List[str] = []
        raw_ids: List[Any] = []
        ids: List[int] = []
        index = 0
        for row in dataset:
            texts.append(row[text_col])
            if id_col:
                raw_ids.append(row[id_col])
            else:
                ids.append(index)
                index += 1
            if len(texts) >= batch_size:
                batch_ids = _normalize_example_ids(raw_ids) if id_col else ids
                yield {"texts": texts, "example_ids": batch_ids}
                texts = []
                raw_ids = []
                ids = []
        if texts:
            batch_ids = _normalize_example_ids(raw_ids) if id_col else ids
            yield {"texts": texts, "example_ids": batch_ids}
        return

    size = len(dataset)
    for start in range(0, size, batch_size):
        slice_ = dataset[start : start + batch_size]
        texts = slice_[text_col]
        if id_col:
            ids = _normalize_example_ids(slice_[id_col])
        else:
            ids = list(range(start, start + len(texts)))
        yield {"texts": texts, "example_ids": ids}


_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _hash_to_int64(value: Any) -> int:
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    unsigned = int.from_bytes(digest, "little", signed=False)
    if unsigned > _INT64_MAX:
        return unsigned - (1 << 64)
    return unsigned


def _normalize_example_ids(raw_ids: Sequence[Any]) -> List[int]:
    normalized: List[int] = []
    for value in raw_ids:
        if isinstance(value, bool):
            normalized.append(int(value))
            continue
        if isinstance(value, int):
            if _INT64_MIN <= value <= _INT64_MAX:
                normalized.append(int(value))
            else:
                normalized.append(_hash_to_int64(value))
            continue
        try:
            if hasattr(value, "item"):
                scalar = value.item()
                if isinstance(scalar, (int, bool)):
                    if _INT64_MIN <= int(scalar) <= _INT64_MAX:
                        normalized.append(int(scalar))
                    else:
                        normalized.append(_hash_to_int64(scalar))
                    continue
                if isinstance(scalar, float) and scalar.is_integer():
                    normalized.append(int(scalar))
                    continue
            if isinstance(value, float) and value.is_integer():
                normalized.append(int(value))
                continue
        except Exception:
            pass
        normalized.append(_hash_to_int64(value))
    return normalized


def _alias_module(module, target_name: str, replace_existing: bool) -> None:
    """
    Register the module under a new name while keeping parent attributes in sync.
    """
    if target_name in sys.modules and not replace_existing:
        return
    sys.modules[target_name] = module
    if "." not in target_name:
        return
    parent_name, attr_name = target_name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is None:
        try:
            parent = importlib.import_module(parent_name)
        except ImportError:
            parent = ModuleType(parent_name)
            parent.__path__ = []
            sys.modules[parent_name] = parent
    if replace_existing or not hasattr(parent, attr_name):
        setattr(parent, attr_name, module)


def _module_exists(name: str) -> bool:
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    return True


def patch_modeling_modules(root: Path = Path("models")) -> None:
    """
    Ensure local modeling files mirror transformers.models.* automatically.
    """
    root = Path(root)
    if not root.exists():
        return
    root = root.resolve()
    default_models_root = (Path(__file__).parent / "models").resolve()
    clear_models_namespace = root != default_models_root
    saved_model_modules: Dict[str, ModuleType] = {}
    if clear_models_namespace:
        for name in list(sys.modules):
            if name == "models" or name.startswith("models."):
                saved_model_modules[name] = sys.modules.pop(name)
    import_root = str(root.parent)
    sys.path.insert(0, import_root)
    try:
        for file_path in root.rglob("*.py"):
            rel_parts = file_path.relative_to(root).with_suffix("").parts
            if not rel_parts:
                continue
            is_package = rel_parts[-1] == "__init__"
            if is_package:
                rel_parts = rel_parts[:-1]
                if not rel_parts:
                    continue
                target_package = ".".join(["transformers", "models", *rel_parts])
                if _module_exists(target_package):
                    # Avoid clobbering upstream packages (e.g., transformers.models.llama).
                    continue
            module = None
            completed_depth = 0
            for depth in range(1, len(rel_parts) + 1):
                local_name = ".".join(["models", *rel_parts[:depth]])
                target_name = ".".join(["transformers", "models", *rel_parts[:depth]])
                try:
                    module = importlib.import_module(local_name)
                    completed_depth = depth
                except ImportError:
                    # If a local module fails to import, bail on deeper descendants.
                    module = None
                    break
            if module is None:
                continue
            replace = completed_depth == len(rel_parts)
            _alias_module(module, target_name, replace_existing=replace)
    finally:
        try:
            sys.path.remove(import_root)
        except ValueError:
            pass


def _snapshot_state_path(data_root: Path) -> Path:
    return Path(data_root) / _SNAPSHOT_STATE_FILENAME


def _load_snapshot_state(data_root: Path) -> Optional[Dict[str, str]]:
    path = _snapshot_state_path(data_root)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_snapshot_state(data_root: Path, repo_id: str, sha: Optional[str]) -> None:
    path = _snapshot_state_path(data_root)
    if not sha:
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        return
    payload = {"repo_id": repo_id, "sha": sha}
    try:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass
        if clear_models_namespace:
            for name in list(sys.modules):
                if name == "models" or name.startswith("models."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved_model_modules)


def pull_remote_dataset(settings: OutputSettings) -> None:
    if not settings.hf_repo_id:
        return
    target_root = Path(settings.data_root)
    api = HfApi()
    repo_sha: Optional[str] = None
    try:
        repo_info = api.repo_info(settings.hf_repo_id, repo_type="dataset")
        repo_sha = getattr(repo_info, "sha", None)
    except RepositoryNotFoundError:
        print(f"[sniff] Dataset repo {settings.hf_repo_id} not found; skipping pull.")
        _write_snapshot_state(target_root, settings.hf_repo_id, None)
        return
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to query dataset repo {settings.hf_repo_id}: {err}")
        return
    state = _load_snapshot_state(target_root)
    if (
        repo_sha
        and state
        and state.get("repo_id") == settings.hf_repo_id
        and state.get("sha") == repo_sha
        and target_root.exists()
    ):
        print(f"[sniff] Dataset {settings.hf_repo_id} already synced at {repo_sha}; skipping pull.")
        return
    tmp_dir = Path(tempfile.mkdtemp(prefix="sniff-pull-"))
    try:
        snapshot_download(
            repo_id=settings.hf_repo_id,
            repo_type="dataset",
            local_dir=str(tmp_dir),
            force_download=True,
            token=True,
            revision=repo_sha,
        )
        repo_root = tmp_dir
        children = list(tmp_dir.iterdir())
        if len(children) == 1 and children[0].is_dir():
            repo_root = children[0]
        if target_root.exists():
            shutil.rmtree(target_root)
        shutil.move(str(repo_root), str(target_root))
        if repo_sha:
            _write_snapshot_state(target_root, settings.hf_repo_id, repo_sha)
    except RepositoryNotFoundError:
        print(f"[sniff] Dataset repo {settings.hf_repo_id} not found; skipping pull.")
        _write_snapshot_state(target_root, settings.hf_repo_id, None)
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to pull dataset repo {settings.hf_repo_id}: {err}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def push_remote_dataset(settings: OutputSettings) -> None:
    if not settings.hf_repo_id:
        return
    api = HfApi()
    try:
        api.repo_info(settings.hf_repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        print(f"[sniff] Creating dataset repo {settings.hf_repo_id}")
        api.create_repo(settings.hf_repo_id, repo_type="dataset", exist_ok=True, private=settings.private)
    try:
        api.upload_folder(
            repo_id=settings.hf_repo_id,
            folder_path=settings.data_root,
            repo_type="dataset",
            commit_message="Update dataset",
        )
        try:
            repo_info = api.repo_info(settings.hf_repo_id, repo_type="dataset")
            sha = getattr(repo_info, "sha", None)
            if sha:
                _write_snapshot_state(Path(settings.data_root), settings.hf_repo_id, sha)
        except HfHubHTTPError:
            pass
    except RepositoryNotFoundError:
        print(f"[sniff] Dataset repo {settings.hf_repo_id} not found; please create it before pushing.")
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to push dataset repo {settings.hf_repo_id}: {err}")


def run_inference(config: SniffConfig) -> None:
    patch_modeling_modules()
    data_root = Path(config.output.data_root)
    pull_remote_dataset(config.output)
    data_root.mkdir(parents=True, exist_ok=True)
    dataset = load_hf_dataset(config.dataset)
    tokenizer = prepare_tokenizer(config.tokenizer, config.model.name)
    readme_path = resolve_readme_path(config.output)

    torch_dtype = resolve_dtype(config.model.dtype)
    model_config = prepare_model_config(config.model)
    model_kwargs = {}
    if model_config is not None:
        model_kwargs["config"] = model_config
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        dtype=torch_dtype,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
        **model_kwargs,
    )
    model.eval()

    sampler_factory = build_sampler_factory(config.capture.sampler, min_bucket_size=config.capture.min_bucket_size)
    sniffer_config = SnifferConfig(
        model_name=config.model.name,
        data_root=data_root,
        readme_path=readme_path,
        capture_queries=config.capture.capture_queries,
        capture_keys=config.capture.capture_keys,
        layers=set(config.capture.layers) if config.capture.layers else None,
        heads=set(config.capture.heads) if config.capture.heads else None,
        sampler_factory=sampler_factory,
        max_rows_per_batch=config.capture.max_rows_per_batch,
        queue_size=config.capture.queue_size,
        write_batch_size=config.output.write_batch_size,
        min_bucket_size=config.capture.min_bucket_size,
        capture_pre_rope=config.capture.capture_pre_rope,
        capture_token_strings=config.capture.capture_token_strings,
        metadata={
            "source_dataset": config.dataset.path,
            "dataset_name": config.dataset.name or "",
            "dataset_split": config.dataset.split,
        },
    )

    primary_device = resolve_primary_device(model)
    autocast_dtype = resolve_dtype(config.inference.autocast_dtype)
    print("Model device:", primary_device)

    total_rows: Optional[int] = None
    if isinstance(dataset, IterableDataset):
        if config.dataset.max_samples is not None:
            total_rows = int(config.dataset.max_samples)
    else:
        total_rows = len(dataset)
    progress = tqdm(total=total_rows, desc="Capturing", unit="rows")
    try:
        with activate_sniffer(sniffer_config):
            for batch in batch_iter(dataset, config.dataset, config.inference.batch_size):
                texts: Sequence[str] = batch["texts"]
                if not texts:
                    continue
                encodings = tokenizer(
                    list(texts),
                    return_tensors="pt",
                    padding=config.tokenizer.padding,
                    truncation=True,
                    max_length=config.tokenizer.max_length,
                )
                batch_example_ids = batch["example_ids"]
                if config.capture.capture_token_strings:
                    input_ids = encodings.get("input_ids")
                    if input_ids is None:
                        raise ValueError("Tokenizer did not return input_ids required for token string capture.")
                    token_id_rows = input_ids.tolist()
                    token_strings = [
                        tokenizer.convert_ids_to_tokens(row, skip_special_tokens=False) for row in token_id_rows
                    ]
                    set_active_token_strings(token_strings)
                attention_mask = encodings.get("attention_mask")
                if attention_mask is not None:
                    valid_lengths = attention_mask.sum(dim=1).tolist()
                else:
                    seq_len = next(iter(encodings.values())).shape[-1]
                    valid_lengths = [seq_len] * len(texts)
                set_active_example_ids(batch_example_ids)
                set_active_sequence_lengths([int(length) for length in valid_lengths])
                inputs = {k: v.to(primary_device) for k, v in encodings.items()}
                with torch.no_grad():
                    use_autocast = primary_device.type == "cuda"
                    context = (
                        torch.autocast(device_type=primary_device.type, dtype=autocast_dtype)
                        if use_autocast
                        else nullcontext()
                    )
                    with context:
                        model(**inputs)
                progress.update(len(texts))
    finally:
        progress.close()
    push_remote_dataset(config.output)


def main():
    args = parse_args()
    config = load_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()
