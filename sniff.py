from __future__ import annotations

import argparse
import importlib
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence

from dotenv import load_dotenv
from tqdm.auto import tqdm

import torch
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoModelForCausalLM, AutoTokenizer

from sniffer import LogUniformSampler, SnifferConfig, activate_sniffer, set_active_example_ids

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


@dataclass
class DatasetSettings:
    path: str
    name: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    id_column: Optional[str] = None
    max_samples: Optional[int] = None


@dataclass
class ModelSettings:
    name: str
    revision: Optional[str] = None
    dtype: str = "float16"
    device_map: Optional[Any] = "auto"
    trust_remote_code: bool = False


@dataclass
class TokenizerSettings:
    name: Optional[str] = None
    max_length: int = 4096
    padding: str = "longest"


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


def load_hf_dataset(settings: DatasetSettings) -> Dataset:
    dataset = load_dataset(settings.path, settings.name, split=settings.split)
    if settings.max_samples is not None:
        dataset = dataset.select(range(min(settings.max_samples, len(dataset))))
    return dataset


def prepare_tokenizer(settings: TokenizerSettings, model_name: str):
    tokenizer_name = settings.name or model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    return tokenizer


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


def build_sampler_factory(settings: SamplerSettings):
    if settings.type != "log_uniform":
        raise ValueError(f"Unsupported sampler type '{settings.type}'.")

    def factory():
        return LogUniformSampler(base_rate=settings.base_rate)

    return factory


def batch_iter(dataset: Dataset, settings: DatasetSettings, batch_size: int) -> Iterable[Dict[str, Any]]:
    size = len(dataset)
    text_col = settings.text_column
    id_col = settings.id_column
    for start in range(0, size, batch_size):
        slice_ = dataset[start : start + batch_size]
        texts = slice_[text_col]
        if id_col:
            ids = slice_[id_col]
        else:
            ids = list(range(start, start + len(texts)))
        yield {"texts": texts, "example_ids": ids}


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
            for depth in range(1, len(rel_parts) + 1):
                local_name = ".".join(["models", *rel_parts[:depth]])
                target_name = ".".join(["transformers", "models", *rel_parts[:depth]])
                try:
                    module = importlib.import_module(local_name)
                except ImportError:
                    # If a local module fails to import, bail on deeper descendants.
                    break
            replace = depth == len(rel_parts)
            _alias_module(module, target_name, replace_existing=replace)
    finally:
        try:
            sys.path.remove(import_root)
        except ValueError:
            pass


def pull_remote_dataset(settings: OutputSettings) -> None:
    if not settings.hf_repo_id:
        return
    try:
        snapshot_download(
            repo_id=settings.hf_repo_id,
            repo_type="dataset",
            local_dir=settings.data_root,
            force_download=True,
        )
    except RepositoryNotFoundError:
        print(f"[sniff] Dataset repo {settings.hf_repo_id} not found; skipping pull.")
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to pull dataset repo {settings.hf_repo_id}: {err}")


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
        api.upload_large_folder(
            repo_id=settings.hf_repo_id,
            folder_path=settings.data_root,
            repo_type="dataset",
            commit_message="Update sniffed QK dataset",
        )
    except RepositoryNotFoundError:
        print(f"[sniff] Dataset repo {settings.hf_repo_id} not found; please create it before pushing.")
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to push dataset repo {settings.hf_repo_id}: {err}")


def run_inference(config: SniffConfig) -> None:
    patch_modeling_modules()
    pull_remote_dataset(config.output)
    dataset = load_hf_dataset(config.dataset)
    tokenizer = prepare_tokenizer(config.tokenizer, config.model.name)

    torch_dtype = resolve_dtype(config.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        revision=config.model.revision,
        dtype=torch_dtype,
        device_map=config.model.device_map,
        trust_remote_code=config.model.trust_remote_code,
    )
    model.eval()

    sampler_factory = build_sampler_factory(config.capture.sampler)
    sniffer_config = SnifferConfig(
        model_name=config.model.name,
        data_root=config.output.data_root,
        readme_path=resolve_readme_path(config.output),
        capture_queries=config.capture.capture_queries,
        capture_keys=config.capture.capture_keys,
        layers=set(config.capture.layers) if config.capture.layers else None,
        heads=set(config.capture.heads) if config.capture.heads else None,
        sampler_factory=sampler_factory,
        metadata={
            "source_dataset": config.dataset.path,
            "dataset_name": config.dataset.name or "",
            "dataset_split": config.dataset.split,
        },
    )

    primary_device = resolve_primary_device(model)
    autocast_dtype = resolve_dtype(config.inference.autocast_dtype)
    print("Model device:", primary_device)

    progress = tqdm(total=len(dataset), desc="Capturing", unit="rows")
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
                set_active_example_ids(batch["example_ids"])
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
