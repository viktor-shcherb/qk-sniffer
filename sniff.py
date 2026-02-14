from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
import hashlib
import re
import shutil
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from dotenv import load_dotenv
from tqdm.auto import tqdm

import torch
import yaml
from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from saver.readme import DatasetReadme
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
class HeadSamplingSettings:
    count: int
    seed: int = 0


@dataclass
class CaptureSettings:
    capture_queries: bool = True
    capture_keys: bool = True
    layers: Optional[List[int]] = None
    heads: Optional[List[int]] = None
    head_sampling: Optional[HeadSamplingSettings] = None
    full_attention_only: bool = False
    sampler: SamplerSettings = field(default_factory=SamplerSettings)
    max_rows_per_batch: Optional[int] = None
    queue_size: int = 32
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
    distributed: bool = False
    backend: str = "auto"
    debug_logging: bool = False
    debug_log_every_n_batches: int = 1
    mps_cleanup_every_batches: int = 0


@dataclass
class OutputSettings:
    data_root: str = "data"
    readme_path: str = "README.md"
    hf_repo_id: Optional[str] = None
    hf_branch: Optional[str] = None
    private: bool = False
    write_batch_size: int = 4096


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


def _debug_log(enabled: bool, message: str) -> None:
    if enabled:
        print(f"[sniff][debug] {message}", flush=True)


def _summarize_tensor_shapes(tensors: Dict[str, torch.Tensor]) -> str:
    parts: List[str] = []
    for name, tensor in tensors.items():
        parts.append(f"{name}:{tuple(tensor.shape)}@{tensor.device}")
    return ", ".join(parts)


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
    head_sampling_raw = capture_raw.get("head_sampling")
    if isinstance(head_sampling_raw, dict):
        capture_raw["head_sampling"] = HeadSamplingSettings(**head_sampling_raw)
    elif head_sampling_raw is None:
        capture_raw["head_sampling"] = None
    else:
        raise ValueError("capture.head_sampling must be an object with fields like count/seed, or null.")
    return SniffConfig(
        dataset=DatasetSettings(**raw["dataset"]),
        model=ModelSettings(**raw["model"]),
        tokenizer=TokenizerSettings(**raw["tokenizer"]),
        inference=InferenceSettings(**raw["inference"]),
        capture=CaptureSettings(**capture_raw),
        output=OutputSettings(**raw["output"]),
    )


def resolve_readme_path(settings: OutputSettings) -> Path:
    _ = settings.readme_path
    return Path(settings.data_root) / "README.md"


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


def _load_raw_model_config(settings: ModelSettings) -> Dict[str, Any]:
    config_path = hf_hub_download(
        repo_id=settings.name,
        filename="config.json",
        revision=settings.revision,
        token=True,
    )
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid config.json payload for {settings.name}: expected a JSON object.")
    return payload


def _build_mistral3_config_with_compat(raw_config: Dict[str, Any]):
    from transformers.models.auto import CONFIG_MAPPING
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    patched = dict(raw_config)
    text_config = patched.get("text_config")
    if isinstance(text_config, dict):
        text_raw = dict(text_config)
        model_type = str(text_raw.get("model_type", "")).lower()
        if model_type == "ministral3" and "ministral3" not in CONFIG_MAPPING:
            if "ministral" in CONFIG_MAPPING:
                text_raw["model_type"] = "ministral"
            else:
                text_raw["model_type"] = "mistral"
        elif model_type == "mistral3" and "mistral3" not in CONFIG_MAPPING:
            # Defensive fallback for older Transformers where only "mistral" exists.
            text_raw["model_type"] = "mistral"
        patched["text_config"] = text_raw
    return Mistral3Config(**patched)


def _prepare_model_config_with_compat(settings: ModelSettings):
    raw_config = _load_raw_model_config(settings)
    if str(raw_config.get("model_type", "")).lower() != "mistral3":
        return None
    return _build_mistral3_config_with_compat(raw_config)


def prepare_model_config(settings: ModelSettings, force: bool = False):
    overrides = settings.config_overrides
    if not force and not overrides:
        return None
    model_config = None
    try:
        model_config = AutoConfig.from_pretrained(
            settings.name,
            revision=settings.revision,
            trust_remote_code=settings.trust_remote_code,
        )
    except Exception:
        model_config = _prepare_model_config_with_compat(settings)
        if model_config is None:
            raise
    if overrides:
        _apply_model_config_overrides(model_config, overrides)
    return model_config


def _load_mistral3_for_conditional_generation_model(
    *,
    settings: ModelSettings,
    model_config,
    torch_dtype: torch.dtype,
    device_map: Optional[Any],
):
    try:
        from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration
    except ImportError as err:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "Mistral 3 requires a Transformers version that includes "
            "transformers.models.mistral3.modeling_mistral3."
        ) from err

    kwargs: Dict[str, Any] = {}
    if model_config is not None:
        kwargs["config"] = model_config
    return Mistral3ForConditionalGeneration.from_pretrained(
        settings.name,
        revision=settings.revision,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=settings.trust_remote_code,
        **kwargs,
    )


def _load_auto_model_for_image_text_to_text(
    *,
    settings: ModelSettings,
    model_config,
    torch_dtype: torch.dtype,
    device_map: Optional[Any],
):
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError as err:  # pragma: no cover - environment-dependent
        raise RuntimeError(
            "This Transformers version does not expose AutoModelForImageTextToText."
        ) from err

    kwargs: Dict[str, Any] = {}
    if model_config is not None:
        kwargs["config"] = model_config
    return AutoModelForImageTextToText.from_pretrained(
        settings.name,
        revision=settings.revision,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=settings.trust_remote_code,
        **kwargs,
    )


def _load_mistral3_model(
    *,
    settings: ModelSettings,
    model_config,
    torch_dtype: torch.dtype,
    device_map: Optional[Any],
):
    try:
        return _load_auto_model_for_image_text_to_text(
            settings=settings,
            model_config=model_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
    except Exception:
        return _load_mistral3_for_conditional_generation_model(
            settings=settings,
            model_config=model_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )


def _load_model_from_pretrained(
    *,
    settings: ModelSettings,
    model_config,
    torch_dtype: torch.dtype,
    device_map: Optional[Any],
):
    config_model_type = str(getattr(model_config, "model_type", "")).lower() if model_config is not None else ""
    if config_model_type == "mistral3":
        return _load_mistral3_model(
            settings=settings,
            model_config=model_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )

    kwargs: Dict[str, Any] = {}
    if model_config is not None:
        kwargs["config"] = model_config
    try:
        return AutoModelForCausalLM.from_pretrained(
            settings.name,
            revision=settings.revision,
            dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=settings.trust_remote_code,
            **kwargs,
        )
    except ValueError as err:
        if "Mistral3Config" not in str(err):
            raise
        compat_config = model_config if model_config is not None else prepare_model_config(settings, True)
        return _load_mistral3_model(
            settings=settings,
            model_config=compat_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )


def resolve_dtype(name: str) -> torch.dtype:
    key = name.lower()
    if key not in DTYPE_ALIASES:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return DTYPE_ALIASES[key]


def resolve_primary_device(model: AutoModelForCausalLM) -> torch.device:
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        for entry in model.hf_device_map.values():
            first_entry = entry[0] if isinstance(entry, (list, tuple)) else entry
            if isinstance(first_entry, int):
                return torch.device(f"cuda:{first_entry}")
            if isinstance(first_entry, str):
                lowered = first_entry.lower()
                if lowered in {"disk", "cpu"}:
                    continue
                try:
                    return torch.device(first_entry)
                except RuntimeError:
                    continue
    try:
        first_param = next(model.parameters())
        return first_param.device
    except (StopIteration, AttributeError, TypeError):
        pass
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class DistributedContext:
    enabled: bool = False
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    is_main: bool = True
    initialized_here: bool = False


def _resolve_distributed_context(settings: InferenceSettings) -> DistributedContext:
    dist = torch.distributed
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    env_rank = int(os.environ.get("RANK", "0"))
    env_local_rank = int(os.environ.get("LOCAL_RANK", str(env_rank)))
    already_initialized = bool(dist.is_available() and dist.is_initialized())
    enabled = bool(settings.distributed or env_world_size > 1 or already_initialized)
    if not enabled:
        return DistributedContext()
    if not dist.is_available():
        raise RuntimeError("Distributed inference requested but torch.distributed is unavailable.")

    backend = settings.backend.lower().strip()
    if backend == "auto":
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL backend requires CUDA devices.")

    if torch.cuda.is_available():
        torch.cuda.set_device(env_local_rank)

    initialized_here = False
    if not dist.is_initialized():
        if env_world_size <= 1:
            raise RuntimeError(
                "Distributed inference requires WORLD_SIZE > 1. "
                "Launch with torchrun (for example: torchrun --nproc_per_node=2 sniff.py --config ...)."
            )
        dist.init_process_group(
            backend=backend,
            rank=env_rank,
            world_size=env_world_size,
        )
        initialized_here = True

    rank = int(dist.get_rank())
    world_size = int(dist.get_world_size())
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main=rank == 0,
        initialized_here=initialized_here,
    )


def _distributed_barrier(ctx: DistributedContext) -> None:
    if not ctx.enabled:
        return
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _destroy_distributed_context(ctx: DistributedContext) -> None:
    if not ctx.enabled or not ctx.initialized_here:
        return
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def _distributed_work_root(data_root: Path) -> Path:
    return data_root.parent / f".{data_root.name}_dist_work"


def _mps_is_available() -> bool:
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    try:
        return bool(backend.is_available())
    except Exception:
        return False


def _normalize_device_map_for_runtime(
    device_map: Optional[Any],
    *,
    distributed: DistributedContext,
) -> Optional[Any]:
    if distributed.enabled:
        return device_map
    if isinstance(device_map, str) and device_map.lower() == "auto" and not torch.cuda.is_available():
        # Avoid Accelerate meta/device-map flows on non-CUDA backends (notably MPS),
        # which can leave placeholder storages unmaterialized during eager inference.
        print("[sniff] Using eager single-device loading (device_map=None) because CUDA is unavailable.")
        return None
    return device_map


def _maybe_place_single_process_model(
    model: AutoModelForCausalLM,
    *,
    distributed: DistributedContext,
    requested_device_map: Optional[Any],
    requested_dtype: torch.dtype,
):
    if distributed.enabled:
        return model
    if requested_device_map is not None:
        return model
    if not hasattr(model, "to"):
        return model

    if torch.cuda.is_available():
        return model.to(device=torch.device("cuda"), dtype=requested_dtype)
    if not _mps_is_available():
        return model

    target_dtype = requested_dtype
    if target_dtype == torch.bfloat16:
        target_dtype = torch.float16
        print("[sniff] Using float16 on MPS (requested bfloat16).")
    return model.to(device=torch.device("mps"), dtype=target_dtype)


def _rank_output_settings(base: OutputSettings, rank_root: Path) -> OutputSettings:
    return OutputSettings(
        data_root=str(rank_root),
        readme_path="README.md",
        hf_repo_id=None,
        hf_branch=base.hf_branch,
        private=base.private,
        write_batch_size=base.write_batch_size,
    )


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


def _iter_model_config_candidates(model: AutoModelForCausalLM) -> Iterable[Any]:
    queue = [getattr(model, "config", None)]
    seen: Set[int] = set()
    nested_attrs = ("text_config", "language_config", "llm_config", "decoder_config")
    while queue:
        candidate = queue.pop(0)
        if candidate is None:
            continue
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        yield candidate
        for attr in nested_attrs:
            nested = getattr(candidate, attr, None)
            if nested is not None:
                queue.append(nested)


def _resolve_model_head_layout(model: AutoModelForCausalLM) -> Tuple[int, int, int]:
    for candidate in _iter_model_config_candidates(model):
        num_layers_raw = getattr(candidate, "num_hidden_layers", None)
        num_query_heads_raw = getattr(candidate, "num_attention_heads", None)
        if num_layers_raw is None or num_query_heads_raw is None:
            continue
        num_key_heads_raw = getattr(candidate, "num_key_value_heads", num_query_heads_raw)
        try:
            num_layers = int(num_layers_raw)
            num_query_heads = int(num_query_heads_raw)
            num_key_heads = int(num_key_heads_raw)
        except (TypeError, ValueError):
            continue
        if num_layers <= 0 or num_query_heads <= 0 or num_key_heads <= 0:
            continue
        if num_query_heads % num_key_heads != 0:
            raise ValueError(
                "Unsupported attention layout: num_attention_heads must be divisible by num_key_value_heads "
                f"(got {num_query_heads} and {num_key_heads})."
            )
        return num_layers, num_query_heads, num_key_heads
    raise ValueError(
        "Unable to resolve attention head layout from model config. "
        "Expected num_hidden_layers and num_attention_heads."
    )


def _resolve_sliding_attention_layers(model: AutoModelForCausalLM, *, num_layers: int) -> Set[int]:
    for candidate in _iter_model_config_candidates(model):
        layer_types = getattr(candidate, "layer_types", None)
        if layer_types is None:
            continue
        if isinstance(layer_types, (str, bytes)) or not isinstance(layer_types, Sequence):
            continue
        if len(layer_types) < num_layers:
            continue
        sliding: Set[int] = set()
        for layer_idx in range(num_layers):
            if str(layer_types[layer_idx]) == "sliding_attention":
                sliding.add(layer_idx)
        return sliding
    return set()


def resolve_head_filters(
    *,
    model: AutoModelForCausalLM,
    capture_settings: CaptureSettings,
) -> Tuple[Optional[Dict[int, Set[int]]], Optional[Dict[int, Set[int]]], Optional[Set[int]], Dict[str, Union[str, int, float]]]:
    explicit_heads = set(capture_settings.heads) if capture_settings.heads else None
    sampling = capture_settings.head_sampling
    if sampling is None:
        return None, None, explicit_heads, {}

    requested_count = int(sampling.count)
    if requested_count < 1:
        raise ValueError("capture.head_sampling.count must be at least 1.")
    seed = int(sampling.seed)
    full_attention_only = bool(capture_settings.full_attention_only)

    num_layers, num_query_heads, num_key_heads = _resolve_model_head_layout(model)
    sliding_layers = _resolve_sliding_attention_layers(model, num_layers=num_layers)
    layers_filter = set(capture_settings.layers) if capture_settings.layers else None
    if layers_filter is not None:
        invalid_layers = sorted(layer for layer in layers_filter if layer < 0 or layer >= num_layers)
        if invalid_layers:
            raise ValueError(
                f"Invalid layer indices {invalid_layers}; model exposes layers 0..{num_layers - 1}."
            )
    if explicit_heads is not None:
        invalid_heads = sorted(head for head in explicit_heads if head < 0 or head >= num_query_heads)
        if invalid_heads:
            raise ValueError(
                f"Invalid query head indices {invalid_heads}; model exposes query heads 0..{num_query_heads - 1}."
            )

    candidates: List[Tuple[int, int]] = []
    excluded_sliding_candidates = 0
    for layer_idx in range(num_layers):
        if layers_filter is not None and layer_idx not in layers_filter:
            continue
        is_sliding_layer = layer_idx in sliding_layers
        for head_idx in range(num_query_heads):
            if explicit_heads is not None and head_idx not in explicit_heads:
                continue
            if full_attention_only and is_sliding_layer:
                excluded_sliding_candidates += 1
                continue
            candidates.append((layer_idx, head_idx))
    if not candidates:
        if full_attention_only:
            raise ValueError(
                "capture.head_sampling produced no candidate query heads after excluding sliding-attention layers."
            )
        raise ValueError("capture.head_sampling produced no candidate query heads after applying filters.")

    selected_count = min(requested_count, len(candidates))
    if selected_count == len(candidates):
        selected = list(candidates)
    else:
        rng = random.Random(seed)
        selected = rng.sample(candidates, selected_count)

    query_heads_by_layer: Dict[int, Set[int]] = {}
    key_heads_by_layer: Dict[int, Set[int]] = {}
    queries_per_key = num_query_heads // num_key_heads
    for layer_idx, query_head in selected:
        query_heads_by_layer.setdefault(layer_idx, set()).add(query_head)
        key_head = query_head // queries_per_key
        key_heads_by_layer.setdefault(layer_idx, set()).add(key_head)

    sampled_key_pairs = sum(len(heads) for heads in key_heads_by_layer.values())
    metadata: Dict[str, Union[str, int, float]] = {
        "head_sampling_strategy": "random_query",
        "head_sampling_seed": seed,
        "head_sampling_requested_query_heads": requested_count,
        "head_sampling_selected_query_heads": selected_count,
        "head_sampling_selected_key_heads": sampled_key_pairs,
        "head_sampling_num_layers": num_layers,
        "head_sampling_num_query_heads": num_query_heads,
        "head_sampling_num_key_heads": num_key_heads,
        "head_sampling_full_attention_only": int(full_attention_only),
        "head_sampling_sliding_layer_count": len(sliding_layers),
        "head_sampling_full_layer_count": num_layers - len(sliding_layers),
        "head_sampling_sliding_candidates_excluded": excluded_sliding_candidates,
    }
    return query_heads_by_layer, key_heads_by_layer, None, metadata


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


def _shard_dataset_for_rank(
    dataset: Dataset | IterableDataset,
    *,
    rank: int,
    world_size: int,
) -> Dataset | IterableDataset:
    if world_size <= 1:
        return dataset
    return dataset.shard(num_shards=world_size, index=rank)


def _estimate_total_rows(
    dataset: Dataset | IterableDataset,
    *,
    dataset_settings: DatasetSettings,
    distributed: DistributedContext,
) -> Optional[int]:
    if isinstance(dataset, IterableDataset):
        if dataset_settings.max_samples is None:
            return None
        if distributed.world_size <= 1:
            return int(dataset_settings.max_samples)
        per_rank = dataset_settings.max_samples // distributed.world_size
        remainder = dataset_settings.max_samples % distributed.world_size
        return per_rank + (1 if distributed.rank < remainder else 0)
    return len(dataset)


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


def pull_remote_dataset(settings: OutputSettings) -> None:
    target_root = Path(settings.data_root)
    if target_root.exists():
        shutil.rmtree(target_root)
    target_root.mkdir(parents=True, exist_ok=True)


def _ensure_main_branch_readme(api: HfApi, repo_id: str) -> None:
    try:
        main_files = api.list_repo_files(
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
        )
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to inspect main branch for {repo_id}: {err}")
        return
    if main_files:
        return
    readme_text = DatasetReadme(Path("README.md"), dataset_name=repo_id).main_branch_text()
    try:
        api.upload_file(
            path_or_fileobj=readme_text.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
            revision="main",
            commit_message="Add default branch README",
        )
        print(f"[sniff] Added README.md to empty main branch for {repo_id}")
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to seed main README for {repo_id}: {err}")


def push_remote_dataset(settings: OutputSettings) -> None:
    if not settings.hf_repo_id:
        return
    api = HfApi()
    revision = settings.hf_branch or None
    try:
        api.repo_info(settings.hf_repo_id, repo_type="dataset")
    except RepositoryNotFoundError:
        print(f"[sniff] Creating dataset repo {settings.hf_repo_id}")
        api.create_repo(settings.hf_repo_id, repo_type="dataset", exist_ok=True, private=settings.private)
    _ensure_main_branch_readme(api, settings.hf_repo_id)
    if revision:
        try:
            api.create_branch(
                settings.hf_repo_id,
                repo_type="dataset",
                branch=revision,
                exist_ok=True,
            )
        except HfHubHTTPError as err:
            print(f"[sniff] Failed to create/use branch {revision} for {settings.hf_repo_id}: {err}")
            return
    try:
        upload_kwargs: Dict[str, Any] = {
            "repo_id": settings.hf_repo_id,
            "folder_path": settings.data_root,
            "repo_type": "dataset",
            "commit_message": "Update dataset",
            "revision": revision,
            "delete_patterns": ["*", "**/*"],
        }
        api.upload_folder(
            **upload_kwargs,
        )
    except RepositoryNotFoundError:
        print(f"[sniff] Dataset repo {settings.hf_repo_id} not found; please create it before pushing.")
    except HfHubHTTPError as err:
        print(f"[sniff] Failed to push dataset repo {settings.hf_repo_id}: {err}")


def _load_rank_model_metadata(rank_root: Path) -> Tuple[Optional[str], Dict[str, Union[str, int, float]]]:
    state_path = rank_root / "_saver_state.json"
    if not state_path.exists():
        return None, {}
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None, {}

    model_name = payload.get("model_name")
    metadata = payload.get("model_metadata")
    if isinstance(model_name, str) and isinstance(metadata, dict):
        cleaned = {
            key: value
            for key, value in metadata.items()
            if isinstance(key, str) and isinstance(value, (str, int, float))
        }
        return model_name, cleaned

    # Best-effort fallback for old state format.
    if isinstance(metadata, dict) and metadata:
        first_model_name = next(iter(metadata.keys()))
        first_model_meta = metadata.get(first_model_name)
        if isinstance(first_model_name, str) and isinstance(first_model_meta, dict):
            cleaned = {
                key: value
                for key, value in first_model_meta.items()
                if isinstance(key, str) and isinstance(value, (str, int, float))
            }
            return first_model_name, cleaned

    return None, {}


def _resolve_sliding_window_value(values: Sequence[Any]) -> Optional[int]:
    non_null = {int(value) for value in values if value is not None}
    if len(non_null) > 1:
        raise ValueError(f"Inconsistent sliding_window values in rank shard: {sorted(non_null)}")
    if not non_null:
        return None
    return next(iter(non_null))


def merge_rank_outputs(rank_roots: Sequence[Path], output_settings: OutputSettings) -> None:
    import numpy as np
    import pyarrow.parquet as pq

    from saver.dataset import CaptureBatch, DatasetSaver

    final_root = Path(output_settings.data_root)
    final_root.mkdir(parents=True, exist_ok=True)
    readme_path = resolve_readme_path(output_settings)
    saver = DatasetSaver(
        root=final_root,
        readme_path=readme_path,
        write_batch_size=max(1, int(output_settings.write_batch_size)),
    )
    try:
        fallback_model_name: Optional[str] = None
        for rank_root in sorted(rank_roots):
            model_name, metadata = _load_rank_model_metadata(rank_root)
            if model_name:
                saver.register_model_metadata(model_name, metadata)
                fallback_model_name = model_name

            for config_dir in sorted(rank_root.iterdir()):
                if not config_dir.is_dir():
                    continue
                if config_dir.name.startswith("_"):
                    continue
                match = _CONFIG_NAME_RE.match(config_dir.name)
                if match is None:
                    continue
                data_file = config_dir / "data.parquet"
                if not data_file.exists():
                    continue
                table = pq.read_table(data_file)
                if table.num_rows == 0:
                    continue
                token_strings: Optional[List[str]] = None
                if "token_str" in table.schema.names:
                    token_strings = table.column("token_str").to_pylist()
                sliding_values = (
                    table.column("sliding_window").to_pylist()
                    if "sliding_window" in table.schema.names
                    else [None] * table.num_rows
                )
                batch = CaptureBatch(
                    model_name=model_name or fallback_model_name or "unknown/model",
                    layer_idx=int(match.group(1)),
                    head_idx=int(match.group(2)),
                    vector_kind=match.group(3),  # type: ignore[arg-type]
                    buckets=np.asarray(table.column("bucket").to_pylist(), dtype=np.int64),
                    example_ids=np.asarray(table.column("example_id").to_pylist(), dtype=np.int64),
                    positions=np.asarray(table.column("position").to_pylist(), dtype=np.int64),
                    vectors=np.asarray(table.column("vector").to_pylist(), dtype=np.float32),
                    sliding_window=_resolve_sliding_window_value(sliding_values),
                    token_strings=token_strings,
                )
                saver.add_batch(batch)
    finally:
        saver.close()


def run_inference(config: SniffConfig) -> None:
    patch_modeling_modules()
    distributed = _resolve_distributed_context(config.inference)
    debug_logging = bool(config.inference.debug_logging)
    debug_log_every_n_batches = max(1, int(config.inference.debug_log_every_n_batches))
    _debug_log(
        debug_logging,
        (
            "runtime setup: "
            f"distributed={distributed.enabled}, world_size={distributed.world_size}, "
            f"rank={distributed.rank}, local_rank={distributed.local_rank}, "
            f"cuda_available={torch.cuda.is_available()}, mps_available={_mps_is_available()}"
        ),
    )
    final_output_settings = config.output
    final_data_root = Path(final_output_settings.data_root)
    rank_output_settings = final_output_settings
    dist_work_root: Optional[Path] = None

    try:
        if distributed.enabled:
            dist_work_root = _distributed_work_root(final_data_root)
            if distributed.is_main:
                if dist_work_root.exists():
                    shutil.rmtree(dist_work_root)
                dist_work_root.mkdir(parents=True, exist_ok=True)
            _distributed_barrier(distributed)
            rank_root = dist_work_root / f"rank{distributed.rank:05d}"
            rank_root.mkdir(parents=True, exist_ok=True)
            rank_output_settings = _rank_output_settings(final_output_settings, rank_root)

            if distributed.is_main:
                pull_remote_dataset(final_output_settings)
            _distributed_barrier(distributed)
        else:
            pull_remote_dataset(final_output_settings)
            final_data_root.mkdir(parents=True, exist_ok=True)

        dataset_load_start = perf_counter()
        dataset = load_hf_dataset(config.dataset)
        _debug_log(
            debug_logging,
            (
                f"dataset loaded in {perf_counter() - dataset_load_start:.2f}s: "
                f"path={config.dataset.path}, split={config.dataset.split}, streaming={config.dataset.streaming}"
            ),
        )
        if distributed.world_size > 1:
            dataset = _shard_dataset_for_rank(
                dataset,
                rank=distributed.rank,
                world_size=distributed.world_size,
            )

        tokenizer_load_start = perf_counter()
        tokenizer = prepare_tokenizer(config.tokenizer, config.model.name)
        _debug_log(debug_logging, f"tokenizer prepared in {perf_counter() - tokenizer_load_start:.2f}s")
        readme_path = resolve_readme_path(rank_output_settings)

        torch_dtype = resolve_dtype(config.model.dtype)
        model_config = prepare_model_config(config.model)
        device_map = config.model.device_map
        if distributed.enabled and torch.cuda.is_available():
            device_map = {"": distributed.local_rank}
        elif distributed.enabled:
            device_map = None
        device_map = _normalize_device_map_for_runtime(device_map, distributed=distributed)

        _debug_log(debug_logging, f"loading model: dtype={torch_dtype}, device_map={device_map!r}")
        model_load_start = perf_counter()
        model = _load_model_from_pretrained(
            settings=config.model,
            model_config=model_config,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        _debug_log(debug_logging, f"model loaded in {perf_counter() - model_load_start:.2f}s")
        model = _maybe_place_single_process_model(
            model,
            distributed=distributed,
            requested_device_map=device_map,
            requested_dtype=torch_dtype,
        )
        model.eval()
        _debug_log(debug_logging, "model switched to eval mode")

        head_filter_start = perf_counter()
        sampled_query_heads, sampled_key_heads, explicit_heads, head_sampling_metadata = resolve_head_filters(
            model=model,
            capture_settings=config.capture,
        )
        _debug_log(debug_logging, f"head filters resolved in {perf_counter() - head_filter_start:.2f}s")

        sampler_factory = build_sampler_factory(config.capture.sampler, min_bucket_size=config.capture.min_bucket_size)
        metadata: Dict[str, Union[str, int, float]] = {
            "source_dataset": config.dataset.path,
            "dataset_name": config.dataset.name or "",
            "dataset_split": config.dataset.split,
            "attention_scope": "full_only" if config.capture.full_attention_only else "all_attention",
            "distributed_world_size": distributed.world_size,
            "dataset_repo": config.output.hf_repo_id or "",
            "dataset_branch": config.output.hf_branch or "",
        }
        metadata.update(head_sampling_metadata)
        sniffer_config = SnifferConfig(
            model_name=config.model.name,
            data_root=rank_output_settings.data_root,
            readme_path=readme_path,
            capture_queries=config.capture.capture_queries,
            capture_keys=config.capture.capture_keys,
            layers=set(config.capture.layers) if config.capture.layers else None,
            heads=explicit_heads,
            sampled_query_heads=sampled_query_heads,
            sampled_key_heads=sampled_key_heads,
            sampler_factory=sampler_factory,
            max_rows_per_batch=config.capture.max_rows_per_batch,
            queue_size=config.capture.queue_size,
            write_batch_size=rank_output_settings.write_batch_size,
            min_bucket_size=config.capture.min_bucket_size,
            capture_pre_rope=config.capture.capture_pre_rope,
            capture_token_strings=config.capture.capture_token_strings,
            full_attention_only=config.capture.full_attention_only,
            metadata=metadata,
            debug_logging=debug_logging,
        )

        if distributed.enabled and torch.cuda.is_available():
            primary_device = torch.device(f"cuda:{distributed.local_rank}")
        else:
            primary_device = resolve_primary_device(model)
        autocast_dtype = resolve_dtype(config.inference.autocast_dtype)
        print("Model device:", primary_device)
        _debug_log(debug_logging, f"autocast dtype={autocast_dtype}")

        total_rows = _estimate_total_rows(
            dataset,
            dataset_settings=config.dataset,
            distributed=distributed,
        )
        _debug_log(debug_logging, f"progress target rows={total_rows}")
        progress = tqdm(
            total=total_rows,
            desc="Capturing",
            unit="rows",
            disable=bool(distributed.enabled and not distributed.is_main),
        )
        local_example_index = 0
        batch_idx = 0
        mps_cleanup_every = max(0, int(config.inference.mps_cleanup_every_batches))
        try:
            with activate_sniffer(sniffer_config) as active_sniffer:
                flush_batch = getattr(active_sniffer, "flush_batch", None)
                consume_debug_stats = getattr(active_sniffer, "consume_debug_stats", None)
                for batch in batch_iter(dataset, config.dataset, config.inference.batch_size):
                    batch_idx += 1
                    texts: Sequence[str] = batch["texts"]
                    if not texts:
                        continue
                    debug_this_batch = debug_logging and (
                        batch_idx <= 3 or batch_idx % debug_log_every_n_batches == 0
                    )
                    batch_start = perf_counter()
                    if debug_this_batch:
                        text_lengths = [len(text) for text in texts]
                        avg_chars = sum(text_lengths) / len(text_lengths)
                        _debug_log(
                            True,
                            (
                                f"batch {batch_idx} start: rows={len(texts)}, "
                                f"text_chars(min/avg/max)="
                                f"{min(text_lengths)}/{avg_chars:.1f}/{max(text_lengths)}"
                            ),
                        )
                        _debug_log(True, f"batch {batch_idx}: tokenizer start")
                    tokenize_start = perf_counter()
                    encodings = tokenizer(
                        list(texts),
                        return_tensors="pt",
                        padding=config.tokenizer.padding,
                        truncation=True,
                        max_length=config.tokenizer.max_length,
                    )
                    if debug_this_batch:
                        tokenized_in = perf_counter() - tokenize_start
                        input_ids = encodings.get("input_ids")
                        seq_len = int(input_ids.shape[-1]) if input_ids is not None else None
                        _debug_log(
                            True,
                            (
                                f"batch {batch_idx}: tokenizer done in {tokenized_in:.3f}s, "
                                f"seq_len={seq_len}, tensors=({_summarize_tensor_shapes(encodings)})"
                            ),
                        )
                    batch_example_ids = batch["example_ids"]
                    if distributed.world_size > 1 and config.dataset.id_column is None:
                        count = len(batch_example_ids)
                        batch_example_ids = [
                            (local_example_index + idx) * distributed.world_size + distributed.rank
                            for idx in range(count)
                        ]
                        local_example_index += count
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
                    if debug_this_batch:
                        valid_min = min(valid_lengths)
                        valid_max = max(valid_lengths)
                        _debug_log(True, f"batch {batch_idx}: valid token lengths min/max={valid_min}/{valid_max}")
                        _debug_log(True, f"batch {batch_idx}: moving tensors to device {primary_device}")
                    move_start = perf_counter()
                    inputs = {k: v.to(primary_device) for k, v in encodings.items()}
                    if debug_this_batch:
                        _debug_log(
                            True,
                            (
                                f"batch {batch_idx}: device transfer done in {perf_counter() - move_start:.3f}s, "
                                f"tensors=({_summarize_tensor_shapes(inputs)})"
                            ),
                        )
                        _debug_log(True, f"batch {batch_idx}: forward start")
                    forward_start = perf_counter()
                    with torch.no_grad():
                        use_autocast = primary_device.type == "cuda"
                        context = (
                            torch.autocast(device_type=primary_device.type, dtype=autocast_dtype)
                            if use_autocast
                            else nullcontext()
                        )
                        with context:
                            model(**inputs)
                    if callable(flush_batch):
                        flush_batch()
                    if debug_this_batch:
                        if callable(consume_debug_stats):
                            stats = consume_debug_stats()
                        else:
                            stats = {
                                "capture_invocations": 0,
                                "capture_time_s": 0.0,
                                "flush_invocations": 0,
                                "flush_time_s": 0.0,
                                "captured_payloads": 0,
                                "captured_rows": 0,
                                "submit_wait_s": 0.0,
                                "pending_configs": 0,
                                "writer_batches_total": 0,
                                "writer_rows_total": 0,
                                "writer_queue_depth": 0,
                            }
                        _debug_log(
                            True,
                            f"batch {batch_idx}: forward done in {perf_counter() - forward_start:.3f}s",
                        )
                        _debug_log(
                            True,
                            (
                                f"batch {batch_idx}: capture stats "
                                f"(hook_calls={int(stats['capture_invocations'])}, "
                                f"hook_s={float(stats['capture_time_s']):.3f}, "
                                f"flush_calls={int(stats['flush_invocations'])}, "
                                f"flush_s={float(stats['flush_time_s']):.3f}, "
                                f"payloads={int(stats['captured_payloads'])}, "
                                f"rows={int(stats['captured_rows'])}, "
                                f"queue_wait_s={float(stats['submit_wait_s']):.3f}, "
                                f"pending_configs={int(stats['pending_configs'])}, "
                                f"writer_total_batches={int(stats['writer_batches_total'])}, "
                                f"writer_total_rows={int(stats['writer_rows_total'])}, "
                                f"writer_queue={int(stats['writer_queue_depth'])})"
                            ),
                        )
                    progress.update(len(texts))
                    if primary_device.type == "mps" and mps_cleanup_every > 0 and batch_idx % mps_cleanup_every == 0:
                        cleanup_start = perf_counter()
                        gc.collect()
                        mps_backend = getattr(torch, "mps", None)
                        if mps_backend is not None and hasattr(mps_backend, "empty_cache"):
                            mps_backend.empty_cache()
                        if debug_this_batch:
                            _debug_log(
                                True,
                                f"batch {batch_idx}: mps cleanup in {perf_counter() - cleanup_start:.3f}s",
                            )
                    if debug_this_batch:
                        _debug_log(
                            True,
                            f"batch {batch_idx} done in {perf_counter() - batch_start:.3f}s",
                        )
        finally:
            progress.close()

        if distributed.enabled:
            _distributed_barrier(distributed)
            if distributed.is_main:
                if dist_work_root is None:
                    raise RuntimeError("Distributed work root is not initialized.")
                rank_roots = [path for path in sorted(dist_work_root.iterdir()) if path.is_dir()]
                merge_rank_outputs(rank_roots, final_output_settings)
            _distributed_barrier(distributed)
            if distributed.is_main:
                push_remote_dataset(final_output_settings)
            _distributed_barrier(distributed)
            if distributed.is_main and dist_work_root is not None:
                shutil.rmtree(dist_work_root, ignore_errors=True)
            _distributed_barrier(distributed)
        else:
            push_remote_dataset(final_output_settings)
    finally:
        _destroy_distributed_context(distributed)


def main():
    args = parse_args()
    config = load_config(args.config)
    run_inference(config)


if __name__ == "__main__":
    main()
