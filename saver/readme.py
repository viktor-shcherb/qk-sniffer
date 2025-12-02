from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import yaml

SECTION_DEFS = {
    "models": {
        "title": "## Available Models",
        "start": "<!-- MODELS_START -->",
        "end": "<!-- MODELS_END -->",
        "default": "- (no captures yet)",
    },
    "columns": {
        "title": "## Dataset Columns",
        "start": "<!-- COLUMNS_START -->",
        "end": "<!-- COLUMNS_END -->",
        "default": "",
    },
    "loading": {
        "title": "## Loading Examples",
        "start": "<!-- LOAD_START -->",
        "end": "<!-- LOAD_END -->",
        "default": "",
    },
}


class DatasetReadme:
    """Manages README front matter and sections for the sniffed-qk dataset."""

    def __init__(
        self,
        path: Union[str, Path],
        data_root: Union[str, Path],
        dataset_name: str = "viktoroo/sniffed-qk",
    ):
        self.path = Path(path)
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.front_matter, self.body = self._load_existing()

    def write(
        self,
        config_splits: Dict[str, Set[str]],
        models: Set[str],
        model_metadata: Dict[str, Dict[str, Union[str, int, float]]],
        bucket_counts: Dict[str, Counter],
    ) -> None:
        front_matter = dict(self.front_matter or {})
        front_matter["configs"] = self._build_config_entries(config_splits)
        front_matter["models"] = self._build_model_entries(models)
        front_text = yaml.safe_dump(front_matter, sort_keys=False).strip()

        body = self.body or self._default_body()
        for section_name in SECTION_DEFS:
            body = self._ensure_section(body, section_name)

        body = self._replace_section(body, "models", self._render_models_section(models, model_metadata, bucket_counts))
        body = self._replace_section(body, "columns", self._render_columns_section())
        body = self._replace_section(body, "loading", self._render_loading_section())

        final_text = f"---\n{front_text}\n---\n\n{body.strip()}\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(final_text, encoding="utf-8")

        self.front_matter = front_matter
        self.body = body

    def _load_existing(self) -> tuple[dict, str]:
        if not self.path.exists():
            return {}, self._default_body()
        raw_text = self.path.read_text(encoding="utf-8")
        if raw_text.startswith("---"):
            parts = raw_text.split("---", 2)
            if len(parts) >= 3:
                front_raw = parts[1].strip()
                body = parts[2].lstrip("\r\n")
                front = yaml.safe_load(front_raw) or {}
                return front, body
        return {}, raw_text

    def _build_config_entries(
        self,
        config_splits: Dict[str, Set[str]],
    ) -> List[dict]:
        metadata = self._build_metadata(config_splits)
        entries: List[dict] = []

        # Combined (all keys + queries)
        entries.append(
                self._make_entry(
                    "all",
                    metadata["all_splits"],
                    lambda split: f"{self.data_root.as_posix()}/{split}/*/*.parquet",
                    default=True,
                    require_splits=False,
                )
            )
        for layer in sorted(metadata["layer_splits"]):
            entries.append(
                self._make_entry(
                    f"layer{layer:02d}",
                    metadata["layer_splits"][layer],
                    lambda split, layer=layer: f"{self.data_root.as_posix()}/{split}/l{layer:02d}h*/*.parquet",
                )
            )
        for layer, head in sorted(metadata["head_splits"]):
            entries.append(
                self._make_entry(
                    f"l{layer:02d}h{head:02d}",
                    metadata["head_splits"][(layer, head)],
                    lambda split, layer=layer, head=head: (
                        f"{self.data_root.as_posix()}/{split}/l{layer:02d}h{head:02d}*/*.parquet"
                    ),
                )
            )

        # Query-only sections
        entries.append(
                self._make_entry(
                    "all_q",
                    metadata["kind_splits"]["q"],
                    lambda split: f"{self.data_root.as_posix()}/{split}/*q/*.parquet",
                )
            )
        for layer in sorted(metadata["layer_kind_splits"]["q"]):
            entries.append(
                self._make_entry(
                    f"layer{layer:02d}_q",
                    metadata["layer_kind_splits"]["q"][layer],
                    lambda split, layer=layer: f"{self.data_root.as_posix()}/{split}/l{layer:02d}h*q/*.parquet",
                )
            )
        entries.extend(self._build_kind_head_entries(config_splits, "q"))

        # Key-only sections
        entries.append(
                self._make_entry(
                    "all_k",
                    metadata["kind_splits"]["k"],
                    lambda split: f"{self.data_root.as_posix()}/{split}/*k/*.parquet",
                )
            )
        for layer in sorted(metadata["layer_kind_splits"]["k"]):
            entries.append(
                self._make_entry(
                    f"layer{layer:02d}_k",
                    metadata["layer_kind_splits"]["k"][layer],
                    lambda split, layer=layer: f"{self.data_root.as_posix()}/{split}/l{layer:02d}h*k/*.parquet",
                )
            )
        entries.extend(self._build_kind_head_entries(config_splits, "k"))

        entries = [entry for entry in entries if entry is not None]
        if not entries:
            entries.append({"config_name": "placeholder", "data_files": []})
        return entries

    def _build_model_entries(self, models: Set[str]) -> List[dict]:
        entries: List[dict] = []
        for model in sorted(models):
            url = f"https://huggingface.co/{model}"
            entries.append({"name": model, "url": url})
        if not entries:
            entries.append({"name": "none", "url": ""})
        return entries

    def _default_body(self) -> str:
        return (
            "# sniffed-qk\n\n"
            "Research dataset of captured key and query vectors sampled from transformer attention blocks. "
            "Each configuration (`lXXhYY{q|k}`) corresponds to a specific layer/head and vector type, "
            "while splits match source model checkpoints.\n\n"
            "## Available Models\n"
            "<!-- MODELS_START -->\n"
            "- (no captures yet)\n"
            "<!-- MODELS_END -->\n\n"
            "## Dataset Columns\n"
            "<!-- COLUMNS_START -->\n"
            "<!-- COLUMNS_END -->\n\n"
            "## Loading Examples\n"
            "<!-- LOAD_START -->\n"
            "<!-- LOAD_END -->\n"
        )

    def _ensure_section(self, text: str, section: str) -> str:
        definition = SECTION_DEFS[section]
        start = definition["start"]
        end = definition["end"]
        if start in text and end in text:
            return text
        block = (
            f"{definition['title']}\n"
            f"{start}\n{definition['default']}\n{end}\n"
        )
        if text and not text.endswith("\n"):
            text += "\n"
        return text + ("\n" if text else "") + block

    def _replace_section(self, text: str, section: str, content: str) -> str:
        definition = SECTION_DEFS[section]
        start_marker = definition["start"]
        end_marker = definition["end"]
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker, start_idx + len(start_marker))
        if start_idx == -1 or end_idx == -1:
            return text
        end_idx += len(end_marker)
        new_block = f"{start_marker}\n{content.strip()}\n{end_marker}"
        return text[:start_idx] + new_block + text[end_idx:]

    def _render_models_section(
        self,
        models: Set[str],
        metadata: Dict[str, Dict[str, Union[str, int, float]]],
        bucket_counts: Dict[str, Counter],
    ) -> str:
        if not models:
            return "- (no captures yet)"
        lines = []
        for model in sorted(models):
            lines.append(f"- [{model}](https://huggingface.co/{model})")
            info = metadata.get(model, {})
            source = info.get("source_dataset", "unknown")
            split = info.get("dataset_split", "unknown")
            lines.append(f"  - dataset: {source} (split: {split})")
            counts = bucket_counts.get(model, Counter())
            if counts:
                bucket_str = ", ".join(f"b{bucket}={counts[bucket]}" for bucket in sorted(counts))
            else:
                bucket_str = "(no samples)"
            lines.append(f"  - buckets: {bucket_str}")
        return "\n".join(lines)

    def _render_columns_section(self) -> str:
        rows = [
            ("bucket", "Log2 bucket identifier used for sampling (lower buckets capture earlier positions)."),
            ("example_id", "Index of the example within the batch when the vector was captured."),
            ("position", "Token position within the example's sequence (0-indexed)."),
            ("vector", "Float32 tensor containing the query or key vector; the config name encodes which."),
            ("sliding_window", "Size of the sliding window for local attention (null implies global causal)."),
        ]
        table_lines = ["| Column | Description |", "| --- | --- |"]
        for column, description in rows:
            table_lines.append(f"| `{column}` | {description} |")
        return "\n".join(table_lines)

    def _render_loading_section(self) -> str:
        instructions = [
            "1. Pick the configuration matching your target layer/head and vector type. "
            "For example, `l00h00q` captures queries from layer 0, head 0.",
            "2. Use the source model identifier as the split. Splits follow the Hugging Face hub naming "
            "pattern (`org/name`).",
            "3. Load the dataset via `datasets.load_dataset` with both the config and split:",
            "```python",
            "from datasets import load_dataset",
            f'ds = load_dataset("{self.dataset_name}", "l00h00q", split="org/name")',
            "```",
            "4. Convert to torch/tensorflow as needed; the `vector` column already stores float32 tensors.",
        ]
        return "\n".join(instructions)

    def _build_metadata(self, config_splits: Dict[str, Set[str]]) -> Dict[str, Union[Set[str], Dict]]:
        layer_splits: Dict[int, Set[str]] = defaultdict(set)
        head_splits: Dict[Tuple[int, int], Set[str]] = defaultdict(set)
        kind_splits: Dict[str, Set[str]] = {"q": set(), "k": set()}
        layer_kind_splits: Dict[str, Dict[int, Set[str]]] = {"q": defaultdict(set), "k": defaultdict(set)}
        all_splits: Set[str] = set()

        for config_name, splits in config_splits.items():
            parsed = self._parse_head_config(config_name)
            if not parsed:
                continue
            layer, head, kind = parsed
            layer_splits[layer].update(splits)
            head_splits[(layer, head)].update(splits)
            kind_splits.setdefault(kind, set()).update(splits)
            layer_kind_splits.setdefault(kind, defaultdict(set))[layer].update(splits)
            all_splits.update(splits)

        return {
            "all_splits": all_splits,
            "layer_splits": layer_splits,
            "head_splits": head_splits,
            "kind_splits": kind_splits,
            "layer_kind_splits": layer_kind_splits,
        }

    def _build_kind_head_entries(
        self,
        config_splits: Dict[str, Set[str]],
        kind: str,
    ) -> List[dict]:
        entries: List[dict] = []
        for config_name in sorted(config_splits):
            if not config_name.endswith(kind):
                continue
            splits = config_splits[config_name]
            if not splits:
                continue
            data_files = [
                {"split": split, "path": f"{self.data_root.as_posix()}/{split}/{config_name}/*.parquet"}
                for split in sorted(splits)
            ]
            entries.append({"config_name": config_name, "data_files": data_files})
        return entries

    def _make_entry(
        self,
        config_name: str,
        splits: Set[str],
        path_fn,
        default: bool = False,
        require_splits: bool = True,
    ) -> Optional[dict]:
        if not splits and require_splits:
            return None
        data_files = [
            {"split": split, "path": path_fn(split)}
            for split in sorted(splits)
        ]
        entry = {"config_name": config_name, "data_files": data_files}
        if default:
            entry["default"] = True
        return entry

    def _parse_head_config(self, config_name: str) -> Optional[Tuple[int, int, str]]:
        if len(config_name) != 7 or not (config_name.startswith("l") and config_name[3] == "h"):
            return None
        try:
            layer = int(config_name[1:3])
            head = int(config_name[4:6])
            kind = config_name[6]
            if kind not in {"q", "k"}:
                return None
            return layer, head, kind
        except ValueError:
            return None
