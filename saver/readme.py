from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Union


class DatasetReadme:
    """Writes a static README for a single model-specialization dataset branch."""

    def __init__(
        self,
        path: Union[str, Path],
        dataset_name: str = "viktoroo/sniffed-qk",
    ):
        self.path = Path(path)
        self.dataset_name = dataset_name

    def write(
        self,
        *,
        model_name: Optional[str],
        metadata: Dict[str, Union[str, int, float]],
        config_names: Sequence[str],
        bucket_counts: Dict[int, int],
    ) -> None:
        if model_name is None and not metadata and not config_names and not bucket_counts:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(self.main_branch_text(), encoding="utf-8")
            return

        model_display = model_name or "unknown"
        source_dataset = str(metadata.get("source_dataset", "unknown"))
        dataset_name = str(metadata.get("dataset_name", ""))
        dataset_split = str(metadata.get("dataset_split", "unknown"))
        attention_scope = str(metadata.get("attention_scope", "unknown"))
        sampling_strategy = str(metadata.get("sampling_strategy", "unknown"))
        min_bucket_size = metadata.get("sampling_min_bucket_size")
        uniform_bucket_size = metadata.get("sampling_bucket_size")
        world_size = metadata.get("distributed_world_size")

        config_lines = "\n".join(f"- `{name}`" for name in sorted(config_names))
        if not config_lines:
            config_lines = "- (no captures yet)"

        bucket_line = self._format_bucket_counts(bucket_counts)
        total_rows = sum(int(v) for v in bucket_counts.values()) if bucket_counts else 0

        extra_lines = self._render_extra_metadata(metadata)
        extra_section = "\n".join(extra_lines) if extra_lines else "- (none)"

        bucket_detail_lines = []
        if min_bucket_size is not None:
            bucket_detail_lines.append(f"- `sampling_min_bucket_size`: `{self._fmt(min_bucket_size)}`")
        if uniform_bucket_size is not None:
            bucket_detail_lines.append(f"- `sampling_bucket_size`: `{self._fmt(uniform_bucket_size)}`")
        if not bucket_detail_lines:
            bucket_detail_lines.append("- Bucket size details are not recorded for this run.")
        bucket_detail = "\n".join(bucket_detail_lines)

        text = (
            "# sniffed-qk (model specialization)\n\n"
            "This branch stores one model-specific specialization of the sniffed-qk dataset. "
            "Use a dedicated Git branch per `(model, source dataset, split, capture settings)` combination.\n\n"
            "## Branching\n"
            "- The default branch (`main`) should contain only `README.md` and no Parquet data.\n"
            "- Specializations are written to dedicated branches selected via `output.hf_branch`.\n"
            "- Load a specialization by pinning the branch with `revision=<branch_name>`.\n\n"
            "## Current Specialization Metadata\n"
            f"- `model`: `{model_display}`\n"
            f"- `source_dataset`: `{source_dataset}`\n"
            f"- `dataset_name`: `{dataset_name}`\n"
            f"- `dataset_split`: `{dataset_split}`\n"
            f"- `attention_scope`: `{attention_scope}`\n"
            f"- `sampling_strategy`: `{sampling_strategy}`\n"
            f"- `distributed_world_size`: `{self._fmt(world_size)}`\n"
            f"- `captured_configs`: `{len(config_names)}`\n"
            f"- `captured_rows`: `{total_rows}`\n"
            f"- `bucket_counts`: {bucket_line}\n\n"
            "## Bucket Parameters\n"
            f"{bucket_detail}\n\n"
            "## Captured Configs\n"
            f"{config_lines}\n\n"
            "## Dataset Layout\n"
            "- `README.md`\n"
            "- `lXXhYYq/data.parquet` for query vectors\n"
            "- `lXXhYYk/data.parquet` for key vectors\n\n"
            "## Dataset Columns\n"
            "| Column | Description |\n"
            "| --- | --- |\n"
            "| `bucket` | Bucket identifier produced by the configured sampling strategy. |\n"
            "| `example_id` | Example identifier for the source sample. |\n"
            "| `position` | Token position inside the source example sequence. |\n"
            "| `token_str` | Optional token string captured for that position. |\n"
            "| `vector` | Float32 Q/K vector payload. |\n"
            "| `sliding_window` | Sliding-window size for local attention; null for full causal. |\n\n"
            "## Generation Code\n"
            "- Entry point: `sniff.py` (`sniff-qk`)\n"
            "- Capture runtime: `sniffer/core.py`\n"
            "- Dataset writing: `saver/dataset.py`\n"
            "- README template: `saver/readme.py`\n\n"
            "## Loading Data\n"
            "```python\n"
            "from datasets import load_dataset\n\n"
            f"branch = \"<your-branch>\"\n"
            f"repo = \"{self.dataset_name}\"\n"
            "config = \"l00h00q\"\n\n"
            "# Local clone / snapshot\n"
            "local_ds = load_dataset(\"parquet\", data_files=f\"{config}/*.parquet\", split=\"train\")\n\n"
            "# Directly from the Hub branch\n"
            "hub_ds = load_dataset(\n"
            "    \"parquet\",\n"
            "    data_files=f\"hf://datasets/{repo}@{branch}/{config}/*.parquet\",\n"
            "    split=\"train\",\n"
            ")\n"
            "```\n\n"
            "## Additional Captured Metadata\n"
            f"{extra_section}\n"
        )

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(text, encoding="utf-8")

    def main_branch_text(self) -> str:
        return (
            "# sniffed-qk\n\n"
            "This is the default branch for the sniffed-qk dataset.\n\n"
            "## How Versions Are Organized\n"
            "- This branch intentionally stores no Parquet data files.\n"
            "- Each model+dataset specialization lives on a dedicated Git branch.\n"
            "- Pick the branch name from your run configuration (`output.hf_branch`) and load data from that branch.\n\n"
            "## Branch Layout\n"
            "- `README.md` at the branch root\n"
            "- `lXXhYYq/data.parquet` (queries)\n"
            "- `lXXhYYk/data.parquet` (keys)\n\n"
            "## Loading Example\n"
            "```python\n"
            "from datasets import load_dataset\n\n"
            f"repo = \"{self.dataset_name}\"\n"
            "branch = \"<model-specialization-branch>\"\n"
            "config = \"l00h00q\"\n\n"
            "ds = load_dataset(\n"
            "    \"parquet\",\n"
            "    data_files=f\"hf://datasets/{repo}@{branch}/{config}/*.parquet\",\n"
            "    split=\"train\",\n"
            ")\n"
            "```\n"
        )

    @staticmethod
    def _format_bucket_counts(bucket_counts: Dict[int, int]) -> str:
        if not bucket_counts:
            return "(no samples)"
        parts = [f"b{bucket}={count}" for bucket, count in sorted(bucket_counts.items())]
        return ", ".join(parts)

    @staticmethod
    def _fmt(value: Optional[Union[str, int, float]]) -> str:
        if value is None:
            return "unknown"
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)

    @staticmethod
    def _render_extra_metadata(metadata: Dict[str, Union[str, int, float]]) -> Sequence[str]:
        known = {
            "source_dataset",
            "dataset_name",
            "dataset_split",
            "attention_scope",
            "sampling_strategy",
            "sampling_min_bucket_size",
            "sampling_bucket_size",
            "distributed_world_size",
        }
        lines = []
        for key in sorted(metadata.keys()):
            if key in known:
                continue
            lines.append(f"- `{key}`: `{metadata[key]}`")
        return lines
