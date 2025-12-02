from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple, Union

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy should usually be installed.
    np = None  # type: ignore[assignment]

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

import pyarrow as pa
import pyarrow.parquet as pq

from .readme import DatasetReadme

VectorKind = Literal["k", "q"]


@dataclass(slots=True)
class CaptureRow:
    """
    Container describing a single query/key sample.

    Attributes:
        model_name: Acts as the HF dataset split (e.g. ``gemma3-9b``).
        layer_idx: Decoder layer index.
        head_idx: Attention head index within the layer.
        vector_kind: ``"q"`` or ``"k"`` – appended to the config name.
        bucket: Log2 bucket id (or any other integer grouping).
        example_id: Identifier of the example/batch item.
        position: Position of the token within the example sequence.
        vector: numpy array, tensor, or sequence of floats representing the vector.
        sliding_window: Size of the sliding window (``None`` for fully causal attention).
    """

    model_name: str
    layer_idx: int
    head_idx: int
    vector_kind: VectorKind
    bucket: int
    example_id: int
    position: int
    vector: Union["np.ndarray", "torch.Tensor", Sequence[float]]
    sliding_window: Optional[int]

    @property
    def config_name(self) -> str:
        return f"l{self.layer_idx:02d}h{self.head_idx:02d}{self.vector_kind}"


class _ParquetSink:
    """Keeps a Parquet writer open for a split/config pair."""

    def __init__(self, root: Path, split: str, config_name: str, compression: str = "zstd"):
        self.root = root
        self.split = split
        self.config_name = config_name
        self.compression = compression
        self.dir_path = self.root / split / config_name
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.file_path = self.dir_path / "data.parquet"
        self._writer: Optional[pq.ParquetWriter] = None
        self._schema: Optional[pa.Schema] = None
        self._row_count = 0

    def write(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return

        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(
                self.file_path,
                schema=self._schema,
                compression=self.compression,
            )

        self._writer.write_table(table)
        self._row_count += table.num_rows

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @property
    def row_count(self) -> int:
        return self._row_count


class DatasetSaver:
    """
    Streams capture rows into Parquet files organised as ``data/{split}/{config}/data.parquet``.

    The split name corresponds to the model name, while the config name encodes the layer/head and
    whether the vectors are keys or queries (e.g. ``l03h07q``).
    """

    def __init__(
        self,
        root: Union[str, Path] = "data",
        compression: str = "zstd",
        readme_path: Union[str, Path] = "README.md",
        dataset_name: str = "viktoroo/sniffed-qk",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.readme_path = Path(readme_path)
        self._sinks: Dict[Tuple[str, str], _ParquetSink] = {}
        self._config_splits: Dict[str, Set[str]] = {}
        self._models: Set[str] = set()
        self._sanitized_splits: Dict[str, str] = {}
        self._position_cache: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {}
        self._model_metadata: Dict[str, Dict[str, Union[str, int, float]]] = {}
        self._bucket_counts: Dict[str, Counter] = defaultdict(Counter)
        self._readme = DatasetReadme(readme_path, self.root, dataset_name)
        self._seed_existing_entries()

    def add(self, row: CaptureRow) -> None:
        self.add_many([row])

    def add_many(self, rows: Iterable[CaptureRow]) -> None:
        batch_by_sink: Dict[Tuple[str, str], Dict[str, list]] = {}

        for row in rows:
            split = row.model_name
            storage_split = self._sanitized_splits.setdefault(split, _sanitize_split(split))
            config = row.config_name
            key = (storage_split, config)
            if key not in batch_by_sink:
                position_cache = self._position_cache.setdefault(key, self._load_existing_positions(storage_split, config))
                batch_by_sink[key] = {
                    "bucket": [],
                    "example_id": [],
                    "position": [],
                    "vector": [],
                    "sliding_window": [],
                    "position_cache": position_cache,
                }

            batch = batch_by_sink[key]
            position_cache: Set[Tuple[int, int]] = batch["position_cache"]
            position_key = (row.example_id, row.position)
            if position_key in position_cache:
                continue
            position_cache.add(position_key)
            batch["bucket"].append(row.bucket)
            batch["example_id"].append(row.example_id)
            batch["position"].append(row.position)
            batch["vector"].append(_to_numpy(row.vector))
            batch["sliding_window"].append(row.sliding_window)
            self._models.add(split)
            self._config_splits.setdefault(config, set()).add(storage_split)
            self._bucket_counts[row.model_name][row.bucket] += 1

        for (storage_split, config), columns in batch_by_sink.items():
            columns = dict(columns)
            columns.pop("position_cache", None)
            if not columns["vector"]:
                continue
            sink = self._sinks.setdefault((storage_split, config), self._create_sink(storage_split, config))
            vector_array = _vectors_to_fixed_list(columns["vector"])
            table = pa.table(
                {
                    "bucket": pa.array(columns["bucket"], type=pa.int32()),
                    "example_id": pa.array(columns["example_id"], type=pa.int32()),
                    "position": pa.array(columns["position"], type=pa.int32()),
                    "vector": vector_array,
                    "sliding_window": pa.array(columns["sliding_window"]),
                }
            )
            sink.write(table)

    def close(self) -> None:
        current_sinks = list(self._sinks.values())
        for sink in current_sinks:
            sink.close()
        self._sinks.clear()
        self._write_readme()

    def __enter__(self) -> "DatasetSaver":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def register_model_metadata(self, model_name: str, metadata: Dict[str, Union[str, int, float]]) -> None:
        sanitized = self._sanitized_splits.setdefault(model_name, _sanitize_split(model_name))
        self._model_metadata.setdefault(model_name, {}).update(metadata)
        (self.root / sanitized).mkdir(parents=True, exist_ok=True)

    def _write_readme(self) -> None:
        self._readme.write(self._config_splits, self._models, self._model_metadata, self._bucket_counts)

    def _seed_existing_entries(self) -> None:
        front_matter = self._readme.front_matter or {}
        for model_entry in front_matter.get("models", []):
            name = model_entry.get("name")
            if name:
                self._models.add(name)
        if not self.root.exists():
            return
        for split_dir in self.root.iterdir():
            if not split_dir.is_dir():
                continue
            split_name = split_dir.name
            for config_dir in split_dir.iterdir():
                if not config_dir.is_dir():
                    continue
                data_file = config_dir / "data.parquet"
                if not data_file.exists():
                    continue
                self._config_splits.setdefault(config_dir.name, set()).add(split_name)

    def _create_sink(self, storage_split: str, config: str) -> _ParquetSink:
        return _ParquetSink(self.root, storage_split, config, compression=self.compression)

    def _load_existing_positions(self, storage_split: str, config: str) -> Set[Tuple[int, int]]:
        cache: Set[Tuple[int, int]] = set()
        data_path = self.root / storage_split / config / "data.parquet"
        if not data_path.exists():
            return cache
        try:
            table = pq.read_table(data_path, columns=["example_id", "position"])
        except Exception:
            return cache
        example_ids = table.column("example_id").to_pylist()
        positions = table.column("position").to_pylist()
        for example_id, position in zip(example_ids, positions):
            cache.add((int(example_id), int(position)))
        return cache


def _to_numpy(vector: Union["np.ndarray", "torch.Tensor", Sequence[float]]) -> "np.ndarray":
    if np is None:
        raise RuntimeError("NumPy is required to convert vectors for storage.")
    if isinstance(vector, np.ndarray):
        return vector.astype("float32", copy=False)
    if torch is not None and isinstance(vector, torch.Tensor):
        return vector.detach().to(dtype=torch.float32, device="cpu").numpy()
    return np.asarray(list(vector), dtype="float32")


def _vectors_to_fixed_list(vectors: List["np.ndarray"]) -> pa.FixedSizeListArray:
    if np is None:
        raise RuntimeError("NumPy is required to convert vectors for storage.")
    stacked = np.stack(vectors, axis=0).astype("float32", copy=False)
    list_size = stacked.shape[-1]
    values = pa.array(stacked.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, list_size)


def _sanitize_split(split: str) -> str:
    return _SPLIT_SANITIZE_RE.sub("_", split)


_SPLIT_SANITIZE_RE = re.compile(r"\W")
