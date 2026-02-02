from __future__ import annotations

import re
import json
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
    token_str: Optional[str] = None

    @property
    def config_name(self) -> str:
        return f"l{self.layer_idx:02d}h{self.head_idx:02d}{self.vector_kind}"


@dataclass(slots=True)
class CaptureBatch:
    model_name: str
    layer_idx: int
    head_idx: int
    vector_kind: VectorKind
    buckets: "np.ndarray"
    example_ids: "np.ndarray"
    positions: "np.ndarray"
    vectors: "np.ndarray"
    sliding_window: Optional[int]
    token_strings: Optional[Sequence[str]] = None

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
        write_batch_size: int = 2048,
        mirror_readme_paths: Optional[Sequence[Union[str, Path]]] = None,
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        raw_readme_path = Path(readme_path).expanduser()
        if raw_readme_path.is_absolute():
            resolved_readme_path = raw_readme_path
        else:
            resolved_readme_path = (self.root / raw_readme_path).expanduser()
        self.readme_path = resolved_readme_path
        seen_readme_paths = {self.readme_path.resolve()}
        mirror_readmes: List[DatasetReadme] = []
        for extra_path in mirror_readme_paths or []:
            extra = Path(extra_path).expanduser()
            if not extra.is_absolute():
                extra = (self.root / extra).expanduser()
            resolved_extra = extra.resolve()
            if resolved_extra in seen_readme_paths:
                continue
            mirror_readmes.append(DatasetReadme(resolved_extra, self.root, dataset_name))
            seen_readme_paths.add(resolved_extra)
        self._mirror_readmes = mirror_readmes
        self._sinks: Dict[Tuple[str, str], _ParquetSink] = {}
        self._config_splits: Dict[str, Set[str]] = {}
        self._models: Set[str] = set()
        self._sanitized_splits: Dict[str, str] = {}
        self._position_cache: Dict[Tuple[str, str], Set[Tuple[int, int]]] = {}
        self._model_metadata: Dict[str, Dict[str, Union[str, int, float]]] = {}
        self._bucket_counts: Dict[str, Counter] = defaultdict(Counter)
        self._readme = DatasetReadme(self.readme_path, self.root, dataset_name)
        self._write_batch_size = max(1, int(write_batch_size))
        self._pending: Dict[Tuple[str, str], Dict[str, List]] = {}
        self._token_columns: Dict[Tuple[str, str], bool] = {}
        self._state_path = self.root / "_saver_state.json"
        self._load_state()
        for model_name in list(self._model_metadata.keys()):
            self._ensure_sanitized_metadata(model_name)
        self._seed_existing_entries()

    def add(self, row: CaptureRow) -> None:
        self.add_many([row])

    def add_many(self, rows: Iterable[CaptureRow]) -> None:
        if np is None:
            raise RuntimeError("NumPy is required to convert vectors for storage.")
        for row in rows:
            vector = _to_numpy(row.vector)
            batch = CaptureBatch(
                model_name=row.model_name,
                layer_idx=row.layer_idx,
                head_idx=row.head_idx,
                vector_kind=row.vector_kind,
                buckets=np.asarray([row.bucket], dtype=np.int64),
                example_ids=np.asarray([row.example_id], dtype=np.int64),
                positions=np.asarray([row.position], dtype=np.int64),
                vectors=np.asarray([vector], dtype=np.float32),
                sliding_window=row.sliding_window,
                token_strings=[row.token_str] if row.token_str is not None else None,
            )
            self.add_batch(batch)

    def add_batch(self, batch: CaptureBatch) -> None:
        if np is None:
            raise RuntimeError("NumPy is required to convert vectors for storage.")
        buckets = np.asarray(batch.buckets, dtype=np.int64).reshape(-1)
        example_ids = np.asarray(batch.example_ids, dtype=np.int64).reshape(-1)
        positions = np.asarray(batch.positions, dtype=np.int64).reshape(-1)
        vectors = np.asarray(batch.vectors, dtype=np.float32)
        row_count = buckets.shape[0]
        if row_count == 0:
            return
        token_strings_raw = batch.token_strings
        if token_strings_raw is not None and len(token_strings_raw) != row_count:
            raise ValueError("token_strings length must match number of rows in CaptureBatch.")
        if not (example_ids.shape[0] == positions.shape[0] == row_count == vectors.shape[0]):
            raise ValueError("CaptureBatch columns must share the same length.")

        split = batch.model_name
        storage_split = self._sanitized_splits.setdefault(split, _sanitize_split(split))
        self._ensure_sanitized_metadata(split)
        config = batch.config_name
        key = (storage_split, config)
        position_cache = self._position_cache.setdefault(key, self._load_existing_positions(storage_split, config))

        keep_mask = np.ones(row_count, dtype=bool)
        for idx in range(row_count):
            position_key = (int(example_ids[idx]), int(positions[idx]))
            if position_key in position_cache:
                keep_mask[idx] = False
            else:
                position_cache.add(position_key)

        if not keep_mask.any():
            return

        buckets = buckets[keep_mask].astype("int32", copy=False)
        example_ids = example_ids[keep_mask].astype("int32", copy=False)
        positions = positions[keep_mask].astype("int32", copy=False)
        vectors = vectors[keep_mask].astype("float32", copy=False)
        sliding_window_values: List = (
            [int(batch.sliding_window)] * buckets.shape[0]
            if batch.sliding_window is not None
            else [None] * buckets.shape[0]
        )

        token_strings: Optional[List[Optional[str]]] = None
        has_tokens = token_strings_raw is not None
        token_policy = self._token_columns.get(key)
        if token_policy is None:
            self._token_columns[key] = has_tokens
        elif token_policy and not has_tokens:
            token_strings = [None] * buckets.shape[0]
            has_tokens = True
        elif not token_policy and has_tokens:
            raise ValueError(
                f"Token strings provided for {key} after writing data without token strings."
            )
        if has_tokens and token_strings is None:
            token_strings = np.asarray(token_strings_raw, dtype=object)[keep_mask].tolist()

        columns = {
            "bucket": buckets.tolist(),
            "example_id": example_ids.tolist(),
            "position": positions.tolist(),
            "vector": [np.copy(vec) for vec in vectors],
            "sliding_window": sliding_window_values,
        }
        if token_strings is not None:
            columns["token_str"] = token_strings

        self._models.add(split)
        self._config_splits.setdefault(config, set()).add(storage_split)
        for bucket in buckets.tolist():
            self._bucket_counts[split][bucket] += 1

        self._append_pending(key, columns)

    def _append_pending(self, key: Tuple[str, str], columns: Dict[str, List]) -> None:
        pending = self._pending.setdefault(key, self._empty_pending())
        for name, values in columns.items():
            if name not in pending:
                pending[name] = []
            pending[name].extend(values)
        if len(pending["vector"]) >= self._write_batch_size:
            self._flush_pending(key)

    def _flush_pending(self, key: Tuple[str, str]) -> None:
        pending = self._pending.get(key)
        if not pending or not pending["vector"]:
            return
        storage_split, config = key
        sink = self._sinks.setdefault(key, self._create_sink(storage_split, config))
        vector_array = _vectors_to_fixed_list(pending["vector"])
        data = {
            "bucket": pa.array(pending["bucket"], type=pa.int32()),
            "example_id": pa.array(pending["example_id"], type=pa.int32()),
            "position": pa.array(pending["position"], type=pa.int32()),
            "vector": vector_array,
            "sliding_window": pa.array(pending["sliding_window"]),
        }
        if "token_str" in pending:
            data["token_str"] = pa.array(pending["token_str"], type=pa.string())
        table = pa.table(data)
        sink.write(table)
        for values in pending.values():
            values.clear()

    @staticmethod
    def _empty_pending() -> Dict[str, List]:
        return {"bucket": [], "example_id": [], "position": [], "vector": [], "sliding_window": []}

    def close(self) -> None:
        for key in list(self._pending.keys()):
            self._flush_pending(key)
        current_sinks = list(self._sinks.values())
        for sink in current_sinks:
            sink.close()
        self._sinks.clear()
        self._pending.clear()
        self._write_readme()
        self._save_state()

    def __enter__(self) -> "DatasetSaver":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def register_model_metadata(self, model_name: str, metadata: Dict[str, Union[str, int, float]]) -> None:
        sanitized = self._ensure_sanitized_metadata(model_name)
        self._model_metadata.setdefault(model_name, {}).update(metadata)
        self._models.add(model_name)
        (self.root / sanitized).mkdir(parents=True, exist_ok=True)

    def _write_readme(self) -> None:
        self._readme.write(self._config_splits, self._models, self._model_metadata, self._bucket_counts)
        for mirror in self._mirror_readmes:
            mirror.write(self._config_splits, self._models, self._model_metadata, self._bucket_counts)

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

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return

        metadata = data.get("model_metadata", {})
        if isinstance(metadata, dict):
            for model_name, meta in metadata.items():
                if isinstance(meta, dict):
                    self._model_metadata.setdefault(model_name, {}).update(meta)
                    self._models.add(model_name)
                    self._ensure_sanitized_metadata(model_name)

        bucket_counts = data.get("bucket_counts", {})
        if isinstance(bucket_counts, dict):
            for model_name, counts in bucket_counts.items():
                if not isinstance(counts, dict):
                    continue
                counter = Counter()
                for bucket_key, value in counts.items():
                    try:
                        bucket_idx = int(bucket_key)
                        counter[bucket_idx] = int(value)
                    except (TypeError, ValueError):
                        continue
                if counter:
                    self._bucket_counts[model_name].update(counter)
                    self._models.add(model_name)
                    self._ensure_sanitized_metadata(model_name)

    def _save_state(self) -> None:
        bucket_counts = {
            model: {str(bucket): int(count) for bucket, count in counts.items()}
            for model, counts in self._bucket_counts.items()
            if counts
        }
        state = {
            "model_metadata": self._model_metadata,
            "bucket_counts": bucket_counts,
        }
        try:
            self._state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass

    def _ensure_sanitized_metadata(self, model_name: str) -> str:
        sanitized = self._sanitized_splits.setdefault(model_name, _sanitize_split(model_name))
        meta = self._model_metadata.setdefault(model_name, {})
        if meta.get("sanitized_split") != sanitized:
            meta["sanitized_split"] = sanitized
        return sanitized


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
