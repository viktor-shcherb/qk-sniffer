from __future__ import annotations

import json
import re
from collections import Counter
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
    """Container describing a single query/key sample."""

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
    """Keeps a Parquet writer open for a config."""

    def __init__(self, root: Path, config_name: str, compression: str = "zstd"):
        self.root = root
        self.config_name = config_name
        self.compression = compression
        self.dir_path = self.root / config_name
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
    """Streams capture rows into ``<root>/<config>/data.parquet`` files."""

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

        # README is always rooted at dataset root.
        self.readme_path = self.root / "README.md"
        self._readme = DatasetReadme(self.readme_path, dataset_name)
        mirror_readmes: List[DatasetReadme] = []
        for path in mirror_readme_paths or []:
            mirror_path = Path(path).expanduser()
            if not mirror_path.is_absolute():
                mirror_path = self.root / mirror_path
            mirror_readmes.append(DatasetReadme(mirror_path, dataset_name))
        self._mirror_readmes = mirror_readmes

        self._sinks: Dict[str, _ParquetSink] = {}
        self._config_names: Set[str] = set()
        self._position_cache: Dict[str, Set[Tuple[int, int]]] = {}
        self._bucket_counts: Counter = Counter()

        self._model_name: Optional[str] = None
        self._model_metadata: Dict[str, Union[str, int, float]] = {}

        self._write_batch_size = max(1, int(write_batch_size))
        self._pending: Dict[str, Dict[str, List]] = {}
        self._token_columns: Dict[str, bool] = {}

        self._state_path = self.root / "_saver_state.json"
        self._load_state()
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

        self._set_model_name(batch.model_name)

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

        config = batch.config_name
        self._config_names.add(config)
        key = config
        position_cache = self._position_cache.setdefault(key, self._load_existing_positions(config))

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
            raise ValueError(f"Token strings provided for {key} after writing data without token strings.")
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

        for bucket in buckets.tolist():
            self._bucket_counts[int(bucket)] += 1

        self._append_pending(key, columns)

    def _append_pending(self, key: str, columns: Dict[str, List]) -> None:
        pending = self._pending.setdefault(key, self._empty_pending())
        for name, values in columns.items():
            if name not in pending:
                pending[name] = []
            pending[name].extend(values)
        if len(pending["vector"]) >= self._write_batch_size:
            self._flush_pending(key)

    def _flush_pending(self, key: str) -> None:
        pending = self._pending.get(key)
        if not pending or not pending["vector"]:
            return

        sink = self._sinks.setdefault(key, self._create_sink(key))
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
        self._set_model_name(model_name)
        self._model_metadata.update(metadata)

    def _set_model_name(self, model_name: str) -> None:
        if self._model_name is None:
            self._model_name = model_name
            return
        if self._model_name != model_name:
            raise ValueError(
                "Dataset branch layout only supports one model per branch. "
                f"Existing model is '{self._model_name}', got '{model_name}'."
            )

    def _write_readme(self) -> None:
        self._readme.write(
            model_name=self._model_name,
            metadata=dict(self._model_metadata),
            config_names=sorted(self._config_names),
            bucket_counts={int(k): int(v) for k, v in self._bucket_counts.items()},
        )
        for mirror in self._mirror_readmes:
            mirror.write(
                model_name=self._model_name,
                metadata=dict(self._model_metadata),
                config_names=sorted(self._config_names),
                bucket_counts={int(k): int(v) for k, v in self._bucket_counts.items()},
            )

    def _seed_existing_entries(self) -> None:
        if not self.root.exists():
            return
        for config_dir in self.root.iterdir():
            if not config_dir.is_dir():
                continue
            if not _CONFIG_NAME_RE.match(config_dir.name):
                continue
            data_file = config_dir / "data.parquet"
            if data_file.exists():
                self._config_names.add(config_dir.name)

    def _create_sink(self, config: str) -> _ParquetSink:
        return _ParquetSink(self.root, config, compression=self.compression)

    def _load_existing_positions(self, config: str) -> Set[Tuple[int, int]]:
        cache: Set[Tuple[int, int]] = set()
        data_path = self.root / config / "data.parquet"
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

        # New format.
        model_name = data.get("model_name")
        if isinstance(model_name, str):
            self._model_name = model_name
        metadata = data.get("model_metadata")
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if isinstance(key, str) and isinstance(value, (str, int, float)):
                    self._model_metadata[key] = value

        bucket_counts = data.get("bucket_counts")
        if isinstance(bucket_counts, dict):
            for bucket_key, value in bucket_counts.items():
                try:
                    bucket_idx = int(bucket_key)
                    self._bucket_counts[bucket_idx] = int(value)
                except (TypeError, ValueError):
                    continue

        # Best-effort old format fallback.
        if self._model_name is None:
            old_metadata = data.get("model_metadata")
            if isinstance(old_metadata, dict) and old_metadata:
                first_key = next(iter(old_metadata.keys()))
                first_meta = old_metadata.get(first_key)
                if isinstance(first_key, str) and isinstance(first_meta, dict):
                    self._model_name = first_key
                    for key, value in first_meta.items():
                        if isinstance(key, str) and isinstance(value, (str, int, float)):
                            self._model_metadata[key] = value
        if not self._bucket_counts:
            old_bucket_counts = data.get("bucket_counts")
            if isinstance(old_bucket_counts, dict) and old_bucket_counts:
                if self._model_name and isinstance(old_bucket_counts.get(self._model_name), dict):
                    old_counts = old_bucket_counts[self._model_name]
                else:
                    first_value = next(iter(old_bucket_counts.values()))
                    old_counts = first_value if isinstance(first_value, dict) else {}
                if isinstance(old_counts, dict):
                    for bucket_key, value in old_counts.items():
                        try:
                            self._bucket_counts[int(bucket_key)] = int(value)
                        except (TypeError, ValueError):
                            continue

    def _save_state(self) -> None:
        state = {
            "model_name": self._model_name,
            "model_metadata": self._model_metadata,
            "bucket_counts": {str(bucket): int(count) for bucket, count in self._bucket_counts.items() if count},
        }
        try:
            self._state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        except Exception:
            pass


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


_CONFIG_NAME_RE = re.compile(r"^l\d{2}h\d{2}[qk]$")
