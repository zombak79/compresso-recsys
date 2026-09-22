"""Opt-in cold-start infrastructure for named item-feature matrices.

The catalog preserves modalities; it does not choose a fusion rule or run an
encoder. Models may override registration and persistence for other stored
representations without changing the single-matrix cold-start family.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Literal

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models.cold_start import (
    _NOT_INSTALLED,
    BaseColdStartRecommender,
    CandidateConflict,
    ItemFeatures,
    _freeze_features,
    _replace_feature_rows,
    _stack_features,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_metadata,
    take_features,
)
from compresso_recsys.models.identifiers import ItemVocabulary, canonical_item_ids
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
    _decoded_item_id,
    _encoded_item_id,
)

__all__ = [
    "BaseMultiModalRecommender",
    "MultiModalCandidateCatalog",
    "MultiModalCandidateSelection",
    "MultiModalItemFeatures",
    "MutableMultiModalCandidateCatalog",
]

MultiModalItemFeatures = Mapping[str, ItemFeatures]
_StoredFeatures = Mapping[str, csr_matrix | np.ndarray]


def _canonical_features(
    features: MultiModalItemFeatures,
    *,
    n_items: int,
    dtype: np.dtype,
    dimensions: Mapping[str, int] | None = None,
) -> dict[str, csr_matrix | np.ndarray]:
    if not isinstance(features, Mapping):
        raise TypeError("item_features must be a mapping of modality names to matrices")
    if not features:
        raise ValueError("item_features must contain at least one modality")
    if any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("modality names must be non-empty strings")
    if dimensions is not None and set(features) != set(dimensions):
        raise ValueError(
            "modality names must match the installed schema; "
            f"missing={sorted(set(dimensions) - set(features))}, "
            f"extra={sorted(set(features) - set(dimensions))}"
        )
    result = {}
    # Use the installed order on updates, never the caller's dictionary order.
    for name in features if dimensions is None else dimensions:
        matrix = canonical_item_features(features[name], dtype=dtype)
        values = matrix.data if isinstance(matrix, csr_matrix) else matrix
        if not np.all(np.isfinite(values)):
            raise ValueError(f"item_features[{name!r}] must remain finite in {dtype}")
        if matrix.shape[0] != n_items:
            raise ValueError(
                f"item_features[{name!r}] has {matrix.shape[0]} rows, "
                f"but item_ids has {n_items} entries"
            )
        if dimensions is not None and matrix.shape[1] != dimensions[name]:
            raise ValueError(
                f"item_features[{name!r}] has {matrix.shape[1]} columns, "
                f"expected {dimensions[name]}"
            )
        result[name] = matrix
    return result


def _canonical_dtype(dtype: str | np.dtype) -> np.dtype:
    # NumPy interprets None as float64; an unset config must not choose precision.
    if dtype is None:
        raise ValueError("dtype must be float32 or float64")
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    return resolved


def _encode_metadata_label(label: Hashable) -> dict:
    """Reuse the safe scalar codec, adding null, NaN and tuple metadata labels."""
    if label is None:
        return {"type": "none"}
    # Pandas can coerce missing column labels to NaN; keep it distinct from None.
    if isinstance(label, (float, np.floating)) and np.isnan(label):
        return {"type": "nan"}
    if isinstance(label, tuple):
        return {"type": "tuple", "value": [_encode_metadata_label(v) for v in label]}
    try:
        return _encoded_item_id(label)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"unsupported metadata column/index label: {label!r} "
            f"({type(label).__name__})"
        ) from error


def _decode_metadata_label(state: object) -> Hashable:
    if isinstance(state, dict):
        if state.get("type") == "none":
            return None
        if state.get("type") == "nan":
            return np.nan
        if state.get("type") == "tuple" and isinstance(state.get("value"), list):
            return tuple(_decode_metadata_label(v) for v in state["value"])
    return _decoded_item_id(state)


def _metadata_columns_state(columns: pd.Index) -> dict:
    state = {
        "labels": [_encode_metadata_label(label) for label in columns],
        "names": [_encode_metadata_label(name) for name in columns.names],
    }
    if isinstance(columns, pd.MultiIndex):
        state["kind"] = "multi"
        state["level_dtypes"] = [str(level.dtype) for level in columns.levels]
    elif isinstance(columns, pd.RangeIndex):
        state.update(
            kind="range", start=columns.start, stop=columns.stop, step=columns.step
        )
    else:
        state.update(kind="index", dtype=str(columns.dtype))
    return state


def _restore_metadata_columns(state: object, *, n_columns: int) -> pd.Index:
    try:
        if not isinstance(state, dict):
            raise TypeError("column description must be an object")
        labels, names = state["labels"], state["names"]
        if not isinstance(labels, list) or not isinstance(names, list):
            raise TypeError("column labels and names must be lists")
        if len(labels) != n_columns:
            raise ValueError("column count does not match metadata")
        labels = [_decode_metadata_label(label) for label in labels]
        names = [_decode_metadata_label(name) for name in names]
        if state["kind"] == "multi":
            columns = pd.MultiIndex.from_tuples(labels, names=names)
            if "level_dtypes" in state:
                dtypes = state["level_dtypes"]
                if (
                    not isinstance(dtypes, list)
                    or len(dtypes) != columns.nlevels
                    or any(not isinstance(dtype, str) for dtype in dtypes)
                ):
                    raise ValueError("invalid multi-level column dtypes")
                # Iterating a level with NaN can turn its integer labels into
                # floats. Restore the level dtype independently of missing codes.
                columns = columns.set_levels(
                    [
                        level.astype(dtype)
                        for level, dtype in zip(columns.levels, dtypes)
                    ]
                )
            return columns
        if len(names) != 1:
            raise ValueError("flat columns must have one index name")
        if state["kind"] == "range":
            columns = pd.RangeIndex(
                state["start"], state["stop"], state["step"], name=names[0]
            )
            if len(columns) != n_columns or columns.tolist() != labels:
                raise ValueError("range does not match column labels")
            return columns
        if state["kind"] == "index" and isinstance(state.get("dtype"), str):
            return pd.Index(
                labels, dtype=state["dtype"], name=names[0], tupleize_cols=False
            )
        raise ValueError("unsupported column index description")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid checkpoint metadata column description") from error


@dataclass(frozen=True, init=False)
class MultiModalCandidateCatalog:
    """One published snapshot with aligned, read-only modality matrices.

    Snapshot consistency does not pin a catalog version across an entire
    batched prediction. Model authors needing that guarantee must capture one
    snapshot for the whole prediction themselves.
    """

    item_ids: np.ndarray
    item_features: _StoredFeatures
    feature_dims: Mapping[str, int]
    _metadata: pd.DataFrame | None = field(repr=False)
    feature_space_id: str | None
    version: int
    id_to_row: Mapping[Hashable, int]

    def __init__(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
        version: int = 1,
        dtype: str | np.dtype = "float32",
    ) -> None:
        vocabulary = ItemVocabulary.from_ids(item_ids)
        features = _canonical_features(
            item_features, n_items=vocabulary.n_items, dtype=_canonical_dtype(dtype)
        )
        if (
            isinstance(version, bool)
            or not isinstance(version, (int, np.integer))
            or version < 1
        ):
            raise ValueError("catalog version must be an integer >= 1")
        object.__setattr__(self, "item_ids", vocabulary.item_ids)
        object.__setattr__(self, "id_to_row", vocabulary.id_to_row)
        object.__setattr__(
            self,
            "item_features",
            MappingProxyType(
                {name: _freeze_features(matrix) for name, matrix in features.items()}
            ),
        )
        object.__setattr__(
            self,
            "feature_dims",
            MappingProxyType(
                {name: int(matrix.shape[1]) for name, matrix in features.items()}
            ),
        )
        object.__setattr__(
            self,
            "_metadata",
            canonical_metadata(metadata, item_ids=vocabulary.item_ids),
        )
        object.__setattr__(
            self, "feature_space_id", canonical_feature_space_id(feature_space_id)
        )
        object.__setattr__(self, "version", int(version))

    @property
    def n_items(self) -> int:
        return int(self.item_ids.size)

    @property
    def metadata(self) -> pd.DataFrame | None:
        return None if self._metadata is None else self._metadata.copy(deep=True)

    def rows_for(self, item_ids: Sequence[Hashable]) -> np.ndarray:
        ids = canonical_item_ids(item_ids, name="candidate_ids")
        rows = np.empty(ids.size, dtype=np.int64)
        for position, item_id in enumerate(ids.tolist()):
            try:
                rows[position] = self.id_to_row[item_id]
            except KeyError as error:
                raise KeyError(f"unknown candidate item ID: {item_id!r}") from error
        return rows

    def ids_for(self, rows: np.ndarray | torch.Tensor) -> np.ndarray:
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().numpy()
        rows = np.asarray(rows)
        if not np.issubdtype(rows.dtype, np.integer):
            raise TypeError("candidate rows must contain integers")
        if rows.size and (rows.min() < 0 or rows.max() >= self.n_items):
            raise IndexError("candidate rows are out of bounds")
        return self.item_ids[rows]


@dataclass(frozen=True)
class MultiModalCandidateSelection:
    """Selected feature rows and their mappings into one complete snapshot.

    Row indices and both mappings use int64 for compatibility with Torch indices.
    """

    catalog: MultiModalCandidateCatalog
    rows: np.ndarray
    features: _StoredFeatures
    source_to_candidate: np.ndarray
    candidate_to_local: np.ndarray


class MutableMultiModalCandidateCatalog:
    """Candidate-only lifecycle for named feature matrices.

    All mutations validate before publication. The optional callback runs
    *after* publication, under the catalog lock; if it raises, the new snapshot
    remains installed. Adding candidates never extends the source vocabulary.
    Callbacks must not wait for workers that acquire this catalog's lock (for
    example through snapshot/selection reads or mutations). Pass the supplied
    immutable snapshot to workers instead. Same-thread reads are reentrant.

    Every build/update supplies all registered modalities for its item rows.
    There is no implicit encoding, normalization, fusion or popularity feature.
    """

    def __init__(
        self,
        *,
        on_publish: Callable[[MultiModalCandidateCatalog], None] | None = None,
    ) -> None:
        self._lock = RLock()
        self._on_publish = on_publish
        self._snapshot: MultiModalCandidateCatalog | None = None
        self._source_vocabulary: ItemVocabulary | None = None
        self._dtype: np.dtype | None = None

    @property
    def is_installed(self) -> bool:
        return self._snapshot is not None

    def snapshot(self) -> MultiModalCandidateCatalog:
        with self._lock:
            if self._snapshot is None:
                raise RuntimeError(_NOT_INSTALLED)
            return self._snapshot

    @property
    def n_items(self) -> int | None:
        return None if self._snapshot is None else self._snapshot.n_items

    @property
    def source_vocabulary(self) -> ItemVocabulary | None:
        return self._source_vocabulary

    @property
    def source_item_ids(self) -> np.ndarray | None:
        return (
            None
            if self._source_vocabulary is None
            else self._source_vocabulary.item_ids
        )

    @property
    def source_id_to_row(self) -> Mapping[Hashable, int] | None:
        return (
            None
            if self._source_vocabulary is None
            else self._source_vocabulary.id_to_row
        )

    @property
    def feature_dims(self) -> Mapping[str, int] | None:
        """Installed dimensions describe the stored representation of each modality."""
        return None if self._snapshot is None else self._snapshot.feature_dims

    @property
    def feature_space_id(self) -> str | None:
        return None if self._snapshot is None else self._snapshot.feature_space_id

    def _publish(
        self, snapshot: MultiModalCandidateCatalog
    ) -> MultiModalCandidateCatalog:
        # Callers hold the lock and finish all validation before reaching here.
        self._snapshot = snapshot
        if self._on_publish is not None:
            self._on_publish(snapshot)
        return snapshot

    def install(
        self,
        *,
        source_item_ids: Sequence[Hashable] | np.ndarray,
        candidate_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
        dtype: str | np.dtype = "float32",
    ) -> MultiModalCandidateCatalog:
        """Install fitted source IDs and initial candidates with those same IDs.

        Dimensions are inferred from the supplied stored representations. A
        refit may install a new schema; later candidate mutations cannot.
        Omitting ``dtype`` uses float32; an explicit None is rejected.

        Reinstallation advances this catalog's current version and replaces
        all candidates, including additions published before it. It does not
        merge candidates from an earlier fit into a potentially new schema.
        The lock orders catalog publications only; the model must coordinate
        changes to its other fitted state with concurrent prediction itself.
        """
        vocabulary = ItemVocabulary.from_ids(source_item_ids, name="source_item_ids")
        resolved_dtype = _canonical_dtype(dtype)
        with self._lock:
            # Choose the version under the publication lock so a refit cannot
            # reuse an earlier version or race another catalog mutation.
            current = self._snapshot
            snapshot = MultiModalCandidateCatalog(
                item_ids=vocabulary.item_ids,
                item_features=candidate_features,
                metadata=metadata,
                feature_space_id=feature_space_id,
                version=1 if current is None else current.version + 1,
                dtype=resolved_dtype,
            )
            # Validate the complete replacement before changing fitted source
            # state; validation failure must leave the previous catalog intact.
            self._source_vocabulary = vocabulary
            self._dtype = resolved_dtype
            return self._publish(snapshot)

    def _resolve_feature_space_id(self, value: str | None) -> str | None:
        resolved = canonical_feature_space_id(value)
        installed = self.snapshot().feature_space_id
        if resolved is not None and resolved != installed:
            raise ValueError(
                "feature_space_id must match the feature space used to fit the model"
            )
        return installed

    def _prepare_features(
        self,
        item_ids: np.ndarray,
        item_features: MultiModalItemFeatures,
    ) -> dict[str, csr_matrix | np.ndarray]:
        current = self.snapshot()
        assert self._dtype is not None
        return _canonical_features(
            item_features,
            n_items=item_ids.size,
            dtype=self._dtype,
            dimensions=current.feature_dims,
        )

    def build(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> MultiModalCandidateCatalog:
        """Replace all candidates while preserving the fitted source vocabulary."""
        ids = canonical_item_ids(item_ids)
        with self._lock:
            features = self._prepare_features(ids, item_features)
            return self._publish(
                MultiModalCandidateCatalog(
                    item_ids=ids,
                    item_features=features,
                    metadata=metadata,
                    feature_space_id=self._resolve_feature_space_id(feature_space_id),
                    version=self.snapshot().version + 1,
                    dtype=self._dtype,
                )
            )

    def update(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> MultiModalCandidateCatalog:
        """Append or replace complete modality rows for the supplied item IDs."""
        if on_conflict not in {"error", "replace", "ignore"}:
            raise ValueError("on_conflict must be 'error', 'replace', or 'ignore'")
        ids = canonical_item_ids(item_ids)
        incoming_metadata = canonical_metadata(metadata, item_ids=ids)
        with self._lock:
            current = self.snapshot()
            incoming = self._prepare_features(ids, item_features)
            space = self._resolve_feature_space_id(feature_space_id)
            conflicts = np.array(
                [item_id in current.id_to_row for item_id in ids], dtype=bool
            )
            if on_conflict == "error" and conflicts.any():
                raise ValueError(
                    f"candidate item ID already exists: {ids[np.flatnonzero(conflicts)[0]]!r}"
                )
            replacements = (
                np.flatnonzero(conflicts)
                if on_conflict == "replace"
                else np.empty(0, dtype=np.int64)
            )
            old_rows = np.array(
                [current.id_to_row[ids[row]] for row in replacements], dtype=np.int64
            )
            additions = np.flatnonzero(~conflicts)
            if not replacements.size and not additions.size:
                return current
            features = {}
            for name, matrix in current.item_features.items():
                updated = _replace_feature_rows(
                    matrix, old_rows, take_features(incoming[name], replacements)
                )
                if additions.size:
                    updated = _stack_features(
                        updated, take_features(incoming[name], additions)
                    )
                features[name] = updated
            combined_ids = np.concatenate((current.item_ids, ids[additions]))
            combined_metadata = _updated_metadata(
                current,
                incoming_metadata,
                replacements,
                old_rows,
                additions,
                combined_ids=combined_ids,
            )
            return self._publish(
                MultiModalCandidateCatalog(
                    item_ids=combined_ids,
                    item_features=features,
                    metadata=combined_metadata,
                    feature_space_id=space,
                    version=current.version + 1,
                    dtype=self._dtype,
                )
            )

    def remove(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> MultiModalCandidateCatalog:
        """Remove candidates only; fitted history items remain valid."""
        if missing not in {"error", "ignore"}:
            raise ValueError("missing must be 'error' or 'ignore'")
        ids = canonical_item_ids(item_ids)
        with self._lock:
            current = self.snapshot()
            unknown = [item_id for item_id in ids if item_id not in current.id_to_row]
            if unknown and missing == "error":
                raise KeyError(f"unknown candidate item ID: {unknown[0]!r}")
            removed = set(ids) & set(current.id_to_row)
            if not removed:
                return current
            rows = np.array(
                [
                    i
                    for i, item_id in enumerate(current.item_ids)
                    if item_id not in removed
                ],
                dtype=np.int64,
            )
            if not rows.size:
                raise ValueError("candidate catalog must contain at least one item")
            metadata = current.metadata
            return self._publish(
                MultiModalCandidateCatalog(
                    item_ids=current.item_ids[rows],
                    item_features={
                        name: take_features(matrix, rows)
                        for name, matrix in current.item_features.items()
                    },
                    metadata=None
                    if metadata is None
                    else metadata.iloc[rows].reset_index(drop=True),
                    feature_space_id=current.feature_space_id,
                    version=current.version + 1,
                    dtype=self._dtype,
                )
            )

    def align_source(
        self,
        source: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
    ) -> csr_matrix:
        if self._source_vocabulary is None:
            raise RuntimeError(_NOT_INSTALLED)
        return self._source_vocabulary.align_csr(source, item_ids=item_ids)

    def resolve_selection(
        self,
        candidate_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> MultiModalCandidateSelection:
        """Capture consistent catalog/source state, then prepare rows unlocked."""
        with self._lock:
            current = self.snapshot()
            # A reinstall replaces both references. Capture the pair together,
            # then use these immutable objects without blocking other readers
            # or publications during feature copies and mapping construction.
            source_vocabulary = self._source_vocabulary
            assert source_vocabulary is not None
        rows = (
            np.arange(current.n_items, dtype=np.int64)
            if candidate_ids is None
            else np.sort(current.rows_for(candidate_ids))
        )
        features = (
            current.item_features
            if rows.size == current.n_items
            else MappingProxyType(
                {
                    name: _freeze_features(take_features(matrix, rows))
                    for name, matrix in current.item_features.items()
                }
            )
        )
        source_to_candidate = np.array(
            [
                current.id_to_row.get(item_id, -1)
                for item_id in source_vocabulary.item_ids
            ],
            dtype=np.int64,
        )
        candidate_to_local = np.full(current.n_items, -1, dtype=np.int64)
        candidate_to_local[rows] = np.arange(rows.size, dtype=np.int64)
        return MultiModalCandidateSelection(
            current,
            rows,
            features,
            source_to_candidate,
            candidate_to_local,
        )

    def _save_checkpoint(
        self,
        writer: ModelCheckpointWriter,
        *,
        prefix: str = "catalog",
    ) -> None:
        """Save the multimodal schema and one matrix per modality."""
        with self._lock:
            current = self.snapshot()
            assert self._source_vocabulary is not None and self._dtype is not None
            writer.write_item_ids(
                f"{prefix}/source_item_ids.json", self._source_vocabulary.item_ids
            )
            writer.write_item_ids(f"{prefix}/candidate_item_ids.json", current.item_ids)
            modalities = []
            for index, (name, matrix) in enumerate(current.item_features.items()):
                # Names are data, never filesystem paths.
                storage = writer.write_features(f"{prefix}/features/{index}", matrix)
                modalities.append(
                    {"name": name, "n_features": matrix.shape[1], "storage": storage}
                )
            metadata = current.metadata
            metadata_columns = None
            if metadata is not None:
                # Parquet stringifies labels: 7 and "7" would collide. Persist
                # labels separately and use unique physical names for the data.
                metadata_columns = _metadata_columns_state(metadata.columns)
                metadata.columns = [
                    f"column_{index}" for index in range(metadata.shape[1])
                ]
                writer.write_dataframe(f"{prefix}/metadata.parquet", metadata)
            writer.write_json(
                f"{prefix}/state.json",
                {
                    "format": "multimodal",
                    "version": 2,
                    "modalities": modalities,
                    "dtype": self._dtype.str,
                    "feature_space_id": current.feature_space_id,
                    "catalog_version": current.version,
                    "metadata_columns": metadata_columns,
                    "metadata_dtypes": None
                    if metadata is None
                    else {name: str(dtype) for name, dtype in metadata.dtypes.items()},
                },
            )

    def _load_checkpoint(
        self,
        reader: ModelCheckpointReader,
        *,
        prefix: str = "catalog",
    ) -> MultiModalCandidateCatalog:
        """Validate the full checkpoint before replacing any installed state."""
        state = reader.read_json(f"{prefix}/state.json")
        if state.get("format") != "multimodal" or state.get("version") not in (1, 2):
            raise ValueError("unsupported multimodal catalog checkpoint format")
        for key in ("dtype", "feature_space_id", "catalog_version"):
            if key not in state:
                raise ValueError(f"multimodal catalog checkpoint is missing {key!r}")
        dtype = _canonical_dtype(state["dtype"])
        modalities = state.get("modalities")
        if not isinstance(modalities, list) or not modalities:
            raise ValueError("checkpoint must describe at least one modality")
        features = {}
        for index, modality in enumerate(modalities):
            if not isinstance(modality, dict):
                raise ValueError(  # noqa: TRY004 - invalid serialized checkpoint data
                    "checkpoint modality schema must be an object"
                )
            name = modality.get("name")
            if not isinstance(name, str) or not name.strip() or name in features:
                raise ValueError(
                    "checkpoint modality names must be unique non-empty strings"
                )
            matrix = reader.read_features(
                f"{prefix}/features/{index}", storage=modality.get("storage")
            )
            if matrix.ndim != 2 or matrix.shape[1] != modality.get("n_features"):
                raise ValueError(
                    f"checkpoint modality {name!r} has incompatible dimensions"
                )
            features[name] = matrix
        if "metadata_dtypes" not in state:
            raise ValueError("checkpoint metadata dtype description is missing")
        metadata_dtypes = state["metadata_dtypes"]
        candidate_ids = reader.read_item_ids(f"{prefix}/candidate_item_ids.json")
        metadata = None
        if metadata_dtypes is not None:
            if not isinstance(metadata_dtypes, dict) or any(
                not isinstance(value, str) for value in metadata_dtypes.values()
            ):
                raise ValueError("invalid checkpoint metadata dtype description")
            metadata = reader.read_dataframe(f"{prefix}/metadata.parquet")
            if set(metadata.columns) != set(metadata_dtypes):
                raise ValueError(
                    "checkpoint metadata columns do not match their schema"
                )
            try:
                metadata = metadata.astype(metadata_dtypes)
            except (TypeError, ValueError) as error:
                raise ValueError("cannot restore checkpoint metadata dtypes") from error
            if state["version"] == 2:
                expected = [f"column_{index}" for index in range(metadata.shape[1])]
                if metadata.columns.tolist() != expected:
                    raise ValueError("checkpoint metadata storage columns are invalid")
                metadata.columns = _restore_metadata_columns(
                    state.get("metadata_columns"), n_columns=metadata.shape[1]
                )
                # Parquet without an index cannot retain rows of a zero-column
                # frame; its row count comes from the aligned candidate IDs.
                if metadata.shape[1] == 0:
                    metadata = metadata.reindex(range(candidate_ids.size))
        elif (
            reader.exists(f"{prefix}/metadata.parquet")
            or state.get("metadata_columns") is not None
        ):
            raise ValueError("checkpoint metadata dtype description is missing")
        vocabulary = ItemVocabulary.from_ids(
            reader.read_item_ids(f"{prefix}/source_item_ids.json"),
            name="source_item_ids",
        )
        snapshot = MultiModalCandidateCatalog(
            item_ids=candidate_ids,
            item_features=features,
            metadata=metadata,
            feature_space_id=state["feature_space_id"],
            version=state["catalog_version"],
            dtype=dtype,
        )
        with self._lock:
            self._source_vocabulary = vocabulary
            self._dtype = dtype
            return self._publish(snapshot)


def _updated_metadata(
    current: MultiModalCandidateCatalog,
    incoming: pd.DataFrame | None,
    replacements: np.ndarray,
    old_rows: np.ndarray,
    additions: np.ndarray,
    *,
    combined_ids: np.ndarray,
) -> pd.DataFrame | None:
    old = current.metadata
    if old is None and incoming is None:
        return None
    old = pd.DataFrame(index=range(current.n_items)) if old is None else old
    columns = (
        old.columns
        if incoming is None
        else old.columns.union(incoming.columns, sort=False)
    )
    result = old.reindex(columns=columns)
    introduced = [column for column in columns if column not in old.columns]
    if introduced:
        result[introduced] = result[introduced].astype(object)
    if additions.size:
        result = result.reindex(range(len(combined_ids)))
        if incoming is not None:
            for column in incoming.columns:
                result.iloc[current.n_items :, result.columns.get_loc(column)] = (
                    incoming.iloc[additions][column].to_numpy()
                )
    if incoming is not None and replacements.size:
        result.loc[old_rows, incoming.columns] = incoming.iloc[replacements].to_numpy()
    if "item_id" in result.columns:
        result["item_id"] = combined_ids
    return result.reset_index(drop=True)


class BaseMultiModalRecommender(BaseColdStartRecommender):
    """Optional CSR-history base with a default named-matrix candidate catalog.

    Implement ``fit``, ``is_fitted`` and ``predict_on_batch``. Fitting installs
    the source IDs and stored matrices with ``self.candidates.install(...)``.
    Recommendation, history alignment and prediction batching are inherited.

    Initial installation, rebuilding and updates must produce the same stored
    representation using the same fitted transformation. Override candidate
    methods if a model must encode incoming matrices first. Other storage and
    scoring designs are equally valid; no fusion rule is imposed by this base.

    The default persistence hooks save/load the catalog. Overrides saving model
    state should call ``super()`` as well, or manage catalog persistence themselves.
    The model still supplies its checkpoint construction/configuration hooks.

    Mapping-specific overrides intentionally specialize the existing runtime
    interface; the matrix base and its annotations are not made generic here.
    """

    candidates: MutableMultiModalCandidateCatalog  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.candidates = MutableMultiModalCandidateCatalog(
            on_publish=self._on_catalog_published
        )

    @abstractmethod
    def fit(  # type: ignore[override]
        self,
        interactions: csr_matrix,
        item_features: MultiModalItemFeatures,
        **kwargs,
    ) -> BaseMultiModalRecommender:
        """Fit the model and install its initial multimodal candidate catalog."""

    def _on_catalog_published(self, catalog: MultiModalCandidateCatalog) -> None:  # type: ignore[override]
        """Invalidate caches under the catalog lock; raising does not roll back.

        Do not wait for another thread that reads or mutates this catalog: it
        needs the same lock. Workers can instead use the immutable ``catalog``
        argument directly without reacquiring the owner's lock.
        """

    def build_candidates(  # type: ignore[override]
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> MultiModalCandidateCatalog:
        """Replace candidates using already prepared stored representations."""
        return self.candidates.build(
            item_ids=item_ids,
            item_features=item_features,
            metadata=metadata,
            feature_space_id=feature_space_id,
        )

    def update_candidates(  # type: ignore[override]
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> MultiModalCandidateCatalog:
        """Append or replace candidates using already prepared representations."""
        return self.candidates.update(
            item_ids=item_ids,
            item_features=item_features,
            metadata=metadata,
            on_conflict=on_conflict,
            feature_space_id=feature_space_id,
        )

    def remove_candidates(  # type: ignore[override]
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> MultiModalCandidateCatalog:
        return self.candidates.remove(item_ids, missing=missing)

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        self.candidates._save_checkpoint(writer)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        self.candidates._load_checkpoint(reader)
