from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Hashable, Literal, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack, issparse, isspmatrix_csr, vstack

from compresso import SRPTensor
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.base import Recommender

__all__ = [
    "CandidateCatalog",
    "ColdStartRecommender",
    "ItemVocabulary",
]

ItemFeatures = csr_matrix | SRPTensor | np.ndarray | torch.Tensor
CandidateConflict = Literal["error", "replace", "ignore"]


def canonical_item_ids(
    item_ids: Sequence[Hashable] | np.ndarray,
    *,
    expected_rows: int | None = None,
    expected_rows_name: str = "item_features",
    expected_rows_unit: str = "rows",
    name: str = "item_ids",
) -> np.ndarray:
    if isinstance(item_ids, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence of item IDs")
    try:
        values = list(item_ids)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a one-dimensional sequence of item IDs"
        ) from error
    ids = np.empty(len(values), dtype=object)
    ids[:] = values
    if expected_rows is not None and ids.size != expected_rows:
        raise ValueError(
            f"{name} has {ids.size} entries, but {expected_rows_name} has "
            f"{expected_rows} {expected_rows_unit}"
        )
    if ids.size < 1:
        raise ValueError(f"{name} must contain at least one item")

    seen: dict[Hashable, int] = {}
    for position, item_id in enumerate(ids.tolist()):
        try:
            hash(item_id)
        except TypeError as error:
            raise TypeError(
                f"{name} entry at position {position} is not hashable"
            ) from error
        try:
            missing = bool(pd.isna(item_id))
        except (TypeError, ValueError):
            missing = False
        if item_id is None or missing:
            raise ValueError(f"{name} must not contain missing IDs")
        if item_id in seen:
            raise ValueError(f"{name} must not contain duplicate IDs: {item_id!r}")
        seen[item_id] = position

    ids.setflags(write=False)
    return ids


def canonical_metadata(
    metadata: pd.DataFrame | None,
    *,
    item_ids: np.ndarray,
) -> pd.DataFrame | None:
    if metadata is None:
        return None
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas.DataFrame or None")
    if len(metadata) != item_ids.size:
        raise ValueError(
            f"metadata has {len(metadata)} rows, but item_ids has "
            f"{item_ids.size} entries"
        )
    out = metadata.reset_index(drop=True).copy(deep=True)
    if "item_id" in out.columns:
        metadata_ids = canonical_item_ids(
            out["item_id"].tolist(),
            expected_rows=item_ids.size,
            name="metadata['item_id']",
        )
        if not np.array_equal(metadata_ids, item_ids):
            raise ValueError("metadata['item_id'] must match item_ids in row order")
    return out


def canonical_feature_space_id(feature_space_id: str | None) -> str | None:
    if feature_space_id is None:
        return None
    if not isinstance(feature_space_id, str) or not feature_space_id.strip():
        raise ValueError("feature_space_id must be a non-empty string or None")
    return feature_space_id


def _torch_sparse_to_csr(features: torch.Tensor) -> csr_matrix:
    coo = features.detach().cpu().to_sparse_coo().coalesce()
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    return csr_matrix(
        (values, (indices[0], indices[1])),
        shape=tuple(features.shape),
    )


def canonical_item_features(
    features: ItemFeatures,
    *,
    dtype: np.dtype,
) -> csr_matrix | np.ndarray:
    if isinstance(features, SRPTensor):
        if features.dim() != 2:
            raise ValueError("item_features must be two-dimensional")
        if torch.is_complex(features.vals):
            raise TypeError("item_features must contain real numeric values")
        torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
        features = features.to(device="cpu", dtype=torch_dtype).to_scipy_csr()
    elif isinstance(features, torch.Tensor):
        if features.ndim != 2:
            raise ValueError("item_features must be two-dimensional")
        if torch.is_complex(features):
            raise TypeError("item_features must contain real numeric values")
        torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
        features = features.detach().to(device="cpu", dtype=torch_dtype)
        if features.layout == torch.strided:
            features = features.numpy()
        else:
            features = _torch_sparse_to_csr(features)

    if isspmatrix_csr(features):
        out = canonical_csr(features, name="item_features")
        if out.ndim != 2:
            raise ValueError("item_features must be two-dimensional")
        if out.shape[0] < 1 or out.shape[1] < 1:
            raise ValueError(
                "item_features must contain at least one item and one feature"
            )
        if np.iscomplexobj(out.data):
            raise TypeError("item_features must contain real numeric values")
        return out.astype(dtype, copy=False)

    if not isinstance(features, np.ndarray):
        raise TypeError(
            "item_features must be a scipy.sparse.csr_matrix, "
            "compresso.SRPTensor, numpy.ndarray, or torch.Tensor"
        )
    if features.ndim != 2:
        raise ValueError("item_features must be two-dimensional")
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("item_features must contain at least one item and one feature")
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("item_features must contain numeric values")
    if np.iscomplexobj(features):
        raise TypeError("item_features must contain real numeric values")
    if not np.all(np.isfinite(features)):
        raise ValueError("item_features values must be finite")
    return np.asarray(features, dtype=dtype, order="C")


def append_column(
    matrix: csr_matrix | np.ndarray,
    column: np.ndarray,
) -> csr_matrix | np.ndarray:
    if isspmatrix_csr(matrix):
        return hstack((matrix, csr_matrix(column[:, None])), format="csr")
    return np.concatenate((matrix, column[:, None]), axis=1)


def take_features(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
) -> csr_matrix | np.ndarray:
    selected = features[rows]
    return selected.tocsr() if issparse(selected) else np.asarray(selected)


def _freeze_features(features: csr_matrix | np.ndarray) -> csr_matrix | np.ndarray:
    out = features.copy()
    if isspmatrix_csr(out):
        out.data.setflags(write=False)
        out.indices.setflags(write=False)
        out.indptr.setflags(write=False)
    else:
        out.setflags(write=False)
    return out


def _stack_features(
    top: csr_matrix | np.ndarray,
    bottom: csr_matrix | np.ndarray,
) -> csr_matrix | np.ndarray:
    if isspmatrix_csr(top) or isspmatrix_csr(bottom):
        return vstack((csr_matrix(top), csr_matrix(bottom)), format="csr")
    return np.concatenate((top, bottom), axis=0)


def _replace_feature_rows(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
    replacements: csr_matrix | np.ndarray,
) -> csr_matrix | np.ndarray:
    if rows.size == 0:
        return features
    if isinstance(features, np.ndarray) and isinstance(replacements, np.ndarray):
        out = features.copy()
        out[rows] = replacements
        return out
    out = csr_matrix(features).tolil(copy=True)
    out[rows] = csr_matrix(replacements)
    return out.tocsr()


@dataclass(frozen=True)
class ItemVocabulary:
    """Immutable stable-ID vocabulary for sparse matrix columns."""

    item_ids: np.ndarray
    id_to_row: Mapping[Hashable, int]

    @classmethod
    def from_ids(
        cls,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        name: str = "item_ids",
    ) -> ItemVocabulary:
        ids = canonical_item_ids(item_ids, name=name)
        frozen_ids = ids.copy()
        frozen_ids.setflags(write=False)
        return cls(
            item_ids=frozen_ids,
            id_to_row=MappingProxyType(
                {item_id: row for row, item_id in enumerate(frozen_ids.tolist())}
            ),
        )

    @property
    def n_items(self) -> int:
        """Number of item IDs in the vocabulary."""
        return int(self.item_ids.size)

    def align_csr(
        self,
        matrix: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        name: str = "source",
    ) -> csr_matrix:
        """Select and reorder sparse columns to match this vocabulary."""
        if not isspmatrix_csr(matrix):
            raise TypeError(f"{name} must be a scipy.sparse.csr_matrix")
        if item_ids is self.item_ids and matrix.shape[1] == self.n_items:
            return matrix
        external_ids = canonical_item_ids(
            item_ids,
            expected_rows=matrix.shape[1],
            expected_rows_name=name,
            expected_rows_unit="columns",
        )
        if np.array_equal(external_ids, self.item_ids):
            return matrix
        external_to_column = {
            item_id: column for column, item_id in enumerate(external_ids.tolist())
        }
        missing = [
            item_id
            for item_id in self.item_ids.tolist()
            if item_id not in external_to_column
        ]
        if missing:
            raise ValueError(
                f"{name} item_ids is missing fitted source item ID: {missing[0]!r}"
            )
        columns = np.fromiter(
            (external_to_column[item_id] for item_id in self.item_ids.tolist()),
            dtype=np.int64,
            count=self.n_items,
        )
        return matrix[:, columns].tocsr()


@dataclass(frozen=True, init=False)
class CandidateCatalog:
    """Immutable snapshot of a feature-based candidate catalog."""

    item_ids: np.ndarray
    item_features: csr_matrix | np.ndarray
    _metadata: pd.DataFrame | None = field(repr=False)
    feature_space_id: str | None
    version: int
    id_to_row: Mapping[Hashable, int]

    def __init__(
        self,
        *,
        item_ids: np.ndarray,
        item_features: csr_matrix | np.ndarray,
        metadata: pd.DataFrame | None,
        feature_space_id: str | None,
        version: int,
        id_to_row: Mapping[Hashable, int],
    ) -> None:
        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "item_features", item_features)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "feature_space_id", feature_space_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "id_to_row", id_to_row)

    @property
    def n_items(self) -> int:
        """Number of candidates in this snapshot."""
        return int(self.item_ids.size)

    @property
    def metadata(self) -> pd.DataFrame | None:
        """Return a defensive copy of metadata aligned with candidate rows."""
        return None if self._metadata is None else self._metadata.copy(deep=True)

    def rows_for(self, item_ids: Sequence[Hashable]) -> np.ndarray:
        """Resolve stable item IDs to candidate rows in request order."""
        ids = canonical_item_ids(item_ids, name="candidate_ids")
        rows = np.empty(ids.size, dtype=np.int64)
        for position, item_id in enumerate(ids.tolist()):
            try:
                rows[position] = self.id_to_row[item_id]
            except KeyError as error:
                raise KeyError(f"unknown candidate item ID: {item_id!r}") from error
        return rows

    def ids_for(self, rows: np.ndarray | torch.Tensor) -> np.ndarray:
        """Resolve candidate row indices to stable item IDs."""
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().numpy()
        row_array = np.asarray(rows)
        if not np.issubdtype(row_array.dtype, np.integer):
            raise TypeError("candidate rows must contain integers")
        if row_array.size and (
            int(row_array.min()) < 0 or int(row_array.max()) >= self.n_items
        ):
            raise IndexError("candidate row is out of bounds")
        return self.item_ids[row_array]


@dataclass(frozen=True)
class CandidateSelection:
    catalog: CandidateCatalog
    rows: np.ndarray
    features: csr_matrix | np.ndarray
    source_to_candidate: np.ndarray
    candidate_to_local: np.ndarray


def _make_catalog(
    *,
    item_ids: np.ndarray,
    item_features: csr_matrix | np.ndarray,
    metadata: pd.DataFrame | None,
    feature_space_id: str | None,
    version: int,
) -> CandidateCatalog:
    vocabulary = ItemVocabulary.from_ids(item_ids)
    return CandidateCatalog(
        item_ids=vocabulary.item_ids,
        item_features=_freeze_features(item_features),
        metadata=None if metadata is None else metadata.copy(deep=True),
        feature_space_id=feature_space_id,
        version=int(version),
        id_to_row=vocabulary.id_to_row,
    )


@runtime_checkable
class ColdStartRecommender(Recommender, Protocol):
    """Recommender with distinct identified source and candidate spaces."""

    source_vocabulary_: ItemVocabulary | None

    @property
    def candidates(self) -> CandidateCatalog: ...

    def align_source(
        self,
        source: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
    ) -> csr_matrix: ...

    def build_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> CandidateCatalog: ...

    def update_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> CandidateCatalog: ...

    def remove_candidates(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> CandidateCatalog: ...


class FeatureCatalogMixin:
    """Reusable source-vocabulary and feature-catalog implementation."""

    def _init_feature_catalog_state(self) -> None:
        self._catalog_lock = RLock()
        self._candidates: CandidateCatalog | None = None
        self.source_vocabulary_: ItemVocabulary | None = None
        self.source_item_ids_: np.ndarray | None = None
        self.source_id_to_row_: Mapping[Hashable, int] | None = None
        self.source_popularity_: np.ndarray | None = None
        self.feature_space_id_: str | None = None
        self.n_input_features_: int | None = None
        self._feature_catalog_dtype: np.dtype | None = None
        self._feature_catalog_include_popularity = False

    @property
    def candidates(self) -> CandidateCatalog:
        """Current immutable candidate-catalog snapshot."""
        catalog = self._candidates
        if catalog is None:
            raise RuntimeError("model must be fitted before accessing candidates")
        return catalog

    @property
    def n_candidates_(self) -> int | None:
        """Number of current candidates, or ``None`` before fitting."""
        return None if self._candidates is None else self._candidates.n_items

    def _install_feature_catalog(
        self,
        *,
        source_item_ids: np.ndarray,
        source_popularity: np.ndarray,
        n_input_features: int,
        candidate_features: csr_matrix | np.ndarray,
        metadata: pd.DataFrame | None,
        feature_space_id: str | None,
        dtype: np.dtype,
        include_popularity: bool,
    ) -> CandidateCatalog:
        """Atomically replace the complete candidate catalog."""
        vocabulary = ItemVocabulary.from_ids(source_item_ids)
        popularity = np.asarray(source_popularity, dtype=dtype).copy()
        popularity.setflags(write=False)
        catalog = _make_catalog(
            item_ids=vocabulary.item_ids,
            item_features=candidate_features,
            metadata=metadata,
            feature_space_id=feature_space_id,
            version=1,
        )
        self.source_vocabulary_ = vocabulary
        self.source_item_ids_ = vocabulary.item_ids
        self.source_id_to_row_ = vocabulary.id_to_row
        self.source_popularity_ = popularity
        self.feature_space_id_ = feature_space_id
        self.n_input_features_ = int(n_input_features)
        self._feature_catalog_dtype = np.dtype(dtype)
        self._feature_catalog_include_popularity = bool(include_popularity)
        with self._catalog_lock:
            self._candidates = catalog
            self._on_catalog_published(catalog)
        return catalog

    def _on_catalog_published(self, catalog: CandidateCatalog) -> None:
        pass

    def _prepare_catalog_features(
        self,
        item_ids: np.ndarray,
        item_features: ItemFeatures,
    ) -> csr_matrix | np.ndarray:
        if (
            self.n_input_features_ is None
            or self.source_id_to_row_ is None
            or self.source_popularity_ is None
            or self._feature_catalog_dtype is None
        ):
            raise RuntimeError("model must be fitted before changing candidates")
        features = canonical_item_features(
            item_features,
            dtype=self._feature_catalog_dtype,
        )
        if features.shape[0] != item_ids.size:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but item_ids "
                f"has {item_ids.size} entries"
            )
        if features.shape[1] != self.n_input_features_:
            raise ValueError(
                f"item_features has {features.shape[1]} columns, but the model "
                f"was fitted with {self.n_input_features_} input features"
            )
        if self._feature_catalog_include_popularity:
            popularity = np.zeros(item_ids.size, dtype=self._feature_catalog_dtype)
            for row, item_id in enumerate(item_ids.tolist()):
                source_row = self.source_id_to_row_.get(item_id)
                if source_row is not None:
                    popularity[row] = self.source_popularity_[source_row]
            features = append_column(features, popularity)
        return features

    def _resolve_catalog_feature_space_id(
        self,
        feature_space_id: str | None,
    ) -> str | None:
        resolved = canonical_feature_space_id(feature_space_id)
        if resolved is None:
            return self.feature_space_id_
        if resolved != self.feature_space_id_:
            raise ValueError(
                "feature_space_id must match the feature space used to fit the "
                "model; set feature_space_id during fit to enable this check"
            )
        return resolved

    def build_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Add or update candidates and atomically publish a new snapshot."""
        ids = canonical_item_ids(item_ids)
        candidate_metadata = canonical_metadata(metadata, item_ids=ids)
        features = self._prepare_catalog_features(ids, item_features)
        resolved_space = self._resolve_catalog_feature_space_id(feature_space_id)
        with self._catalog_lock:
            current = self.candidates
            catalog = _make_catalog(
                item_ids=ids,
                item_features=features,
                metadata=candidate_metadata,
                feature_space_id=resolved_space,
                version=current.version + 1,
            )
            self._candidates = catalog
            self._on_catalog_published(catalog)
        return catalog

    def update_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Remove registered candidates and publish a new snapshot."""
        if on_conflict not in {"error", "replace", "ignore"}:
            raise ValueError("on_conflict must be 'error', 'replace', or 'ignore'")
        ids = canonical_item_ids(item_ids)
        incoming_metadata = canonical_metadata(metadata, item_ids=ids)
        incoming_features = self._prepare_catalog_features(ids, item_features)
        resolved_space = self._resolve_catalog_feature_space_id(feature_space_id)
        with self._catalog_lock:
            current = self.candidates
            conflicts = np.array(
                [item_id in current.id_to_row for item_id in ids.tolist()],
                dtype=bool,
            )
            if on_conflict == "error" and bool(conflicts.any()):
                first = ids[int(np.flatnonzero(conflicts)[0])]
                raise ValueError(f"candidate item ID already exists: {first!r}")
            replace_input_rows = (
                np.flatnonzero(conflicts)
                if on_conflict == "replace"
                else np.empty(0, dtype=np.int64)
            )
            replace_catalog_rows = np.asarray(
                [current.id_to_row[ids[row]] for row in replace_input_rows],
                dtype=np.int64,
            )
            additions = np.flatnonzero(~conflicts)
            if replace_input_rows.size == 0 and additions.size == 0:
                return current
            features = _replace_feature_rows(
                current.item_features,
                replace_catalog_rows,
                take_features(incoming_features, replace_input_rows),
            )
            if additions.size:
                features = _stack_features(
                    features,
                    take_features(incoming_features, additions),
                )
                combined_ids = np.concatenate((current.item_ids, ids[additions]))
            else:
                combined_ids = current.item_ids.copy()
            combined_metadata = self._updated_metadata(
                current=current,
                incoming=incoming_metadata,
                replace_input_rows=replace_input_rows,
                replace_catalog_rows=replace_catalog_rows,
                addition_input_rows=additions,
            )
            catalog = _make_catalog(
                item_ids=combined_ids,
                item_features=features,
                metadata=combined_metadata,
                feature_space_id=resolved_space,
                version=current.version + 1,
            )
            self._candidates = catalog
            self._on_catalog_published(catalog)
        return catalog

    @staticmethod
    def _updated_metadata(
        *,
        current: CandidateCatalog,
        incoming: pd.DataFrame | None,
        replace_input_rows: np.ndarray,
        replace_catalog_rows: np.ndarray,
        addition_input_rows: np.ndarray,
    ) -> pd.DataFrame | None:
        if current.metadata is None and incoming is None:
            return None
        old = (
            current.metadata.copy(deep=True)
            if current.metadata is not None
            else pd.DataFrame(index=range(current.n_items))
        )
        new = incoming if incoming is not None else pd.DataFrame()
        columns = old.columns.union(new.columns, sort=False)
        result = old.reindex(columns=columns)
        if addition_input_rows.size:
            additions = pd.DataFrame(
                pd.NA,
                index=range(addition_input_rows.size),
                columns=columns,
            )
            if incoming is not None:
                additions.loc[:, incoming.columns] = (
                    incoming.iloc[addition_input_rows].reset_index(drop=True).to_numpy()
                )
            result = pd.concat((result, additions), ignore_index=True)
        if incoming is not None and replace_input_rows.size:
            result.loc[replace_catalog_rows, incoming.columns] = incoming.iloc[
                replace_input_rows
            ].to_numpy()
        return result.reset_index(drop=True)

    def remove_candidates(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> CandidateCatalog:
        if missing not in {"error", "ignore"}:
            raise ValueError("missing must be 'error' or 'ignore'")
        ids = canonical_item_ids(item_ids)
        with self._catalog_lock:
            current = self.candidates
            unknown = [
                item_id for item_id in ids.tolist() if item_id not in current.id_to_row
            ]
            if unknown and missing == "error":
                raise KeyError(f"unknown candidate item ID: {unknown[0]!r}")
            removed = {
                item_id for item_id in ids.tolist() if item_id in current.id_to_row
            }
            if not removed:
                return current
            keep = np.asarray(
                [item_id not in removed for item_id in current.item_ids.tolist()],
                dtype=bool,
            )
            if not bool(keep.any()):
                raise ValueError("candidate catalog must contain at least one item")
            rows = np.flatnonzero(keep)
            metadata = (
                None
                if current.metadata is None
                else current.metadata.iloc[rows].reset_index(drop=True)
            )
            catalog = _make_catalog(
                item_ids=current.item_ids[rows],
                item_features=take_features(current.item_features, rows),
                metadata=metadata,
                feature_space_id=current.feature_space_id,
                version=current.version + 1,
            )
            self._candidates = catalog
            self._on_catalog_published(catalog)
        return catalog

    def align_source(
        self,
        source: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
    ) -> csr_matrix:
        """Align external sparse columns to the fitted source vocabulary."""
        if self.source_vocabulary_ is None:
            raise RuntimeError("model must be fitted before aligning source data")
        return self.source_vocabulary_.align_csr(source, item_ids=item_ids)

    def _resolve_candidate_selection(
        self,
        candidate_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> CandidateSelection:
        catalog = self.candidates
        rows = (
            np.arange(catalog.n_items, dtype=np.int64)
            if candidate_ids is None
            else np.sort(catalog.rows_for(candidate_ids))
        )
        features = (
            catalog.item_features
            if rows.size == catalog.n_items
            else take_features(catalog.item_features, rows)
        )
        assert self.source_item_ids_ is not None
        source_to_candidate = np.fromiter(
            (
                catalog.id_to_row.get(item_id, -1)
                for item_id in self.source_item_ids_.tolist()
            ),
            dtype=np.int64,
            count=self.source_item_ids_.size,
        )
        candidate_to_local = np.full(catalog.n_items, -1, dtype=np.int64)
        candidate_to_local[rows] = np.arange(rows.size, dtype=np.int64)
        return CandidateSelection(
            catalog=catalog,
            rows=rows,
            features=features,
            source_to_candidate=source_to_candidate,
            candidate_to_local=candidate_to_local,
        )
