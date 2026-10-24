"""The catalog that grows, and the bookkeeping that keeps it consistent.

Adding or replacing an item has to keep the feature matrices, the identifiers
and any stored scores agreeing with each other, under a lock, which is why
this is far longer than the fixed snapshot it produces.
"""

from __future__ import annotations

from threading import RLock
from typing import (
    Callable,
    Hashable,
    Literal,
    Mapping,
    Sequence,
)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.models.core.identifiers import (
    ItemVocabulary,
    canonical_item_ids,
)
from compresso_recsys.models.core.features import (
    ItemFeatures,
    _replace_feature_rows,
    _stack_features,
    append_column,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_metadata,
    take_features,
)
from compresso_recsys.models.core.catalog import (
    _NOT_INSTALLED,
    CandidateCatalog,
    CandidateConflict,
    CandidateSelection,
    _make_catalog,
)


__all__ = ["MutableCandidateCatalog"]


class MutableCandidateCatalog:
    """The lifecycle around a :class:`CandidateCatalog`, as an owned object.

    :class:`CandidateCatalog` is an immutable snapshot and needs nothing. What
    used to be stuck inside :class:`BaseColdStartRecommender` was the *lifecycle*
    around it: the lock, the current snapshot, the fitted source vocabulary, and
    the dozen methods that publish, extend, shrink and align against them.

    While that lived on a base class, "cold-capable" meant "inherits
    :class:`BaseColdStartRecommender`". Adding a second axis -- a model that reads
    ordered histories rather than a matrix -- then forced a choice between
    multiple inheritance and a fourth base class for two independent ideas. An
    owned object removes the choice: any model can hold one.

    Composition rather than a mixin, because the state is what decides it. A mixin
    would not encapsulate these attributes, it would install them on whatever
    class it is mixed into -- and two stateful mixins initialising through
    ``super().__init__()`` is where MRO pain lives. This has its own
    ``__init__``, its own lock and its own tests, and a model could own two if
    that ever made sense::

        class SequentialContentRNN(BaseSequentialRecommender):
            def __init__(self) -> None:
                self.candidates = MutableCandidateCatalog()

            def predict_on_batch(self, source, *, k, exclude_seen=True):
                catalog = self.candidates.snapshot()

    Reads go through :meth:`snapshot`, deliberately, rather than through
    forwarded properties. A snapshot is a consistent view: several reads off one
    snapshot cannot straddle a concurrent republish, which forwarding
    ``n_items``, ``item_ids`` and ``rows_for`` separately would silently allow.

    ``on_publish`` is called with each new snapshot while the lock is held, which
    is how an owner drops caches derived from the previous one.
    """

    def __init__(
        self,
        *,
        on_publish: Callable[[CandidateCatalog], None] | None = None,
    ) -> None:
        self._on_publish = on_publish
        self._lock = RLock()
        self._snapshot: CandidateCatalog | None = None
        self._source_vocabulary: ItemVocabulary | None = None
        self._source_item_ids: np.ndarray | None = None
        self._source_id_to_row: Mapping[Hashable, int] | None = None
        self._source_popularity: np.ndarray | None = None
        self._feature_space_id: str | None = None
        self._n_input_features: int | None = None
        self._dtype: np.dtype | None = None
        self._include_popularity = False

    # -- reading ------------------------------------------------------------

    @property
    def is_installed(self) -> bool:
        """Whether a catalog has been published yet."""
        return self._snapshot is not None

    def snapshot(self) -> CandidateCatalog:
        """The current immutable snapshot.

        Take one and read every field off it, rather than reading fields off
        this object one at a time: only the snapshot is guaranteed internally
        consistent against a concurrent :meth:`build`, :meth:`update` or
        :meth:`remove`.
        """
        catalog = self._snapshot
        if catalog is None:
            raise RuntimeError(_NOT_INSTALLED)
        return catalog

    @property
    def n_items(self) -> int | None:
        """Number of current candidates, or ``None`` before installation."""
        return None if self._snapshot is None else self._snapshot.n_items

    @property
    def source_vocabulary(self) -> ItemVocabulary | None:
        """Item space a source matrix must be expressed over."""
        return self._source_vocabulary

    @property
    def source_item_ids(self) -> np.ndarray | None:
        """Stable IDs of the fitted source items, in column order."""
        return self._source_item_ids

    @property
    def source_id_to_row(self) -> Mapping[Hashable, int] | None:
        """Source item ID to source column."""
        return self._source_id_to_row

    @property
    def source_popularity(self) -> np.ndarray | None:
        """Per-source-item popularity recorded at installation."""
        return self._source_popularity

    @property
    def feature_space_id(self) -> str | None:
        """Identifier of the feature space, when one was declared."""
        return self._feature_space_id

    @property
    def n_input_features(self) -> int | None:
        """Feature columns every candidate must supply."""
        return self._n_input_features

    # -- lifecycle ----------------------------------------------------------

    def install(
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
        self._source_vocabulary = vocabulary
        self._source_item_ids = vocabulary.item_ids
        self._source_id_to_row = vocabulary.id_to_row
        self._source_popularity = popularity
        self._feature_space_id = feature_space_id
        self._n_input_features = int(n_input_features)
        self._dtype = np.dtype(dtype)
        self._include_popularity = bool(include_popularity)
        with self._lock:
            self._snapshot = catalog
            self._notify(catalog)
        return catalog
    def _notify(self, catalog: CandidateCatalog) -> None:
        """Tell the owner a new snapshot is live, so it can drop stale caches."""
        if self._on_publish is not None:
            self._on_publish(catalog)

    def _prepare_features(
        self,
        item_ids: np.ndarray,
        item_features: ItemFeatures,
    ) -> csr_matrix | np.ndarray:
        if (
            self._n_input_features is None
            or self._source_id_to_row is None
            or self._source_popularity is None
            or self._dtype is None
        ):
            raise RuntimeError(_NOT_INSTALLED)
        features = canonical_item_features(
            item_features,
            dtype=self._dtype,
        )
        if features.shape[0] != item_ids.size:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but item_ids "
                f"has {item_ids.size} entries"
            )
        if features.shape[1] != self._n_input_features:
            raise ValueError(
                f"item_features has {features.shape[1]} columns, but the model "
                f"was fitted with {self._n_input_features} input features"
            )
        if self._include_popularity:
            popularity = np.zeros(item_ids.size, dtype=self._dtype)
            for row, item_id in enumerate(item_ids.tolist()):
                source_row = self._source_id_to_row.get(item_id)
                if source_row is not None:
                    popularity[row] = self._source_popularity[source_row]
            features = append_column(features, popularity)
        return features

    def _resolve_feature_space_id(
        self,
        feature_space_id: str | None,
    ) -> str | None:
        resolved = canonical_feature_space_id(feature_space_id)
        if resolved is None:
            return self._feature_space_id
        if resolved != self._feature_space_id:
            raise ValueError(
                "feature_space_id must match the feature space used to fit the "
                "model; set feature_space_id during fit to enable this check"
            )
        return resolved

    def build(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Atomically replace the complete catalog and publish a new snapshot."""
        ids = canonical_item_ids(item_ids)
        candidate_metadata = canonical_metadata(metadata, item_ids=ids)
        features = self._prepare_features(ids, item_features)
        resolved_space = self._resolve_feature_space_id(feature_space_id)
        with self._lock:
            current = self.snapshot()
            catalog = _make_catalog(
                item_ids=ids,
                item_features=features,
                metadata=candidate_metadata,
                feature_space_id=resolved_space,
                version=current.version + 1,
            )
            self._snapshot = catalog
            self._notify(catalog)
        return catalog

    def update(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Add or update candidates and atomically publish a new snapshot."""
        if on_conflict not in {"error", "replace", "ignore"}:
            raise ValueError("on_conflict must be 'error', 'replace', or 'ignore'")
        ids = canonical_item_ids(item_ids)
        incoming_metadata = canonical_metadata(metadata, item_ids=ids)
        incoming_features = self._prepare_features(ids, item_features)
        resolved_space = self._resolve_feature_space_id(feature_space_id)
        with self._lock:
            current = self.snapshot()
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
            self._snapshot = catalog
            self._notify(catalog)
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
        # A column only the incoming frame carries is missing for every
        # pre-existing item, so it holds NA and has to accept whatever the
        # incoming values are. Reindexing alone would default it to float64,
        # which then rejects non-numeric incoming values.
        introduced = [column for column in columns if column not in old.columns]
        if introduced:
            result[introduced] = result[introduced].astype(object)
        if addition_input_rows.size:
            # Extend the index rather than concatenating an all-NA frame.
            # Concat resolves result dtypes while excluding all-NA columns,
            # which pandas warns about and will stop doing, and which silently
            # widened float and datetime metadata to object on newer pandas.
            result = result.reset_index(drop=True)
            start = len(result)
            result = result.reindex(range(start + int(addition_input_rows.size)))
            if incoming is not None:
                block = incoming.iloc[addition_input_rows]
                for column in incoming.columns:
                    # Per column, so each one promotes on its own terms.
                    result.iloc[start:, result.columns.get_loc(column)] = block[
                        column
                    ].to_numpy()
        if incoming is not None and replace_input_rows.size:
            result.loc[replace_catalog_rows, incoming.columns] = incoming.iloc[
                replace_input_rows
            ].to_numpy()
        return result.reset_index(drop=True)

    def remove(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> CandidateCatalog:
        """Remove registered candidates and publish a new snapshot."""
        if missing not in {"error", "ignore"}:
            raise ValueError("missing must be 'error' or 'ignore'")
        ids = canonical_item_ids(item_ids)
        with self._lock:
            current = self.snapshot()
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
            self._snapshot = catalog
            self._notify(catalog)
        return catalog

    def align_source(
        self,
        source: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
    ) -> csr_matrix:
        """Align external sparse columns to the fitted source vocabulary."""
        if self._source_vocabulary is None:
            raise RuntimeError(_NOT_INSTALLED)
        return self._source_vocabulary.align_csr(source, item_ids=item_ids)

    def resolve_selection(
        self,
        candidate_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> CandidateSelection:
        catalog = self.snapshot()
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
        assert self._source_item_ids is not None
        source_to_candidate = np.fromiter(
            (
                catalog.id_to_row.get(item_id, -1)
                for item_id in self._source_item_ids.tolist()
            ),
            dtype=np.int64,
            count=self._source_item_ids.size,
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

    def _save_checkpoint(
        self,
        writer: ModelCheckpointWriter,
        *,
        prefix: str = "catalog",
    ) -> None:
        """Persist the fitted source vocabulary and current published snapshot."""
        catalog = self.snapshot()
        if (
            self._source_item_ids is None
            or self._source_popularity is None
            or self._n_input_features is None
            or self._dtype is None
        ):
            raise RuntimeError(_NOT_INSTALLED)
        writer.write_item_ids(
            f"{prefix}/source_item_ids.json",
            self._source_item_ids,
        )
        writer.write_item_ids(
            f"{prefix}/candidate_item_ids.json",
            catalog.item_ids,
        )
        writer.write_numpy(
            f"{prefix}/source_popularity.npy",
            self._source_popularity,
        )
        feature_storage = writer.write_features(
            f"{prefix}/candidate_features",
            catalog.item_features,
        )
        metadata = catalog.metadata
        if metadata is not None:
            writer.write_dataframe(f"{prefix}/metadata.parquet", metadata)
        writer.write_json(
            f"{prefix}/state.json",
            {
                "feature_storage": feature_storage,
                "feature_space_id": self._feature_space_id,
                "n_input_features": self._n_input_features,
                "dtype": self._dtype.str,
                "include_popularity": self._include_popularity,
                "catalog_version": catalog.version,
                "has_metadata": metadata is not None,
                "metadata_dtypes": (
                    None
                    if metadata is None
                    else {str(column): str(dtype) for column, dtype in metadata.dtypes.items()}
                ),
            },
        )

    def _load_checkpoint(
        self,
        reader: ModelCheckpointReader,
        *,
        prefix: str = "catalog",
    ) -> CandidateCatalog:
        """Restore an exact catalog snapshot without replaying its mutations."""
        state = reader.read_json(f"{prefix}/state.json")
        dtype = np.dtype(state["dtype"])
        n_input_features = int(state["n_input_features"])
        include_popularity = bool(state["include_popularity"])
        feature_space_id = canonical_feature_space_id(
            state.get("feature_space_id")
        )
        source_ids = canonical_item_ids(
            reader.read_item_ids(f"{prefix}/source_item_ids.json"),
            name="source_item_ids",
        )
        candidate_ids = canonical_item_ids(
            reader.read_item_ids(f"{prefix}/candidate_item_ids.json"),
            name="candidate_item_ids",
        )
        popularity = np.asarray(
            reader.read_numpy(f"{prefix}/source_popularity.npy"),
            dtype=dtype,
        )
        if popularity.ndim != 1 or popularity.size != source_ids.size:
            raise ValueError(
                "catalog source_popularity must align with source_item_ids"
            )
        if not np.all(np.isfinite(popularity)):
            raise ValueError("catalog source_popularity must be finite")
        features = canonical_item_features(
            reader.read_features(
                f"{prefix}/candidate_features",
                storage=str(state["feature_storage"]),
            ),
            dtype=dtype,
        )
        if features.shape[0] != candidate_ids.size:
            raise ValueError(
                "catalog candidate features must align with candidate item IDs"
            )
        expected_features = n_input_features + int(include_popularity)
        if features.shape[1] != expected_features:
            raise ValueError(
                f"catalog has {features.shape[1]} feature columns, expected "
                f"{expected_features}"
            )
        metadata = (
            reader.read_dataframe(f"{prefix}/metadata.parquet")
            if bool(state.get("has_metadata", False))
            else None
        )
        if metadata is not None:
            if len(metadata) != candidate_ids.size:
                raise ValueError(
                    "catalog metadata must align with candidate item IDs"
                )
            metadata = metadata.reset_index(drop=True).copy(deep=True)
            dtypes = state.get("metadata_dtypes")
            if not isinstance(dtypes, dict):
                raise ValueError("catalog metadata dtype description is missing")
            for column, dtype in dtypes.items():
                if column not in metadata.columns:
                    raise ValueError(
                        f"catalog metadata is missing column {column!r}"
                    )
                try:
                    metadata[column] = metadata[column].astype(str(dtype))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"catalog metadata column {column!r} cannot restore "
                        f"dtype {dtype!r}"
                    ) from error
        version = int(state["catalog_version"])
        if version < 1:
            raise ValueError("catalog version must be >= 1")

        vocabulary = ItemVocabulary.from_ids(source_ids, name="source_item_ids")
        frozen_popularity = popularity.copy()
        frozen_popularity.setflags(write=False)
        catalog = _make_catalog(
            item_ids=candidate_ids,
            item_features=features,
            metadata=metadata,
            feature_space_id=feature_space_id,
            version=version,
        )
        with self._lock:
            self._source_vocabulary = vocabulary
            self._source_item_ids = vocabulary.item_ids
            self._source_id_to_row = vocabulary.id_to_row
            self._source_popularity = frozen_popularity
            self._feature_space_id = feature_space_id
            self._n_input_features = n_input_features
            self._dtype = dtype
            self._include_popularity = include_popularity
            self._snapshot = catalog
            self._notify(catalog)
        return catalog
