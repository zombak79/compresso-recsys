from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack, issparse, isspmatrix_csr, vstack

from compresso import SRPTensor
from compresso_recsys._reporting import _INHERIT, _Inherit, _Reporter, _format_duration
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.base import (
    BaseIdentifiedRecommender,
    BasePersistableRecommender,
    Recommender,
    SequentialRecommender,
    _accepts_reporting_keywords,
)
from compresso_recsys.models.identifiers import (
    ItemVocabulary,
    canonical_item_ids,
)
from compresso_recsys.sequences import ItemSequences

__all__ = [
    "BaseColdStartRecommender",
    "CandidateCatalog",
    "ColdStartRecommender",
    "ItemVocabulary",
    "MutableCandidateCatalog",
    "WarmCatalogAdapter",
]

ItemFeatures = csr_matrix | SRPTensor | np.ndarray | torch.Tensor
CandidateConflict = Literal["error", "replace", "ignore"]

_NOT_INSTALLED = (
    "no candidate catalog is installed: the model has not been fitted, or "
    "install() was never called on the catalog"
)


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


class WarmCatalogAdapter(BaseIdentifiedRecommender):
    """Expose a fixed-catalog recommender in a larger identified catalog.

    The wrapped model continues to consume and rank only its training items.
    :meth:`align_source` expresses a source over the expanded catalog in the
    fitted item space, while :meth:`predict_on_batch` remaps the resulting ranked
    columns back into that catalog. Cold candidates remain valid target items but
    can never be emitted by the wrapped model.

    Stage catalogs must follow the checkpoint invariant: the training IDs are an
    exact ordered prefix and cold items are appended. A ``csr_matrix`` is projected
    to that fitted prefix. An :class:`~compresso_recsys.ItemSequences` is passed
    through whole, because its warm indices already mean the same thing and the
    wrapped model's tokenizer turns appended cold indices into ``unk`` without
    deleting positions. Rows survive either way, so alignment with the targets is
    preserved.

    This is mandatory whenever the model's item space is narrower than the
    evaluation catalog, which the ``temporal`` split mode guarantees by
    construction. It is also worth reaching for under ``leave_last_out``, where
    the catalogs do match but items whose every occurrence falls in a held-out
    tail are still absent from training -- and the model families do not treat
    such columns alike. A softmax next-item objective pushes every non-target
    logit down on every step, and a never-trained item is never a target, so it
    is buried: on MovieLens-1M such items land at the 95th rank percentile for
    :class:`~compresso_recsys.models.SimpleRNNTrainer` against the 60th for
    :class:`~compresso_recsys.models.ELSATrainer`, which leaves them near their
    initialization. Neither number is about recommendation quality, so a
    comparison spanning both families is sounder with the cold items made
    unreachable for each. Whether it matters is a question about the data rather
    than the protocol: count the evaluation rows whose target is absent from
    training before deciding.

    Parameters
    ----------
    model:
        Fitted recommender whose prediction columns follow ``train_item_ids``.
    train_item_ids:
        Item IDs in the exact column order used to fit ``model``.
    catalog_item_ids:
        Expanded source and target catalog. ``train_item_ids`` must be its exact
        ordered prefix; additional cold items are appended after it.
    """

    def __init__(
        self,
        model: Recommender | SequentialRecommender,
        train_item_ids: Sequence[Hashable] | np.ndarray,
        catalog_item_ids: Sequence[Hashable] | np.ndarray,
    ) -> None:
        if not isinstance(model, Recommender):
            raise TypeError("model must implement predict_on_batch(source, *, k)")
        train_vocabulary = ItemVocabulary.from_ids(
            train_item_ids,
            name="train_item_ids",
        )
        catalog_vocabulary = ItemVocabulary.from_ids(
            catalog_item_ids,
            name="catalog_item_ids",
        )
        missing = [
            item_id
            for item_id in train_vocabulary.item_ids.tolist()
            if item_id not in catalog_vocabulary.id_to_row
        ]
        if missing:
            raise ValueError(
                "catalog_item_ids is missing training item ID: "
                f"{missing[0]!r}"
            )
        if not np.array_equal(
            train_vocabulary.item_ids,
            catalog_vocabulary.item_ids[: train_vocabulary.n_items],
        ):
            raise ValueError(
                "train_item_ids must be an exact ordered prefix of "
                "catalog_item_ids; checkpoint stage catalogs may only grow by "
                "appending cold items"
            )

        train_to_catalog = np.fromiter(
            (
                catalog_vocabulary.id_to_row[item_id]
                for item_id in train_vocabulary.item_ids.tolist()
            ),
            dtype=np.int64,
            count=train_vocabulary.n_items,
        )
        train_to_catalog.setflags(write=False)

        self.model = model
        self._train_vocabulary = train_vocabulary
        self._catalog_vocabulary = catalog_vocabulary
        self.train_item_ids = train_vocabulary.item_ids
        self.catalog_item_ids = catalog_vocabulary.item_ids
        self.train_to_catalog = train_to_catalog
        self.catalog_size = catalog_vocabulary.n_items
        self._identity_alignment = np.array_equal(
            self.train_item_ids,
            self.catalog_item_ids,
        )
        self._mapping_lock = RLock()
        self._mapping_by_device: dict[torch.device, torch.Tensor] = {}

        if isinstance(model, BaseIdentifiedRecommender) and not np.array_equal(
            model.source_item_ids,
            self.train_item_ids,
        ):
            raise ValueError(
                "model source_item_ids must match train_item_ids in row order"
            )

    @property
    def source_item_ids(self) -> np.ndarray:
        """Stable IDs accepted from the expanded stage catalog."""
        return self.catalog_item_ids

    @property
    def candidate_item_ids(self) -> np.ndarray:
        """Stable IDs in the expanded output catalog."""
        return self.catalog_item_ids

    def _recommend_vocabularies(
        self,
    ) -> tuple[ItemVocabulary, ItemVocabulary]:
        return self._catalog_vocabulary, self._catalog_vocabulary

    def _prediction_reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        if isinstance(self.model, BaseIdentifiedRecommender):
            return self.model._prediction_reporter(logger, show_progress)
        return super()._prediction_reporter(logger, show_progress)

    def _scoreable_candidate_rows(
        self,
        vocabulary: ItemVocabulary,
    ) -> np.ndarray:
        del vocabulary
        return self.train_to_catalog

    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> csr_matrix | ItemSequences:
        if not isinstance(self.model, BaseIdentifiedRecommender):
            raise TypeError(
                "recommend() requires the wrapped model to inherit an "
                "identified recommender base"
            )
        return self.model._recommendation_source(rows, vocabulary=vocabulary)

    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        if not isinstance(self.model, BaseIdentifiedRecommender):
            raise TypeError(
                "recommend() requires the wrapped model to inherit an "
                "identified recommender base"
            )
        if isinstance(source, csr_matrix):
            source = self.align_source(source)
        predictions = self.model._predict_identified(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )
        n_rows = source.shape[0] if isinstance(source, csr_matrix) else source.n_rows
        return self._remap_predictions(predictions, n_rows=n_rows)

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        if not isinstance(self.model, BaseIdentifiedRecommender):
            raise TypeError(
                "recommend() requires the wrapped model to inherit an "
                "identified recommender base"
            )
        if isinstance(source, csr_matrix):
            source = self.align_source(source)
        predictions = self.model._predict_identified_with_reporting(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            reporter=reporter,
        )
        n_rows = source.shape[0] if isinstance(source, csr_matrix) else source.n_rows
        return self._remap_predictions(predictions, n_rows=n_rows)

    def align_source(self, source: csr_matrix) -> csr_matrix:
        """Select the fitted training-item columns from an expanded-catalog matrix.

        Matrices only. A history needs no alignment: a sequential model's
        tokenizer maps an out-of-catalog index to its own ``unk`` token, keeping
        the position, and *projecting* one instead would delete interior events
        and thereby assert transitions that never happened. Pass sequences
        straight to :meth:`predict_on_batch`.
        """
        if isinstance(source, ItemSequences):
            raise TypeError(
                "sequences need no alignment: a model's tokenizer turns an "
                "out-of-catalog index into its own 'unk' token, in place. "
                "Dropping those items instead would join their neighbours as if "
                "they had been consecutive. Pass the sequences to "
                "predict_on_batch() directly"
            )
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.catalog_size:
            raise ValueError(
                f"source has {source.shape[1]} items, but catalog_item_ids has "
                f"{self.catalog_size} entries"
            )
        if self._identity_alignment:
            return source
        return source[:, self.train_to_catalog].tocsr()

    def _mapping_on(self, device: torch.device) -> torch.Tensor:
        with self._mapping_lock:
            mapping = self._mapping_by_device.get(device)
            if mapping is None:
                mapping = torch.tensor(
                    self.train_to_catalog,
                    dtype=torch.long,
                    device=device,
                )
                self._mapping_by_device[device] = mapping
            return mapping

    def _remap_predictions(
        self,
        predictions: SRPTensor,
        *,
        n_rows: int,
    ) -> SRPTensor:
        """Express wrapped-model prediction columns in the expanded catalog."""
        if not isinstance(predictions, SRPTensor):
            raise TypeError("model prediction must be an SRPTensor")
        if predictions.rows != n_rows:
            raise ValueError("model prediction rows must match the source rows")
        if predictions.cols_total != len(self.train_item_ids):
            raise ValueError(
                "model prediction items must match train_item_ids: expected "
                f"{len(self.train_item_ids)}, got {predictions.cols_total}"
            )

        mapping = self._mapping_on(predictions.cols.device)
        return SRPTensor(
            cols=mapping[predictions.cols],
            vals=predictions.vals,
            shape=(predictions.rows, self.catalog_size),
            validate=False,
        )

    def predict_on_batch(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict warm items and express their columns in the full catalog."""
        if isinstance(source, ItemSequences):
            # Unaligned by design: the model's tokenizer decides what an
            # out-of-catalog index becomes, so a history arrives in catalog
            # space and only the prediction columns need widening.
            n_rows, n_items = source.n_rows, source.n_items
            if n_items != self.catalog_size:
                raise ValueError(
                    f"source spans {n_items} items, but catalog_item_ids has "
                    f"{self.catalog_size} entries"
                )
        else:
            source = canonical_csr(source, name="source")
            n_rows, n_items = source.shape
            if n_items != len(self.train_item_ids):
                raise ValueError(
                    f"source has {n_items} items, but train_item_ids has "
                    f"{len(self.train_item_ids)} entries; call align_source() first"
                )
        if candidate_ids is not None:
            self._train_vocabulary.rows_for(
                candidate_ids,
                name="candidate_ids",
            )
        kwargs = (
            {}
            if candidate_ids is None
            else {"candidate_ids": candidate_ids}
        )
        predictions = self.model.predict_on_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            **kwargs,
        )
        return self._remap_predictions(predictions, n_rows=n_rows)


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


@runtime_checkable
class ColdStartRecommender(Recommender, Protocol):
    """Recommender with distinct identified source and candidate spaces.

    The source vocabulary is no longer a member here: it lives on the catalog
    the model owns, reachable as ``model.candidates.source_vocabulary``.
    """

    @property
    def candidates(self) -> MutableCandidateCatalog: ...

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


class BaseColdStartRecommender(BasePersistableRecommender):
    """Reusable base for feature-driven cold-start recommenders that read a matrix.

    Subclasses implement :meth:`fit`, :attr:`is_fitted`, and
    :meth:`predict_on_batch`. The catalog lifecycle is *owned* rather than
    inherited: :attr:`candidates` is a :class:`MutableCandidateCatalog` holding
    the fitted source vocabulary, the current snapshot and the operations over
    them. The methods below are a facade over it, kept because they are the
    documented model surface.

    That composition is why this class is only about reading a ``csr_matrix``
    source. A cold-capable model that reads ordered histories owns the same
    catalog from :class:`~compresso_recsys.models.BaseSequentialRecommender`
    instead, rather than needing a fourth base class or multiple inheritance.

    Subclass constructors must call ``super().__init__()``. During fitting, call
    ``self.candidates.install(...)`` after learning the source encoder to publish
    the initial catalog.
    """

    def __init__(self) -> None:
        # The hook is passed in rather than discovered, so the catalog notifies
        # its owner without knowing what an owner is.
        self.candidates = MutableCandidateCatalog(
            on_publish=self._on_catalog_published
        )

    @property
    def source_item_ids(self) -> np.ndarray:
        """Stable IDs accepted in recommendation histories."""
        item_ids = self.candidates.source_item_ids
        if item_ids is None:
            raise RuntimeError(_NOT_INSTALLED)
        return item_ids

    @property
    def candidate_item_ids(self) -> np.ndarray:
        """Stable IDs in the current candidate snapshot."""
        return self.candidates.snapshot().item_ids

    def _recommend_vocabularies(
        self,
    ) -> tuple[ItemVocabulary, ItemVocabulary]:
        source = self.candidates.source_vocabulary
        if source is None:
            raise RuntimeError(_NOT_INSTALLED)
        candidate = ItemVocabulary.from_ids(
            self.candidates.snapshot().item_ids,
            name="candidate_item_ids",
        )
        return source, candidate

    def _restore_source_item_ids(self, item_ids: np.ndarray) -> None:
        source = self.candidates.source_item_ids
        if source is None or not np.array_equal(source, item_ids):
            raise ValueError(
                "checkpoint identity does not match the cold-start source catalog"
            )

    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> csr_matrix:
        lengths = np.fromiter((row.size for row in rows), dtype=np.int64)
        row_indices = np.repeat(np.arange(len(rows), dtype=np.int64), lengths)
        columns = (
            np.concatenate(rows)
            if rows
            else np.empty(0, dtype=np.int64)
        )
        source = csr_matrix(
            (
                np.ones(columns.size, dtype=np.float32),
                (row_indices, columns),
            ),
            shape=(len(rows), vocabulary.n_items),
        )
        source.sum_duplicates()
        source.data.fill(1.0)
        return source

    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        if not isinstance(source, csr_matrix):
            raise TypeError("cold-start recommendations require a CSR source")
        return self.predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        if not isinstance(source, csr_matrix):
            raise TypeError("cold-start recommendations require a CSR source")
        predict = self.predict
        if not _accepts_reporting_keywords(predict):
            predict = BaseColdStartRecommender.predict.__get__(self)
        return predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            logger=reporter,
            show_progress=_INHERIT,
        )

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""

    @abstractmethod
    def fit(
        self,
        interactions: csr_matrix,
        item_features: ItemFeatures,
        **kwargs,
    ) -> BaseColdStartRecommender:
        """Fit a source encoder and publish the initial candidate catalog."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return ranked predictions against the current candidate catalog."""

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        """Validate source columns against the fitted source vocabulary."""
        vocabulary = self.candidates.source_vocabulary
        if not self.is_fitted or vocabulary is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction"
            )
        source = canonical_csr(source, name="source")
        if source.shape[1] != vocabulary.n_items:
            raise ValueError(
                f"source has {source.shape[1]} items, but "
                f"{type(self).__name__} was fitted with "
                f"{vocabulary.n_items} source items"
            )
        return source

    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SRPTensor:
        """Predict all source rows by repeatedly calling ``predict_on_batch``."""
        reporter = self._prediction_reporter(logger, show_progress)
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        catalog = self.candidates.snapshot()
        selected_items = (
            catalog.n_items
            if candidate_ids is None
            else catalog.rows_for(candidate_ids).size
        )
        if not 1 <= int(k) <= selected_items:
            raise ValueError(f"k must be in [1, {selected_items}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.shape[0]} rows | "
            f"{steps} batches of {batch_size}"
        )
        for step, start in enumerate(
            reporter.wrap(
                starts,
                total=steps,
                desc=f"{type(self).__name__} predict@{k}",
            ),
            start=1,
        ):
            result = self.predict_on_batch(
                source[start : start + batch_size],
                k=k,
                exclude_seen=exclude_seen,
                candidate_ids=candidate_ids,
            )
            if result.cols_total != catalog.n_items:
                raise ValueError(
                    "predict_on_batch() item count must match the candidate catalog"
                )
            columns.append(result.cols)
            values.append(result.vals)
            log_steps = reporter.log_every_n_steps
            if log_steps and step % log_steps == 0:
                reporter.step(
                    f"predict@{k} step {step}/{steps}",
                    step,
                    steps,
                    started,
                )

        if not columns:
            prediction = self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                candidate_ids=candidate_ids,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=(source.shape[0], catalog.n_items),
                validate=False,
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.shape[0]} rows"
        )
        return prediction

    def _on_catalog_published(self, catalog: CandidateCatalog) -> None:
        """Called with each new snapshot, for dropping caches derived from it."""

    def build_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Atomically replace the complete candidate catalog."""
        return self.candidates.build(
            item_ids=item_ids,
            item_features=item_features,
            metadata=metadata,
            feature_space_id=feature_space_id,
        )

    def update_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Add or update candidates and atomically publish a new snapshot."""
        return self.candidates.update(
            item_ids=item_ids,
            item_features=item_features,
            metadata=metadata,
            on_conflict=on_conflict,
            feature_space_id=feature_space_id,
        )

    def remove_candidates(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> CandidateCatalog:
        """Remove registered candidates and publish a new snapshot."""
        return self.candidates.remove(item_ids, missing=missing)

    def align_source(
        self,
        source: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
    ) -> csr_matrix:
        """Align external sparse columns to the fitted source vocabulary."""
        return self.candidates.align_source(source, item_ids=item_ids)


class _LinearFeatureRecommenderMixin(BaseColdStartRecommender):
    """Shared prediction path for linear fixed-feature cold-start models."""

    _model_name = "model"

    def _on_catalog_published(self, catalog: CandidateCatalog) -> None:
        self.decoder_features_ = catalog.item_features

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if (
            not self.is_fitted
            or self.n_items_ is None
            or self.train_item_indices_ is None
            or self.train_item_mask_ is None
        ):
            raise RuntimeError(
                f"{self._model_name} must be fitted before prediction"
            )
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items_:
            raise ValueError(
                f"source has {source.shape[1]} items, but {self._model_name} "
                f"was fitted with {self.n_items_} items"
            )
        if source.data.size and not np.all(source.data == 1):
            raise ValueError("source must contain binary implicit values equal to 1")

        cold_positions = np.flatnonzero(~self.train_item_mask_[source.indices])
        if cold_positions.size:
            cold_item = int(source.indices[cold_positions[0]])
            raise ValueError(
                f"source contains item {cold_item}, which has no fitted encoder row"
            )
        return source

    def user_profiles(self, source: csr_matrix) -> np.ndarray:
        """Transform binary source histories into item-feature profiles."""
        source = self._prepare_source(source)
        return self._profiles_from_prepared_source(source)

    def _profiles_from_prepared_source(self, source: csr_matrix) -> np.ndarray:
        assert self.encoder_ is not None
        assert self.train_item_indices_ is not None
        return np.asarray(
            source[:, self.train_item_indices_] @ self.encoder_,
            dtype=self.dtype,
        )

    def _score_profiles(
        self,
        profiles: np.ndarray,
        *,
        candidate_features: csr_matrix | np.ndarray,
    ) -> np.ndarray:
        if isspmatrix_csr(candidate_features):
            scores = (candidate_features @ profiles.T).T
        else:
            scores = profiles @ candidate_features.T
        return np.asarray(scores, dtype=self.dtype)

    def _predict_prepared_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        catalog: CandidateCatalog,
        candidate_rows: np.ndarray,
        candidate_features: csr_matrix | np.ndarray,
        source_to_candidate_rows: np.ndarray,
        candidate_to_local: np.ndarray,
    ) -> SRPTensor:
        if not 1 <= int(k) <= candidate_rows.size:
            raise ValueError(f"k must be in [1, {candidate_rows.size}], got {k}")

        seen_counts = np.diff(source.indptr)
        source_rows = np.repeat(
            np.arange(source.shape[0], dtype=np.int64),
            seen_counts,
        )
        seen_candidate_rows = source_to_candidate_rows[source.indices]
        registered = seen_candidate_rows >= 0
        seen_local_rows = np.full(seen_candidate_rows.shape, -1, dtype=np.int64)
        seen_local_rows[registered] = candidate_to_local[
            seen_candidate_rows[registered]
        ]
        selected_seen = seen_local_rows >= 0

        if exclude_seen:
            selected_seen_counts = np.bincount(
                source_rows[selected_seen],
                minlength=source.shape[0],
            )
            available_counts = candidate_rows.size - selected_seen_counts
            if available_counts.size and np.any(available_counts < k):
                row = int(np.flatnonzero(available_counts < k)[0])
                raise ValueError(
                    f"source row {row} has only {available_counts[row]} unseen "
                    f"items among the selected candidates, fewer than k={k}"
                )

        if source.shape[0] == 0:
            value_dtype = torch.from_numpy(np.empty(0, dtype=self.dtype)).dtype
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long),
                vals=torch.empty((0, k), dtype=value_dtype),
                shape=(0, catalog.n_items),
            )

        scores = self._score_profiles(
            self._profiles_from_prepared_source(source),
            candidate_features=candidate_features,
        )
        if exclude_seen and bool(selected_seen.any()):
            scores[source_rows[selected_seen], seen_local_rows[selected_seen]] = -np.inf
        local_predictions = SRPTensor.from_dense(
            torch.from_numpy(scores),
            k=int(k),
            score_mode="raw",
        )
        global_columns = torch.from_numpy(candidate_rows).to(
            local_predictions.cols.device
        )[local_predictions.cols]
        return SRPTensor(
            cols=global_columns,
            vals=local_predictions.vals,
            shape=(source.shape[0], catalog.n_items),
        )

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for one source batch."""
        source = self._prepare_source(source)
        selection = self.candidates.resolve_selection(candidate_ids)
        return self._predict_prepared_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            catalog=selection.catalog,
            candidate_rows=selection.rows,
            candidate_features=selection.features,
            source_to_candidate_rows=selection.source_to_candidate,
            candidate_to_local=selection.candidate_to_local,
        )

    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for all source rows in batches."""
        reporter = self._prediction_reporter(logger, show_progress)
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        selection = self.candidates.resolve_selection(candidate_ids)
        if not 1 <= int(k) <= selection.rows.size:
            raise ValueError(f"k must be in [1, {selection.rows.size}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.shape[0]} rows | "
            f"{steps} batches of {batch_size}"
        )
        for step, start in enumerate(
            reporter.wrap(
                starts,
                total=steps,
                desc=f"{self._model_name} predict@{k}",
            ),
            start=1,
        ):
            end = min(start + batch_size, source.shape[0])
            predictions = self._predict_prepared_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
                catalog=selection.catalog,
                candidate_rows=selection.rows,
                candidate_features=selection.features,
                source_to_candidate_rows=selection.source_to_candidate,
                candidate_to_local=selection.candidate_to_local,
            )
            columns.append(predictions.cols)
            values.append(predictions.vals)
            log_steps = reporter.log_every_n_steps
            if log_steps and step % log_steps == 0:
                reporter.step(
                    f"predict@{k} step {step}/{steps}",
                    step,
                    steps,
                    started,
                )

        if not columns:
            prediction = self._predict_prepared_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                catalog=selection.catalog,
                candidate_rows=selection.rows,
                candidate_features=selection.features,
                source_to_candidate_rows=selection.source_to_candidate,
                candidate_to_local=selection.candidate_to_local,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=(source.shape[0], selection.catalog.n_items),
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.shape[0]} rows"
        )
        return prediction
