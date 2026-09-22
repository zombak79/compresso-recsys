"""The cold-start recommender contract, and the base every one of them shares.

The catalog, the feature canonicalization and the warm-model adapter moved to
siblings; this module keeps the contract and its base implementation, and
re-exports the rest so ``models.core.cold_start`` still names the whole family.
"""

from __future__ import annotations

from abc import abstractmethod
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
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.base import (
    BaseIdentifiedRecommender,
    BasePersistableRecommender,
    Recommender,
    SequentialRecommender,
    _accepts_reporting_keywords,
)
from compresso_recsys.models.core.identifiers import (
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

# Re-exported, not used here: callers have always reached these through this
# module, and the split below it should not change that.
from compresso_recsys.models.core.features import (  # noqa: F401
    _freeze_features,
    _replace_feature_rows,
    _stack_features,
    _torch_sparse_to_csr,
    append_column,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_metadata,
    take_features,
)
from compresso_recsys.models.core.adapters import WarmCatalogAdapter  # noqa: F401



from compresso_recsys.models.core.adapters import WarmCatalogAdapter
from compresso_recsys.models.core.catalog import (
    CandidateCatalog,
    CandidateSelection,
    MutableCandidateCatalog,
    _make_catalog,
)
# Re-exported: callers have always reached these through this module.


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
