"""The base for models scoring a user-item interaction matrix."""

from __future__ import annotations


from abc import ABC, abstractmethod
import time
from typing import (
    Any,
    ClassVar,
    Hashable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.core.identifiers import ItemVocabulary, Recommendations
from compresso_recsys.models.base.identified import _accepts_reporting_keywords
from compresso_recsys.models.base.persistable import BasePersistableRecommender


class BaseCollaborativeRecommender(BasePersistableRecommender):
    """Reusable base for fixed-catalog collaborative recommenders.

    Implementors provide :meth:`fit`, :attr:`is_fitted`, :attr:`n_items`, and
    :meth:`predict_on_batch`. The base validates source matrices and supplies a
    memory-bounded :meth:`predict` implementation that concatenates ranked
    batches without materializing a complete score matrix.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""

    @property
    @abstractmethod
    def n_items(self) -> int | None:
        """Number of fitted item columns, or ``None`` before fitting."""

    @abstractmethod
    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> BaseCollaborativeRecommender:
        """Fit the model from a user-item CSR interaction matrix."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return ranked predictions for one source batch."""

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
            raise TypeError("collaborative recommendations require a CSR source")
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
            raise TypeError("collaborative recommendations require a CSR source")
        predict = self.predict
        if not _accepts_reporting_keywords(predict):
            # Fall back to base batching for extensions overriding the released
            # predict() signature without a logger keyword.
            predict = BaseCollaborativeRecommender.predict.__get__(self)
        return predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            logger=reporter,
            show_progress=_INHERIT,
        )

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        """Validate a source matrix against the fitted item catalog."""
        if not self.is_fitted or self.n_items is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction"
            )
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items:
            raise ValueError(
                f"source has {source.shape[1]} items, but "
                f"{type(self).__name__} was fitted with {self.n_items} items"
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
        candidate_count = int(self._candidate_rows(candidate_ids).size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

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
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            result = self.predict_on_batch(
                source[start : start + batch_size],
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
            if result.cols_total != source.shape[1]:
                raise ValueError(
                    "predict_on_batch() item count must match the fitted catalog"
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
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            prediction = self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=source.shape,
                validate=False,
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.shape[0]} rows"
        )
        return prediction
