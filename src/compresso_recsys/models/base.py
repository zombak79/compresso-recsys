from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.models._validation import canonical_csr

__all__ = ["BaseCollaborativeRecommender", "Recommender"]


@runtime_checkable
class Recommender(Protocol):
    """A fitted recommender that produces ranked predictions for one batch."""

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        """Return top-``k`` predictions, optionally excluding source items."""


class BaseCollaborativeRecommender(ABC):
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
    def fit(self, interactions: csr_matrix) -> BaseCollaborativeRecommender:
        """Fit the model from a user-item CSR interaction matrix."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        """Return ranked predictions for one source batch."""

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
        show_progress: bool = False,
    ) -> SRPTensor:
        """Predict all source rows by repeatedly calling ``predict_on_batch``."""
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 1 <= int(k) <= source.shape[1]:
            raise ValueError(f"k must be in [1, {source.shape[1]}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                starts = tqdm(
                    starts,
                    desc=f"{type(self).__name__} predict@{k}",
                )
            except Exception:  # pragma: no cover - optional display helper
                pass
        for start in starts:
            result = self.predict_on_batch(
                source[start : start + batch_size],
                k=k,
                exclude_seen=exclude_seen,
            )
            if result.cols_total != source.shape[1]:
                raise ValueError(
                    "predict_on_batch() item count must match the fitted catalog"
                )
            columns.append(result.cols)
            values.append(result.vals)

        if not columns:
            return self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
            )
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=source.shape,
            validate=False,
        )
