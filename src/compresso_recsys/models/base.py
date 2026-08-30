from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models._validation import canonical_csr

__all__ = [
    "BaseCollaborativeRecommender",
    "BaseSequentialRecommender",
    "Recommender",
    "SequentialRecommender",
]


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


@runtime_checkable
class SequentialRecommender(Protocol):
    """A fitted recommender that ranks from chronological histories.

    The same contract as :class:`Recommender` with a different source type. Kept
    structural, like its sibling, so a model satisfies it by having the method
    rather than by inheriting anything.
    """

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        """Return top-``k`` predictions, optionally excluding source items."""


class BaseSequentialRecommender(ABC):
    """Reusable base for recommenders that read chronological histories.

    Parallel to :class:`BaseCollaborativeRecommender` rather than derived from
    it. The two differ only in how a user's history arrives — a CSR row of items
    interacted with, or an ordered history that keeps repeats — and crossing that
    with cold-start capability in the type hierarchy would give four classes for
    two ideas. Candidate capability is composed instead: a model that scores
    unseen items owns a catalog rather than inheriting one.

    Implementors provide :attr:`is_fitted`, :attr:`n_items`, and
    :meth:`predict_on_batch`. ``fit`` is deliberately absent from the contract:
    trainers follow the package's existing shape, where
    ``SomeTrainer(config).fit(data)`` returns a fitted model and the model owes
    only the prediction contract.

    Two properties this base is careful not to assume.

    **The source vocabulary need not equal the candidate catalog.**
    :attr:`n_items` describes what can be *scored*. A history may be expressed
    over a different, usually smaller, vocabulary — a truncated context, a
    hashed one — and a cold-capable model scores candidates that never appear in
    any history at all. Nothing here compares the two.

    **Truncation is not exclusion.** ``exclude_seen=True`` must mask every item
    in the *full* history handed to it, even where the encoder reads only a
    suffix. A model that attends to the last 200 interactions must still refuse
    to recommend the 201st.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""

    @property
    @abstractmethod
    def n_items(self) -> int | None:
        """Number of scoreable candidates, or ``None`` before fitting."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        """Return ranked predictions for one batch of histories."""

    def _prepare_source(self, source: ItemSequences) -> ItemSequences:
        """Check a batch of histories against the fitted model."""
        if not self.is_fitted or self.n_items is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction"
            )
        if not isinstance(source, ItemSequences):
            raise TypeError(
                f"{type(self).__name__} predicts from ItemSequences, got "
                f"{type(source).__name__}"
            )
        return source

    @staticmethod
    def _check_unseen_capacity(
        source: ItemSequences,
        *,
        n_items: int,
        k: int,
    ) -> None:
        """Require every row to contain at least ``k`` scoreable unseen items."""
        for row in range(source.n_rows):
            history = source.row(row)
            scoreable = history[history < n_items]
            available = n_items - int(np.unique(scoreable).size)
            if available < k:
                raise ValueError(
                    f"source row {row} has only {available} unseen items, "
                    f"fewer than k={k}"
                )

    def predict(
        self,
        source: ItemSequences,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        show_progress: bool = False,
    ) -> SRPTensor:
        """Predict all histories by repeatedly calling ``predict_on_batch``."""
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 1 <= int(k) <= self.n_items:
            raise ValueError(f"k must be in [1, {self.n_items}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.n_rows, batch_size)
        if show_progress:
            try:
                from tqdm.auto import tqdm

                starts = tqdm(starts, desc=f"{type(self).__name__} predict@{k}")
            except Exception:  # pragma: no cover - optional display helper
                pass
        for start in starts:
            result = self.predict_on_batch(
                source.take_rows(start, start + batch_size),
                k=k,
                exclude_seen=exclude_seen,
            )
            if result.cols_total != self.n_items:
                raise ValueError(
                    "predict_on_batch() item count must match the candidate catalog"
                )
            columns.append(result.cols)
            values.append(result.vals)

        if not columns:
            return self.predict_on_batch(source, k=k, exclude_seen=exclude_seen)
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=(source.n_rows, self.n_items),
        )
