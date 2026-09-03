from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.models._ranking import (
    mask_seen_numpy,
    rank_numpy_scores,
    validate_candidate_topk,
)
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.base import BaseCollaborativeRecommender
from compresso_recsys.models.identifiers import ItemVocabulary
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = [
    "PopularityBaseline",
    "PopularityBaselineConfig",
    "RandomBaseline",
    "RandomBaselineConfig",
]

_UINT64_MASK = (1 << 64) - 1


@dataclass(frozen=True)
class RandomBaselineConfig:
    """Configuration for :class:`RandomBaseline`.

    ``seed`` determines a stable pseudorandom ranking for each distinct source
    history. Predictions are invariant to evaluation batch size and checkpoint
    round trips.
    """

    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")


@dataclass(frozen=True)
class PopularityBaselineConfig:
    """Configuration for :class:`PopularityBaseline`.

    When ``use_values`` is false, popularity counts users with a nonzero
    interaction. When true, it sums the interaction values instead.
    """

    use_values: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.use_values, (bool, np.bool_)):
            raise TypeError("use_values must be a boolean")


class _FixedCatalogBaseline(BaseCollaborativeRecommender):
    n_items_: int | None

    @property
    def is_fitted(self) -> bool:
        return self.n_items_ is not None

    @property
    def n_items(self) -> int | None:
        return self.n_items_

    def _prepare_catalog(
        self,
        interactions: csr_matrix,
        item_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> tuple[csr_matrix, ItemVocabulary]:
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        vocabulary = self._prepare_item_vocabulary(
            item_ids,
            n_items=int(interactions.shape[1]),
        )
        return interactions, vocabulary

    def _checkpoint_state(self, reader: ModelCheckpointReader) -> int:
        state = reader.read_json("state/baseline.json")
        n_items = state.get("n_items")
        if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items < 1:
            raise ValueError("baseline n_items must be a positive integer")
        return n_items


class RandomBaseline(_FixedCatalogBaseline):
    """Deterministic random-ranking baseline for a fixed item catalog."""

    checkpoint_type = "random_baseline"

    def __init__(self, config: RandomBaselineConfig | None = None) -> None:
        self.cfg = config if config is not None else RandomBaselineConfig()
        self.n_items_: int | None = None

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> RandomBaseline:
        """Record the fitted catalog used by the random baseline."""
        interactions, vocabulary = self._prepare_catalog(interactions, item_ids)
        self.n_items_ = int(interactions.shape[1])
        self._publish_item_vocabulary(vocabulary)
        return self

    @staticmethod
    def _history_key(indices: np.ndarray, seed: int) -> int:
        value = (int(seed) ^ 0xCBF29CE484222325) & _UINT64_MASK
        for index in indices.tolist():
            value ^= (int(index) + 1) & _UINT64_MASK
            value = (value * 0x100000001B3) & _UINT64_MASK
        return value

    @staticmethod
    def _random_scores(key: int, candidate_rows: np.ndarray) -> np.ndarray:
        values = np.asarray(candidate_rows, dtype=np.uint64)
        values = values + np.uint64(key)
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
        values ^= values >> np.uint64(31)
        return (values >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        source = self._prepare_source(source)
        candidate_rows = self._candidate_rows(candidate_ids)
        validate_candidate_topk(
            source,
            candidate_rows,
            k=k,
            exclude_seen=exclude_seen,
        )
        scores = np.empty((source.shape[0], candidate_rows.size), dtype=np.float64)
        for row in range(source.shape[0]):
            key = self._history_key(
                source.indices[source.indptr[row] : source.indptr[row + 1]],
                int(self.cfg.seed),
            )
            scores[row] = self._random_scores(key, candidate_rows)
        if exclude_seen:
            mask_seen_numpy(scores, source, candidate_rows)
        return rank_numpy_scores(
            scores,
            candidate_rows=candidate_rows,
            shape=source.shape,
            k=k,
        )

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> RandomBaseline:
        del device
        model = cls(RandomBaselineConfig(**config))
        model.n_items_ = model._checkpoint_state(reader)
        return model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.n_items_ is not None
        writer.write_json("state/baseline.json", {"n_items": self.n_items_})


class PopularityBaseline(_FixedCatalogBaseline):
    """Non-personalized baseline ranking items by training popularity."""

    checkpoint_type = "popularity_baseline"

    def __init__(self, config: PopularityBaselineConfig | None = None) -> None:
        self.cfg = config if config is not None else PopularityBaselineConfig()
        self.popularity_: np.ndarray | None = None
        self.n_items_: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.popularity_ is not None

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> PopularityBaseline:
        """Count item popularity in the fitted interaction matrix."""
        interactions, vocabulary = self._prepare_catalog(interactions, item_ids)
        if np.any(interactions.data < 0):
            raise ValueError("interactions must contain nonnegative values")
        if self.cfg.use_values:
            popularity = np.asarray(interactions.sum(axis=0)).ravel()
        else:
            popularity = np.asarray(interactions.getnnz(axis=0))
        popularity = popularity.astype(np.float64, copy=False)
        self.popularity_ = popularity
        self.n_items_ = int(interactions.shape[1])
        self._publish_item_vocabulary(vocabulary)
        return self

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        source = self._prepare_source(source)
        candidate_rows = self._candidate_rows(candidate_ids)
        validate_candidate_topk(
            source,
            candidate_rows,
            k=k,
            exclude_seen=exclude_seen,
        )
        assert self.popularity_ is not None
        scores = np.broadcast_to(
            self.popularity_[candidate_rows],
            (source.shape[0], candidate_rows.size),
        ).copy()
        if exclude_seen:
            mask_seen_numpy(scores, source, candidate_rows)
        return rank_numpy_scores(
            scores,
            candidate_rows=candidate_rows,
            shape=source.shape,
            k=k,
        )

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> PopularityBaseline:
        del device
        return cls(PopularityBaselineConfig(**config))

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.popularity_ is not None and self.n_items_ is not None
        writer.write_json("state/baseline.json", {"n_items": self.n_items_})
        writer.write_numpy("state/popularity.npy", self.popularity_)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        n_items = self._checkpoint_state(reader)
        popularity = reader.read_numpy("state/popularity.npy")
        if popularity.shape != (n_items,) or popularity.dtype != np.float64:
            raise ValueError(
                "popularity state must be a float64 vector with n_items entries"
            )
        if not np.isfinite(popularity).all() or np.any(popularity < 0):
            raise ValueError("popularity state must contain finite nonnegative values")
        self.n_items_ = n_items
        self.popularity_ = popularity
