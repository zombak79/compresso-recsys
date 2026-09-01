from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Literal, Sequence

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
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = ["UserKNNConfig", "UserKNNRecommender"]

UserKNNDataType = Literal["float32", "float64"]


@dataclass(frozen=True)
class UserKNNConfig:
    """Configuration for cosine user-neighborhood collaborative filtering."""

    n_neighbors: int = 100
    dtype: UserKNNDataType = "float32"
    n_jobs: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.n_neighbors, (bool, np.bool_)) or not isinstance(
            self.n_neighbors, (int, np.integer)
        ):
            raise TypeError("n_neighbors must be an integer")
        if self.n_neighbors < 1:
            raise ValueError("n_neighbors must be >= 1")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        if self.n_jobs is not None and (
            isinstance(self.n_jobs, (bool, np.bool_))
            or not isinstance(self.n_jobs, (int, np.integer))
            or self.n_jobs == 0
        ):
            raise ValueError("n_jobs must be None or a nonzero integer")


class UserKNNRecommender(BaseCollaborativeRecommender):
    """User-user cosine KNN using fitted users as the neighbor population."""

    checkpoint_type = "user_knn"

    def __init__(self, config: UserKNNConfig | None = None) -> None:
        self.cfg = config if config is not None else UserKNNConfig()
        self.training_interactions_: csr_matrix | None = None
        self.n_items_: int | None = None
        self._index: Any | None = None

    @property
    def is_fitted(self) -> bool:
        return self.training_interactions_ is not None and self._index is not None

    @property
    def n_items(self) -> int | None:
        return self.n_items_

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.cfg.dtype)

    @staticmethod
    def _nearest_neighbors_class():
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "UserKNNRecommender requires scikit-learn; install "
                "compresso-recsys[knn]"
            ) from error
        return NearestNeighbors

    def _build_index(self) -> None:
        if self.training_interactions_ is None:
            raise RuntimeError("UserKNN training interactions are unavailable")
        nearest_neighbors = self._nearest_neighbors_class()
        self._index = nearest_neighbors(
            metric="cosine",
            algorithm="brute",
            n_jobs=self.cfg.n_jobs,
        ).fit(self.training_interactions_)

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> UserKNNRecommender:
        """Store fitted users and build the transient cosine-neighbor index."""
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        if np.any(interactions.data < 0):
            raise ValueError("interactions must contain nonnegative values")
        self.training_interactions_ = interactions.astype(self.dtype, copy=True)
        self.n_items_ = int(interactions.shape[1])
        self._set_item_ids(item_ids, n_items=self.n_items_)
        self._build_index()
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
        assert self.training_interactions_ is not None and self._index is not None
        neighbor_count = min(
            int(self.cfg.n_neighbors),
            int(self.training_interactions_.shape[0]),
        )
        scores = np.zeros(
            (source.shape[0], candidate_rows.size),
            dtype=self.dtype,
        )
        if source.shape[0]:
            distances, neighbors = self._index.kneighbors(
                source.astype(self.dtype, copy=False),
                n_neighbors=neighbor_count,
                return_distance=True,
            )
            similarities = np.maximum(0.0, 1.0 - distances)
            for row in range(source.shape[0]):
                weights = similarities[row]
                normalizer = float(np.abs(weights).sum())
                if normalizer == 0.0:
                    continue
                neighbor_values = self.training_interactions_[
                    neighbors[row]
                ][:, candidate_rows]
                scores[row] = np.asarray(
                    weights @ neighbor_values,
                    dtype=self.dtype,
                ).ravel() / normalizer
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
    ) -> UserKNNRecommender:
        del reader, device
        return cls(UserKNNConfig(**config))

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.training_interactions_ is not None
        writer.write_sparse(
            "state/training_interactions.npz",
            self.training_interactions_,
        )

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        interactions = reader.read_sparse("state/training_interactions.npz")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "UserKNN training interactions must contain users and items"
            )
        if interactions.dtype != self.dtype:
            raise ValueError(
                "UserKNN training interactions use "
                f"{interactions.dtype}, expected {self.dtype}"
            )
        if not np.isfinite(interactions.data).all() or np.any(interactions.data < 0):
            raise ValueError(
                "UserKNN training interactions must contain finite "
                "nonnegative values"
            )
        self.training_interactions_ = interactions
        self.n_items_ = int(interactions.shape[1])

    def _finish_checkpoint_load(self) -> None:
        self._build_index()
