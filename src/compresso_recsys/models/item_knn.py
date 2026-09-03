from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

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

__all__ = ["ItemKNNConfig", "ItemKNNRecommender"]

ItemKNNDataType = Literal["float32", "float64"]


@dataclass(frozen=True)
class ItemKNNConfig:
    """Configuration for cosine item-neighborhood collaborative filtering."""

    n_neighbors: int = 100
    dtype: ItemKNNDataType = "float32"
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


class ItemKNNRecommender(BaseCollaborativeRecommender):
    """Item-item cosine KNN fitted over item interaction vectors."""

    checkpoint_type = "item_knn"

    def __init__(self, config: ItemKNNConfig | None = None) -> None:
        self.cfg = config if config is not None else ItemKNNConfig()
        self.similarity_: csr_matrix | None = None
        self.n_items_: int | None = None

    @property
    def is_fitted(self) -> bool:
        return self.similarity_ is not None

    @property
    def n_items(self) -> int | None:
        return self.n_items_

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.cfg.dtype)

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> ItemKNNRecommender:
        """Build a sparse cosine-neighbor graph over item columns."""
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        if np.any(interactions.data < 0):
            raise ValueError("interactions must contain nonnegative values")
        try:
            from sklearn.neighbors import NearestNeighbors
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "ItemKNNRecommender requires scikit-learn; install "
                "compresso-recsys[knn]"
            ) from error

        n_items = int(interactions.shape[1])
        vocabulary = self._prepare_item_vocabulary(item_ids, n_items=n_items)
        item_vectors = interactions.T.tocsr().astype(self.dtype, copy=False)
        requested = min(n_items, int(self.cfg.n_neighbors) + 1)
        index = NearestNeighbors(
            n_neighbors=requested,
            metric="cosine",
            algorithm="brute",
            n_jobs=self.cfg.n_jobs,
        ).fit(item_vectors)
        distances, neighbors = index.kneighbors(item_vectors, return_distance=True)

        rows: list[int] = []
        columns: list[int] = []
        values: list[float] = []
        for item in range(n_items):
            kept = 0
            for neighbor, distance in zip(
                neighbors[item].tolist(),
                distances[item].tolist(),
                strict=True,
            ):
                if neighbor == item:
                    continue
                similarity = max(0.0, 1.0 - float(distance))
                if similarity > 0.0:
                    rows.append(item)
                    columns.append(int(neighbor))
                    values.append(similarity)
                kept += 1
                if kept == int(self.cfg.n_neighbors):
                    break

        similarity = csr_matrix(
            (
                np.asarray(values, dtype=self.dtype),
                (np.asarray(rows, dtype=np.int64), np.asarray(columns, dtype=np.int64)),
            ),
            shape=(n_items, n_items),
            dtype=self.dtype,
        )
        self.similarity_ = similarity
        self.n_items_ = n_items
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
        assert self.similarity_ is not None
        selected_similarity = self.similarity_[candidate_rows]
        scores = np.asarray(
            (source @ selected_similarity.T).toarray(),
            dtype=self.dtype,
        )
        normalizer = np.asarray(abs(selected_similarity).sum(axis=1)).ravel()
        np.divide(
            scores,
            normalizer,
            out=scores,
            where=normalizer > 0,
        )
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
    ) -> ItemKNNRecommender:
        del reader, device
        return cls(ItemKNNConfig(**config))

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.similarity_ is not None
        writer.write_sparse("state/similarity.npz", self.similarity_)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        similarity = reader.read_sparse("state/similarity.npz")
        if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
            raise ValueError("ItemKNN similarity must be a square matrix")
        if similarity.shape[0] < 1:
            raise ValueError("ItemKNN similarity must contain at least one item")
        if similarity.dtype != self.dtype:
            raise ValueError(
                f"ItemKNN similarity uses {similarity.dtype}, expected {self.dtype}"
            )
        if not np.isfinite(similarity.data).all() or np.any(similarity.data < 0):
            raise ValueError(
                "ItemKNN similarity must contain finite nonnegative values"
            )
        self.similarity_ = similarity
        self.n_items_ = int(similarity.shape[0])
