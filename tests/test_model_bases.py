from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.models import (
    BaseColdStartRecommender,
    BaseCollaborativeRecommender,
    EASE,
    ELSATrainer,
    TEASER,
    TEASERGDTrainer,
)


class _PopularityModel(BaseCollaborativeRecommender):
    def __init__(self) -> None:
        self.popularity_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.popularity_ is not None

    @property
    def n_items(self) -> int | None:
        return None if self.popularity_ is None else int(self.popularity_.size)

    def fit(self, interactions: csr_matrix) -> _PopularityModel:
        interactions = interactions.tocsr()
        self.popularity_ = np.asarray(interactions.sum(axis=0)).ravel().astype(
            np.float32
        )
        return self

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        source = self._prepare_source(source)
        assert self.popularity_ is not None
        scores = np.broadcast_to(
            self.popularity_,
            source.shape,
        ).copy()
        if exclude_seen:
            rows = np.repeat(np.arange(source.shape[0]), np.diff(source.indptr))
            scores[rows, source.indices] = -np.inf
        return SRPTensor.from_dense(
            torch.from_numpy(scores),
            k=k,
            score_mode="raw",
        )


class _ContentModel(BaseColdStartRecommender):
    def __init__(self) -> None:
        super().__init__()
        self.source_features_: np.ndarray | None = None

    @property
    def is_fitted(self) -> bool:
        return self.source_features_ is not None

    def fit(
        self,
        interactions: csr_matrix,
        item_features: np.ndarray,
        **kwargs,
    ) -> _ContentModel:
        item_ids = np.asarray(kwargs["item_ids"], dtype=object)
        features = np.asarray(item_features, dtype=np.float32)
        if interactions.shape[1] != len(item_ids):
            raise ValueError("interactions and item_ids must have matching items")
        self.source_features_ = features.copy()
        self._install_feature_catalog(
            source_item_ids=item_ids,
            source_popularity=np.zeros(len(item_ids), dtype=np.float32),
            n_input_features=features.shape[1],
            candidate_features=features,
            metadata=None,
            feature_space_id=None,
            dtype=np.dtype("float32"),
            include_popularity=False,
        )
        return self

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids=None,
    ) -> SRPTensor:
        source = self._prepare_source(source)
        assert self.source_features_ is not None
        selection = self._resolve_candidate_selection(candidate_ids)
        profiles = np.asarray(source @ self.source_features_)
        scores = profiles @ np.asarray(selection.features).T
        if exclude_seen:
            rows = np.repeat(np.arange(source.shape[0]), np.diff(source.indptr))
            seen_catalog = selection.source_to_candidate[source.indices]
            seen_local = selection.candidate_to_local[seen_catalog]
            selected = seen_local >= 0
            scores[rows[selected], seen_local[selected]] = -np.inf
        local = SRPTensor.from_dense(
            torch.from_numpy(scores.astype(np.float32, copy=False)),
            k=k,
            score_mode="raw",
        )
        catalog_rows = torch.from_numpy(selection.rows)
        return SRPTensor(
            cols=catalog_rows[local.cols],
            vals=local.vals,
            shape=(source.shape[0], self.candidates.n_items),
        )


def test_abstract_bases_cannot_be_instantiated_directly():
    with pytest.raises(TypeError, match="abstract"):
        BaseCollaborativeRecommender()
    with pytest.raises(TypeError, match="abstract"):
        BaseColdStartRecommender()


def test_collaborative_base_supplies_batched_predict_and_validation():
    interactions = csr_matrix(
        np.asarray([[1, 0, 1], [0, 1, 1]], dtype=np.float32)
    )
    model = _PopularityModel().fit(interactions)

    predictions = model.predict(interactions, k=1, batch_size=1)

    assert predictions.shape == interactions.shape
    assert predictions.rows == 2
    with pytest.raises(ValueError, match="source has 4 items"):
        model.predict(csr_matrix((1, 4), dtype=np.float32), k=1)


def test_cold_start_base_supplies_catalog_lifecycle_and_batched_predict():
    interactions = csr_matrix(
        np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    )
    features = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    model = _ContentModel().fit(
        interactions,
        features,
        item_ids=["a", "b"],
    )
    model.build_candidates(
        item_ids=["a", "b", "cold"],
        item_features=np.asarray([[1, 0], [0, 1], [1, 1]], dtype=np.float32),
    )

    predictions = model.predict(
        interactions,
        k=1,
        batch_size=1,
        exclude_seen=True,
    )

    assert predictions.shape == (2, 3)
    assert model.candidates.item_ids.tolist() == ["a", "b", "cold"]
    assert predictions.cols[:, 0].tolist() == [2, 2]


def test_builtin_models_use_the_public_bases():
    assert isinstance(EASE(), BaseCollaborativeRecommender)
    assert isinstance(ELSATrainer(), BaseCollaborativeRecommender)
    assert isinstance(TEASER(), BaseColdStartRecommender)
    assert isinstance(TEASERGDTrainer(), BaseColdStartRecommender)

