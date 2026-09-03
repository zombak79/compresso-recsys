from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

pytest.importorskip("sklearn")

from compresso_recsys.models import UserKNNConfig, UserKNNRecommender


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 0, 1, 1, 0],
                [0, 1, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )


def test_user_knn_matches_normalized_weighted_neighbor_formula():
    training = csr_matrix(
        np.array([[1, 1, 0], [1, 0, 1]], dtype=np.float32)
    )
    source = csr_matrix(np.array([[1, 0, 0]], dtype=np.float32))
    model = UserKNNRecommender(UserKNNConfig(n_neighbors=2)).fit(training)

    predictions = model.predict_on_batch(source, k=3, exclude_seen=False)

    actual = np.empty(3, dtype=np.float32)
    actual[predictions.cols[0].numpy()] = predictions.vals[0].numpy()
    np.testing.assert_allclose(actual, [1.0, 0.5, 0.5], atol=1e-7)


def test_user_knn_prediction_contract(interactions):
    model = UserKNNRecommender(UserKNNConfig(n_neighbors=3)).fit(
        interactions,
        item_ids=[f"item-{i}" for i in range(5)],
    )
    source = csr_matrix(np.array([[1, 0, 0, 0, 0]], dtype=np.float32))

    predictions = model.predict_on_batch(
        source,
        k=2,
        candidate_ids=["item-1", "item-2", "item-3"],
    )

    assert predictions.shape == source.shape
    assert set(predictions.cols[0].tolist()) <= {1, 2, 3}
    assert 0 not in predictions.cols[0].tolist()


def test_failed_user_knn_fit_does_not_publish_state(interactions):
    model = UserKNNRecommender(UserKNNConfig(n_neighbors=2))

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        model.fit(interactions, item_ids=["too-short"])

    assert not model.is_fitted
    assert model.n_items is None


def test_failed_user_knn_refit_preserves_state(interactions):
    item_ids = np.array([f"item-{index}" for index in range(interactions.shape[1])])
    model = UserKNNRecommender(UserKNNConfig(n_neighbors=2)).fit(
        interactions,
        item_ids=item_ids,
    )
    old_interactions = model.training_interactions_.copy()
    old_index = model._index
    wider = csr_matrix((interactions.shape[0], interactions.shape[1] + 1))

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        model.fit(wider, item_ids=["too-short"])

    assert model.n_items == interactions.shape[1]
    assert model._index is old_index
    np.testing.assert_array_equal(model.source_item_ids, item_ids)
    np.testing.assert_array_equal(
        model.training_interactions_.toarray(),
        old_interactions.toarray(),
    )


def test_user_knn_index_failure_preserves_state(interactions, monkeypatch):
    model = UserKNNRecommender(UserKNNConfig(n_neighbors=2)).fit(interactions)
    old_interactions = model.training_interactions_.copy()
    old_index = model._index

    def fail_index(_interactions):
        raise RuntimeError("index failed")

    monkeypatch.setattr(model, "_make_index", fail_index)
    with pytest.raises(RuntimeError, match="index failed"):
        model.fit(interactions)

    assert model.is_fitted
    assert model._index is old_index
    np.testing.assert_array_equal(
        model.training_interactions_.toarray(),
        old_interactions.toarray(),
    )


def test_user_knn_round_trip_rebuilds_transient_index(tmp_path, interactions):
    model = UserKNNRecommender(UserKNNConfig(n_neighbors=3)).fit(interactions)
    source = interactions[:2]
    before = model.predict_on_batch(source, k=2)
    path = tmp_path / "user-knn.zip"

    model.save(path)
    restored = UserKNNRecommender.load(path)
    after = restored.predict_on_batch(source, k=2)

    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)
    np.testing.assert_array_equal(
        restored.training_interactions_.toarray(),
        model.training_interactions_.toarray(),
    )
    assert restored._index is not None


def test_user_knn_rejects_insufficient_unseen_candidates(interactions):
    model = UserKNNRecommender().fit(interactions)
    source = csr_matrix(np.array([[1, 1, 1, 1, 0]], dtype=np.float32))

    with pytest.raises(ValueError, match="only 1 unseen items"):
        model.predict_on_batch(source, k=2)


def test_user_knn_config_validation():
    with pytest.raises(ValueError, match="n_neighbors"):
        UserKNNConfig(n_neighbors=0)
    with pytest.raises(ValueError, match="dtype"):
        UserKNNConfig(dtype="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_jobs"):
        UserKNNConfig(n_jobs=0)
