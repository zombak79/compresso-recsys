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
