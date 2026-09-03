from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

pytest.importorskip("sklearn")

from compresso_recsys.models import ItemKNNConfig, ItemKNNRecommender


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0],
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [0, 0, 1, 1, 1],
                [0, 1, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )


def test_item_knn_builds_sparse_neighbor_graph(interactions):
    model = ItemKNNRecommender(ItemKNNConfig(n_neighbors=2)).fit(interactions)

    assert model.similarity_.shape == (5, 5)
    assert model.similarity_.nnz <= 10
    np.testing.assert_array_equal(model.similarity_.diagonal(), 0)
    assert np.all(model.similarity_.data > 0)


def test_item_knn_prediction_contract(interactions):
    model = ItemKNNRecommender(ItemKNNConfig(n_neighbors=3)).fit(
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


def test_failed_item_knn_fit_does_not_publish_state(interactions):
    model = ItemKNNRecommender(ItemKNNConfig(n_neighbors=2))

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        model.fit(interactions, item_ids=["too-short"])

    assert not model.is_fitted
    assert model.n_items is None


def test_failed_item_knn_refit_preserves_state(interactions):
    item_ids = np.array([f"item-{index}" for index in range(interactions.shape[1])])
    model = ItemKNNRecommender(ItemKNNConfig(n_neighbors=2)).fit(
        interactions,
        item_ids=item_ids,
    )
    old_similarity = model.similarity_.copy()
    wider = csr_matrix((interactions.shape[0], interactions.shape[1] + 1))

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        model.fit(wider, item_ids=["too-short"])

    assert model.n_items == interactions.shape[1]
    np.testing.assert_array_equal(model.source_item_ids, item_ids)
    np.testing.assert_array_equal(
        model.similarity_.toarray(),
        old_similarity.toarray(),
    )


def test_item_knn_uses_normalized_target_neighborhood_scores(interactions):
    model = ItemKNNRecommender(ItemKNNConfig(n_neighbors=3)).fit(interactions)
    source = csr_matrix(np.array([[1, 0, 0, 0, 0]], dtype=np.float32))

    predictions = model.predict_on_batch(
        source,
        k=interactions.shape[1],
        exclude_seen=False,
    )

    normalizer = np.asarray(abs(model.similarity_).sum(axis=1)).ravel()
    expected = np.zeros(interactions.shape[1], dtype=np.float32)
    np.divide(
        model.similarity_[:, 0].toarray().ravel(),
        normalizer,
        out=expected,
        where=normalizer > 0,
    )
    actual = np.empty_like(expected)
    actual[predictions.cols[0].numpy()] = predictions.vals[0].numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-7)


def test_item_knn_round_trip_needs_only_similarity_graph(
    tmp_path, interactions
):
    model = ItemKNNRecommender(ItemKNNConfig(n_neighbors=3)).fit(interactions)
    source = interactions[:2]
    before = model.predict_on_batch(source, k=2)
    path = tmp_path / "knn.zip"

    model.save(path)
    restored = ItemKNNRecommender.load(path)
    after = restored.predict_on_batch(source, k=2)

    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)
    np.testing.assert_array_equal(
        restored.similarity_.toarray(), model.similarity_.toarray()
    )


def test_item_knn_config_validation():
    with pytest.raises(ValueError, match="n_neighbors"):
        ItemKNNConfig(n_neighbors=0)
    with pytest.raises(ValueError, match="dtype"):
        ItemKNNConfig(dtype="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_jobs"):
        ItemKNNConfig(n_jobs=0)
