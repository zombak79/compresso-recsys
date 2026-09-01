from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models import (
    PopularityBaseline,
    PopularityBaselineConfig,
    RandomBaseline,
    RandomBaselineConfig,
)


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def source() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
        )
    )


def test_popularity_counts_interacting_users(interactions):
    model = PopularityBaseline().fit(interactions)

    np.testing.assert_array_equal(model.popularity_, [2, 3, 2, 2, 2, 1])


def test_popularity_can_sum_interaction_values():
    interactions = csr_matrix(np.array([[2, 0], [3, 4]], dtype=np.float32))

    model = PopularityBaseline(
        PopularityBaselineConfig(use_values=True)
    ).fit(interactions)

    np.testing.assert_array_equal(model.popularity_, [5, 4])


def test_popularity_rejects_negative_interactions():
    interactions = csr_matrix(np.array([[1, -1]], dtype=np.float32))

    with pytest.raises(ValueError, match="nonnegative"):
        PopularityBaseline().fit(interactions)


@pytest.mark.parametrize("model", [RandomBaseline(), PopularityBaseline()])
def test_baselines_exclude_seen_and_respect_candidates(model, interactions, source):
    item_ids = np.array([f"item-{i}" for i in range(interactions.shape[1])])
    model.fit(interactions, item_ids=item_ids)

    predictions = model.predict_on_batch(
        source,
        k=2,
        candidate_ids=["item-0", "item-3", "item-4", "item-5"],
    )

    allowed = {0, 3, 4, 5}
    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()) <= allowed
        assert set(predictions.cols[row].tolist()).isdisjoint(source[row].indices)


def test_random_predictions_are_batch_invariant(interactions, source):
    model = RandomBaseline(RandomBaselineConfig(seed=17)).fit(interactions)

    together = model.predict(source, k=3, batch_size=source.shape[0])
    split = model.predict(source, k=3, batch_size=1)

    torch.testing.assert_close(together.cols, split.cols)
    torch.testing.assert_close(together.vals, split.vals)


def test_random_seed_changes_rankings(interactions, source):
    first = RandomBaseline(RandomBaselineConfig(seed=1)).fit(interactions)
    second = RandomBaseline(RandomBaselineConfig(seed=2)).fit(interactions)

    left = first.predict_on_batch(source, k=4, exclude_seen=False)
    right = second.predict_on_batch(source, k=4, exclude_seen=False)

    assert not torch.equal(left.cols, right.cols)


@pytest.mark.parametrize("model", [RandomBaseline(), PopularityBaseline()])
def test_baseline_round_trip_preserves_predictions(
    tmp_path, model, interactions, source
):
    item_ids = np.array([f"item-{i}" for i in range(interactions.shape[1])])
    model.fit(interactions, item_ids=item_ids)
    before = model.predict_on_batch(source, k=3)
    path = tmp_path / f"{type(model).__name__}.zip"

    model.save(path)
    restored = type(model).load(path)
    after = restored.predict_on_batch(source, k=3)

    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)
    np.testing.assert_array_equal(restored.source_item_ids, item_ids)


@pytest.mark.parametrize("model", [RandomBaseline(), PopularityBaseline()])
def test_baselines_reject_insufficient_unseen_candidates(model, interactions):
    model.fit(interactions)
    source = csr_matrix(np.array([[1, 1, 1, 1, 1, 0]], dtype=np.float32))

    with pytest.raises(ValueError, match="only 1 unseen items"):
        model.predict_on_batch(source, k=2)


def test_baseline_config_validation():
    with pytest.raises(TypeError, match="seed"):
        RandomBaselineConfig(seed=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="use_values"):
        PopularityBaselineConfig(use_values=1)  # type: ignore[arg-type]
