from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso_recsys.evaluation import (
    evaluate_ranked_predictions,
    evaluate_recommender,
)
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import EASE, EASEConfig, Recommender


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
            dtype=np.float64,
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
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )


def test_ease_matches_closed_form_reference(interactions):
    config = EASEConfig(l2=7.5, dtype="float64")
    model = EASE(config)

    returned = model.fit(interactions)

    gram = interactions.T.dot(interactions).toarray()
    diagonal = np.diag_indices(gram.shape[0])
    gram[diagonal] += config.l2
    precision = np.linalg.inv(gram)
    expected = precision / -np.diag(precision)
    expected[diagonal] = 0
    assert returned is model
    assert model.is_fitted
    assert model.n_items_ == interactions.shape[1]
    np.testing.assert_allclose(model.coefficients_, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(np.diag(model.coefficients_), 0)


def test_ease_defaults_to_float32():
    config = EASEConfig()

    assert config.l2 == 500.0
    assert config.dtype == "float32"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_ease_preserves_configured_dtype(interactions, source, dtype):
    model = EASE(EASEConfig(dtype=dtype)).fit(interactions)

    predictions = model.predict_on_batch(source, k=2)

    assert model.coefficients_.dtype == np.dtype(dtype)
    assert predictions.vals.numpy().dtype == np.dtype(dtype)


def test_predict_matches_predict_on_batch(interactions, source):
    model = EASE().fit(interactions)

    expected = model.predict_on_batch(source, k=3)
    actual = model.predict(source, k=3, batch_size=2)

    assert actual.shape == source.shape
    assert actual.k == 3
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_predictions_exclude_seen_items(interactions, source):
    model = EASE().fit(interactions)

    predictions = model.predict(source, k=3, batch_size=2)

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()).isdisjoint(source[row].indices)


def test_seen_item_mask_can_be_disabled(interactions, source):
    model = EASE().fit(interactions)

    predictions = model.predict(
        source,
        k=source.shape[1],
        batch_size=2,
        exclude_seen=False,
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()) == set(range(source.shape[1]))


def test_empty_source_returns_empty_srp(interactions):
    model = EASE(EASEConfig(dtype="float32")).fit(interactions)
    source = csr_matrix((0, interactions.shape[1]), dtype=np.float32)

    predictions = model.predict(source, k=3)

    assert predictions.shape == source.shape
    assert predictions.cols.shape == (0, 3)
    assert predictions.vals.shape == (0, 3)
    assert predictions.vals.dtype == torch.float32


@pytest.mark.parametrize("l2", [0, -1, np.nan, np.inf])
def test_ease_config_rejects_invalid_regularization(l2):
    with pytest.raises(ValueError, match="l2"):
        EASEConfig(l2=l2)


def test_ease_config_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="dtype"):
        EASEConfig(dtype="float16")  # type: ignore[arg-type]


def test_fit_requires_nonempty_finite_csr():
    model = EASE()

    with pytest.raises(TypeError, match="csr_matrix"):
        model.fit(csc_matrix(np.eye(2)))
    with pytest.raises(ValueError, match="at least one"):
        model.fit(csr_matrix((0, 2)))
    with pytest.raises(ValueError, match="finite"):
        model.fit(csr_matrix(np.array([[1.0, np.nan]])))


def test_failed_ease_fit_does_not_publish_state(interactions):
    model = EASE()

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        model.fit(interactions, item_ids=["too-short"])

    assert not model.is_fitted
    assert model.n_items is None


def test_failed_ease_refit_preserves_state(interactions):
    item_ids = np.array([f"item-{index}" for index in range(interactions.shape[1])])
    model = EASE().fit(interactions, item_ids=item_ids)
    old_coefficients = model.coefficients_.copy()
    wider = csr_matrix((interactions.shape[0], interactions.shape[1] + 1))

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        model.fit(wider, item_ids=["too-short"])

    assert model.n_items == interactions.shape[1]
    np.testing.assert_array_equal(model.source_item_ids, item_ids)
    np.testing.assert_array_equal(model.coefficients_, old_coefficients)


def test_prediction_requires_fitted_model(source):
    with pytest.raises(RuntimeError, match="fitted"):
        EASE().predict_on_batch(source, k=2)


def test_prediction_validates_shape_and_parameters(interactions, source):
    model = EASE().fit(interactions)

    with pytest.raises(ValueError, match="fitted with"):
        model.predict_on_batch(csr_matrix((1, interactions.shape[1] + 1)), k=1)
    with pytest.raises(ValueError, match=r"k must be in \[1"):
        model.predict_on_batch(source, k=0)
    with pytest.raises(ValueError, match="batch_size"):
        model.predict(source, k=2, batch_size=0)
    with pytest.raises(TypeError, match="csr_matrix"):
        model.predict_on_batch(csc_matrix(source), k=2)


def test_prediction_rejects_rows_with_too_few_unseen_items(interactions):
    model = EASE().fit(interactions)
    source = csr_matrix(np.array([[1, 1, 1, 1, 1, 0]], dtype=np.float64))

    with pytest.raises(ValueError, match="only 1 unseen items"):
        model.predict_on_batch(source, k=2)


def test_ease_implements_recommender_protocol():
    assert isinstance(EASE(), Recommender)


def test_ease_streaming_evaluation_matches_materialized_predictions(
    interactions,
    source,
):
    model = EASE(EASEConfig(l2=20)).fit(interactions)
    targets = csr_matrix(
        (
            np.ones(source.shape[0], dtype=np.float64),
            (
                np.arange(source.shape[0]),
                np.array([1, 3, 2, 5]),
            ),
        ),
        shape=source.shape,
    )

    streamed = evaluate_recommender(
        model,
        source=source,
        targets=targets,
        metrics=[CalibratedRecall([1, 3]), NDCG(3)],
        batch_size=2,
    )
    materialized = evaluate_ranked_predictions(
        predictions=model.predict(source, k=3, batch_size=2),
        targets=targets,
        metrics=[CalibratedRecall([1, 3]), NDCG(3)],
        batch_size=2,
    )

    assert streamed == materialized
