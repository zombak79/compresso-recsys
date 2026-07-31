from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import (
    evaluate_ranked_predictions,
    evaluate_recommender,
)
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import LEMSA, LEMSAConfig, Recommender


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
def item_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.2, 0.0],
            [0.8, 0.5, 0.0],
            [0.0, 1.0, 0.2],
            [0.0, 0.7, 1.0],
            [0.2, 0.0, 1.0],
            [1.0, 0.0, 0.6],
        ],
        dtype=np.float64,
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


def _literal_lemsa(
    interactions: np.ndarray,
    features: np.ndarray,
    *,
    l2_encoder: float,
    epochs: int,
    encoder_init: str = "zeros",
    update_batch_size: int | None = 1,
) -> np.ndarray:
    encoder = (
        np.zeros_like(features) if encoder_init == "zeros" else features.copy()
    )
    batch_size = (
        interactions.shape[1]
        if update_batch_size is None
        else update_batch_size
    )
    for _ in range(epochs):
        for block_start in range(0, interactions.shape[1], batch_size):
            block_end = min(block_start + batch_size, interactions.shape[1])
            next_rows = encoder[block_start:block_end].copy()
            for offset, item in enumerate(range(block_start, block_end)):
                users = np.flatnonzero(interactions[:, item])
                keep = np.arange(interactions.shape[1]) != item
                decoder = features[keep]
                if users.size == 0:
                    next_rows[offset] = 0
                    continue
                restricted = interactions[np.ix_(users, keep)]
                context = restricted @ encoder[keep]
                residual = restricted - context @ decoder.T
                matrix = users.size * decoder.T @ decoder
                matrix.flat[:: matrix.shape[0] + 1] += l2_encoder
                right_hand_side = decoder.T @ residual.T @ np.ones(users.size)
                next_rows[offset] = np.linalg.solve(matrix, right_hand_side)
            encoder[block_start:block_end] = next_rows
    return encoder


def _fit(
    interactions: csr_matrix,
    item_features,
    *,
    solver: str = "eigen",
    epochs: int = 2,
    encoder_init: str = "zeros",
    update_batch_size: int | None = 1,
) -> LEMSA:
    return LEMSA(
        LEMSAConfig(
            l2_encoder=0.17,
            epochs=epochs,
            solver=solver,
            encoder_init=encoder_init,
            update_batch_size=update_batch_size,
            precompute_batch_size=2,
            dtype="float64",
        )
    ).fit(interactions, item_features)


def test_defaults_are_production_oriented():
    config = LEMSAConfig()

    assert config.l2_encoder == 0.05
    assert config.epochs == 10
    assert config.solver == "eigen"
    assert config.encoder_init == "zeros"
    assert config.tolerance is None
    assert config.update_batch_size == 1
    assert config.precompute_batch_size == 8192
    assert config.dtype == "float32"


@pytest.mark.parametrize("solver", ["direct", "eigen"])
@pytest.mark.parametrize("encoder_init", ["zeros", "features"])
def test_fit_matches_literal_gated_coordinate_updates(
    interactions,
    item_features,
    solver,
    encoder_init,
):
    model = _fit(
        interactions,
        item_features,
        solver=solver,
        epochs=3,
        encoder_init=encoder_init,
    )
    expected = _literal_lemsa(
        interactions.toarray(),
        item_features,
        l2_encoder=0.17,
        epochs=3,
        encoder_init=encoder_init,
    )

    np.testing.assert_allclose(model.encoder_, expected, rtol=1e-11, atol=1e-11)
    assert model.n_epochs_ == 3
    assert len(model.fit_history_) == 3
    assert all(row["solver_fallbacks"] == 0 for row in model.fit_history_)


@pytest.mark.parametrize("solver", ["direct", "eigen"])
@pytest.mark.parametrize("update_batch_size", [2, None])
def test_fit_matches_literal_snapshot_block_updates(
    interactions,
    item_features,
    solver,
    update_batch_size,
):
    model = _fit(
        interactions,
        item_features,
        solver=solver,
        epochs=3,
        update_batch_size=update_batch_size,
    )
    expected = _literal_lemsa(
        interactions.toarray(),
        item_features,
        l2_encoder=0.17,
        epochs=3,
        update_batch_size=update_batch_size,
    )

    np.testing.assert_allclose(model.encoder_, expected, rtol=1e-11, atol=1e-11)


def test_snapshot_block_size_changes_update_schedule(interactions, item_features):
    sequential = _fit(
        interactions,
        item_features,
        epochs=2,
        update_batch_size=1,
    )
    full_sweep = _fit(
        interactions,
        item_features,
        epochs=2,
        update_batch_size=None,
    )

    assert not np.allclose(sequential.encoder_, full_sweep.encoder_)


def test_eigen_and_direct_solvers_produce_identical_scores(
    interactions,
    item_features,
    source,
):
    direct = _fit(interactions, item_features, solver="direct", epochs=4)
    eigen = _fit(interactions, item_features, solver="eigen", epochs=4)

    np.testing.assert_allclose(eigen.encoder_, direct.encoder_, rtol=1e-11, atol=1e-11)
    direct_scores = direct.predict_on_batch(
        source,
        k=source.shape[1],
        exclude_seen=False,
    ).to_dense()
    eigen_scores = eigen.predict_on_batch(
        source,
        k=source.shape[1],
        exclude_seen=False,
    ).to_dense()
    torch.testing.assert_close(eigen_scores, direct_scores, rtol=1e-11, atol=1e-11)
    assert eigen.feature_rotation_.shape == (3, 3)
    assert eigen.feature_eigenvalues_.shape == (3,)
    assert direct.feature_rotation_ is None
    assert direct.feature_eigenvalues_ is None


def test_default_float32_path_preserves_prediction_dtype(
    interactions,
    item_features,
):
    model = LEMSA(LEMSAConfig(epochs=1)).fit(interactions, item_features)
    empty_source = csr_matrix((0, interactions.shape[1]), dtype=np.float32)

    predictions = model.predict(empty_source, k=3)

    assert model.encoder_.dtype == np.float32
    assert predictions.cols.shape == (0, 3)
    assert predictions.vals.dtype == torch.float32


def test_supported_item_feature_types_are_equivalent(
    interactions,
    item_features,
    source,
):
    srp_features = SRPTensor.from_dense(
        torch.from_numpy(item_features),
        k=2,
        score_mode="raw",
    )
    feature_inputs = [
        item_features,
        csr_matrix(item_features),
        torch.from_numpy(item_features),
        torch.from_numpy(item_features).to_sparse(),
        srp_features,
    ]
    models = [_fit(interactions, features) for features in feature_inputs]
    expected = models[0].predict_on_batch(source, k=3)

    for model in models[1:]:
        np.testing.assert_allclose(
            model.encoder_,
            models[0].encoder_,
            rtol=1e-11,
            atol=1e-11,
        )
        actual = model.predict_on_batch(source, k=3)
        torch.testing.assert_close(actual.cols, expected.cols)
        torch.testing.assert_close(actual.vals, expected.vals)


def test_zero_support_rows_are_zero_even_with_feature_initialization(item_features):
    interactions = csr_matrix(
        np.array(
            [
                [1, 1, 0],
                [0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    model = _fit(
        interactions,
        item_features[:3],
        solver="eigen",
        epochs=1,
        encoder_init="features",
    )

    np.testing.assert_array_equal(model.encoder_[2], np.zeros(3))


def test_fit_uses_gating_instead_of_a_global_zero_diagonal(
    interactions,
    item_features,
):
    model = _fit(interactions, item_features, epochs=3)

    diagonal = np.sum(model.encoder_ * item_features, axis=1)
    assert np.any(np.abs(diagonal) > 1e-8)


def test_cold_items_are_decoder_only_candidates(item_features):
    train_indices = np.array([0, 2, 4], dtype=np.int64)
    interactions = csr_matrix(
        np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    model = LEMSA(LEMSAConfig(epochs=2, dtype="float64")).fit(
        interactions,
        item_features,
        train_item_indices=train_indices,
        item_ids=list("ABCDEF"),
    )
    source = csr_matrix([[1, 0, 0, 0, 1, 0]], dtype=np.float64)

    predictions = model.predict_on_batch(
        source,
        k=source.shape[1],
        exclude_seen=False,
    )
    profiles = np.asarray(source[:, train_indices] @ model.encoder_)
    expected_scores = profiles @ item_features.T

    np.testing.assert_allclose(
        predictions.to_dense().numpy(),
        expected_scores,
        rtol=1e-11,
        atol=1e-11,
    )
    with pytest.raises(ValueError, match="no fitted encoder row"):
        model.predict_on_batch(csr_matrix([[0, 1, 0, 0, 0, 0]]), k=2)


def test_candidate_catalog_can_be_rebuilt_and_updated(
    interactions,
    item_features,
):
    metadata = pd.DataFrame(
        {"item_id": list("ABCDEF"), "title": list("abcdef")}
    )
    model = LEMSA(LEMSAConfig(epochs=1, dtype="float64")).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
        metadata=metadata,
        feature_space_id="language-model@revision",
    )

    model.update_candidates(
        item_ids=["G"],
        item_features=item_features[[0]] * 0.5,
        metadata=pd.DataFrame({"item_id": ["G"], "title": ["g"]}),
    )
    assert model.candidates.item_ids.tolist() == list("ABCDEFG")
    assert model.candidates.version == 2
    assert model.candidates.metadata.iloc[-1]["title"] == "g"

    model.build_candidates(
        item_ids=["F", "G"],
        item_features=item_features[[5, 0]],
    )
    predictions = model.predict_on_batch(
        csr_matrix([[1, 0, 0, 0, 0, 0]], dtype=np.float64),
        k=2,
    )
    assert predictions.shape == (1, 2)
    assert set(predictions.cols[0].tolist()) == {0, 1}


def test_align_source_and_user_profiles_use_original_feature_space(
    interactions,
    item_features,
    source,
):
    columns = np.array([0, 2, 4], dtype=np.int64)
    item_ids = np.array(list("ABCDEF"), dtype=object)
    model = LEMSA(LEMSAConfig(epochs=2, dtype="float64")).fit(
        interactions[:, columns],
        item_features[columns],
        item_ids=item_ids[columns],
    )
    model.build_candidates(item_ids=item_ids, item_features=item_features)

    aligned = model.align_source(source, item_ids=item_ids)
    profiles = model.user_profiles(aligned)

    np.testing.assert_array_equal(aligned.toarray(), source[:, columns].toarray())
    np.testing.assert_allclose(
        profiles,
        source[:, columns] @ model.encoder_,
        rtol=1e-12,
        atol=1e-12,
    )


def test_streaming_evaluation_matches_materialized_predictions(
    interactions,
    item_features,
    source,
):
    model = _fit(interactions, item_features)
    targets = csr_matrix(
        (
            np.ones(source.shape[0], dtype=np.float64),
            (np.arange(source.shape[0]), np.array([1, 3, 2, 5])),
        ),
        shape=source.shape,
    )
    metrics = [CalibratedRecall([1, 3]), NDCG(3)]

    streamed = evaluate_recommender(
        model,
        source=source,
        targets=targets,
        metrics=metrics,
        batch_size=2,
    )
    materialized = evaluate_ranked_predictions(
        predictions=model.predict(source, k=3, batch_size=2),
        targets=targets,
        metrics=metrics,
        batch_size=2,
    )

    assert streamed == materialized


def test_tolerance_can_stop_after_first_sweep(interactions, item_features):
    model = LEMSA(
        LEMSAConfig(
            epochs=10,
            tolerance=2.0,
            dtype="float64",
        )
    ).fit(interactions, item_features)

    assert model.n_epochs_ == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("l2_encoder", 0),
        ("l2_encoder", np.nan),
        ("epochs", 0),
        ("epochs", 1.5),
        ("solver", "iterative"),
        ("encoder_init", "xavier"),
        ("tolerance", 0),
        ("tolerance", np.inf),
        ("update_batch_size", 0),
        ("update_batch_size", 1.5),
        ("update_batch_size", True),
        ("precompute_batch_size", 0),
        ("precompute_batch_size", True),
        ("dtype", "float16"),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        LEMSAConfig(**{field: value})


def test_fit_and_prediction_validate_inputs(interactions, item_features, source):
    model = LEMSA(LEMSAConfig(epochs=1))

    with pytest.raises(TypeError, match="csr_matrix"):
        model.fit(csc_matrix(interactions), item_features)
    with pytest.raises(ValueError, match="binary"):
        model.fit(interactions * 2, item_features)
    with pytest.raises(ValueError, match="rows"):
        model.fit(interactions, item_features[:-1])
    with pytest.raises(ValueError, match="feature_names"):
        model.fit(interactions, item_features, feature_names=["too", "short"])
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_on_batch(source, k=2)

    model.fit(interactions, item_features)
    with pytest.raises(ValueError, match="fitted with"):
        model.predict_on_batch(csr_matrix((1, source.shape[1] + 1)), k=1)
    with pytest.raises(TypeError, match="csr_matrix"):
        model.predict_on_batch(csc_matrix(source), k=1)
    with pytest.raises(ValueError, match="binary"):
        model.predict_on_batch(source * 2, k=1)


def test_lemsa_implements_recommender_protocol():
    assert isinstance(LEMSA(), Recommender)
