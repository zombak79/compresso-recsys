from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import (
    ColdStartRecommender,
    Recommender,
    TEASERGD,
    TEASERGDConfig,
    TEASERGDTrainer,
)
from compresso_recsys.models.teaser_gd import (
    _initialize_encoder_from_features,
    _teaser_reconstruction_loss,
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
def item_features() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )


def _config(**overrides) -> TEASERGDConfig:
    values = dict(
        epochs=2,
        batch_size=2,
        max_output=4,
        lr=0.01,
        show_progress=False,
        include_popularity=False,
        coefficient_regularization_samples=16,
        seed=3,
    )
    values.update(overrides)
    return TEASERGDConfig(**values)


def test_forward_matches_explicit_diagonal_free_coefficients():
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 2.0], [-1.0, 1.0]],
    )
    model = TEASERGD(3, 2, use_relu=False)
    with torch.no_grad():
        model.encoder.copy_(torch.tensor([[0.5, 1.0], [2.0, -0.5], [1.5, 0.25]]))
    x = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])

    actual = model(
        x,
        sources=torch.arange(3),
        source_features=features,
        candidate_features=features,
        source_candidate_positions=torch.arange(3),
    )
    coefficients = model.encoder @ features.T
    expected_coefficients = coefficients - torch.diag(torch.diag(coefficients))

    torch.testing.assert_close(actual, x @ expected_coefficients)


def test_forward_handles_source_prefix_and_sampled_candidates():
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0]],
    )
    model = TEASERGD(4, 2, use_relu=False)
    with torch.no_grad():
        model.encoder.copy_(torch.arange(8, dtype=torch.float32).reshape(4, 2) / 4)
    sources = torch.tensor([2, 0])
    candidates = torch.tensor([2, 0, 3])
    x = torch.tensor([[1.0, 2.0]])

    actual = model(
        x,
        sources=sources,
        source_features=features[sources],
        candidate_features=features[candidates],
        source_candidate_positions=torch.tensor([0, 1]),
    )
    coefficients = model.encoder[sources] @ features[candidates].T
    coefficients[0, 0] = 0
    coefficients[1, 1] = 0

    torch.testing.assert_close(actual, x @ coefficients)


def test_forward_preserves_encoder_gradients():
    model = TEASERGD(3, 2, use_relu=False)
    features = torch.tensor([[1.0, 0.0], [0.5, 1.0], [0.0, 2.0]])
    result = model(
        torch.tensor([[1.0, 1.0]]),
        sources=torch.tensor([0, 2]),
        source_features=features[[0, 2]],
        candidate_features=features,
        source_candidate_positions=torch.tensor([0, 2]),
    )
    result.square().sum().backward()

    assert model.encoder.grad is not None
    assert torch.isfinite(model.encoder.grad).all()
    assert torch.count_nonzero(model.encoder.grad[[0, 2]]) > 0
    assert torch.count_nonzero(model.encoder.grad[1]) == 0


def test_optional_encoder_normalization_matches_explicit_l2_normalization():
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    model = TEASERGD(3, 2, use_relu=False, normalize_encoder=True)
    with torch.no_grad():
        model.encoder.copy_(torch.tensor([[3.0, 4.0], [0.0, 2.0], [-2.0, 0.0]]))
    x = torch.tensor([[1.0, 0.0, 1.0]])

    actual = model(
        x,
        sources=torch.arange(3),
        source_features=features,
        candidate_features=features,
        source_candidate_positions=torch.arange(3),
    )
    encoder = torch.nn.functional.normalize(model.encoder, dim=-1)
    coefficients = encoder @ features.T
    coefficients.fill_diagonal_(0)

    torch.testing.assert_close(actual, x @ coefficients)
    torch.testing.assert_close(
        model.encoder_weights().norm(dim=-1),
        torch.ones(3),
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_feature_encoder_initialization_supports_dense_and_csr(sparse):
    features = np.array(
        [[1.0, 0.0, 2.0], [0.0, -1.0, 0.0], [3.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    model = TEASERGD(3, 3)

    _initialize_encoder_from_features(
        model,
        csr_matrix(features) if sparse else features,
    )

    torch.testing.assert_close(model.encoder, torch.from_numpy(features))


def test_exact_coefficient_norm_matches_materialized_matrix():
    model = TEASERGD(3, 2, use_relu=False)
    features = torch.randn(3, 2)
    coefficients = model.encoder @ features.T
    coefficients = coefficients - torch.diag(torch.diag(coefficients))

    actual = model.exact_coefficient_squared_norm(features)

    torch.testing.assert_close(actual, coefficients.square().sum())


def test_teaser_reconstruction_matches_frobenius_error():
    predictions = torch.tensor([[1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])

    sampled = _teaser_reconstruction_loss(
        predictions,
        targets,
        n_source_items=2,
        n_items=6,
    )
    squared = (predictions - targets).square()
    expected = torch.cat((squared[:, :2], squared[:, 2:] * 2.0), dim=1)

    torch.testing.assert_close(sampled, expected.sum(dim=-1).mean())


def test_teaser_reconstruction_full_output_needs_no_weighting():
    predictions = torch.randn(3, 6)
    targets = torch.randn(3, 6)

    actual = _teaser_reconstruction_loss(
        predictions,
        targets,
        n_source_items=2,
        n_items=6,
    )

    torch.testing.assert_close(
        actual,
        (predictions - targets).square().sum(dim=-1).mean(),
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_fit_predict_and_candidate_catalog(interactions, item_features, sparse):
    features = csr_matrix(item_features) if sparse else item_features
    item_ids = np.array([f"item-{index}" for index in range(6)])
    metadata = pd.DataFrame({"item_id": item_ids, "title": list("ABCDEF")})
    model = TEASERGDTrainer(_config()).fit(
        interactions,
        features,
        item_ids=item_ids,
        metadata=metadata,
        feature_space_id="test-features",
    )

    predictions = model.predict(interactions, k=3, batch_size=2)

    assert predictions.shape == interactions.shape
    assert len(model.history) == 2
    assert all(np.isfinite(record["loss"]) for record in model.history)
    assert model.candidates.metadata.equals(metadata)
    assert model.candidates.feature_space_id == "test-features"
    for row, seen in enumerate(interactions):
        assert set(predictions.cols[row].tolist()).isdisjoint(seen.indices)


def test_predict_matches_predict_on_batch(interactions, item_features):
    model = TEASERGDTrainer(_config(epochs=1)).fit(interactions, item_features)

    batched = model.predict(interactions, k=3, batch_size=2)
    direct = model.predict_on_batch(interactions, k=3)

    torch.testing.assert_close(batched.cols, direct.cols)
    torch.testing.assert_close(batched.vals, direct.vals)


def test_full_output_training_and_empty_prediction(interactions, item_features):
    model = TEASERGDTrainer(
        _config(epochs=1, max_output=None, coefficient_regularization_samples=0)
    ).fit(interactions, csr_matrix(item_features))

    result = model.predict_on_batch(
        csr_matrix((0, interactions.shape[1]), dtype=np.float32),
        k=2,
    )

    assert result.shape == (0, interactions.shape[1])
    assert result.cols.shape == (0, 2)
    assert model._training_tensor_cache is not None


def test_loss_modes_use_distinct_documented_scales(interactions, item_features):
    common = dict(
        epochs=1,
        batch_size=interactions.shape[0],
        max_output=None,
        lr=1e-3,
        show_progress=False,
        include_popularity=False,
        coefficient_regularization_samples=0,
        l2_coefficients=0.0,
        seed=9,
        use_relu=False,
    )
    normalized = TEASERGDTrainer(TEASERGDConfig(**common, loss="normalized_mse")).fit(
        interactions, item_features
    )
    teaser = TEASERGDTrainer(TEASERGDConfig(**common, loss="teaser")).fit(
        interactions,
        item_features,
    )

    assert normalized.history[0]["reconstruction"] <= 2.0
    assert teaser.history[0]["reconstruction"] > 2.0
    expected_encoder_ratio = item_features.size / interactions.shape[0]
    assert teaser.history[0]["encoder_l2"] == pytest.approx(
        normalized.history[0]["encoder_l2"] * expected_encoder_ratio
    )


def test_trainer_applies_feature_initialization_before_training(
    interactions,
    item_features,
    monkeypatch,
):
    def no_update(self, batch, training_features):
        return {
            "loss": 0.0,
            "reconstruction": 0.0,
            "coefficient_l2": 0.0,
            "encoder_l2": 0.0,
        }

    monkeypatch.setattr(TEASERGDTrainer, "_train_step", no_update)
    model = TEASERGDTrainer(
        _config(
            epochs=1,
            encoder_init="features",
            normalize_encoder=True,
        )
    ).fit(interactions, csr_matrix(item_features))

    torch.testing.assert_close(
        model._base_model.encoder.cpu(),
        torch.from_numpy(item_features),
    )
    torch.testing.assert_close(
        model._base_model.encoder_weights().norm(dim=-1).cpu(),
        torch.ones(item_features.shape[0]),
    )


def test_new_candidates_are_scored_without_refitting(interactions, item_features):
    model = TEASERGDTrainer(_config(epochs=1)).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
    )
    old_encoder = model._base_model.encoder.detach().clone()
    catalog = model.update_candidates(
        item_ids=["new"],
        item_features=np.array([[2.0, 1.0, 0.5]], dtype=np.float32),
    )

    result = model.predict_on_batch(interactions[:1], k=3, exclude_seen=False)

    assert catalog.n_items == 7
    assert result.shape == (1, 7)
    torch.testing.assert_close(model._base_model.encoder, old_encoder)


def test_allowlist_returns_global_catalog_rows(interactions, item_features):
    model = TEASERGDTrainer(_config(epochs=1)).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
    )

    result = model.predict_on_batch(
        interactions[:2],
        k=2,
        exclude_seen=False,
        candidate_ids=["F", "B", "D"],
    )

    assert set(result.cols.ravel().tolist()) <= {1, 3, 5}
    assert result.shape == (2, 6)


def test_align_source_is_sparse_and_predictions_match(interactions, item_features):
    model = TEASERGDTrainer(_config(epochs=1)).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
    )
    external_ids = list("GFCBADE")
    external = csr_matrix(
        np.column_stack(
            (
                np.zeros(interactions.shape[0], dtype=np.float32),
                interactions.toarray()[:, [5, 2, 1, 0, 3, 4]],
            )
        )
    )

    aligned = model.align_source(external, item_ids=external_ids)

    assert isinstance(aligned, csr_matrix)
    np.testing.assert_array_equal(aligned.toarray(), interactions.toarray())
    expected = model.predict_on_batch(interactions, k=3)
    actual = model.predict_on_batch(aligned, k=3)
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_train_item_subset_supports_cold_candidates(item_features):
    interactions = csr_matrix(
        np.array([[1, 0, 1, 0, 0, 0], [0, 0, 1, 0, 1, 0]], dtype=np.float32)
    )
    model = TEASERGDTrainer(_config(epochs=1)).fit(
        interactions,
        item_features,
        train_item_indices=[0, 2, 4],
    )

    result = model.predict_on_batch(interactions, k=3, exclude_seen=False)

    assert model._base_model.encoder.shape == (3, 3)
    assert result.shape == interactions.shape
    with pytest.raises(ValueError, match="no fitted encoder row"):
        model.predict_on_batch(csr_matrix([[0, 1, 0, 0, 0, 0]]), k=2)


def test_full_candidate_tensor_is_cached_and_invalidated(interactions, item_features):
    model = TEASERGDTrainer(_config(epochs=1)).fit(interactions, item_features)
    selection = model._resolve_candidate_selection(None)

    first = model._selection_tensor(selection)
    second = model._selection_tensor(selection)
    assert first is second

    model.update_candidates(
        item_ids=["new"],
        item_features=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )
    third = model._selection_tensor(model._resolve_candidate_selection(None))
    assert third is not first


def test_streaming_evaluation(interactions, item_features):
    model = TEASERGDTrainer(_config(epochs=1)).fit(interactions, item_features)
    targets = csr_matrix(
        np.array(
            [
                [0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )

    result = evaluate_recommender(
        model,
        source=interactions,
        targets=targets,
        metrics=[CalibratedRecall(2), NDCG(3)],
        batch_size=2,
    )

    assert set(result) == {"recall@2", "ndcg@3", "n_eval_users"}
    assert result["n_eval_users"] == 5


def test_protocols_and_configuration_validation():
    config = _config()
    model = TEASERGDTrainer(config)
    assert isinstance(model, Recommender)
    assert isinstance(model, ColdStartRecommender)
    assert config.encoder_init == "xavier"
    assert config.normalize_encoder is False

    for kwargs in (
        {"batch_size": 0},
        {"epochs": 0},
        {"max_output": 0},
        {"coefficient_regularization_samples": -1},
        {"lr": np.inf},
        {"l2_coefficients": -1},
        {"optimizer": "SGD"},
        {"loss": "mse"},
        {"encoder_init": "zeros"},
        {"normalize_encoder": "yes"},
    ):
        with pytest.raises(ValueError):
            TEASERGDConfig(**kwargs)
