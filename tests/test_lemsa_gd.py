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
    LEMSAGD,
    LEMSAGDConfig,
    LEMSAGDTrainer,
    Recommender,
)
from compresso_recsys.models._batching import (
    LeaveOneOutInteractionBatchSampler,
    SymmetricInteractionBatchSampler,
)
from compresso_recsys.models.lemsa_gd import _cross_reconstruction_loss


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 1],
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


def _config(**overrides) -> LEMSAGDConfig:
    values = dict(
        epochs=2,
        batch_size=2,
        max_output=5,
        lr=0.01,
        show_progress=False,
        seed=7,
        use_relu=False,
    )
    values.update(overrides)
    return LEMSAGDConfig(**values)


def test_forward_matches_unconstrained_coefficients():
    features = torch.tensor(
        [[1.0, 0.0], [1.0, 2.0], [-1.0, 1.0]],
    )
    model = LEMSAGD(3, 2, use_relu=False)
    with torch.no_grad():
        model.encoder.copy_(
            torch.tensor([[0.5, 1.0], [2.0, -0.5], [1.5, 0.25]])
        )
    x = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]])

    actual = model(
        x,
        sources=torch.arange(3),
        candidate_features=features,
    )

    torch.testing.assert_close(actual, x @ model.encoder @ features.T)


def test_cross_reconstruction_ignores_source_coordinates_and_gradients():
    predictions = torch.tensor(
        [[0.4, -0.2, 10.0]],
        requires_grad=True,
    )
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    source = torch.tensor([[1.0]])
    source_positions = torch.tensor([2])

    first = _cross_reconstruction_loss(
        predictions,
        targets,
        source=source,
        source_positions=source_positions,
    )
    changed = _cross_reconstruction_loss(
        torch.tensor([[0.4, -0.2, -999.0]]),
        targets,
        source=source,
        source_positions=source_positions,
    )
    first.backward()

    torch.testing.assert_close(first.detach(), changed)
    assert predictions.grad[0, 2] == 0
    assert torch.count_nonzero(predictions.grad[0, :2]) > 0


def test_symmetric_sampler_produces_disjoint_nonempty_complete_views(
    interactions,
):
    sampler = SymmetricInteractionBatchSampler(
        interactions,
        device=torch.device("cpu"),
        batch_size=interactions.shape[0],
        shuffle=False,
        max_output=5,
        seed=3,
        split_probability=0.5,
    )

    batch = sampler[0]
    x = batch.x.to_dense()
    y = batch.y.to_dense()

    assert torch.all(x.sum(dim=1) > 0)
    assert torch.all(y.sum(dim=1) > 0)
    assert torch.count_nonzero(x * y) == 0
    expected_counts = torch.from_numpy(
        np.diff(interactions.indptr).astype(np.float32)
    )
    torch.testing.assert_close((x + y).sum(dim=1), expected_counts)
    assert batch.candidates is not None
    torch.testing.assert_close(
        batch.candidates[: batch.sources.numel()],
        batch.sources,
    )


def test_symmetric_sampler_is_reproducible_and_skips_short_histories():
    interactions = csr_matrix(
        np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 1, 0],
                [0, 1, 1, 1],
            ],
            dtype=np.float32,
        )
    )
    kwargs = dict(
        device=torch.device("cpu"),
        batch_size=2,
        shuffle=False,
        max_output=None,
        seed=11,
        split_probability=0.5,
    )
    first = SymmetricInteractionBatchSampler(interactions, **kwargs)
    second = SymmetricInteractionBatchSampler(interactions, **kwargs)

    assert len(first) == 1
    torch.testing.assert_close(first[0].x.to_dense(), second[0].x.to_dense())


def test_sampler_rejects_data_without_splittable_users():
    with pytest.raises(ValueError, match="two interactions"):
        SymmetricInteractionBatchSampler(
            csr_matrix(np.eye(3, dtype=np.float32)),
            device=torch.device("cpu"),
            batch_size=2,
            shuffle=False,
            max_output=None,
            seed=0,
            split_probability=0.5,
        )


@pytest.mark.parametrize("max_output", [None, 4])
def test_leave_one_out_sampler_visits_every_interaction_exactly_once(
    interactions,
    max_output,
):
    sampler = LeaveOneOutInteractionBatchSampler(
        interactions,
        device=torch.device("cpu"),
        batch_size=4,
        shuffle=False,
        max_output=max_output,
        seed=5,
    )
    visited: list[tuple[int, int]] = []

    for batch_index in range(len(sampler)):
        batch = sampler[batch_index]
        x = batch.x.to_dense()
        candidates = batch.candidates
        for row in range(x.shape[0]):
            user = int(batch.user_rows[row])
            target = int(batch.target_items[row])
            source_items = set(
                batch.sources[torch.nonzero(x[row], as_tuple=False).flatten()].tolist()
            )
            original = set(interactions[user].indices.tolist())

            assert target not in source_items
            assert source_items | {target} == original
            if candidates is None:
                assert int(batch.target_positions[row]) == target
            else:
                assert int(candidates[batch.target_positions[row]]) == target
            visited.append((user, target))

    expected = [
        (user, int(item))
        for user in range(interactions.shape[0])
        for item in interactions[user].indices
    ]
    assert sampler.n_examples == interactions.nnz
    assert sorted(visited) == sorted(expected)


def test_leave_one_out_sampler_skips_single_interaction_users():
    interactions = csr_matrix(
        np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    sampler = LeaveOneOutInteractionBatchSampler(
        interactions,
        device=torch.device("cpu"),
        batch_size=4,
        shuffle=False,
        max_output=None,
        seed=0,
    )

    batch = sampler[0]

    assert sampler.n_examples == 2
    assert set(batch.user_rows.tolist()) == {1}


@pytest.mark.parametrize("sparse", [False, True])
def test_fit_predict_and_candidate_catalog(
    interactions,
    item_features,
    sparse,
):
    features = csr_matrix(item_features) if sparse else item_features
    item_ids = np.array([f"item-{index}" for index in range(6)])
    metadata = pd.DataFrame({"item_id": item_ids, "title": list("ABCDEF")})
    model = LEMSAGDTrainer(_config()).fit(
        interactions,
        features,
        item_ids=item_ids,
        metadata=metadata,
        feature_space_id="test-features",
    )

    predictions = model.predict(interactions, k=2, batch_size=2)

    assert predictions.shape == interactions.shape
    assert len(model.history) == 2
    assert set(model.history[0]) == {
        "loss",
        "reconstruction",
        "encoder_l2",
        "epoch",
        "lr",
    }
    assert all(np.isfinite(record["loss"]) for record in model.history)
    assert model.candidates.metadata.equals(metadata)
    for row, seen in enumerate(interactions):
        assert set(predictions.cols[row].tolist()).isdisjoint(seen.indices)


def test_training_is_deterministic(interactions, item_features):
    first = LEMSAGDTrainer(_config()).fit(interactions, item_features)
    second = LEMSAGDTrainer(_config()).fit(interactions, item_features)

    torch.testing.assert_close(
        first._base_model.encoder,
        second._base_model.encoder,
    )
    assert first.history == second.history


def test_symmetric_training_mode_remains_available(interactions, item_features):
    model = LEMSAGDTrainer(
        _config(epochs=1, training_mode="symmetric")
    ).fit(interactions, item_features)

    assert len(model.history) == 1
    assert np.isfinite(model.history[0]["loss"])


def test_train_item_subset_supports_cold_candidates(item_features):
    interactions = csr_matrix(
        np.array(
            [
                [1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0],
            ],
            dtype=np.float32,
        )
    )
    model = LEMSAGDTrainer(_config(epochs=1)).fit(
        interactions,
        item_features,
        train_item_indices=[0, 2, 4],
    )

    result = model.predict_on_batch(interactions, k=3, exclude_seen=False)

    assert model._base_model.encoder.shape == (3, 3)
    assert result.shape == interactions.shape
    with pytest.raises(ValueError, match="no fitted encoder row"):
        model.predict_on_batch(csr_matrix([[0, 1, 0, 0, 0, 0]]), k=2)


def test_streaming_evaluation(interactions, item_features):
    model = LEMSAGDTrainer(_config(epochs=1)).fit(interactions, item_features)
    targets = csr_matrix(
        np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )

    result = evaluate_recommender(
        model,
        source=interactions,
        targets=targets,
        metrics=[CalibratedRecall(2), NDCG(2)],
        batch_size=2,
    )

    assert set(result) == {"recall@2", "ndcg@2", "n_eval_users"}
    assert result["n_eval_users"] == 5


def test_protocols_defaults_and_configuration_validation():
    config = LEMSAGDConfig()
    model = LEMSAGDTrainer(config)

    assert isinstance(model, Recommender)
    assert isinstance(model, ColdStartRecommender)
    assert config.split_probability == 0.5
    assert config.training_mode == "leave_one_out"
    assert config.include_popularity is False
    assert config.l2_encoder == 0.0

    for kwargs in (
        {"batch_size": 0},
        {"epochs": 0},
        {"max_output": 0},
        {"lr": np.inf},
        {"l2_encoder": -1},
        {"split_probability": 0},
        {"split_probability": 1},
        {"split_probability": np.nan},
        {"training_mode": "pairs"},
        {"optimizer": "SGD"},
        {"encoder_init": "zeros"},
        {"normalize_encoder": "yes"},
    ):
        with pytest.raises(ValueError):
            LEMSAGDConfig(**kwargs)
