from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.sparse import csc_matrix, csr_matrix

from compresso_recsys.evaluation import (
    evaluate_ranked_predictions,
    evaluate_recommender,
)
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import ELSA, ELSAConfig, ELSATrainer, Recommender
from compresso_recsys.models.elsa import _ELSAInteractionDataset


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 0, 0],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def source() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )


def _trainer(**overrides) -> ELSATrainer:
    defaults = {
        "latent_dim": 4,
        "batch_size": 2,
        "max_output": 6,
        "epochs": 2,
        "lr": 1e-2,
        "show_progress": False,
        "seed": 7,
    }
    defaults.update(overrides)
    return ELSATrainer(ELSAConfig(**defaults))


def test_elsa_forward_matches_restricted_reference():
    model = ELSA(input_dim=4, latent_dim=2, use_relu=False)
    with torch.no_grad():
        model.A.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
        )
    x = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    sources = torch.tensor([0, 2])
    candidates = torch.tensor([1, 2, 3])
    x_out = torch.tensor([[0.0, 1.0, 0.0], [0.0, 3.0, 0.0]])

    actual = model(
        x,
        sources=sources,
        candidates=candidates,
        x_out=x_out,
    )

    embeddings = F.normalize(model.A, dim=-1)
    expected = (x @ embeddings[sources]) @ embeddings[candidates].T - x_out
    torch.testing.assert_close(actual, expected)


def test_candidate_sampler_preserves_positives_and_samples_unique_negatives(
    interactions,
):
    dataset = _ELSAInteractionDataset(
        interactions,
        device=torch.device("cpu"),
        batch_size=2,
        shuffle=False,
        max_output=5,
        seed=3,
    )

    x, y, sources, candidates = dataset[0]

    assert x.is_sparse
    assert y.is_sparse
    assert candidates is not None
    assert sources.tolist() == [0, 1, 2]
    assert candidates.shape == (5,)
    assert len(set(candidates.tolist())) == 5
    assert candidates[: len(sources)].tolist() == sources.tolist()
    assert set(candidates[len(sources) :].tolist()).isdisjoint(sources.tolist())
    assert x.shape == (2, 3)
    assert y.shape == (2, 5)
    x_dense = x.to_dense()
    y_dense = y.to_dense()
    torch.testing.assert_close(x_dense, y_dense[:, : len(sources)])
    assert torch.count_nonzero(y_dense[:, len(sources) :]) == 0


def test_candidate_sampler_uses_full_catalog_when_unlimited(interactions):
    dataset = _ELSAInteractionDataset(
        interactions,
        device=torch.device("cpu"),
        batch_size=2,
        shuffle=False,
        max_output=None,
        seed=3,
    )

    x, y, sources, candidates = dataset[0]

    assert x.is_sparse
    assert y.is_sparse
    assert candidates is None
    assert y.shape == (2, interactions.shape[1])
    torch.testing.assert_close(
        y.to_dense(),
        torch.from_numpy(interactions[:2].toarray()),
    )
    torch.testing.assert_close(
        x.to_dense(),
        y.to_dense()[:, sources],
    )


def test_candidate_sampler_never_drops_positives_to_meet_budget(interactions):
    dataset = _ELSAInteractionDataset(
        interactions,
        device=torch.device("cpu"),
        batch_size=4,
        shuffle=False,
        max_output=2,
        seed=3,
    )

    _, y, sources, candidates = dataset[0]

    assert candidates is not None
    assert sources.tolist() == [0, 1, 2, 3, 4]
    assert candidates.tolist() == sources.tolist()
    assert y.shape == (4, 5)


def test_candidate_sampling_is_reproducible(interactions):
    kwargs = {
        "device": torch.device("cpu"),
        "batch_size": 2,
        "shuffle": False,
        "max_output": 5,
        "seed": 11,
    }
    first = _ELSAInteractionDataset(interactions, **kwargs)
    second = _ELSAInteractionDataset(interactions, **kwargs)

    first_candidates = first[0][3]
    second_candidates = second[0][3]
    assert first_candidates is not None
    assert second_candidates is not None
    torch.testing.assert_close(first_candidates, second_candidates)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("latent_dim", 0),
        ("batch_size", 0),
        ("max_output", 0),
        ("epochs", 0),
        ("lr", 0),
        ("lr", np.nan),
        ("weight_decay", -1),
        ("weight_decay", np.inf),
        ("optimizer", "SGD"),
    ],
)
def test_elsa_config_rejects_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        ELSAConfig(**{name: value})


def test_fit_records_history_and_returns_trainer(interactions):
    trainer = _trainer()

    returned = trainer.fit(interactions)

    assert returned is trainer
    assert trainer.is_built
    assert trainer.is_fitted
    assert trainer.input_dim == interactions.shape[1]
    assert len(trainer.history) == trainer.cfg.epochs
    assert all(np.isfinite(record["loss"]) for record in trainer.history)
    assert all(np.isfinite(record["cosine_loss"]) for record in trainer.history)


def test_build_is_idempotent_only_for_same_input_dimension():
    trainer = _trainer().build(8)
    model = trainer.elsa

    assert trainer.build(8) is trainer
    assert trainer.elsa is model
    with pytest.raises(ValueError, match="already built"):
        trainer.build(9)


def test_prediction_requires_fitted_model(source):
    with pytest.raises(RuntimeError, match="fitted"):
        _trainer().predict_on_batch(source, k=2)


def test_fit_and_prediction_validate_inputs(interactions, source):
    with pytest.raises(TypeError, match="csr_matrix"):
        _trainer().fit(csc_matrix(interactions))
    with pytest.raises(ValueError, match="at least one"):
        _trainer().fit(csr_matrix((0, interactions.shape[1])))
    with pytest.raises(ValueError, match="finite"):
        _trainer().fit(csr_matrix(np.array([[1.0, np.nan]])))

    trainer = _trainer(epochs=1).fit(interactions)
    with pytest.raises(TypeError, match="csr_matrix"):
        trainer.predict_on_batch(csc_matrix(source), k=2)
    with pytest.raises(ValueError, match="expected"):
        trainer.predict_on_batch(csr_matrix((1, source.shape[1] + 1)), k=1)
    with pytest.raises(ValueError, match=r"k must be in \[1"):
        trainer.predict_on_batch(source, k=0)
    with pytest.raises(ValueError, match="batch_size"):
        trainer.predict(source, k=2, batch_size=0)


def test_predict_matches_predict_on_batch_across_batch_sizes(interactions, source):
    trainer = _trainer(epochs=1).fit(interactions)

    expected = trainer.predict_on_batch(source, k=3)
    actual = trainer.predict(source, k=3, batch_size=2, show_progress=False)

    assert actual.shape == source.shape
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_predictions_exclude_seen_items_by_default(interactions, source):
    predictions = _trainer(epochs=1).fit(interactions).predict(
        source,
        k=3,
        show_progress=False,
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()).isdisjoint(source[row].indices)


def test_seen_item_mask_can_be_disabled(interactions, source):
    trainer = _trainer(epochs=1).fit(interactions)

    predictions = trainer.predict(
        source,
        k=source.shape[1],
        batch_size=2,
        exclude_seen=False,
        show_progress=False,
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()) == set(range(source.shape[1]))


def test_prediction_rejects_rows_with_too_few_unseen_items(interactions):
    trainer = _trainer(epochs=1).fit(interactions)
    source = csr_matrix(
        np.array([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="only 1 unseen"):
        trainer.predict_on_batch(source, k=2)


def test_empty_source_returns_empty_srp(interactions):
    trainer = _trainer(epochs=1).fit(interactions)
    source = csr_matrix((0, interactions.shape[1]), dtype=np.float32)

    predictions = trainer.predict(source, k=3, show_progress=False)

    assert predictions.shape == source.shape
    assert predictions.cols.shape == (0, 3)
    assert predictions.vals.shape == (0, 3)
    assert predictions.vals.dtype == torch.float32


def test_elsa_trainer_implements_recommender_protocol():
    assert isinstance(_trainer(), Recommender)


def test_streaming_evaluation_matches_materialized_predictions(
    interactions,
    source,
):
    trainer = _trainer(epochs=1).fit(interactions)
    targets = csr_matrix(
        (
            np.ones(source.shape[0], dtype=np.float32),
            (
                np.arange(source.shape[0]),
                np.array([1, 3, 2, 5]),
            ),
        ),
        shape=source.shape,
    )
    metrics = [CalibratedRecall([1, 3]), NDCG(3)]

    streamed = evaluate_recommender(
        trainer,
        source=source,
        targets=targets,
        metrics=metrics,
        batch_size=2,
    )
    materialized = evaluate_ranked_predictions(
        predictions=trainer.predict(
            source,
            k=3,
            batch_size=2,
            show_progress=False,
        ),
        targets=targets,
        metrics=metrics,
        batch_size=2,
    )

    assert streamed == materialized
