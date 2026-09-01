from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models import MultVAE, MultVAEConfig, MultVAETrainer


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


def _config(**changes) -> MultVAEConfig:
    values = {
        "latent_dim": 3,
        "hidden_dim": 5,
        "dropout": 0.2,
        "epochs": 2,
        "batch_size": 2,
        "lr": 1e-2,
        "kl_cap": 0.2,
        "kl_anneal_steps": 3,
        "show_progress": False,
        "seed": 7,
    }
    values.update(changes)
    return MultVAEConfig(**values)


def test_mult_vae_module_shapes_and_sampling_modes():
    model = MultVAE(n_items=6, latent_dim=3, hidden_dim=5, dropout=0.0)
    inputs = torch.eye(6)[:2]

    model.train()
    sampled_first, mean, log_variance = model(inputs)
    sampled_second, _, _ = model(inputs)
    deterministic_first, _, _ = model(inputs, sample=False)
    deterministic_second, _, _ = model(inputs, sample=False)

    assert sampled_first.shape == mean.shape[:1] + (6,)
    assert mean.shape == log_variance.shape == (2, 3)
    assert not torch.equal(sampled_first, sampled_second)
    torch.testing.assert_close(deterministic_first, deterministic_second)


def test_mult_vae_fit_records_annealed_loss_components(interactions):
    trainer = MultVAETrainer(_config()).fit(interactions)

    assert trainer.is_fitted
    assert trainer.n_items == interactions.shape[1]
    assert trainer._updates == 6
    assert len(trainer.history) == 2
    assert trainer.history[0]["kl_weight"] < trainer.history[1]["kl_weight"]
    assert trainer.history[1]["kl_weight"] == pytest.approx(0.2)
    for entry in trainer.history:
        assert set(entry) == {
            "epoch",
            "loss",
            "reconstruction_loss",
            "kl_loss",
            "kl_weight",
        }
        assert all(np.isfinite(value) for value in entry.values())


def test_mult_vae_zero_anneal_uses_cap_immediately(interactions):
    trainer = MultVAETrainer(_config(epochs=1, kl_anneal_steps=0)).fit(interactions)

    assert trainer.history[0]["kl_weight"] == pytest.approx(0.2)


def test_mult_vae_prediction_is_deterministic_and_filters_candidates(interactions):
    item_ids = np.array([f"item-{i}" for i in range(interactions.shape[1])])
    trainer = MultVAETrainer(_config()).fit(interactions, item_ids=item_ids)
    source = interactions[:2]

    first = trainer.predict_on_batch(
        source,
        k=2,
        candidate_ids=["item-0", "item-2", "item-3", "item-5"],
    )
    second = trainer.predict_on_batch(
        source,
        k=2,
        candidate_ids=["item-0", "item-2", "item-3", "item-5"],
    )

    torch.testing.assert_close(first.cols, second.cols)
    torch.testing.assert_close(first.vals, second.vals)
    for row in range(source.shape[0]):
        assert set(first.cols[row].tolist()) <= {0, 2, 3, 5}
        assert set(first.cols[row].tolist()).isdisjoint(source[row].indices)


def test_mult_vae_round_trip_with_optimizer(tmp_path, interactions):
    trainer = MultVAETrainer(_config()).fit(interactions)
    source = interactions[:2]
    before = trainer.predict_on_batch(source, k=2)
    path = tmp_path / "mult-vae.zip"

    trainer.save(path, include_optimizer=True)
    restored = MultVAETrainer.load(path, load_optimizer=True)
    after = restored.predict_on_batch(source, k=2)

    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)
    assert restored.history == trainer.history
    assert restored._updates == trainer._updates
    assert restored.optimizer is not None
    assert restored.optimizer.state_dict()["state"]
    assert restored.to("cpu") is restored


def test_mult_vae_refit_rebuilds_catalog(interactions):
    trainer = MultVAETrainer(_config()).fit(interactions)
    wider = csr_matrix(np.pad(interactions.toarray(), ((0, 0), (0, 1))))

    trainer.fit(wider)

    assert trainer.n_items == wider.shape[1]
    assert trainer.model.n_items == wider.shape[1]


def test_mult_vae_rejects_invalid_training_data():
    trainer = MultVAETrainer(_config())
    with pytest.raises(ValueError, match="nonempty user"):
        trainer.fit(csr_matrix((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="nonnegative"):
        trainer.fit(csr_matrix(np.array([[1, -1, 0]], dtype=np.float32)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"latent_dim": 0}, "latent_dim"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"lr": 0.0}, "lr"),
        ({"weight_decay": -1.0}, "weight_decay"),
        ({"kl_cap": -0.1}, "kl_cap"),
        ({"kl_anneal_steps": -1}, "kl_anneal_steps"),
    ],
)
def test_mult_vae_config_validation(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _config(**changes)
