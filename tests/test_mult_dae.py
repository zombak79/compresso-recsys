from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models import MultDAE, MultDAEConfig, MultDAETrainer


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


def _config(**changes) -> MultDAEConfig:
    values = {
        "latent_dim": 4,
        "dropout": 0.2,
        "epochs": 2,
        "batch_size": 2,
        "lr": 1e-2,
        "show_progress": False,
        "seed": 7,
    }
    values.update(changes)
    return MultDAEConfig(**values)


def test_mult_dae_module_shape_and_eval_determinism():
    model = MultDAE(n_items=6, latent_dim=3, dropout=0.5).eval()
    inputs = torch.eye(6)[:2]

    first = model(inputs)
    second = model(inputs)

    assert first.shape == (2, 6)
    torch.testing.assert_close(first, second)


def test_mult_dae_fit_records_history_and_rebuilds_catalog(interactions):
    trainer = MultDAETrainer(_config()).fit(interactions)

    assert trainer.is_fitted
    assert trainer.n_items == interactions.shape[1]
    assert len(trainer.history) == 2
    assert all(
        set(entry) == {"epoch", "reconstruction_loss"}
        and np.isfinite(entry["reconstruction_loss"])
        for entry in trainer.history
    )

    wider = csr_matrix(np.pad(interactions.toarray(), ((0, 0), (0, 1))))
    trainer.fit(wider)
    assert trainer.n_items == wider.shape[1]
    assert trainer.model.n_items == wider.shape[1]


def test_mult_dae_failed_initial_fit_leaves_trainer_unfitted(
    interactions,
    monkeypatch,
    tmp_path,
):
    trainer = MultDAETrainer(_config())

    def fail_train_step(target):
        del target
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(trainer, "_train_step", fail_train_step)

    with pytest.raises(RuntimeError, match="injected training failure"):
        trainer.fit(interactions)

    assert not trainer.is_fitted
    assert trainer.n_items is None
    assert trainer.model is None
    assert trainer.optimizer is None
    assert trainer.history == []
    assert trainer.training_data_preloaded_ is None
    assert "_item_vocabulary" not in trainer.__dict__
    with pytest.raises(RuntimeError, match="must be fitted"):
        trainer.predict_on_batch(interactions[:1], k=1)
    with pytest.raises(RuntimeError, match="must be fitted"):
        trainer.save(tmp_path / "failed-mult-dae.zip")


def test_mult_dae_failed_refit_preserves_previous_model(
    interactions,
    tmp_path,
):
    item_ids = np.array([f"item-{i}" for i in range(interactions.shape[1])])
    trainer = MultDAETrainer(_config()).fit(interactions, item_ids=item_ids)
    before = trainer.predict_on_batch(interactions[:2], k=2)
    previous_model = trainer.model
    previous_optimizer = trainer.optimizer
    previous_history = trainer.history
    previous_preloaded = trainer.training_data_preloaded_
    wider = csr_matrix(np.pad(interactions.toarray(), ((0, 0), (0, 1))))

    with pytest.raises(ValueError, match="item_ids has 1 entries"):
        trainer.fit(wider, item_ids=["invalid"])

    assert trainer.is_fitted
    assert trainer.n_items == interactions.shape[1]
    assert trainer.model is previous_model
    assert trainer.optimizer is previous_optimizer
    assert trainer.history is previous_history
    assert trainer.training_data_preloaded_ is previous_preloaded
    np.testing.assert_array_equal(trainer.source_item_ids, item_ids)
    after = trainer.predict_on_batch(interactions[:2], k=2)
    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)

    path = tmp_path / "preserved-mult-dae.zip"
    trainer.save(path)
    restored = MultDAETrainer.load(path)
    assert restored.is_fitted
    assert restored.n_items == interactions.shape[1]
    np.testing.assert_array_equal(restored.source_item_ids, item_ids)


def test_mult_dae_prediction_excludes_seen_and_selects_candidates(interactions):
    item_ids = np.array([f"item-{i}" for i in range(interactions.shape[1])])
    trainer = MultDAETrainer(_config()).fit(interactions, item_ids=item_ids)
    source = interactions[:2]

    predictions = trainer.predict_on_batch(
        source,
        k=2,
        candidate_ids=["item-0", "item-2", "item-3", "item-5"],
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()) <= {0, 2, 3, 5}
        assert set(predictions.cols[row].tolist()).isdisjoint(source[row].indices)


def test_mult_dae_round_trip_with_optimizer(tmp_path, interactions):
    trainer = MultDAETrainer(_config()).fit(interactions)
    source = interactions[:2]
    before = trainer.predict_on_batch(source, k=2)
    path = tmp_path / "mult-dae.zip"

    trainer.save(path, include_optimizer=True)
    restored = MultDAETrainer.load(path, load_optimizer=True)
    after = restored.predict_on_batch(source, k=2)

    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)
    assert restored.history == trainer.history
    assert restored.optimizer is not None
    assert restored.optimizer.state_dict()["state"]
    assert restored.to("cpu") is restored


def test_mult_dae_l2_regularizes_only_weight_matrices():
    trainer = MultDAETrainer(_config(l2_reg=2e-5))
    trainer._n_items = 6
    trainer.model = trainer._build_model()
    trainer._build_checkpoint_optimizer()

    assert trainer.optimizer is not None
    weight_group, bias_group = trainer.optimizer.param_groups
    assert weight_group["weight_decay"] == pytest.approx(4e-5)
    assert bias_group["weight_decay"] == 0.0
    assert {id(parameter) for parameter in weight_group["params"]} == {
        id(trainer.model.encoder.weight),
        id(trainer.model.decoder.weight),
    }
    assert {id(parameter) for parameter in bias_group["params"]} == {
        id(trainer.model.encoder.bias),
        id(trainer.model.decoder.bias),
    }


def test_mult_dae_preloaded_and_streamed_training_match(interactions):
    streamed = MultDAETrainer(
        _config(dropout=0.0, preload_training_data=False)
    ).fit(interactions)
    preloaded = MultDAETrainer(
        _config(dropout=0.0, preload_training_data=True)
    ).fit(interactions)

    assert streamed.training_data_preloaded_ is False
    assert preloaded.training_data_preloaded_ is True
    assert streamed.history == pytest.approx(preloaded.history)
    for streamed_parameter, preloaded_parameter in zip(
        streamed.model.parameters(), preloaded.model.parameters(), strict=True
    ):
        torch.testing.assert_close(streamed_parameter, preloaded_parameter)


def test_mult_dae_rejects_invalid_training_data():
    trainer = MultDAETrainer(_config())
    with pytest.raises(ValueError, match="nonempty user"):
        trainer.fit(csr_matrix((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="nonnegative"):
        trainer.fit(csr_matrix(np.array([[1, -1, 0]], dtype=np.float32)))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"latent_dim": 0}, "latent_dim"),
        ({"dropout": 1.0}, "dropout"),
        ({"epochs": 0}, "epochs"),
        ({"batch_size": 0}, "batch_size"),
        ({"lr": 0.0}, "lr"),
        ({"l2_reg": -1.0}, "l2_reg"),
        ({"preload_training_data": None}, "preload_training_data"),
    ],
)
def test_mult_dae_config_validation(changes, message):
    with pytest.raises((TypeError, ValueError), match=message):
        _config(**changes)
