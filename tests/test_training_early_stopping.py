"""Validation between epochs and early stopping on it."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.metrics import NDCG
from compresso_recsys.models import (
    ItemTokenizer,
    MultDAEConfig,
    MultDAETrainer,
    MultVAEConfig,
    MultVAETrainer,
    SequenceBatcher,
    SimpleRNNConfig,
    SimpleRNNTrainer,
    TEASERGDConfig,
    TEASERGDTrainer,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.training import EarlyStopping, Validation, _ValidationMonitor

N_USERS, N_ITEMS = 40, 12


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    seen = rng.random((N_USERS, N_ITEMS)) < 0.35
    seen[np.arange(N_USERS), np.arange(N_USERS) % N_ITEMS] = True
    relevant = ~seen & (rng.random((N_USERS, N_ITEMS)) < 0.3)
    relevant[np.arange(N_USERS), (np.arange(N_USERS) + 1) % N_ITEMS] = True
    seen &= ~relevant
    return csr_matrix(seen.astype(np.float32)), csr_matrix(relevant.astype(np.float32))


def _dae(epochs: int = 6) -> MultDAETrainer:
    return MultDAETrainer(MultDAEConfig(latent_dim=8, epochs=epochs, batch_size=16,
                                        show_progress=False, seed=0))


def _validation(data, **kwargs) -> Validation:
    x, targets = data
    return Validation(x, targets, [NDCG(5)], sample_ids=np.arange(N_USERS), **kwargs)


@pytest.fixture
def scripted(monkeypatch):
    """Replace the evaluation by a fixed sequence of scores; keep each epoch's weights."""
    seen: dict[str, list] = {"states": []}

    def install(scores):
        it = iter(scores)

        def fake(self):
            module = self.owner.model
            seen["states"].append({k: v.detach().clone() for k, v in module.state_dict().items()})
            return {"ndcg@5": float(next(it))}

        monkeypatch.setattr(_ValidationMonitor, "_evaluate", fake)
        return seen

    return install


def test_stops_after_patience_and_restores_the_best_weights(data, scripted):
    seen = scripted([0.1, 0.3, 0.2, 0.25, 0.2, 0.9])
    model = _dae(epochs=6).fit(data[0], validation=_validation(data),
                               early_stopping=EarlyStopping("ndcg@5", patience=2))
    assert len(model.history) == 4  # best at epoch 2, then two without improvement
    assert [r["val_ndcg@5"] for r in model.history] == [0.1, 0.3, 0.2, 0.25]
    assert model.best_ == {"epoch": 2, "val_ndcg@5": 0.3}
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, seen["states"][1][key])
    assert model.is_fitted


def test_restore_best_false_keeps_the_last_weights(data, scripted):
    seen = scripted([0.3, 0.1, 0.1])
    model = _dae().fit(data[0], validation=_validation(data),
                       early_stopping=EarlyStopping("ndcg@5", patience=2, restore_best=False))
    assert len(model.history) == 3
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, seen["states"][-1][key])


def test_the_last_epoch_restores_the_best_even_without_a_stop(data, scripted):
    seen = scripted([0.5, 0.1, 0.2])
    model = _dae(epochs=3).fit(data[0], validation=_validation(data),
                               early_stopping=EarlyStopping("ndcg@5", patience=5))
    assert len(model.history) == 3 and model.best_["epoch"] == 1
    for key, value in model.model.state_dict().items():
        torch.testing.assert_close(value, seen["states"][0][key])


def test_min_epochs_delays_the_patience_count(data, scripted):
    scripted([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    model = _dae().fit(data[0], validation=_validation(data),
                       early_stopping=EarlyStopping("ndcg@5", patience=1, min_epochs=3))
    assert len(model.history) == 4


def test_min_mode_and_min_delta(data, scripted):
    scripted([1.0, 0.95, 0.5, 0.49, 0.48])
    model = _dae().fit(data[0], validation=_validation(data),
                       early_stopping=EarlyStopping("ndcg@5", patience=2, mode="min", min_delta=0.05))
    assert model.best_["epoch"] == 3  # 0.95 is not 0.05 better than 1.0; 0.49, 0.48 are not either
    assert len(model.history) == 5


def test_every_validates_on_schedule_and_on_the_last_epoch(data, scripted):
    scripted([0.1, 0.2, 0.3])
    model = _dae(epochs=5).fit(data[0], validation=_validation(data, every=2))
    validated = [int(r["epoch"]) for r in model.history if "val_ndcg@5" in r]
    assert validated == [2, 4, 5]
    assert model.best_ is None  # validation alone records, it does not select


def test_validation_does_not_change_training(data):
    x, _ = data
    plain = _dae(epochs=3).fit(x)
    validated = _dae(epochs=3).fit(x, validation=_validation(data))
    for key, value in plain.model.state_dict().items():
        torch.testing.assert_close(validated.model.state_dict()[key], value)
    assert [r["reconstruction_loss"] for r in validated.history] == [
        r["reconstruction_loss"] for r in plain.history
    ]
    assert all(np.isfinite(r["val_ndcg@5"]) for r in validated.history)
    assert "val_ndcg@5" not in plain.history[0] and plain.best_ is None


def test_early_stopping_needs_validation(data):
    with pytest.raises(ValueError, match="needs validation"):
        _dae().fit(data[0], early_stopping=EarlyStopping("ndcg@5", patience=1))


def test_unknown_metric_is_named(data):
    with pytest.raises(KeyError, match="recall@5"):
        _dae().fit(data[0], validation=_validation(data),
                   early_stopping=EarlyStopping("recall@5", patience=1))


@pytest.mark.parametrize(
    "build",
    [
        lambda: Validation(None, None, every=0),
        lambda: Validation(None, None, batch_size=0),
        lambda: EarlyStopping("ndcg@5", patience=0),
        lambda: EarlyStopping("ndcg@5", patience=1, mode="best"),
        lambda: EarlyStopping("ndcg@5", patience=1, min_delta=-1.0),
        lambda: EarlyStopping("ndcg@5", patience=1, min_epochs=-1),
    ],
)
def test_arguments_are_checked(build):
    with pytest.raises(ValueError):
        build()


def _assert_real_validation(model, epochs):
    assert len(model.history) <= epochs
    assert all(np.isfinite(r["val_ndcg@5"]) for r in model.history)
    assert model.best_ is not None and 1 <= model.best_["epoch"] <= len(model.history)
    assert model.is_fitted


def test_autoencoder_scores_mid_fit(data):
    model = MultVAETrainer(MultVAEConfig(latent_dim=8, hidden_dim=16, epochs=4, batch_size=16,
                                         show_progress=False, seed=0))
    model.fit(data[0], validation=_validation(data),
              early_stopping=EarlyStopping("ndcg@5", patience=2))
    _assert_real_validation(model, 4)


def test_sequential_scores_mid_fit(data):
    x, targets = data
    sequences = ItemSequences(values=x.indices.astype(np.int64),
                              indptr=x.indptr.astype(np.int64), n_items=N_ITEMS)
    model = SimpleRNNTrainer(
        SimpleRNNConfig(embedding_dim=8, hidden_dim=16, epochs=4, batch_size=16,
                        show_progress=False, seed=0),
        SequenceBatcher(ItemTokenizer(N_ITEMS), max_length=None),
    )
    model.fit(sequences, validation=Validation(sequences, targets, [NDCG(5)],
                                               sample_ids=np.arange(N_USERS)),
              early_stopping=EarlyStopping("ndcg@5", patience=2))
    _assert_real_validation(model, 4)


def test_teaser_gd_scores_mid_fit(data):
    x, targets = data
    features = np.random.default_rng(1).random((N_ITEMS, 4)).astype(np.float32)
    model = TEASERGDTrainer(TEASERGDConfig(epochs=3, batch_size=8, max_output=4, lr=0.01,
                                           show_progress=False, include_popularity=False,
                                           coefficient_regularization_samples=16, seed=3))
    model.fit(x, features, validation=_validation(data),
              early_stopping=EarlyStopping("ndcg@5", patience=2))
    _assert_real_validation(model, 3)


def test_elsa_scores_mid_fit_and_stops(data):
    from compresso_recsys.models import ELSAConfig, ELSATrainer

    def elsa():
        return ELSATrainer(ELSAConfig(latent_dim=8, epochs=6, batch_size=16, show_progress=False, seed=0))

    real = elsa().fit(data[0], validation=_validation(data),
                      early_stopping=EarlyStopping("ndcg@5", patience=2))
    _assert_real_validation(real, 6)
    plain = elsa().fit(data[0])
    validated = elsa().fit(data[0], validation=_validation(data))
    for key, value in plain._checkpoint_module().state_dict().items():
        torch.testing.assert_close(validated._checkpoint_module().state_dict()[key], value)


def test_elsa_restores_the_best_weights(data, monkeypatch):
    from compresso_recsys.models import ELSAConfig, ELSATrainer

    scores = iter([0.1, 0.4, 0.2, 0.3])
    states = []

    def fake(self):
        states.append({k: v.detach().clone() for k, v in self.owner._checkpoint_module().state_dict().items()})
        return {"ndcg@5": next(scores)}

    monkeypatch.setattr(_ValidationMonitor, "_evaluate", fake)
    model = ELSATrainer(ELSAConfig(latent_dim=8, epochs=6, batch_size=16, show_progress=False, seed=0))
    model.fit(data[0], validation=_validation(data), early_stopping=EarlyStopping("ndcg@5", patience=2))
    assert len(model.history) == 4 and model.best_["epoch"] == 2
    for key, value in model._checkpoint_module().state_dict().items():
        torch.testing.assert_close(value, states[1][key])
