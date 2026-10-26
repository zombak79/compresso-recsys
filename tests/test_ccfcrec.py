from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import NDCG, CalibratedRecall
from compresso_recsys.models import (
    CCFCRec,
    CCFCRecConfig,
    CCFCRecTrainer,
    MMConcatWrapper,
    MMConcatWrapperConfig,
)
from compresso_recsys.models.ccfcrec.trainer import _CooccurrenceSampler
from compresso_recsys.training import EarlyStopping, Validation, _ValidationMonitor

N_USERS = 30
N_WARM = 18
N_ITEMS = 24
N_FEATURES = 8


@pytest.fixture
def data() -> tuple[csr_matrix, np.ndarray]:
    """Two user groups, each liking one half of the items; items 18-23 are cold."""
    rng = np.random.default_rng(0)
    groups = np.arange(N_USERS) % 2
    item_groups = np.arange(N_ITEMS) % 2
    dense = np.zeros((N_USERS, N_ITEMS), dtype=np.float32)
    for user, group in enumerate(groups):
        liked = np.flatnonzero(item_groups[:N_WARM] == group)
        dense[user, rng.choice(liked, size=5, replace=False)] = 1.0
    centers = rng.normal(size=(2, N_FEATURES))
    features = centers[item_groups] + 0.1 * rng.normal(size=(N_ITEMS, N_FEATURES))
    return csr_matrix(dense), features.astype(np.float32)


def _config(**overrides) -> CCFCRecConfig:
    values = dict(
        embedding_dim=8,
        hidden_dim=16,
        n_positive_items=3,
        n_negative_items=4,
        batch_size=16,
        epochs=3,
        lr=1e-2,
        show_progress=False,
    )
    values.update(overrides)
    return CCFCRecConfig(**values)


def _fit(data, **overrides) -> CCFCRecTrainer:
    interactions, features = data
    return CCFCRecTrainer(_config(**overrides)).fit(interactions, features)


def _cold_targets() -> csr_matrix:
    rows = np.arange(N_USERS)
    return csr_matrix(
        (
            np.ones(N_USERS, dtype=np.float32),
            (rows, N_WARM + rows % (N_ITEMS - N_WARM)),
        ),
        shape=(N_USERS, N_ITEMS),
    )


@pytest.mark.parametrize(
    "field, value",
    [
        ("contrastive_weight", -0.1),
        ("weight_decay", -1.0),
        ("temperature", 0.0),
        ("n_positive_items", 0),
        ("n_negative_items", 0),
        ("lr", 0.0),
        ("inference", "id"),
        ("embedding_dim", 0),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        CCFCRecConfig(**{field: value})


def test_fit_validates_inputs(data):
    interactions, features = data
    trainer = CCFCRecTrainer(_config())
    with pytest.raises(ValueError, match="item_features has"):
        trainer.fit(interactions, features[:-1])
    with pytest.raises(ValueError, match="binary"):
        trainer.fit(interactions * 2, features)
    with pytest.raises(ValueError, match="outside train_item_indices"):
        trainer.fit(interactions, features, train_item_indices=np.arange(5))
    with pytest.raises(ValueError, match="needs validation"):
        trainer.fit(interactions, features, early_stopping=EarlyStopping("ndcg@5", 2))


def test_prediction_contract_needs_no_user_ids(data):
    interactions, _ = data
    model = _fit(data)
    assert getattr(model, "accepts_user_ids", False) is False
    predictions = model.predict_on_batch(interactions[:7], k=5)
    assert predictions.shape == (7, N_ITEMS)
    cols = predictions.cols.numpy()
    assert all(np.unique(row).size == row.size for row in cols)
    assert np.all(np.diff(predictions.vals.numpy(), axis=1) <= 0)
    seen = interactions[:7].toarray().astype(bool)
    assert not seen[np.arange(7)[:, None], cols].any()
    batched = model.predict(interactions, k=5, batch_size=4)
    whole = model.predict_on_batch(interactions, k=5)
    torch.testing.assert_close(batched.cols, whole.cols)
    torch.testing.assert_close(batched.vals, whole.vals)
    with pytest.raises(ValueError, match="source has"):
        model.predict_on_batch(interactions[:, :5], k=2)


def test_losses_match_explicit_formulas():
    torch.manual_seed(0)
    q = torch.randn(3, 4)
    positives = torch.randn(3, 2, 4)
    negatives = torch.randn(3, 5, 4)
    tau = 0.3
    loss = CCFCRec.contrastive_loss(q, positives, negatives, tau)
    pos = np.einsum("bd,bpd->bp", q.numpy(), positives.numpy()) / tau
    neg = np.einsum("bd,bnd->bn", q.numpy(), negatives.numpy()) / tau
    # Eq. 4 for every (item, positive) pair against the item's negatives.
    expected = -np.log(
        np.exp(pos) / (np.exp(pos) + np.exp(neg).sum(axis=1, keepdims=True))
    ).mean()
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5)

    item, plus, minus = torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4)
    bpr = CCFCRec.bpr_loss(item, plus, minus)
    margin = (item * plus).sum(1).numpy() - (item * minus).sum(1).numpy()
    np.testing.assert_allclose(
        float(bpr), np.mean(-np.log(1 / (1 + np.exp(-margin)))), rtol=1e-5
    )


def test_cooccurrence_sampler_follows_the_paper_definition():
    # 0-1 and 1-2 share a user (0 and 2 do not); 3-4 share one; item 5 is alone.
    training = csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    sampler = _CooccurrenceSampler(training, max_attempts=50)
    rng = np.random.default_rng(0)
    items = np.array([0, 1, 2, 3, 4] * 20)
    positives = sampler.positives(items, 4, rng)
    groups = {0: {1}, 1: {0, 2}, 2: {1}, 3: {4}, 4: {3}}
    for item, row in zip(items, positives):
        assert set(row.tolist()) <= groups[int(item)]
    assert {int(v) for v in positives[items == 1].ravel()} == {0, 2}
    negatives = sampler.negatives.sample(items, 6, rng)
    for item, row in zip(items, negatives):
        assert not set(row.tolist()) & (groups[int(item)] | {int(item)})
    lonely = sampler.positives(np.array([5]), 3, rng)
    assert lonely.tolist() == [[5, 5, 5]] and sampler.self_positives == 3


def test_user_embedding_is_the_sum_of_history_columns(data):
    interactions, features = data
    model = _fit(data)
    history = interactions[:4].toarray()
    expected = torch.from_numpy(history[:, :N_WARM]) @ model.module.user_columns
    torch.testing.assert_close(model.user_embeddings(interactions[:4]), expected)
    # Cold items in a history have no column and do not change the embedding.
    extended = interactions[:4].tolil()
    extended[:, N_WARM] = 1.0
    torch.testing.assert_close(
        model.user_embeddings(extended.tocsr()), expected, rtol=1e-5, atol=1e-6
    )


def test_loss_decreases_and_learns_groups(data):
    interactions, _ = data
    model = _fit(data, epochs=30)
    losses = [record["loss"] for record in model.history]
    assert losses[-1] < losses[0]
    predictions = model.predict_on_batch(interactions, k=3)
    groups = np.arange(N_USERS) % 2
    hits = (predictions.cols.numpy() % 2) == groups[:, None]
    assert hits.mean() > 0.7


@pytest.mark.parametrize("inference", ["content", "hybrid"])
def test_items_score_by_documented_vectors(data, inference):
    interactions, features = data
    model = _fit(data, inference=inference)
    module = model.module
    users = model.user_embeddings(interactions[:4])
    with torch.no_grad():
        expected_items = module.content(torch.from_numpy(features))
        if inference == "hybrid":
            training = torch.from_numpy(interactions.toarray()[:, :N_WARM].T)
            expected_items[:N_WARM] = training @ module.item_columns
    predictions = model.predict_on_batch(
        interactions[:4], k=N_ITEMS - 5, exclude_seen=False
    )
    gathered = torch.gather(users @ expected_items.T, 1, predictions.cols)
    torch.testing.assert_close(predictions.vals, gathered, rtol=1e-4, atol=1e-4)
    all_rows = model.predict_on_batch(interactions, k=N_ITEMS - 5, exclude_seen=False)
    assert (all_rows.cols.numpy() >= N_WARM).any()


def test_update_and_remove_candidates_without_refit(data):
    interactions, features = data
    model = _fit(data)
    version = model.candidates.snapshot().version
    new_features = features[:2] + 0.05
    model.update_candidates(item_ids=["new-a", "new-b"], item_features=new_features)
    catalog = model.candidates.snapshot()
    assert catalog.version > version and catalog.n_items == N_ITEMS + 2
    predictions = model.predict_on_batch(
        interactions[:3], k=2, candidate_ids=["new-a", "new-b"]
    )
    with torch.no_grad():
        content = model.module.content(torch.from_numpy(new_features))
    expected = model.user_embeddings(interactions[:3]) @ content.T
    new_rows = catalog.rows_for(["new-a", "new-b"])
    local = np.searchsorted(new_rows, predictions.cols.numpy())
    torch.testing.assert_close(
        predictions.vals,
        torch.gather(expected, 1, torch.from_numpy(local)),
        rtol=1e-5,
        atol=1e-5,
    )
    model.remove_candidates(["new-a"])
    assert "new-a" not in model.candidates.snapshot().item_ids.tolist()
    model.predict_on_batch(interactions[:3], k=5)


def test_any_history_is_scored(data):
    _, _ = data
    model = _fit(data)
    unseen = csr_matrix(
        (np.ones(3, dtype=np.float32), ([0, 0, 1], [0, 2, 1])), shape=(3, N_ITEMS)
    )
    predictions = model.predict_on_batch(unseen, k=4)
    assert torch.isfinite(predictions.vals).all()
    # An empty history has the zero embedding.
    assert torch.all(predictions.vals[2] == 0)


def test_save_load_round_trip(data, tmp_path):
    interactions, features = data
    model = _fit(data, inference="hybrid")
    model.update_candidates(item_ids=["late"], item_features=features[:1])
    expected = model.predict_on_batch(interactions, k=6)
    path = tmp_path / "ccfcrec.zip"
    model.save(path, include_optimizer=True)
    loaded = CCFCRecTrainer.load(path, load_optimizer=True)
    assert loaded.cfg == model.cfg
    np.testing.assert_array_equal(
        loaded.candidates.snapshot().item_ids, model.candidates.snapshot().item_ids
    )
    torch.testing.assert_close(loaded.module.cooccurrence, model.module.cooccurrence)
    actual = loaded.predict_on_batch(interactions, k=6)
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)
    assert loaded.history == model.history


def test_same_seed_is_deterministic(data):
    interactions, _ = data
    first = _fit(data)
    second = _fit(data)
    other = _fit(data, seed=1)
    for name, value in first.module.state_dict().items():
        torch.testing.assert_close(value, second.module.state_dict()[name])
    assert not torch.equal(first.module.user_columns, other.module.user_columns)
    left = first.predict_on_batch(interactions, k=5)
    right = second.predict_on_batch(interactions, k=5)
    torch.testing.assert_close(left.cols, right.cols)


def test_validation_matches_post_hoc_evaluation_and_does_not_change_training(
    data, tmp_path
):
    interactions, features = data
    validation = Validation(interactions, _cold_targets(), [NDCG(10)])
    plain = _fit(data, epochs=4, inference="hybrid")
    model = CCFCRecTrainer(_config(epochs=4, inference="hybrid")).fit(
        interactions, features, validation=validation
    )
    for key, value in plain.module.state_dict().items():
        torch.testing.assert_close(value, model.module.state_dict()[key])
    values = [record["val_ndcg@10"] for record in model.history]
    assert len(values) == 4 and len(set(values)) > 1
    # The last validation scored the final weights and co-occurrence buffer,
    # not a candidate matrix cached at an earlier epoch.
    path = tmp_path / "validated.zip"
    model.save(path)
    post_hoc = evaluate_recommender(
        CCFCRecTrainer.load(path),
        source=interactions,
        targets=_cold_targets(),
        metrics=[NDCG(10)],
    )
    assert values[-1] == pytest.approx(post_hoc.metrics["ndcg@10"])


def test_early_stopping_restores_best_weights_and_buffer(data, monkeypatch):
    interactions, features = data
    scores = iter([0.1, 0.4, 0.2, 0.3, 0.9, 0.9])
    states = []

    def scripted(self):
        # Score once, as a real validation would, so the candidate cache fills.
        self.owner._is_fitted = True
        self.owner.predict_on_batch(interactions[:3], k=4)
        self.owner._is_fitted = False
        states.append(
            {k: v.detach().clone() for k, v in self.owner.module.state_dict().items()}
        )
        return {"ndcg@10": next(scores)}

    monkeypatch.setattr(_ValidationMonitor, "_evaluate", scripted)
    model = CCFCRecTrainer(_config(epochs=6, inference="hybrid")).fit(
        interactions,
        features,
        validation=Validation(interactions, _cold_targets(), [NDCG(10)]),
        early_stopping=EarlyStopping("ndcg@10", patience=2),
    )
    assert len(model.history) == 4
    assert model.best_["epoch"] == 2
    for key, value in model.module.state_dict().items():
        torch.testing.assert_close(value, states[1][key])
    with torch.no_grad():
        expected = model.module.content(torch.from_numpy(features))
        expected[:N_WARM] = model.module.cooccurrence
    scores_now = model.predict_on_batch(interactions[:3], k=4, exclude_seen=False)
    direct = model.user_embeddings(interactions[:3]) @ expected.T
    torch.testing.assert_close(
        scores_now.vals, torch.gather(direct, 1, scores_now.cols), rtol=1e-5, atol=1e-5
    )


@pytest.mark.parametrize("missing", ["zero", "mean"])
def test_mm_concat_handles_missing_modalities(data, missing, tmp_path):
    interactions, features = data
    modalities = {"text": features, "image": features[:, :5] * 2}
    available = np.ones(N_ITEMS, dtype=bool)
    available[[3, 20]] = False
    wrapper = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=CCFCRecTrainer,
            model_config=_config(),
            modalities=["text", "image"],
            missing=missing,
            normalize=True,
        )
    )
    wrapper.fit(
        interactions, item_features=modalities, modality_masks={"image": available}
    )
    assert wrapper.inner_model.module.n_features == N_FEATURES + 5
    assert wrapper.accepts_user_ids is False
    evaluated = evaluate_recommender(
        wrapper,
        source=interactions,
        targets=_cold_targets(),
        metrics=[CalibratedRecall(5)],
    )
    # Not a user-identity model: the evaluator records no identity channel.
    assert "user_identity" not in evaluated.metadata
    predictions = wrapper.predict_on_batch(interactions[:4], k=5)
    path = tmp_path / "wrapped.zip"
    wrapper.save(path)
    again = MMConcatWrapper.load(path).predict_on_batch(interactions[:4], k=5)
    torch.testing.assert_close(again.cols, predictions.cols)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_smoke(data):
    interactions, _ = data
    model = _fit(data, device="cuda", epochs=1)
    predictions = model.predict_on_batch(interactions[:4], k=3)
    assert predictions.cols.device.type == "cuda"
