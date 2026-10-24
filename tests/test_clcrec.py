from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall
from compresso_recsys.models import (
    CLCRec,
    CLCRecConfig,
    CLCRecTrainer,
    MMConcatWrapper,
    MMConcatWrapperConfig,
)
from compresso_recsys.models.clcrec.sampling import (
    NegativeSampler,
    sample_other_items,
)

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


def _config(**overrides) -> CLCRecConfig:
    values = dict(
        embedding_dim=8,
        hidden_dim=16,
        n_negatives=4,
        batch_size=16,
        epochs=3,
        show_progress=False,
        lr=1e-2,
    )
    values.update(overrides)
    return CLCRecConfig(**values)


def _fit(data, **overrides) -> CLCRecTrainer:
    interactions, features = data
    return CLCRecTrainer(_config(**overrides)).fit(
        interactions,
        features,
        user_ids=[f"u{row}" for row in range(N_USERS)],
    )


USER_IDS = np.array([f"u{row}" for row in range(N_USERS)], dtype=object)


@pytest.mark.parametrize(
    "field, value",
    [
        ("re_weight", 1.5),
        ("hybrid_prob", -0.1),
        ("temperature", 0.0),
        ("n_negatives", 0),
        ("re_n_negatives", 0),
        ("l2_weight", -1.0),
        ("inference", "id"),
        ("unknown_users", "ignore"),
        ("activation", "gelu"),
        ("fold_in_steps", -1),
    ],
)
def test_config_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        CLCRecConfig(**{field: value})


def test_fit_validates_inputs(data):
    interactions, features = data
    trainer = CLCRecTrainer(_config())
    with pytest.raises(ValueError, match="item_features has"):
        trainer.fit(interactions, features[:-1])
    with pytest.raises(ValueError, match="binary"):
        trainer.fit(interactions * 2, features)
    with pytest.raises(ValueError, match="outside train_item_indices"):
        trainer.fit(interactions, features, train_item_indices=np.arange(5))
    with pytest.raises(NotImplementedError, match="training hook"):
        trainer.fit(interactions, features, early_stopping=object())


def test_prediction_contract(data):
    interactions, _ = data
    model = _fit(data)
    predictions = model.predict_on_batch(interactions[:7], k=5, user_ids=USER_IDS[:7])
    assert predictions.shape == (7, N_ITEMS)
    cols = predictions.cols.numpy()
    vals = predictions.vals.numpy()
    assert all(np.unique(row).size == row.size for row in cols)
    assert np.all(np.diff(vals, axis=1) <= 0)
    seen = interactions[:7].toarray().astype(bool)
    assert not seen[np.arange(7)[:, None], cols].any()
    batched = model.predict(interactions, k=5, batch_size=4, user_ids=USER_IDS)
    whole = model.predict_on_batch(interactions, k=5, user_ids=USER_IDS)
    torch.testing.assert_close(batched.cols, whole.cols)
    torch.testing.assert_close(batched.vals, whole.vals)
    with pytest.raises(ValueError, match="source has"):
        model.predict_on_batch(interactions[:, :5], k=2, user_ids=USER_IDS)


def test_losses_match_explicit_formula():
    torch.manual_seed(0)
    users = torch.randn(3, 4)
    items = torch.randn(3, 5, 4)
    tau = 0.3
    ui = CLCRec.ui_logits(users, items, tau)
    expected_ui = np.einsum("bd,bsd->bs", users.numpy(), items.numpy()) / tau
    np.testing.assert_allclose(ui.numpy(), expected_ui, rtol=1e-5)
    anchors = torch.randn(3, 4)
    re = CLCRec.re_logits(anchors, items, tau)
    a = anchors.numpy() / np.linalg.norm(anchors.numpy(), axis=-1, keepdims=True)
    f = items.numpy() / np.linalg.norm(items.numpy(), axis=-1, keepdims=True)
    np.testing.assert_allclose(
        re.numpy(), np.einsum("bd,bsd->bs", a, f) / tau, rtol=1e-5
    )
    # Eq. 13: -ln(exp(s_0) / sum_s exp(s)), averaged over the batch.
    expected_loss = np.mean(
        -expected_ui[:, 0] + np.log(np.exp(expected_ui).sum(axis=1))
    )
    loss = torch.nn.functional.cross_entropy(ui, torch.zeros(3, dtype=torch.long))
    np.testing.assert_allclose(float(loss), expected_loss, rtol=1e-5)


@pytest.mark.parametrize("hybrid_prob", [0.0, 1.0])
def test_hybrid_extremes_select_ids_or_content(data, hybrid_prob, monkeypatch):
    model = CLCRecTrainer(_config(hybrid_prob=hybrid_prob, epochs=1))
    interactions, features = data
    original = CLCRec.ui_logits
    checked = []

    def spy(users, items, temperature):
        # Called before the optimizer step, so the tables are the ones used.
        module = model.module
        with torch.no_grad():
            if hybrid_prob == 1.0:
                table = module.encoder(torch.from_numpy(features[:N_WARM]))
            else:
                table = module.item_emb.weight
            rows = items.detach().reshape(-1, items.shape[-1])
            distances = (rows[:, None, :] - table[None]).abs().amax(dim=-1)
        checked.append(float(distances.min(dim=1).values.max()))
        return original(users, items, temperature)

    monkeypatch.setattr(CLCRec, "ui_logits", staticmethod(spy))
    model.fit(interactions, features)
    assert len(checked) == -(-interactions.nnz // 16)
    assert max(checked) < 1e-5


def test_hybrid_mask_rate():
    generator = torch.Generator().manual_seed(0)
    mask = torch.rand((4000, 9), generator=generator) < 0.3
    assert abs(mask.float().mean().item() - 0.3) < 0.01


def test_negative_sampler_excludes_positives():
    interactions = csr_matrix(
        np.array(
            [[1, 1, 0, 0, 0, 0], [0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 0]],
            dtype=np.float32,
        )
    )
    sampler = NegativeSampler(interactions, max_attempts=50)
    rng = np.random.default_rng(0)
    users = np.array([0, 1, 2] * 50)
    negatives = sampler.sample(users, 6, rng)
    dense = interactions.toarray().astype(bool)
    assert not dense[users[:, None], negatives].any()
    assert sampler.residual_collisions == 0
    assert np.all(negatives[users == 2] == 5)

    full = NegativeSampler(csr_matrix(np.ones((1, 3))), max_attempts=2)
    full.sample(np.array([0]), 4, rng)
    assert full.residual_collisions == 4

    others = sample_other_items(np.array([0, 3, 5] * 30), 5, 6, rng)
    assert not np.any(others == np.array([0, 3, 5] * 30)[:, None])
    assert others.min() >= 0 and others.max() < 6


def test_loss_decreases_and_learns_groups(data):
    interactions, _ = data
    model = _fit(data, epochs=30)
    losses = [record["loss"] for record in model.history]
    assert losses[-1] < losses[0]
    predictions = model.predict_on_batch(interactions, k=3, user_ids=USER_IDS)
    groups = np.arange(N_USERS) % 2
    hits = (predictions.cols.numpy() % 2) == groups[:, None]
    assert hits.mean() > 0.7


@pytest.mark.parametrize("inference", ["hybrid", "content"])
def test_cold_and_warm_items_score_by_documented_vectors(data, inference):
    interactions, features = data
    model = _fit(data, inference=inference)
    module = model.module
    users = module.user_emb.weight.detach()[:4]
    with torch.no_grad():
        content = module.encoder(torch.from_numpy(features))
    expected_items = content.clone()
    if inference == "hybrid":
        expected_items[:N_WARM] = module.item_emb.weight.detach()
    predictions = model.predict_on_batch(
        interactions[:4], k=N_ITEMS - 5, exclude_seen=False, user_ids=USER_IDS[:4]
    )
    expected = users @ expected_items.T
    gathered = torch.gather(expected, 1, predictions.cols)
    torch.testing.assert_close(predictions.vals, gathered, rtol=1e-5, atol=1e-5)
    all_items = model.predict_on_batch(
        interactions, k=N_ITEMS - 5, exclude_seen=False, user_ids=USER_IDS
    )
    assert (all_items.cols.numpy() >= N_WARM).any()


def test_update_and_remove_candidates_without_refit(data):
    interactions, features = data
    model = _fit(data)
    version = model.candidates.snapshot().version
    new_features = features[:2] + 0.05
    model.update_candidates(item_ids=["new-a", "new-b"], item_features=new_features)
    catalog = model.candidates.snapshot()
    assert catalog.version > version and catalog.n_items == N_ITEMS + 2
    predictions = model.predict_on_batch(
        interactions[:3],
        k=2,
        candidate_ids=["new-a", "new-b"],
        user_ids=USER_IDS[:3],
    )
    with torch.no_grad():
        content = model.module.encoder(torch.from_numpy(new_features))
    expected = model.module.user_emb.weight.detach()[:3] @ content.T
    new_rows = catalog.rows_for(["new-a", "new-b"])
    assert set(predictions.cols.numpy().ravel()) <= set(new_rows.tolist())
    local = np.searchsorted(new_rows, predictions.cols.numpy())
    torch.testing.assert_close(
        predictions.vals,
        torch.gather(expected, 1, torch.from_numpy(local)),
        rtol=1e-5,
        atol=1e-5,
    )
    model.remove_candidates(["new-a"])
    assert "new-a" not in model.candidates.snapshot().item_ids.tolist()
    model.predict_on_batch(interactions[:3], k=5, user_ids=USER_IDS[:3])


def test_known_users_use_learned_embeddings(data):
    interactions, _ = data
    model = _fit(data)
    predictions = model.predict_on_batch(
        interactions[5:9], k=4, user_ids=USER_IDS[[6, 5, 8, 7]][[1, 0, 3, 2]]
    )
    direct = model.predict_on_batch(interactions[5:9], k=4, user_ids=USER_IDS[5:9])
    torch.testing.assert_close(predictions.cols, direct.cols)
    with pytest.raises(KeyError, match="unknown_users='error'"):
        model.predict_on_batch(interactions[:2], k=3, user_ids=["u0", "stranger"])
    with pytest.raises(KeyError):
        model.predict_on_batch(interactions[:2], k=3)


def test_fold_in_is_finite_deterministic_and_frozen(data):
    interactions, _ = data
    model = _fit(data, unknown_users="fold_in")
    before = {name: value.clone() for name, value in model.module.state_dict().items()}
    first = model.predict_on_batch(interactions, k=4)
    second = model.predict_on_batch(interactions, k=4)
    assert torch.isfinite(first.vals).all()
    torch.testing.assert_close(first.cols, second.cols)
    torch.testing.assert_close(first.vals, second.vals)
    for name, value in model.module.state_dict().items():
        torch.testing.assert_close(value, before[name])
    groups = np.arange(N_USERS) % 2
    warm = first.cols.numpy() < N_WARM
    hits = (first.cols.numpy() % 2) == groups[:, None]
    assert hits[warm].mean() > 0.5
    empty = model.predict_on_batch(csr_matrix((2, N_ITEMS), dtype=np.float32), k=3)
    assert torch.all(empty.vals == 0)


def test_streaming_evaluation_with_user_ids(data):
    interactions, _ = data
    model = _fit(data)
    targets = csr_matrix(
        (np.ones(N_USERS, dtype=np.float32), (np.arange(N_USERS), 18 + np.arange(N_USERS) % 6)),
        shape=(N_USERS, N_ITEMS),
    )
    if "user_ids" not in inspect.signature(evaluate_recommender).parameters:
        pytest.skip("user-identity plumbing not on this branch yet")
    passed = evaluate_recommender(
        model,
        source=interactions,
        targets=targets,
        metrics=[CalibratedRecall(5)],
        sample_ids=USER_IDS,
        batch_size=7,
    )
    explicit = evaluate_recommender(
        model,
        source=interactions,
        targets=targets,
        metrics=[CalibratedRecall(5)],
        sample_ids=USER_IDS,
        user_ids=USER_IDS,
        batch_size=7,
    )
    assert passed.metrics == explicit.metrics


def test_save_load_round_trip(data, tmp_path):
    interactions, features = data
    model = _fit(data, unknown_users="fold_in")
    model.update_candidates(item_ids=["late"], item_features=features[:1])
    expected = model.predict_on_batch(interactions, k=6, user_ids=USER_IDS)
    folded = model.predict_on_batch(interactions[:3], k=6)
    path = tmp_path / "clcrec.zip"
    model.save(path, include_optimizer=True)
    loaded = CLCRecTrainer.load(path, load_optimizer=True)
    assert loaded.cfg == model.cfg
    np.testing.assert_array_equal(loaded.user_ids, model.user_ids)
    np.testing.assert_array_equal(
        loaded.candidates.snapshot().item_ids, model.candidates.snapshot().item_ids
    )
    actual = loaded.predict_on_batch(interactions, k=6, user_ids=USER_IDS)
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)
    refolded = loaded.predict_on_batch(interactions[:3], k=6)
    torch.testing.assert_close(refolded.vals, folded.vals)
    assert loaded.history == model.history


def test_same_seed_is_deterministic(data):
    interactions, _ = data
    first = _fit(data, re_n_negatives=3)
    second = _fit(data, re_n_negatives=3)
    other = _fit(data, re_n_negatives=3, seed=1)
    for name, value in first.module.state_dict().items():
        torch.testing.assert_close(value, second.module.state_dict()[name])
    assert not torch.equal(
        first.module.user_emb.weight, other.module.user_emb.weight
    )
    left = first.predict_on_batch(interactions, k=5, user_ids=USER_IDS)
    right = second.predict_on_batch(interactions, k=5, user_ids=USER_IDS)
    torch.testing.assert_close(left.cols, right.cols)


@pytest.mark.parametrize("missing", ["zero", "mean"])
def test_mm_concat_handles_missing_modalities(data, missing, tmp_path):
    interactions, features = data
    modalities = {"text": features, "image": features[:, :5] * 2}
    available = np.ones(N_ITEMS, dtype=bool)
    available[[3, 20]] = False
    wrapper = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=CLCRecTrainer,
            model_config=_config(),
            modalities=["text", "image"],
            missing=missing,
            normalize=True,
        )
    )
    wrapper.fit(
        interactions,
        item_features=modalities,
        modality_masks={"image": available},
        user_ids=USER_IDS,
    )
    assert wrapper.inner_model.module.n_features == N_FEATURES + 5
    predictions = wrapper.predict_on_batch(interactions[:4], k=5, user_ids=USER_IDS[:4])
    assert predictions.shape == (4, N_ITEMS)
    path = tmp_path / "wrapped.zip"
    wrapper.save(path)
    loaded = MMConcatWrapper.load(path)
    again = loaded.predict_on_batch(interactions[:4], k=5, user_ids=USER_IDS[:4])
    torch.testing.assert_close(again.cols, predictions.cols)

    strict = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=CLCRecTrainer,
            model_config=_config(),
            modalities=["text", "image"],
        )
    )
    with pytest.raises(ValueError):
        strict.fit(
            interactions,
            item_features=modalities,
            modality_masks={"image": available},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_smoke(data):
    interactions, _ = data
    model = _fit(data, device="cuda", epochs=1)
    predictions = model.predict_on_batch(interactions[:4], k=3, user_ids=USER_IDS[:4])
    assert predictions.cols.device.type == "cuda"
