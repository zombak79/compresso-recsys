"""Tests for :mod:`compresso_recsys.models.semco`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import NDCG, CalibratedRecall
from compresso_recsys.models import (
    RandomBaseline,
    RandomBaselineConfig,
    SEMCoConfig,
    SEMCoTrainer,
)

N_USERS = 120
N_ITEMS = 200
DIM = 24
HISTORY = 12


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


requires_accelerator = pytest.mark.skipif(
    _accelerator() is None, reason="needs a non-CPU device"
)


def _dataset(seed: int = 0):
    """Clustered item features plus interactions drawn from those clusters.

    The cluster structure is what gives a content encoder something to find.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 10
    centers = rng.normal(size=(n_clusters, DIM))
    assign = rng.integers(0, n_clusters, N_ITEMS)
    features = (centers[assign] + 0.4 * rng.normal(size=(N_ITEMS, DIM))).astype(
        np.float32
    )

    rows, cols = [], []
    for user in range(N_USERS):
        cluster = rng.integers(0, n_clusters)
        pool = np.flatnonzero(assign == cluster)
        picked = rng.choice(pool, size=min(HISTORY, pool.size), replace=False)
        rows.extend([user] * picked.size)
        cols.extend(picked.tolist())

    interactions = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(N_USERS, N_ITEMS),
    )
    return interactions, features, assign


def _holdout(interactions, *, target_frac=0.34, seed=0):
    """Split each user's history into source and target, both always non-empty."""
    rng = np.random.default_rng(seed)
    src_rows, src_cols, tgt_rows, tgt_cols = [], [], [], []
    for row in range(interactions.shape[0]):
        items = interactions[row].indices.copy()
        rng.shuffle(items)
        n_target = max(1, int(round(items.size * target_frac)))
        n_target = min(n_target, items.size - 1)
        src_rows.extend([row] * (items.size - n_target))
        src_cols.extend(items[n_target:].tolist())
        tgt_rows.extend([row] * n_target)
        tgt_cols.extend(items[:n_target].tolist())

    shape = interactions.shape
    source = csr_matrix(
        (np.ones(len(src_cols), dtype=np.float32), (src_rows, src_cols)), shape=shape
    )
    targets = csr_matrix(
        (np.ones(len(tgt_cols), dtype=np.float32), (tgt_rows, tgt_cols)), shape=shape
    )
    return source, targets


def _mm(features, *, image=None):
    """Wrap the fixture matrix as a modality mapping."""
    modalities = {"text": features}
    if image is not None:
        modalities["image"] = image
    return modalities


def _fit(interactions, features, **overrides):
    """Small, fast config. Keyword arguments override the defaults."""
    defaults = {
        "hidden_size": 32,
        "emb_size": 16,
        "epochs": 2,
        "batch_size": 64,
        "show_progress": False,
    }
    return SEMCoTrainer(SEMCoConfig(**{**defaults, **overrides})).fit(
        interactions, _mm(features)
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"emb_size": 0},
        {"emb_size": True},
        {"hidden_size": -1},
        {"temperature": 0.0},
        {"temperature": float("inf")},
        {"lr": 0.0},
        {"weight_decay": -1.0},
        {"objective": "argmax"},
        {"lr_schedule": "linear"},
        {"seed": 1.5},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        SEMCoConfig(**kwargs)


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_fit_returns_self_and_marks_fitted():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    assert model.is_fitted
    assert len(model.history) == 2
    assert model.candidates.n_items == N_ITEMS


def test_fit_rejects_mismatched_feature_rows():
    interactions, features, _ = _dataset()
    with pytest.raises(ValueError):
        _fit(interactions, features[:-1])


def test_fit_rejects_negative_interactions():
    interactions, features, _ = _dataset()
    broken = interactions.copy()
    broken.data[0] = -1.0
    with pytest.raises(ValueError):
        _fit(broken, features)


def test_failed_fit_leaves_previous_state_intact():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    before = model.candidates.n_items
    with pytest.raises(ValueError):
        model.fit(interactions, _mm(features[:-1]))
    assert model.is_fitted
    assert model.candidates.n_items == before


def test_predict_before_fit_raises():
    interactions, _, _ = _dataset()
    with pytest.raises(RuntimeError):
        SEMCoTrainer().predict_on_batch(interactions, k=5)


# ---------------------------------------------------------------------------
# Prediction contract
# ---------------------------------------------------------------------------


def test_predict_on_batch_shape_and_type():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    out = model.predict_on_batch(interactions[:16], k=10)
    assert isinstance(out, SRPTensor)
    assert out.cols.shape == (16, 10)
    assert out.shape == (16, N_ITEMS)


def test_exclude_seen_masks_history():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    out = model.predict_on_batch(interactions[:8], k=10, exclude_seen=True)
    for row in range(8):
        seen = set(interactions[row].indices.tolist())
        assert not seen & set(out.cols[row].tolist())


def test_empty_source_batch_returns_empty_result():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    empty = csr_matrix((0, N_ITEMS), dtype=np.float32)
    out = model.predict_on_batch(empty, k=5)
    assert out.cols.shape == (0, 5)


def test_k_out_of_range_raises():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    with pytest.raises(ValueError):
        model.predict_on_batch(interactions[:4], k=N_ITEMS + 1)


def test_candidate_ids_restrict_results_but_keep_catalog_columns():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    allowed = model.candidate_item_ids[:40]
    out = model.predict_on_batch(interactions[:4], k=5, candidate_ids=allowed)
    returned = model.candidates.snapshot().ids_for(out.cols.reshape(-1))
    assert set(returned.tolist()) <= set(allowed.tolist())


def test_batching_does_not_change_predictions():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    whole = model.predict(interactions, k=10, batch_size=N_USERS)
    split = model.predict(interactions, k=10, batch_size=7)
    assert torch.equal(whole.cols, split.cols)


# ---------------------------------------------------------------------------
# Cold items
# ---------------------------------------------------------------------------


def test_candidate_subset_scores_match_full_catalog():
    """A candidate_ids selection must not change an item's own score.

    Regression test for BatchNorm: encoding a subset rather than the whole
    catalog would make an item's score depend on what was selected with it.
    """
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    subset = model.candidate_item_ids[:25]

    full = model.predict_on_batch(interactions[:4], k=N_ITEMS, exclude_seen=False)
    part = model.predict_on_batch(
        interactions[:4], k=10, exclude_seen=False, candidate_ids=subset
    )

    snapshot = model.candidates.snapshot()
    wanted = set(snapshot.rows_for(subset).tolist())
    for row in range(4):
        full_scores = {
            int(c): float(v)
            for c, v in zip(full.cols[row], full.vals[row])
            if int(c) in wanted
        }
        for col, val in zip(part.cols[row], part.vals[row]):
            assert full_scores[int(col)] == pytest.approx(float(val), abs=1e-5)


def test_two_modalities_train_and_persist(tmp_path):
    """Fuser path: two modalities of different widths, through a round trip."""
    interactions, features, _ = _dataset()
    rng = np.random.default_rng(7)
    image = rng.normal(size=(N_ITEMS, DIM * 2)).astype(np.float32)

    model = SEMCoTrainer(
        SEMCoConfig(hidden_size=32, emb_size=16, epochs=2, show_progress=False)
    ).fit(interactions, _mm(features, image=image))
    assert dict(model.candidates.snapshot().feature_dims) == {
        "text": DIM,
        "image": DIM * 2,
    }

    before = model.predict_on_batch(interactions[:8], k=10)
    path = tmp_path / "semco_mm.ckpt"
    model.save(path)
    after = SEMCoTrainer.load(path).predict_on_batch(interactions[:8], k=10)
    assert torch.equal(before.cols, after.cols)
    assert torch.allclose(before.vals, after.vals)


def test_update_with_missing_modality_is_rejected():
    """The catalog validates the schema on every update."""
    interactions, features, _ = _dataset()
    rng = np.random.default_rng(8)
    image = rng.normal(size=(N_ITEMS, DIM * 2)).astype(np.float32)
    model = SEMCoTrainer(
        SEMCoConfig(hidden_size=32, emb_size=16, epochs=2, show_progress=False)
    ).fit(interactions, _mm(features, image=image))

    with pytest.raises(ValueError):
        model.update_candidates(
            item_ids=np.array(["partial"], dtype=object),
            item_features={"text": rng.normal(size=(1, DIM)).astype(np.float32)},
        )


def test_fit_rejects_a_non_mapping_and_an_empty_mapping():
    interactions, features, _ = _dataset()
    with pytest.raises(TypeError):
        SEMCoTrainer(SEMCoConfig(show_progress=False)).fit(interactions, features)
    with pytest.raises(ValueError):
        SEMCoTrainer(SEMCoConfig(show_progress=False)).fit(interactions, {})


def test_registering_one_item_at_a_time_gives_distinct_embeddings():
    """Items added singly must not collapse onto one vector.

    Standardizing a single row zeroes every channel, so encoding one item at a
    time would give them all the same embedding.
    """
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    rng = np.random.default_rng(3)
    for i in range(3):
        model.update_candidates(
            item_ids=np.array([f"solo-{i}"], dtype=object),
            item_features={"text": rng.normal(size=(1, DIM)).astype(np.float32)},
        )
    added = np.array([f"solo-{i}" for i in range(3)], dtype=object)
    out = model.predict_on_batch(
        interactions[:1], k=3, exclude_seen=False, candidate_ids=added
    )
    scores = out.vals[0].numpy()
    assert len(set(np.round(scores, 6).tolist())) > 1, "added items collapsed"


def test_item_held_out_of_training_may_appear_in_a_source_history():
    """SEMCo has no per-item parameters, so a cold item still encodes."""
    interactions, features, _ = _dataset()
    warm = np.arange(N_ITEMS - 20)
    model = SEMCoTrainer(
        SEMCoConfig(hidden_size=32, emb_size=16, epochs=2, show_progress=False)
    ).fit(interactions, _mm(features), train_item_indices=warm)
    history = csr_matrix(
        (np.ones(2, dtype=np.float32), ([0, 0], [N_ITEMS - 5, N_ITEMS - 3])),
        shape=(1, N_ITEMS),
    )
    out = model.predict_on_batch(history, k=5)
    assert np.isfinite(out.vals.numpy()).all()


def test_cold_items_are_recommendable_without_retraining():
    """The point of the model: an item absent from training can be ranked."""
    interactions, features, _ = _dataset()
    warm = np.arange(N_ITEMS - 20)
    model = SEMCoTrainer(
        SEMCoConfig(hidden_size=32, emb_size=16, epochs=2, show_progress=False)
    ).fit(interactions, _mm(features), train_item_indices=warm)
    cold_ids = model.candidate_item_ids[N_ITEMS - 20 :]
    out = model.predict_on_batch(interactions[:4], k=5, candidate_ids=cold_ids)
    assert np.isfinite(out.vals.numpy()).all()


def test_update_candidates_registers_new_items():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    rng = np.random.default_rng(1)
    new_features = rng.normal(size=(5, DIM)).astype(np.float32)
    new_ids = np.array([f"new-{i}" for i in range(5)], dtype=object)
    model.update_candidates(item_ids=new_ids, item_features={"text": new_features})
    assert model.candidates.n_items == N_ITEMS + 5
    out = model.predict_on_batch(interactions[:2], k=3, candidate_ids=new_ids)
    assert out.cols.shape == (2, 3)


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_beats_random_on_clustered_data(seed):
    """SEMCo must clear a random ranking on data where content carries signal.

    The bar is a ratio against RandomBaseline on the same split rather than an
    absolute nDCG floor, so it does not need retuning when the fixture changes.
    """
    interactions, features, _ = _dataset(seed)
    source, targets = _holdout(interactions, seed=seed)
    metrics = [CalibratedRecall(20), NDCG(20)]

    model = _fit(source, features, epochs=10, seed=seed)
    learned = evaluate_recommender(
        model, source=source, targets=targets, metrics=metrics, batch_size=64
    )

    baseline = RandomBaseline(RandomBaselineConfig(seed=seed)).fit(source)
    chance = evaluate_recommender(
        baseline, source=source, targets=targets, metrics=metrics, batch_size=64
    )

    assert learned["ndcg@20"] > 4.0 * chance["ndcg@20"]
    assert learned["calibrated_recall@20"] > 4.0 * chance["calibrated_recall@20"]


def test_entmax_loss_matches_reference_formula():
    """Pin the entmax objective against an independent computation."""
    entmax = pytest.importorskip("entmax")

    torch.manual_seed(0)
    tau = 8.0
    users = torch.nn.functional.normalize(torch.randn(6, 5), dim=1)
    items = torch.nn.functional.normalize(torch.randn(6, 5), dim=1)

    z = (users @ items.T) / tau
    p_star = entmax.sparsemax(z / tau, dim=1)
    omega = (1.0 - p_star.pow(2.0).sum(dim=-1)) / 2.0
    expected = (
        torch.einsum("ij,ij->i", p_star - torch.eye(6), z).mean() + omega.mean()
    )

    model = SEMCoTrainer(
        SEMCoConfig(objective="sparsemax", temperature=tau, show_progress=False)
    )
    assert float(model._loss(users, items)) == pytest.approx(
        float(expected), rel=1e-6
    )


@pytest.mark.parametrize("objective", ["softmax", "entmax15", "sparsemax"])
def test_each_objective_trains(objective):
    interactions, features, _ = _dataset()
    temperature = {"softmax": 0.2, "entmax15": 2.0, "sparsemax": 10.0}[objective]
    model = _fit(
        interactions, features, objective=objective, temperature=temperature
    )
    assert np.isfinite(model.history[-1]["loss"])


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_save_load_round_trip_preserves_predictions(tmp_path):
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    before = model.predict_on_batch(interactions[:8], k=10)

    path = tmp_path / "semco.ckpt"
    model.save(path)
    restored = SEMCoTrainer.load(path)

    after = restored.predict_on_batch(interactions[:8], k=10)
    assert torch.equal(before.cols, after.cols)
    assert torch.allclose(before.vals, after.vals)
    assert np.array_equal(restored.candidate_item_ids, model.candidate_item_ids)


def test_save_requires_fitted_model(tmp_path):
    with pytest.raises(RuntimeError):
        SEMCoTrainer().save(tmp_path / "unfitted.ckpt")


@requires_accelerator
def test_moves_to_accelerator_and_back():
    interactions, features, _ = _dataset()
    model = _fit(interactions, features)
    cpu = model.predict_on_batch(interactions[:4], k=5)
    model.to(_accelerator())
    moved = model.predict_on_batch(interactions[:4], k=5)
    assert torch.equal(cpu.cols, moved.cols)


# ---------------------------------------------------------------------------
# Evaluation integration
# ---------------------------------------------------------------------------


def test_works_through_evaluate_recommender():
    interactions, features, _ = _dataset()
    source, targets = _holdout(interactions)
    model = _fit(source, features)
    result = evaluate_recommender(
        model,
        source=source,
        targets=targets,
        metrics=[CalibratedRecall(20), NDCG(20)],
        batch_size=32,
    )
    assert 0.0 <= result["ndcg@20"] <= 1.0
    assert 0.0 <= result["calibrated_recall@20"] <= 1.0