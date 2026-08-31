from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import NDCG, CalibratedRecall
from compresso_recsys.models import ContentRecommender, ContentRecommenderConfig
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout

N_ITEMS = 400
DIM = 16
N_USERS = 120
HISTORY = 20


def _accelerator() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


requires_accelerator = pytest.mark.skipif(
    _accelerator() is None, reason="needs a non-CPU device"
)


def _holdout(seed: int = 0):
    """Clustered item features plus a source/target split per user."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(10, DIM))
    assign = rng.integers(0, 10, N_ITEMS)
    features = (centers[assign] + 0.5 * rng.normal(size=(N_ITEMS, DIM))).astype(
        np.float32
    )
    # Unequal row norms, so normalization is observable.
    features *= rng.uniform(0.5, 3.0, size=(N_ITEMS, 1)).astype(np.float32)

    source_indices, target_indices = [], []
    for _ in range(N_USERS):
        pool = np.flatnonzero(assign == rng.integers(0, 10))
        take = min(HISTORY + 5, pool.size)
        picked = rng.choice(pool, size=take, replace=False)
        source_indices.append(np.sort(picked[: take - 5]))
        target_indices.append(np.sort(picked[take - 5 :]))

    def to_csr(index_lists):
        return csr_matrix(
            (
                np.ones(sum(map(len, index_lists)), dtype=np.float32),
                np.concatenate(index_lists),
                np.cumsum([0] + [len(i) for i in index_lists]),
            ),
            shape=(N_USERS, N_ITEMS),
        )

    item_ids = np.array([f"item-{i}" for i in range(N_ITEMS)], dtype=object)
    return features, item_ids, source_indices, target_indices, to_csr(
        source_indices
    ), to_csr(target_indices)


def _fitted(config: ContentRecommenderConfig | None = None, seed: int = 0):
    features, item_ids, src_idx, tgt_idx, source, targets = _holdout(seed)
    model = ContentRecommender(config).fit(features, item_ids=item_ids)
    return model, features, item_ids, src_idx, tgt_idx, source, targets


def test_content_recommender_matches_evaluate_item_embeddings_with_holdout():
    model, features, _ids, src_idx, tgt_idx, source, targets = _fitted()

    reference = {}
    for k in (20, 50):
        metrics = evaluate_item_embeddings_with_holdout(
            item_embeddings=features,
            source_indices=src_idx,
            target_indices=tgt_idx,
            k=k,
            score_batch_size=64,
        )
        reference.update(
            {name: value for name, value in metrics.items() if name != "n_scored_rows"}
        )

    result = evaluate_recommender(
        model,
        source=source,
        targets=targets,
        metrics=[CalibratedRecall([20, 50]), NDCG(50)],
        batch_size=32,
    )

    for key in ("calibrated_recall@20", "calibrated_recall@50"):
        assert result[key] == pytest.approx(reference[key], abs=1e-9)
    assert result["ndcg@50"] == pytest.approx(reference["ndcg@50"], abs=1e-9)


def test_content_recommender_excludes_seen_items_by_default():
    model, _f, _ids, _si, _ti, source, _t = _fitted()

    predictions = model.predict_on_batch(source[:16], k=10)

    for row in range(16):
        seen = set(source[row].indices.tolist())
        assert not seen & set(predictions.cols[row].tolist())


def test_content_recommender_can_keep_seen_items():
    model, _f, _ids, _si, _ti, source, _t = _fitted()

    kept = model.predict_on_batch(source[:8], k=10, exclude_seen=False)

    overlap = sum(
        len(set(source[row].indices.tolist()) & set(kept.cols[row].tolist()))
        for row in range(8)
    )
    assert overlap > 0


def test_content_recommender_elsa_forward_is_inert_when_excluding_seen():
    """``-x`` and ReLU only touch entries that masking overwrites with -inf."""
    _m, features, item_ids, _si, _ti, source, targets = _fitted()
    scores = {}
    for elsa_forward in (True, False):
        model = ContentRecommender(
            ContentRecommenderConfig(elsa_forward=elsa_forward)
        ).fit(features, item_ids=item_ids)
        result = evaluate_recommender(
            model,
            source=source,
            targets=targets,
            metrics=[CalibratedRecall(20)],
            batch_size=32,
        )
        scores[elsa_forward] = result["calibrated_recall@20"]

    assert scores[True] == pytest.approx(scores[False], abs=1e-12)


def test_content_recommender_normalization_changes_the_ranking():
    _m, features, item_ids, _si, _ti, source, targets = _fitted()
    scores = {}
    for normalize in (True, False):
        model = ContentRecommender(
            ContentRecommenderConfig(normalize=normalize)
        ).fit(features, item_ids=item_ids)
        scores[normalize] = evaluate_recommender(
            model,
            source=source,
            targets=targets,
            metrics=[CalibratedRecall(20)],
            batch_size=32,
        )["calibrated_recall@20"]

    assert scores[True] != pytest.approx(scores[False], abs=1e-6)


def test_content_recommender_masks_seen_within_a_candidate_subset():
    model, _f, item_ids, _si, _ti, source, _t = _fitted()
    subset = item_ids[:120]
    allowed = set(range(120))

    predictions = model.predict_on_batch(source[:16], k=10, candidate_ids=subset)

    assert set(predictions.cols.flatten().tolist()) <= allowed
    for row in range(16):
        seen = set(source[row].indices.tolist())
        assert not seen & set(predictions.cols[row].tolist())


def test_content_recommender_scores_items_added_after_fit():
    model, features, item_ids, _si, _ti, source, _t = _fitted()
    new_features = features[:3] * 1.5

    model.update_candidates(
        item_ids=["new-a", "new-b", "new-c"], item_features=new_features
    )
    aligned = model.align_source(source, item_ids=item_ids)
    predictions = model.predict(aligned, k=20)

    assert predictions.shape == (N_USERS, N_ITEMS + 3)
    assert model.candidates.n_items == N_ITEMS + 3


def test_content_recommender_rejects_k_above_candidate_count():
    model, _f, _ids, _si, _ti, source, _t = _fitted()

    with pytest.raises(ValueError, match=r"k must be in \[1, "):
        model.predict_on_batch(source[:4], k=N_ITEMS + 1)


def test_content_recommender_reports_too_few_unseen_candidates():
    model, _f, item_ids, _si, _ti, source, _t = _fitted()

    with pytest.raises(ValueError, match="unseen items"):
        model.predict_on_batch(
            source[:4], k=20, candidate_ids=item_ids[:20]
        )


def test_content_recommender_validates_inputs():
    features, item_ids, _si, _ti, source, _t = _holdout()

    with pytest.raises(ValueError, match="dtype must be one of"):
        ContentRecommenderConfig(dtype="float16")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be 2D"):
        ContentRecommender().fit(features[0], item_ids=item_ids[:1])
    with pytest.raises(ValueError, match="item_ids"):
        ContentRecommender().fit(features, item_ids=item_ids[:10])
    with pytest.raises(ValueError, match="finite"):
        broken = features.copy()
        broken[0, 0] = np.nan
        ContentRecommender().fit(broken, item_ids=item_ids)


def test_content_recommender_requires_fit_before_predict():
    _f, _ids, _si, _ti, source, _t = _holdout()
    model = ContentRecommender()

    assert not model.is_fitted
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict_on_batch(source[:4], k=5)


def test_content_recommender_returns_srp_tensor_with_catalog_width():
    model, _f, _ids, _si, _ti, source, _t = _fitted()

    predictions = model.predict_on_batch(source[:12], k=7)

    assert isinstance(predictions, SRPTensor)
    assert predictions.shape == (12, N_ITEMS)
    assert predictions.cols.shape == (12, 7)


def test_content_recommender_float64_runs():
    features, item_ids, _si, _ti, source, targets = _holdout()
    model = ContentRecommender(ContentRecommenderConfig(dtype="float64")).fit(
        features, item_ids=item_ids
    )

    predictions = model.predict_on_batch(source[:8], k=10)

    assert predictions.vals.dtype == torch.float64


@requires_accelerator
def test_content_recommender_on_accelerator_matches_cpu():
    features, item_ids, _si, _ti, source, targets = _holdout()
    results = {}
    for device in ("cpu", _accelerator()):
        model = ContentRecommender(ContentRecommenderConfig(device=device)).fit(
            features, item_ids=item_ids
        )
        results[device] = evaluate_recommender(
            model,
            source=source,
            targets=targets,
            metrics=[CalibratedRecall(20)],
            batch_size=32,
        )["calibrated_recall@20"]

    assert results["cpu"] == pytest.approx(results[_accelerator()], abs=1e-6)


@requires_accelerator
def test_content_recommender_to_moves_stored_features():
    model, _f, _ids, _si, _ti, source, _t = _fitted()
    assert model.source_features_.device.type == "cpu"

    model.to(_accelerator())

    assert model.source_features_.device.type == _accelerator()
    assert model.device == torch.device(_accelerator())
    assert model.cfg.device == _accelerator()
    assert model.predict_on_batch(source[:8], k=10).cols.shape == (8, 10)
