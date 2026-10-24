from __future__ import annotations

import numpy as np
import pytest
import torch
from compresso import SRPTensor
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import NDCG
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.stats import compare_models


class Recording:
    """Scores items by index, optionally excluding seen ones; records its kwargs."""

    def __init__(self, n_items: int, offset: float = 0.0, reverse: bool = False):
        self.n_items = n_items
        self.offset = offset
        self.reverse = reverse
        self.calls: list[dict] = []

    def predict_on_batch(self, source, *, k, exclude_seen=True, candidate_ids=None, **kwargs):
        self.calls.append({"exclude_seen_passed": "exclude_seen" in kwargs or exclude_seen is not True,
                           "exclude_seen": exclude_seen})
        rows = source.shape[0] if isinstance(source, csr_matrix) else source.n_rows
        order = torch.arange(1, self.n_items + 1) if self.reverse else torch.arange(self.n_items, 0, -1)
        scores = order.to(torch.float32).repeat(rows, 1) + self.offset
        if exclude_seen:
            dense = source.toarray() if isinstance(source, csr_matrix) else None
            if dense is not None:
                scores[torch.from_numpy(dense > 0)] = -torch.inf
        values, cols = scores.topk(k, dim=1)
        return SRPTensor(cols=cols, vals=values, shape=(rows, self.n_items))


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n_rows, n_items = 40, 12
    seen = rng.random((n_rows, n_items)) < 0.2
    relevant = (rng.random((n_rows, n_items)) < 0.2) & ~seen
    relevant[np.arange(n_rows), np.arange(n_rows) % n_items] = True  # every row has a target
    seen[np.arange(n_rows), np.arange(n_rows) % n_items] = False
    source = csr_matrix(seen.astype(np.float32))
    targets = csr_matrix(relevant.astype(np.float32))
    return source, targets, np.array([f"u{i}" for i in range(n_rows)])


def _evaluate(model, data, **kwargs):
    source, targets, ids = data
    return evaluate_recommender(model, source=source, targets=targets, metrics=[NDCG(5)],
                                sample_ids=ids, **kwargs)


def test_exclude_seen_is_forwarded_only_when_set(data):
    model = Recording(12)
    result = _evaluate(model, data)
    assert all(call["exclude_seen"] is True for call in model.calls)  # the model's own default
    assert result.protocol == {"exclude_seen": "model default"}

    model = Recording(12)
    result = _evaluate(model, data, exclude_seen=False)
    assert model.calls and all(call["exclude_seen"] is False for call in model.calls)
    assert result.protocol == {"exclude_seen": False}


def test_exclude_seen_must_be_a_bool_or_none(data):
    with pytest.raises(TypeError, match="exclude_seen"):
        _evaluate(Recording(12), data, exclude_seen="yes")


def test_source_fingerprint_ignores_representation_and_batching(data):
    source, targets, ids = data
    as_sequences = ItemSequences(values=source.indices.astype(np.int64),
                                 indptr=source.indptr.astype(np.int64), n_items=12)
    whole = _evaluate(Recording(12), data, batch_size=64)
    batched = _evaluate(Recording(12), data, batch_size=7)
    seq = evaluate_recommender(Recording(12), source=as_sequences, targets=targets,
                               metrics=[NDCG(5)], sample_ids=ids)
    assert whole.source_fingerprint == batched.source_fingerprint == seq.source_fingerprint


def test_compare_refuses_different_sources(data):
    source, targets, ids = data
    other = source.toarray()
    other[0, :] = 0
    other[0, (0 + 1) % 12] = 1.0
    a = _evaluate(Recording(12), data)
    b = evaluate_recommender(Recording(12, offset=0.5), source=csr_matrix(other), targets=targets,
                             metrics=[NDCG(5)], sample_ids=ids)
    assert a.target_fingerprint == b.target_fingerprint
    with pytest.raises(ValueError, match="different source"):
        compare_models({"a": a, "b": b}, metrics=["ndcg@5"], n_resamples=99)


def test_compare_refuses_different_explicit_exclude_seen(data):
    a = _evaluate(Recording(12), data, exclude_seen=True)
    b = _evaluate(Recording(12), data, exclude_seen=False)
    with pytest.raises(ValueError, match="exclude_seen"):
        compare_models({"a": a, "b": b}, metrics=["ndcg@5"], n_resamples=99)


def test_compare_accepts_model_default_against_explicit(data):
    a = _evaluate(Recording(12), data)
    b = _evaluate(Recording(12, reverse=True), data, exclude_seen=True)
    report = compare_models({"a": a, "b": b}, metrics=["ndcg@5"], n_resamples=99)
    assert len(report) == 1
