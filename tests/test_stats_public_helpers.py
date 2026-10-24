"""adjust_p_values, normalized_mse, and order-independent p-values (issues 9 and 10)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso_recsys.evaluation import EvaluationResult
from compresso_recsys.models.batching import normalized_mse
from compresso_recsys.stats import adjust_p_values, compare_models, compare_pair


def _result(values: dict[str, np.ndarray]) -> EvaluationResult:
    n = len(next(iter(values.values())))
    return EvaluationResult(metrics={k: float(v.mean()) for k, v in values.items()},
                            per_user={k: np.asarray(v, dtype=np.float64) for k, v in values.items()},
                            sample_ids=np.arange(n), n_rows=n, n_scored_rows=n, required_k=10)


@pytest.fixture(scope="module")
def results():
    rng = np.random.default_rng(0)
    base = rng.random(300)
    return {
        name: _result({"ndcg@10": np.clip(base + shift + rng.normal(0, 0.05, 300), 0, 1),
                       "recall@10": np.clip(base * 0.9 + shift + rng.normal(0, 0.05, 300), 0, 1)})
        for name, shift in (("A", 0.0), ("B", 0.01), ("C", -0.02), ("D", 0.003))
    }


def test_a_pair_is_the_same_alone_and_in_any_family_and_order(results):
    alone = compare_pair(results["A"], results["B"], metric="ndcg@10", baseline_name="A",
                         candidate_name="B", n_resamples=499)
    families = [
        compare_models({k: results[k] for k in ("A", "B")}, metrics="ndcg@10", reference="A", n_resamples=499),
        compare_models({k: results[k] for k in ("D", "C", "A", "B")}, metrics=["recall@10", "ndcg@10"],
                       reference="A", n_resamples=499),
    ]
    for report in families:
        row = next(c for c in report.comparisons if c.candidate == "B" and c.metric == "ndcg@10")
        assert row.p_value == alone.p_value
        assert (row.ci_low, row.ci_high, row.difference) == (alone.ci_low, alone.ci_high, alone.difference)
    assert families[1].comparisons[0].adjusted_p_value >= families[1].comparisons[0].p_value


def test_swapping_the_pair_mirrors_the_interval(results):
    forward = compare_pair(results["A"], results["B"], metric="ndcg@10", baseline_name="A", candidate_name="B")
    reverse = compare_pair(results["B"], results["A"], metric="ndcg@10", baseline_name="B", candidate_name="A")
    assert reverse.difference == pytest.approx(-forward.difference)
    assert (reverse.ci_low, reverse.ci_high) == pytest.approx((-forward.ci_high, -forward.ci_low))
    assert reverse.p_value == forward.p_value


def test_adjust_p_values_is_what_compare_models_applies(results):
    report = compare_models(results, metrics=["ndcg@10", "recall@10"], reference="A", n_resamples=499)
    raw = [c.p_value for c in report.comparisons]
    np.testing.assert_allclose(adjust_p_values(raw), [c.adjusted_p_value for c in report.comparisons])
    bonferroni = compare_models(results, metrics=["ndcg@10", "recall@10"], reference="A",
                                n_resamples=499, correction="bonferroni")
    np.testing.assert_allclose(adjust_p_values(raw, method="bonferroni"),
                               [c.adjusted_p_value for c in bonferroni.comparisons])


def test_adjust_p_values_by_hand():
    np.testing.assert_allclose(adjust_p_values([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(adjust_p_values([0.01, 0.04, 0.03], method="bonferroni"), [0.03, 0.12, 0.09])
    np.testing.assert_allclose(adjust_p_values([0.5, 0.9]), [1.0, 1.0])
    assert adjust_p_values([]).shape == (0,)


@pytest.mark.parametrize("bad, method", [
    ([[0.1, 0.2]], "holm"), ([0.1, np.nan], "holm"), ([-0.1], "holm"), ([1.5], "holm"), ([0.1], "bh"),
])
def test_adjust_p_values_validates(bad, method):
    with pytest.raises(ValueError):
        adjust_p_values(bad, method=method)


def test_normalized_mse_compares_directions():
    predictions = torch.tensor([[1.0, 0.0], [3.0, 4.0]])
    targets = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    cosine = torch.tensor([1.0, 0.8])
    assert normalized_mse(predictions, targets) == pytest.approx(float((2 - 2 * cosine).mean()))
    assert normalized_mse(predictions * 7, targets) == pytest.approx(float(normalized_mse(predictions, targets)))
