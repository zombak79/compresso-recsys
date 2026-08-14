from __future__ import annotations

import warnings

import numpy as np
import pytest

from compresso_recsys.evaluation import EvaluationResult
from compresso_recsys.stats import (
    ComparisonReport,
    PairwiseComparison,
    _adjust,
    compare_models,
    compare_pair,
)

N = 400
METRIC = "ndcg@20"


def _result(
    values,
    *,
    metric: str = METRIC,
    sample_ids=None,
    collect: bool = True,
) -> EvaluationResult:
    """Build a result directly, so tests are about statistics rather than ranking."""
    array = np.asarray(values, dtype=np.float32)
    ids = np.arange(array.size) if sample_ids is None else np.asarray(sample_ids)
    return EvaluationResult(
        metrics={metric: float(array.mean(dtype=np.float64))},
        per_user={metric: array} if collect else None,
        sample_ids=ids if collect else None,
        n_rows=int(array.size),
        n_eval_users=int(array.size),
        required_k=20,
    )


def _multi(**columns) -> EvaluationResult:
    arrays = {key: np.asarray(v, dtype=np.float32) for key, v in columns.items()}
    size = next(iter(arrays.values())).size
    return EvaluationResult(
        metrics={k: float(v.mean(dtype=np.float64)) for k, v in arrays.items()},
        per_user=arrays,
        sample_ids=np.arange(size),
        n_rows=size,
        n_eval_users=size,
        required_k=20,
    )


def _quiet(fn, *args, **kwargs):
    """Call ignoring the small-n_effective warning, which some fixtures trigger."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


# --------------------------------------------------------------------------
# degenerate cases with known answers
# --------------------------------------------------------------------------


def test_identical_models_are_inconclusive():
    """Every sign assignment and every resample gives zero, so all are extreme."""
    values = np.random.default_rng(0).random(N)
    same = _result(values)

    comparison = _quiet(compare_pair, same, same, metric=METRIC, n_resamples=999)

    assert comparison.difference == 0.0
    assert (comparison.ci_low, comparison.ci_high) == (0.0, 0.0)
    assert comparison.p_value == 1.0
    assert comparison.direction == "inconclusive"
    assert not comparison.significant
    assert comparison.n_effective == 0


def test_constant_difference_is_exact_and_degenerate():
    base = np.random.default_rng(1).random(N)

    comparison = compare_pair(
        _result(base), _result(base + 0.05), metric=METRIC, n_resamples=999
    )

    assert comparison.difference == pytest.approx(0.05, abs=1e-6)
    assert comparison.ci_low == pytest.approx(0.05, abs=1e-6)
    assert comparison.ci_high == pytest.approx(0.05, abs=1e-6)
    # The smallest value the Monte Carlo p can take at B resamples.
    assert comparison.p_value == pytest.approx(1 / 1000)
    assert comparison.direction == "better"
    assert comparison.n_effective == N


def test_difference_is_candidate_minus_baseline():
    comparison = compare_pair(
        _result(np.full(N, 0.10)),
        _result(np.full(N, 0.30)),
        metric=METRIC,
        n_resamples=199,
    )

    assert comparison.difference == pytest.approx(0.20, abs=1e-6)
    assert comparison.baseline_mean == pytest.approx(0.10, abs=1e-6)
    assert comparison.candidate_mean == pytest.approx(0.30, abs=1e-6)
    assert comparison.direction == "better"


def test_worse_candidate_reports_worse():
    comparison = compare_pair(
        _result(np.full(N, 0.30)),
        _result(np.full(N, 0.10)),
        metric=METRIC,
        n_resamples=199,
    )

    assert comparison.difference == pytest.approx(-0.20, abs=1e-6)
    assert comparison.direction == "worse"


# --------------------------------------------------------------------------
# reproducibility and pairing
# --------------------------------------------------------------------------


def test_fixed_random_state_is_deterministic():
    rng = np.random.default_rng(2)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)

    first = compare_pair(
        _result(base), _result(other), metric=METRIC, n_resamples=499, random_state=7
    )
    second = compare_pair(
        _result(base), _result(other), metric=METRIC, n_resamples=499, random_state=7
    )

    assert first == second


def test_different_random_states_move_only_the_estimates():
    rng = np.random.default_rng(3)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)

    first = compare_pair(
        _result(base), _result(other), metric=METRIC, n_resamples=499, random_state=1
    )
    second = compare_pair(
        _result(base), _result(other), metric=METRIC, n_resamples=499, random_state=2
    )

    # The observed effect is not resampled, so it cannot move.
    assert first.difference == second.difference
    assert first.n_effective == second.n_effective
    assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)


def test_mismatched_sample_ids_raise_an_alignment_error():
    base = np.random.default_rng(4).random(N)

    with pytest.raises(ValueError, match="same samples in the same order"):
        compare_pair(
            _result(base),
            _result(base, sample_ids=np.arange(1, N + 1)),
            metric=METRIC,
            n_resamples=99,
        )


def test_reordered_sample_ids_raise_rather_than_realign():
    base = np.random.default_rng(5).random(N)
    swapped = np.arange(N)
    swapped[[0, 1]] = swapped[[1, 0]]

    with pytest.raises(ValueError, match="same samples in the same order"):
        compare_pair(
            _result(base),
            _result(base, sample_ids=swapped),
            metric=METRIC,
            n_resamples=99,
        )


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_results_without_per_user_values_are_rejected():
    values = np.random.default_rng(6).random(N)

    with pytest.raises(ValueError, match="collect_per_user=False"):
        compare_pair(
            _result(values, collect=False),
            _result(values),
            metric=METRIC,
            n_resamples=99,
        )


def test_missing_metric_is_rejected():
    values = np.random.default_rng(7).random(N)

    with pytest.raises(KeyError, match="recall@20"):
        compare_pair(
            _result(values), _result(values), metric="recall@20", n_resamples=99
        )


def test_fewer_than_two_samples_is_rejected():
    with pytest.raises(ValueError, match="at least 2 evaluable samples"):
        compare_pair(
            _result(np.array([0.5])),
            _result(np.array([0.6])),
            metric=METRIC,
            n_resamples=99,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"confidence_level": 1.0}, "confidence_level"),
        ({"n_resamples": 0}, "n_resamples"),
        ({"alternative": "sideways"}, "alternative"),
        ({"test_method": "jackknife"}, "test_method"),
        ({"resample_batch_size": 0}, "resample_batch_size"),
    ],
)
def test_argument_validation(kwargs, message):
    values = np.random.default_rng(8).random(N)
    defaults = {"metric": METRIC, "n_resamples": 99}

    with pytest.raises(ValueError, match=message):
        compare_pair(_result(values), _result(values), **{**defaults, **kwargs})


def test_relative_difference_is_none_when_the_baseline_is_zero():
    comparison = compare_pair(
        _result(np.zeros(N)),
        _result(np.full(N, 0.2)),
        metric=METRIC,
        n_resamples=199,
    )

    assert comparison.baseline_mean == 0.0
    assert comparison.relative_difference is None


def test_relative_difference_scales_by_the_baseline():
    comparison = compare_pair(
        _result(np.full(N, 0.20)),
        _result(np.full(N, 0.25)),
        metric=METRIC,
        n_resamples=199,
    )

    assert comparison.relative_difference == pytest.approx(0.25, abs=1e-5)


# --------------------------------------------------------------------------
# effective sample size
# --------------------------------------------------------------------------


def test_n_effective_counts_only_untied_users():
    base = np.full(N, 0.4)
    other = base.copy()
    other[:17] += 0.3  # everyone else is an exact tie

    comparison = _quiet(
        compare_pair, _result(base), _result(other), metric=METRIC, n_resamples=499
    )

    assert comparison.n_samples == N
    assert comparison.n_effective == 17


def test_small_effective_sample_warns():
    base = np.full(N, 0.4)
    other = base.copy()
    other[:5] += 0.3

    with pytest.warns(RuntimeWarning, match="nonzero paired difference"):
        compare_pair(_result(base), _result(other), metric=METRIC, n_resamples=299)


def test_large_effective_sample_does_not_warn():
    rng = np.random.default_rng(9)
    base = rng.random(N)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compare_pair(
            _result(base),
            _result(base + rng.normal(0.02, 0.1, N)),
            metric=METRIC,
            n_resamples=299,
        )


# --------------------------------------------------------------------------
# intervals and alternatives
# --------------------------------------------------------------------------


def test_interval_orientation_follows_the_alternative():
    rng = np.random.default_rng(10)
    base = rng.random(N)
    other = base + rng.normal(0.03, 0.1, N)
    kwargs = {"metric": METRIC, "n_resamples": 999, "random_state": 3}

    two = compare_pair(_result(base), _result(other), alternative="two-sided", **kwargs)
    greater = compare_pair(_result(base), _result(other), alternative="greater", **kwargs)
    less = compare_pair(_result(base), _result(other), alternative="less", **kwargs)

    assert np.isfinite(two.ci_low) and np.isfinite(two.ci_high)
    assert np.isfinite(greater.ci_low) and greater.ci_high == np.inf
    assert less.ci_low == -np.inf and np.isfinite(less.ci_high)


def test_one_sided_alternatives_point_opposite_ways():
    rng = np.random.default_rng(11)
    base = rng.random(N)
    other = base + rng.normal(0.05, 0.1, N)
    kwargs = {"metric": METRIC, "n_resamples": 999}

    greater = compare_pair(_result(base), _result(other), alternative="greater", **kwargs)
    less = compare_pair(_result(base), _result(other), alternative="less", **kwargs)

    assert greater.p_value < 0.05
    assert less.p_value > 0.95


def test_p_value_is_bounded_away_from_zero_and_one():
    rng = np.random.default_rng(12)
    base = rng.random(N)
    comparison = compare_pair(
        _result(base),
        _result(base + rng.normal(0.5, 0.01, N)),
        metric=METRIC,
        n_resamples=199,
    )

    assert 1 / 200 <= comparison.p_value <= 1.0


def test_randomization_and_bootstrap_tests_agree():
    rng = np.random.default_rng(13)
    base = rng.random(2000)
    other = base + rng.normal(0.01, 0.15, 2000)
    kwargs = {"metric": METRIC, "n_resamples": 4999, "random_state": 0}

    randomization = compare_pair(
        _result(base), _result(other), test_method="randomization", **kwargs
    )
    bootstrap = compare_pair(
        _result(base), _result(other), test_method="bootstrap", **kwargs
    )

    assert randomization.difference == bootstrap.difference
    assert randomization.p_value == pytest.approx(bootstrap.p_value, abs=0.02)
    assert randomization.test_method == "randomization"
    assert bootstrap.test_method == "bootstrap"


def test_centered_bootstrap_is_a_shift_of_the_ordinary_bootstrap():
    """The identity that lets the bootstrap test skip a second resampling pass."""
    from compresso_recsys.stats import _bootstrap_means

    rng = np.random.default_rng(14)
    d = rng.normal(0.02, 0.2, 500)
    difference = d.mean()

    ordinary = _bootstrap_means(
        d, n_resamples=500, rng=np.random.default_rng(1), resample_batch_size=64
    )
    centered = _bootstrap_means(
        d - difference,
        n_resamples=500,
        rng=np.random.default_rng(1),
        resample_batch_size=64,
    )

    assert np.allclose(centered, ordinary - difference, atol=1e-12)


def test_chunking_does_not_change_results():
    rng = np.random.default_rng(15)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)
    kwargs = {"metric": METRIC, "n_resamples": 997, "random_state": 5}

    small = compare_pair(
        _result(base), _result(other), resample_batch_size=1, **kwargs
    )
    large = compare_pair(
        _result(base), _result(other), resample_batch_size=4096, **kwargs
    )

    assert np.isfinite([small.ci_low, small.ci_high, large.ci_low, large.ci_high]).all()
    assert small.difference == large.difference


# --------------------------------------------------------------------------
# multiple-testing correction
# --------------------------------------------------------------------------


def test_holm_matches_the_hand_computed_fixture():
    adjusted = _adjust(np.array([0.01, 0.03, 0.04]), "holm")

    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_restores_the_original_order():
    adjusted = _adjust(np.array([0.04, 0.01, 0.03]), "holm")

    assert adjusted == pytest.approx([0.06, 0.03, 0.06])


def test_holm_is_monotone_and_clipped():
    adjusted = _adjust(np.array([0.2, 0.5, 0.9]), "holm")

    assert adjusted == pytest.approx([0.6, 1.0, 1.0])
    assert np.all(np.diff(adjusted[np.argsort([0.2, 0.5, 0.9])]) >= 0)


def test_bonferroni_matches_the_closed_form():
    raw = np.array([0.01, 0.03, 0.9])

    assert _adjust(raw, "bonferroni") == pytest.approx(np.minimum(1.0, 3 * raw))


def test_no_correction_passes_p_values_through():
    raw = np.array([0.01, 0.03, 0.04])

    assert _adjust(raw, None) == pytest.approx(raw)


# --------------------------------------------------------------------------
# compare_models
# --------------------------------------------------------------------------


def _three_models(seed: int = 20):
    rng = np.random.default_rng(seed)
    ndcg = rng.random(N)
    recall = rng.random(N)
    return {
        "ELSA": _multi(**{"ndcg@20": ndcg, "recall@20": recall}),
        "TEASER": _multi(
            **{
                "ndcg@20": ndcg + rng.normal(0.05, 0.1, N),
                "recall@20": recall + rng.normal(0.03, 0.1, N),
            }
        ),
        "CSELSA": _multi(
            **{
                "ndcg@20": ndcg + rng.normal(0.02, 0.1, N),
                "recall@20": recall + rng.normal(-0.02, 0.1, N),
            }
        ),
    }


def test_reference_mode_compares_every_model_against_the_reference():
    report = compare_models(
        _three_models(), metrics=["ndcg@20", "recall@20"], reference="ELSA",
        n_resamples=499,
    )

    assert len(report) == 2 * (3 - 1)
    assert {c.baseline for c in report} == {"ELSA"}
    assert {c.candidate for c in report} == {"TEASER", "CSELSA"}
    assert report.reference == "ELSA"


def test_all_pairs_mode_generates_every_unordered_pair():
    report = compare_models(_three_models(), metrics="ndcg@20", n_resamples=499)

    assert len(report) == 3 * (3 - 1) // 2
    assert [(c.baseline, c.candidate) for c in report] == [
        ("ELSA", "TEASER"),
        ("ELSA", "CSELSA"),
        ("TEASER", "CSELSA"),
    ]


def test_correction_spans_every_metric_and_pair():
    report = compare_models(
        _three_models(), metrics=["ndcg@20", "recall@20"], reference="ELSA",
        correction="bonferroni", n_resamples=499,
    )

    assert len(report) == 4
    for comparison in report:
        expected = min(1.0, 4 * comparison.p_value)
        assert comparison.adjusted_p_value == pytest.approx(expected)


def test_uncorrected_report_leaves_p_values_alone():
    report = compare_models(
        _three_models(), metrics="ndcg@20", reference="ELSA",
        correction=None, n_resamples=499,
    )

    for comparison in report:
        assert comparison.adjusted_p_value == comparison.p_value


def test_significance_uses_the_adjusted_p_value():
    report = compare_models(
        _three_models(), metrics=["ndcg@20", "recall@20"], reference="ELSA",
        n_resamples=999,
    )
    alpha = 1.0 - report.confidence_level

    for comparison in report:
        assert comparison.significant == (comparison.adjusted_p_value <= alpha)


def test_significance_rejects_at_exactly_alpha():
    """Monte Carlo p-values are discrete, so equality with alpha is attainable."""
    from compresso_recsys.stats import _with_adjusted

    rng = np.random.default_rng(21)
    base = rng.random(N)
    comparison = compare_pair(
        _result(base), _result(base + rng.normal(0.02, 0.1, N)),
        metric=METRIC, n_resamples=99,
    )

    at_alpha = _with_adjusted(comparison, 0.05, 0.05)

    assert at_alpha.significant


def test_report_metadata_records_the_configuration():
    report = compare_models(
        _three_models(), metrics="ndcg@20", reference="ELSA",
        confidence_level=0.9, n_resamples=299, correction="bonferroni",
        test_method="bootstrap", random_state=11,
    )

    assert report.confidence_level == 0.9
    assert report.n_resamples == 299
    assert report.correction == "bonferroni"
    assert report.test_method == "bootstrap"
    assert report.random_state == 11
    assert report.metrics == ("ndcg@20",)
    assert report.model_names == ("ELSA", "TEASER", "CSELSA")


@pytest.mark.parametrize(
    ("kwargs", "exception", "message"),
    [
        ({"metrics": "ndcg@20", "reference": "GONE"}, ValueError, "not among the models"),
        ({"metrics": []}, ValueError, "at least one metric"),
        ({"metrics": ["ndcg@20", "ndcg@20"]}, ValueError, "metrics must be unique"),
        ({"metrics": "ndcg@20", "correction": "sidak"}, ValueError, "unknown correction"),
    ],
)
def test_compare_models_validation(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        compare_models(_three_models(), n_resamples=99, **kwargs)


def test_compare_models_needs_at_least_two_models():
    models = _three_models()

    with pytest.raises(ValueError, match="at least two models"):
        compare_models({"ELSA": models["ELSA"]}, metrics="ndcg@20", n_resamples=99)


def test_to_frame_has_one_row_per_hypothesis_in_a_fixed_order():
    from compresso_recsys.stats import _FRAME_COLUMNS

    report = compare_models(
        _three_models(), metrics=["ndcg@20", "recall@20"], reference="ELSA",
        n_resamples=299,
    )
    frame = report.to_frame()

    assert len(frame) == len(report)
    assert list(frame.columns) == list(_FRAME_COLUMNS)
    assert "n_effective" in frame.columns
    # Metric-major, then model order from the input mapping.
    assert list(frame["metric"]) == ["ndcg@20"] * 2 + ["recall@20"] * 2


def test_pairwise_comparison_to_dict_round_trips_the_frame_columns():
    from compresso_recsys.stats import _FRAME_COLUMNS

    rng = np.random.default_rng(22)
    base = rng.random(N)
    comparison = compare_pair(
        _result(base), _result(base + rng.normal(0.02, 0.1, N)),
        metric=METRIC, n_resamples=199,
    )

    as_dict = comparison.to_dict()

    assert set(as_dict) == set(_FRAME_COLUMNS)
    assert as_dict["direction"] == comparison.direction
    assert isinstance(comparison, PairwiseComparison)


def test_compare_pair_does_not_correct_a_single_hypothesis():
    rng = np.random.default_rng(23)
    base = rng.random(N)
    comparison = compare_pair(
        _result(base), _result(base + rng.normal(0.02, 0.1, N)),
        metric=METRIC, n_resamples=199,
    )

    assert comparison.adjusted_p_value == comparison.p_value


def test_report_is_iterable_and_sized():
    report = compare_models(_three_models(), metrics="ndcg@20", n_resamples=99)

    assert isinstance(report, ComparisonReport)
    assert len(report) == len(list(report)) == 3


def test_interval_does_not_depend_on_the_test_method():
    """The randomization test consumes draws the bootstrap test does not.

    Sharing one stream made every hypothesis after the first report a different
    interval for identical data, purely because test_method changed.
    """
    rng = np.random.default_rng(30)
    base = rng.random(N)
    models = {
        "A": _multi(**{"m1": base, "m2": rng.random(N)}),
        "B": _multi(
            **{"m1": base + rng.normal(0.01, 0.1, N), "m2": rng.random(N)}
        ),
    }
    kwargs = {"metrics": ["m1", "m2"], "reference": "A", "n_resamples": 499,
              "random_state": 4}

    randomized = compare_models(models, test_method="randomization", **kwargs)
    bootstrapped = compare_models(models, test_method="bootstrap", **kwargs)

    for left, right in zip(randomized, bootstrapped):
        assert left.metric == right.metric
        assert left.ci_low == right.ci_low
        assert left.ci_high == right.ci_high
        assert left.bootstrap_standard_error == right.bootstrap_standard_error
