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


# Fixtures stand in for two models evaluated against one target matrix, so they
# share a fingerprint. Passing None models a result assembled by hand, which
# comparison cannot verify and warns about.
TARGETS = "same-targets"


def _result(
    values,
    *,
    metric: str = METRIC,
    sample_ids=None,
    collect: bool = True,
    fingerprint: str | None = TARGETS,
) -> EvaluationResult:
    """Build a result directly, so tests are about statistics rather than ranking."""
    array = np.asarray(values, dtype=np.float32)
    ids = np.arange(array.size) if sample_ids is None else np.asarray(sample_ids)
    return EvaluationResult(
        metrics={metric: float(array.mean(dtype=np.float64))},
        per_user={metric: array} if collect else None,
        sample_ids=ids if collect else None,
        n_rows=int(array.size),
        n_scored_rows=int(array.size),
        required_k=20,
        target_fingerprint=fingerprint,
    )


def _multi(**columns) -> EvaluationResult:
    arrays = {key: np.asarray(v, dtype=np.float32) for key, v in columns.items()}
    size = next(iter(arrays.values())).size
    return EvaluationResult(
        metrics={k: float(v.mean(dtype=np.float64)) for k, v in arrays.items()},
        per_user=arrays,
        sample_ids=np.arange(size),
        n_rows=size,
        n_scored_rows=size,
        required_k=20,
        target_fingerprint=TARGETS,
    )


def _quiet(fn, *args, **kwargs):
    """Call ignoring the small-n_nonzero warning, which some fixtures trigger."""
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
    assert comparison.n_nonzero == 0


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
    assert comparison.n_nonzero == N


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
    assert first.n_nonzero == second.n_nonzero
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
    with pytest.raises(ValueError, match="at least 2 independent units"):
        compare_pair(
            _result(np.array([0.5])),
            _result(np.array([0.6])),
            metric=METRIC,
            n_resamples=99,
        )


def test_repeated_rows_from_one_independent_unit_are_rejected():
    sample_ids = np.repeat("only-user", 5)

    with pytest.raises(
        ValueError,
        match="got 1 from 5 evaluable samples",
    ):
        compare_pair(
            _result(np.linspace(0.1, 0.5, 5), sample_ids=sample_ids),
            _result(np.linspace(0.2, 0.6, 5), sample_ids=sample_ids),
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


def test_n_nonzero_counts_only_untied_users():
    base = np.full(N, 0.4)
    other = base.copy()
    other[:17] += 0.3  # everyone else is an exact tie

    comparison = _quiet(
        compare_pair, _result(base), _result(other), metric=METRIC, n_resamples=499
    )

    assert comparison.n_samples == N
    assert comparison.n_nonzero == 17
    assert comparison.tie_rate == pytest.approx(1 - 17 / N)


def test_tie_rate_spans_both_extremes():
    base = np.full(N, 0.4)

    all_tied = _quiet(
        compare_pair, _result(base), _result(base), metric=METRIC, n_resamples=99
    )
    none_tied = compare_pair(
        _result(base), _result(base + 0.1), metric=METRIC, n_resamples=99
    )

    assert all_tied.tie_rate == 1.0
    assert none_tied.tie_rate == 0.0


def test_the_estimate_uses_every_sample_not_just_the_untied_ones():
    """Two datasets sharing an n_nonzero can describe entirely different systems.

    This is why n_nonzero is not the sample size: the mean difference divides by
    n_samples, so padding with ties shrinks the effect rather than leaving it be.
    """
    dense = np.full(30, 1.0)
    padded = np.concatenate([dense, np.zeros(9_970)])

    small = compare_pair(
        _result(np.zeros(30)), _result(dense), metric=METRIC, n_resamples=99
    )
    large = _quiet(
        compare_pair,
        _result(np.zeros(10_000)),
        _result(padded),
        metric=METRIC,
        n_resamples=99,
    )

    assert small.n_nonzero == large.n_nonzero == 30
    assert small.difference == pytest.approx(1.0, abs=1e-6)
    assert large.difference == pytest.approx(0.003, abs=1e-6)


def test_small_untied_count_warns_about_discreteness_not_sample_size():
    base = np.full(N, 0.4)
    other = base.copy()
    other[:5] += 0.3

    with pytest.warns(RuntimeWarning, match="highly discrete") as caught:
        compare_pair(_result(base), _result(other), metric=METRIC, n_resamples=299)

    # The estimate is not what is degraded, and the message must not imply it is.
    assert f"still uses all {N} units" in str(caught[0].message)


def test_all_tied_warns_that_the_answer_is_exact_rather_than_degraded():
    """Zero untied users is a different situation from a handful of them."""
    base = np.random.default_rng(31).random(N)

    with pytest.warns(RuntimeWarning, match="mean paired") as caught:
        comparison = compare_pair(
            _result(base), _result(base), metric=METRIC, n_resamples=299
        )

    message = str(caught[0].message)
    assert "exact, not estimated" in message
    assert "highly discrete" not in message
    assert (comparison.difference, comparison.ci_low, comparison.ci_high) == (0.0, 0.0, 0.0)
    assert comparison.p_value == 1.0


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
    # All three counts are reported, so a reader can see how much of the sample
    # was untied rather than inferring it.
    assert {"n_samples", "n_nonzero", "tie_rate"} <= set(frame.columns)
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


# --------------------------------------------------------------------------
# resampling is a function of the hypothesis, not of report order
# --------------------------------------------------------------------------


def _two_models(seed: int = 40):
    rng = np.random.default_rng(seed)
    base = rng.random(N)
    return {
        "EASE": _multi(
            **{"m1": base, "m2": rng.random(N), "unrelated": rng.random(N)}
        ),
        "ELSA": _multi(
            **{
                "m1": base + rng.normal(0.01, 0.1, N),
                "m2": rng.random(N),
                "unrelated": rng.random(N),
            }
        ),
    }


def _raw(comparison):
    """The part of a comparison that must not depend on the rest of the report."""
    return (
        comparison.difference,
        comparison.ci_low,
        comparison.ci_high,
        comparison.p_value,
        comparison.bootstrap_standard_error,
    )


def _find(report, metric):
    return next(c for c in report if c.metric == metric)


def test_reordering_metrics_leaves_raw_comparisons_identical():
    models = _two_models()
    kwargs = {"reference": "EASE", "n_resamples": 499, "random_state": 4}

    forward = compare_models(models, metrics=["m1", "m2"], **kwargs)
    reversed_order = compare_models(models, metrics=["m2", "m1"], **kwargs)

    for metric in ("m1", "m2"):
        assert _raw(_find(forward, metric)) == _raw(_find(reversed_order, metric))


def test_adding_a_metric_leaves_existing_raw_comparisons_unchanged():
    models = _two_models()
    kwargs = {"reference": "EASE", "n_resamples": 499, "random_state": 4}

    without = compare_models(models, metrics=["m1"], **kwargs)
    with_extra = compare_models(models, metrics=["unrelated", "m1"], **kwargs)

    assert _raw(_find(without, "m1")) == _raw(_find(with_extra, "m1"))
    # The family grew, so the adjustment is expected to move even though the
    # raw comparison did not.
    assert _find(without, "m1").adjusted_p_value != _find(with_extra, "m1").adjusted_p_value


def test_reversing_a_pair_mirrors_the_comparison_exactly():
    """Reversal reuses the same draws, so the result mirrors rather than re-noising.

    Orientation follows insertion order, so swapping the mapping swaps baseline
    and candidate. The seed is keyed on the sorted pair, so both directions draw
    identical resample indices and sign assignments.
    """
    models = _two_models()
    kwargs = {"metrics": ["m1"], "n_resamples": 499, "random_state": 4}

    forward = compare_models(models, **kwargs).comparisons[0]
    backward = compare_models(dict(reversed(models.items())), **kwargs).comparisons[0]

    assert (forward.baseline, forward.candidate) == ("EASE", "ELSA")
    assert (backward.baseline, backward.candidate) == ("ELSA", "EASE")

    # Exact: IEEE subtraction is antisymmetric and summation negates term by
    # term, so the mean does too. The p-value counts a symmetric condition over
    # shared sign assignments, and the standard error squares its deviations.
    assert backward.difference == -forward.difference
    assert backward.p_value == forward.p_value
    assert backward.n_samples == forward.n_samples
    assert backward.n_nonzero == forward.n_nonzero
    assert backward.bootstrap_standard_error == forward.bootstrap_standard_error

    # Not exact: np.quantile interpolates as ``a + frac * (b - a)`` between
    # order statistics, and that arithmetic is not sign-symmetric to the last
    # bit. The endpoints mirror to within a couple of ULP, which is a rounding
    # artefact of the interpolation rather than different resampling.
    assert backward.ci_low == pytest.approx(-forward.ci_high, rel=1e-12, abs=1e-15)
    assert backward.ci_high == pytest.approx(-forward.ci_low, rel=1e-12, abs=1e-15)


def test_relative_difference_does_not_mirror_under_reversal():
    """It divides by the baseline mean, and reversal changes which model that is."""
    models = _two_models()
    kwargs = {"metrics": ["m1"], "n_resamples": 499, "random_state": 4}

    forward = compare_models(models, **kwargs).comparisons[0]
    backward = compare_models(dict(reversed(models.items())), **kwargs).comparisons[0]

    assert forward.relative_difference is not None
    assert backward.relative_difference is not None
    assert backward.relative_difference != -forward.relative_difference


def test_random_state_none_stays_nondeterministic():
    models = _two_models()
    kwargs = {"metrics": ["m1"], "reference": "EASE", "n_resamples": 499}

    first = compare_models(models, random_state=None, **kwargs).comparisons[0]
    second = compare_models(models, random_state=None, **kwargs).comparisons[0]

    # Deriving seeds from the literal None would have made these identical.
    assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)
    assert first.difference == second.difference


def test_random_state_none_still_shares_one_draw_across_hypotheses():
    """One nondeterministic draw per call, not one per hypothesis."""
    models = _two_models()
    report = compare_models(
        models, metrics=["m1", "m2"], reference="EASE",
        n_resamples=499, random_state=None,
    )

    assert report.random_state is None
    assert len(report.comparisons) == 2


def test_compare_pair_matches_compare_models_for_the_same_hypothesis():
    """Identity is (metric, unordered pair of names), so the two entry points agree."""
    models = _two_models()

    paired = compare_pair(
        models["EASE"], models["ELSA"],
        metric="m1", baseline_name="EASE", candidate_name="ELSA",
        n_resamples=499, random_state=4,
    )
    from_report = _find(
        compare_models(
            models, metrics=["m1", "m2"], reference="EASE",
            n_resamples=499, random_state=4,
        ),
        "m1",
    )

    assert _raw(paired) == _raw(from_report)


# --------------------------------------------------------------------------
# paired t-test
# --------------------------------------------------------------------------


def test_t_test_agrees_closely_with_the_randomization_test():
    """The literature's finding, as an executable claim."""
    rng = np.random.default_rng(50)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)
    kwargs = {"metric": METRIC, "n_resamples": 9_999, "random_state": 4}

    randomized = compare_pair(_result(base), _result(other), test_method="randomization", **kwargs)
    t_tested = compare_pair(_result(base), _result(other), test_method="t", **kwargs)

    assert t_tested.p_value == pytest.approx(randomized.p_value, abs=0.01)


def _stored(values):
    """Round-trip through the float32 storage and float64 upcast stats sees."""
    return np.asarray(values, dtype=np.float32).astype(np.float64)


def test_t_test_matches_scipy_on_the_same_differences():
    """It must be a one-sample test on d, not an independently derived one."""
    from scipy.stats import ttest_1samp

    rng = np.random.default_rng(51)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)

    comparison = compare_pair(
        _result(base), _result(other), metric=METRIC, n_resamples=99,
        test_method="t", random_state=4,
    )

    expected = ttest_1samp(_stored(other) - _stored(base), 0.0)
    assert comparison.p_value == float(expected.pvalue)


def test_t_test_still_reports_a_bootstrapped_interval():
    """Choosing it makes the p-value deterministic, not the whole comparison."""
    rng = np.random.default_rng(52)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)
    kwargs = {"metric": METRIC, "n_resamples": 499, "test_method": "t"}

    first = compare_pair(_result(base), _result(other), random_state=1, **kwargs)
    second = compare_pair(_result(base), _result(other), random_state=2, **kwargs)

    # Deterministic p-value...
    assert first.p_value == second.p_value
    # ...beside an interval that still moved with the seed.
    assert (first.ci_low, first.ci_high) != (second.ci_low, second.ci_high)
    assert first.interval_method == "percentile"


def test_t_test_has_no_monte_carlo_floor():
    """The whole reason to offer it beside a resampled test."""
    rng = np.random.default_rng(53)
    base = rng.random(N)
    other = base + rng.normal(0.15, 0.1, N)
    kwargs = {"metric": METRIC, "n_resamples": 999, "random_state": 4}

    randomized = compare_pair(_result(base), _result(other), test_method="randomization", **kwargs)
    t_tested = compare_pair(_result(base), _result(other), test_method="t", **kwargs)

    assert randomized.p_value == pytest.approx(1 / 1000)
    assert t_tested.p_value < 1e-6


def test_t_test_leaves_other_hypotheses_untouched():
    """It consumes no test draws, which per-hypothesis seeding makes harmless."""
    models = _two_models()
    kwargs = {"metrics": ["m1", "m2"], "reference": "EASE", "n_resamples": 499,
              "random_state": 4}

    randomized = compare_models(models, test_method="randomization", **kwargs)
    mixed = compare_models(models, test_method="t", **kwargs)

    for left, right in zip(randomized, mixed):
        assert left.metric == right.metric
        assert (left.ci_low, left.ci_high) == (right.ci_low, right.ci_high)


@pytest.mark.parametrize("alternative", ["two-sided", "greater", "less"])
def test_t_test_honours_the_alternative(alternative):
    from scipy.stats import ttest_1samp

    rng = np.random.default_rng(54)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)

    comparison = compare_pair(
        _result(base), _result(other), metric=METRIC, n_resamples=99,
        test_method="t", alternative=alternative, random_state=4,
    )

    expected = ttest_1samp(
        _stored(other) - _stored(base), 0.0, alternative=alternative
    )
    assert comparison.p_value == float(expected.pvalue)


def test_t_test_on_identical_models_reports_one_rather_than_nan():
    """scipy returns nan for 0/0; a p-value of nan is not a result."""
    values = np.random.default_rng(55).random(N)
    same = _result(values)

    comparison = _quiet(
        compare_pair, same, same, metric=METRIC, n_resamples=99, test_method="t"
    )

    assert comparison.p_value == 1.0


def test_t_test_on_a_constant_difference_refuses_rather_than_returning_zero():
    """scipy returns exactly 0.0 from zero variance; that is not a verdict.

    The values survive the float32 round-trip exactly, so every paired
    difference really is identical rather than merely close.
    """
    with pytest.raises(ValueError, match="t statistic is undefined"):
        compare_pair(
            _result(np.zeros(N)), _result(np.full(N, 0.5)), metric=METRIC,
            n_resamples=99, test_method="t",
        )

    # The randomization test is defined here and reports its floor instead.
    randomized = compare_pair(
        _result(np.zeros(N)), _result(np.full(N, 0.5)), metric=METRIC,
        n_resamples=99, test_method="randomization",
    )
    assert randomized.p_value == pytest.approx(1 / 100)


# --------------------------------------------------------------------------
# target provenance
# --------------------------------------------------------------------------


def test_different_targets_are_refused_even_when_ids_match():
    """The failure no identifier check can see.

    Two evaluations on unrelated datasets both number their rows from zero, so
    positional identifiers agree and the comparison would otherwise proceed.
    """
    rng = np.random.default_rng(60)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)

    with pytest.raises(ValueError, match="different target matrices"):
        compare_pair(
            _result(base, fingerprint="dataset-a"),
            _result(other, fingerprint="dataset-b"),
            metric=METRIC,
            n_resamples=99,
        )


def test_missing_fingerprint_warns_rather_than_refusing():
    """Hand-built results stay usable; they just cannot be verified."""
    rng = np.random.default_rng(61)
    base = rng.random(N)
    other = base + rng.normal(0.02, 0.1, N)

    with pytest.warns(RuntimeWarning, match="no target fingerprint"):
        comparison = compare_pair(
            _result(base, fingerprint=None),
            _result(other),
            metric=METRIC,
            n_resamples=99,
        )

    assert comparison.n_samples == N


def test_matching_fingerprints_pass_quietly():
    rng = np.random.default_rng(62)
    base = rng.random(N)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        compare_pair(
            _result(base),
            _result(base + rng.normal(0.02, 0.1, N)),
            metric=METRIC,
            n_resamples=99,
        )


def test_repeated_sample_ids_are_allowed_by_evaluation():
    """The stacked-fold protocol gives one user several rows; that is legitimate.

    Whether those rows may be resampled as independent units is a question for
    paired comparison, not for the result that holds them.
    """
    values = np.random.default_rng(63).random(4)

    result = _result(values, sample_ids=np.array([0, 1, 1, 2]))

    assert result.n_scored_rows == 4


def test_sample_ids_are_read_only():
    result = _result(np.random.default_rng(64).random(8))

    with pytest.raises(ValueError, match="read-only"):
        result.sample_ids[0] = 999


def test_one_sided_without_a_reference_is_refused():
    """Direction would otherwise come from dict insertion order."""
    models = _two_models()

    with pytest.raises(ValueError, match="directional"):
        compare_models(models, metrics=["m1"], alternative="greater", n_resamples=99)


def test_one_sided_with_a_reference_is_accepted():
    models = _two_models()

    report = compare_models(
        models, metrics=["m1"], reference="EASE",
        alternative="greater", n_resamples=99,
    )

    assert report.comparisons[0].baseline == "EASE"


def test_two_sided_without_a_reference_still_compares_every_pair():
    models = _two_models()

    report = compare_models(models, metrics=["m1"], n_resamples=99)

    assert len(report.comparisons) == 1


# --------------------------------------------------------------------------
# rows that share an evaluation unit
# --------------------------------------------------------------------------


def _stacked(values, folds: int = 5, *, jitter: float = 0.0, seed: int = 0):
    """Tile values the way the stacked-fold protocol tiles users.

    ``eval_draws=5`` splits each user several times, so one user owns
    several correlated rows. ``jitter`` makes the folds differ, as real ones do.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(values, dtype=np.float64)
    rows = np.concatenate([
        base + (rng.normal(0.0, jitter, base.size) if jitter else 0.0)
        for _ in range(folds)
    ])
    ids = np.tile(np.arange(base.size), folds)
    return rows, ids


def test_unique_ids_report_one_unit_per_row():
    comparison = compare_pair(
        _result(np.random.default_rng(70).random(N)),
        _result(np.random.default_rng(71).random(N)),
        metric=METRIC, n_resamples=99,
    )

    assert comparison.n_units == comparison.n_samples == N


def test_duplicating_every_row_does_not_narrow_the_interval():
    """The property the whole repeated-row path exists to protect.

    Five identical copies of a dataset carry no more information than one. Row
    resampling would shrink the interval by about sqrt(5) anyway; resampling
    whole users leaves it where it belongs.
    """
    rng = np.random.default_rng(72)
    base = rng.random(200)
    other = base + rng.normal(0.02, 0.1, 200)

    single = compare_pair(
        _result(base), _result(other), metric=METRIC,
        n_resamples=2999, random_state=0,
    )

    rows_b, ids = _stacked(base)
    rows_o, _ = _stacked(other)
    stacked = compare_pair(
        _result(rows_b, sample_ids=ids), _result(rows_o, sample_ids=ids),
        metric=METRIC, n_resamples=2999, random_state=0,
    )

    assert stacked.n_samples == 1000
    assert stacked.n_units == 200
    assert stacked.difference == pytest.approx(single.difference, abs=1e-9)

    single_width = single.ci_high - single.ci_low
    stacked_width = stacked.ci_high - stacked.ci_low
    assert stacked_width == pytest.approx(single_width, rel=0.12)


def test_row_resampling_would_have_narrowed_it():
    """Guards the test above against passing for the wrong reason."""
    rng = np.random.default_rng(73)
    base = rng.random(200)
    other = base + rng.normal(0.02, 0.1, 200)
    rows_b, ids = _stacked(base)
    rows_o, _ = _stacked(other)

    honest = compare_pair(
        _result(rows_b, sample_ids=ids), _result(rows_o, sample_ids=ids),
        metric=METRIC, n_resamples=2999, random_state=0,
    )
    # Same 1000 rows, but every row claiming to be its own user.
    inflated = compare_pair(
        _result(rows_b, sample_ids=np.arange(1000)),
        _result(rows_o, sample_ids=np.arange(1000)),
        metric=METRIC, n_resamples=2999, random_state=0,
    )

    honest_width = honest.ci_high - honest.ci_low
    inflated_width = inflated.ci_high - inflated.ci_low
    assert inflated_width < honest_width
    assert honest_width / inflated_width == pytest.approx(np.sqrt(5), rel=0.2)


def test_correlated_folds_widen_the_interval_over_row_resampling():
    """Real folds differ, so the design effect is between one and the fold count."""
    rng = np.random.default_rng(74)
    base = rng.random(300)
    other = base + rng.normal(0.02, 0.1, 300)
    rows_b, ids = _stacked(base, jitter=0.05, seed=1)
    rows_o, _ = _stacked(other, jitter=0.05, seed=2)

    grouped = compare_pair(
        _result(rows_b, sample_ids=ids), _result(rows_o, sample_ids=ids),
        metric=METRIC, n_resamples=2999, random_state=0,
    )
    as_independent = compare_pair(
        _result(rows_b, sample_ids=np.arange(rows_b.size)),
        _result(rows_o, sample_ids=np.arange(rows_o.size)),
        metric=METRIC, n_resamples=2999, random_state=0,
    )

    ratio = (grouped.ci_high - grouped.ci_low) / (
        as_independent.ci_high - as_independent.ci_low
    )
    assert 1.0 < ratio < np.sqrt(5)


def test_randomization_flips_whole_users():
    """Sign assignments per row would give a far smaller p than the null allows."""
    rng = np.random.default_rng(75)
    base = rng.random(120)
    other = base + rng.normal(0.01, 0.15, 120)
    rows_b, ids = _stacked(base)
    rows_o, _ = _stacked(other)

    grouped = compare_pair(
        _result(rows_b, sample_ids=ids), _result(rows_o, sample_ids=ids),
        metric=METRIC, n_resamples=2999, random_state=0,
    )
    as_independent = compare_pair(
        _result(rows_b, sample_ids=np.arange(600)),
        _result(rows_o, sample_ids=np.arange(600)),
        metric=METRIC, n_resamples=2999, random_state=0,
    )

    assert grouped.p_value > as_independent.p_value


def test_t_test_uses_unit_means_when_rows_repeat():
    from scipy.stats import ttest_1samp

    rng = np.random.default_rng(76)
    base = rng.random(150)
    other = base + rng.normal(0.02, 0.1, 150)
    rows_b, ids = _stacked(base, jitter=0.05, seed=3)
    rows_o, _ = _stacked(other, jitter=0.05, seed=4)

    comparison = compare_pair(
        _result(rows_b, sample_ids=ids), _result(rows_o, sample_ids=ids),
        metric=METRIC, n_resamples=99, test_method="t", random_state=0,
    )

    d = _stored(rows_o) - _stored(rows_b)
    unit_means = np.array([d[ids == u].mean() for u in np.unique(ids)])
    assert comparison.p_value == pytest.approx(
        float(ttest_1samp(unit_means, 0.0).pvalue), rel=1e-9
    )


def test_unequal_row_counts_weight_every_user_the_same():
    """One user is one observation however many rows the protocol gave them.

    Four rows: user A contributes one difference of 1.0, user B contributes
    three of 0.0. Weighting by row gives 0.25 and lets the protocol decide whose
    score counts more; weighting by user gives 0.5, which is the mean over the
    two users actually evaluated.
    """
    ids = np.array(["A", "B", "B", "B"])
    baseline = _result(np.zeros(4), sample_ids=ids)
    candidate = _result(np.array([1.0, 0.0, 0.0, 0.0]), sample_ids=ids)

    comparison = _quiet(
        compare_pair, baseline, candidate, metric=METRIC, n_resamples=99
    )

    assert comparison.n_samples == 4
    assert comparison.n_units == 2
    assert comparison.difference == pytest.approx(0.5)
    assert comparison.candidate_mean - comparison.baseline_mean == pytest.approx(0.5)


@pytest.mark.parametrize("test_method", ["randomization", "bootstrap", "t"])
def test_every_method_tests_the_same_estimand_under_unequal_counts(test_method):
    """The three used to disagree: two kept row weights, the t-test did not."""
    rng = np.random.default_rng(80)
    # Users own 1, 2 or 3 rows, so row and user weighting genuinely differ.
    ids = np.repeat(np.arange(60), rng.integers(1, 4, 60))
    base = rng.random(ids.size)
    other = base + rng.normal(0.05, 0.1, ids.size)

    comparison = compare_pair(
        _result(base, sample_ids=ids), _result(other, sample_ids=ids),
        metric=METRIC, n_resamples=999, test_method=test_method, random_state=0,
    )

    unit_means = np.array([
        (_stored(other) - _stored(base))[ids == u].mean() for u in np.unique(ids)
    ])
    assert comparison.n_units == 60
    assert comparison.difference == pytest.approx(unit_means.mean(), rel=1e-9)
    # The interval brackets the same quantity every method is testing.
    assert comparison.ci_low <= comparison.difference <= comparison.ci_high


def test_tie_rate_counts_tied_users_not_tied_rows():
    """A user whose rows cancel is tied; flipping their sign does nothing."""
    ids = np.array(["A", "A", "B", "B"])
    baseline = _result(np.zeros(4), sample_ids=ids)
    # A's rows cancel to zero; B's do not.
    candidate = _result(np.array([0.4, -0.4, 0.3, 0.3]), sample_ids=ids)

    comparison = _quiet(
        compare_pair, baseline, candidate, metric=METRIC, n_resamples=99
    )

    assert comparison.n_units == 2
    assert comparison.n_nonzero == 1
    assert comparison.tie_rate == pytest.approx(0.5)


# --------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------


def test_show_progress_does_not_change_any_result():
    """It is display only, so every number must be identical either way."""
    models = _two_models()
    kwargs = {"metrics": ["m1", "m2"], "reference": "EASE",
              "n_resamples": 499, "random_state": 4}

    quiet = compare_models(models, **kwargs)
    loud = compare_models(models, show_progress=True, **kwargs)

    assert [c.to_dict() for c in quiet] == [c.to_dict() for c in loud]


@pytest.mark.parametrize("test_method", ["randomization", "bootstrap", "t"])
def test_progress_advances_exactly_one_per_hypothesis(test_method):
    """Fractional within a hypothesis, exactly whole across it, every method."""
    from compresso_recsys.stats import (
        _compare_arrays, _hypothesis_streams, _paired_values,
    )

    models = _two_models()
    x, y, units = _paired_values(
        models["EASE"], models["ELSA"], metric="m1",
        baseline_name="EASE", candidate_name="ELSA",
    )
    interval_rng, test_rng = _hypothesis_streams(
        0, metric="m1", baseline_name="EASE", candidate_name="ELSA"
    )
    seen: list[float] = []

    _compare_arrays(
        x, y, metric="m1", baseline_name="EASE", candidate_name="ELSA",
        confidence_level=0.95, n_resamples=499, alternative="two-sided",
        test_method=test_method, interval_rng=interval_rng, test_rng=test_rng,
        random_state=0, resample_batch_size=8, units=units,
        progress=seen.append,
    )

    assert sum(seen) == pytest.approx(1.0, abs=1e-12)
    assert sum(seen) <= 1.0, "overshooting pushes tqdm past its own total"
    assert len([s for s in seen if s > 0]) > 1, "reported one lump, not progress"


def test_progress_is_silent_without_tqdm(monkeypatch):
    """The work still runs when the optional dependency is absent."""
    import builtins

    real_import = builtins.__import__

    def no_tqdm(name, *args, **kwargs):
        if name.startswith("tqdm"):
            raise ImportError("no tqdm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tqdm)

    comparison = compare_pair(
        _result(np.random.default_rng(90).random(N)),
        _result(np.random.default_rng(91).random(N)),
        metric=METRIC, n_resamples=99, show_progress=True,
    )

    assert comparison.n_samples == N
