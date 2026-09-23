"""The published entry points, and the order they run the pieces in."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager

import numpy as np

from compresso_recsys.evaluation import EvaluationResult
from compresso_recsys.stats.results import (
    Alternative,
    ComparisonReport,
    Correction,
    PairwiseComparison,
    TestMethod,
)
from compresso_recsys.stats.units import (
    _paired_values,
    _unit_sums,
    _validate_common,
)
from compresso_recsys.stats.resampling import (
    MIN_NONZERO_SAMPLES,
    _ZERO_TOLERANCE,
    _base_entropy,
    _bootstrap_means,
    _hypothesis_streams,
    _interval,
    _monte_carlo_p,
    _randomization_means,
    _t_test_p,
)


@contextmanager
def _progress(enabled: bool, total: int, desc: str):
    """Yield a callable advancing a bar by fractions of a hypothesis, or ``None``.

    Counting hypotheses alone would leave the slowest shape unserved: one metric
    on one pair of models over a million users takes about half a minute and
    would show ``0/1`` for all of it. Counting resample chunks alone would give
    a number nobody can size. So the unit is the hypothesis and the advance is
    fractional, which reads sensibly whether a call produces one of them or
    fifty.

    tqdm is imported here rather than required, matching the rest of the
    package: without it the work still runs, silently.
    """
    if not enabled:
        yield None
        return
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        yield None
        return
    # The count is fractional because a hypothesis advances in pieces, so it
    # needs an explicit precision: tqdm's default renders float counts in full
    # and prints things like 7.54405440544057/8.0. One decimal says "part way
    # through the eighth" and nothing more.
    bar = tqdm(
        total=float(total),
        desc=desc,
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| "
            "{n:.1f}/{total:.0f} [{elapsed}<{remaining}]"
        ),
    )

    def update(amount: float) -> None:
        # Callers advance by fractions obtained through division, which need
        # not sum to a whole number. Owning the bar means owning the invariant
        # that it never runs past its own total, so callers can do the
        # arithmetic that reads naturally and leave the edge here.
        room = bar.total - bar.n
        if amount > room:
            amount = room
        if amount > 0.0:
            bar.update(amount)

    try:
        yield update
    finally:
        bar.n = bar.total
        bar.close()


def _adjust(p_values: np.ndarray, correction: Correction) -> np.ndarray:
    """Family-wise adjustment across every hypothesis in one report."""
    if correction is None:
        return p_values.copy()
    n_hypotheses = p_values.shape[0]
    if correction == "bonferroni":
        return np.minimum(1.0, n_hypotheses * p_values)
    if correction != "holm":
        raise ValueError(f"unknown correction: {correction!r}")
    order = np.argsort(p_values, kind="stable")
    scaled = (n_hypotheses - np.arange(n_hypotheses)) * p_values[order]
    adjusted_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return adjusted


def _with_adjusted(
    comparison: PairwiseComparison,
    adjusted: float,
    alpha: float,
) -> PairwiseComparison:
    from dataclasses import replace

    return replace(
        comparison,
        adjusted_p_value=float(adjusted),
        # Monte Carlo p-values are discrete multiples of 1/(B+1), so equality
        # with alpha is attainable and the convention rejects there.
        significant=bool(adjusted <= alpha),
    )


def _compare_arrays(
    x: np.ndarray,
    y: np.ndarray,
    *,
    metric: str,
    baseline_name: str,
    candidate_name: str,
    confidence_level: float,
    n_resamples: int,
    alternative: Alternative,
    test_method: TestMethod,
    interval_rng: np.random.Generator,
    test_rng: np.random.Generator,
    random_state: int | None,
    resample_batch_size: int,
    units: tuple[np.ndarray, int] | None,
    progress: Callable[[float], None] | None = None,
) -> PairwiseComparison:
    """Compare two aligned per-user arrays. Raw p-value only; adjust later."""
    rows = y - x
    n_samples = int(rows.shape[0])

    # Everything downstream works on one difference per independent unit. When
    # a user owns several rows, that is their mean, and the estimand is the
    # mean over users rather than over rows -- a user evaluated five times is
    # one user, not five, and weighting by row count would let the protocol
    # decide whose opinion counts more. With equal row counts the two coincide
    # exactly; with unequal ones only this version answers the question the
    # rest of the module is asking.
    if units is None:
        n_units = n_samples
        d, unit_x, unit_y = rows, x, y
    else:
        codes, n_units = units
        counts = np.bincount(codes, minlength=n_units)
        d = _unit_sums(rows, codes, n_units) / counts
        unit_x = _unit_sums(x, codes, n_units) / counts
        unit_y = _unit_sums(y, codes, n_units) / counts

    n_nonzero = int(np.count_nonzero(d))
    baseline_mean = float(unit_x.mean(dtype=np.float64))
    candidate_mean = float(unit_y.mean(dtype=np.float64))
    # Averaging is linear, so this identity survives the reduction above.
    difference = float(d.mean(dtype=np.float64))

    relative_difference = (
        None
        if abs(baseline_mean) <= _ZERO_TOLERANCE
        else float(difference / abs(baseline_mean))
    )

    # One hypothesis is worth 1.0 on the bar, divided evenly between the
    # resampling passes it will make. The interval always resamples; the
    # randomization test resamples a second time, while the bootstrap test
    # reuses those replicates and the t-test needs none.
    passes = 2 if test_method == "randomization" else 1

    advance: Callable[[float], None] | None = None
    if progress is not None:
        given = 0.0

        def advance(fraction: float) -> None:
            nonlocal given
            step = min(fraction / passes, max(0.0, 1.0 - given))
            given += step
            progress(step)

    bootstrap_means = _bootstrap_means(
        d,
        n_resamples=n_resamples,
        rng=interval_rng,
        resample_batch_size=resample_batch_size,
        progress=advance,
    )
    ci_low, ci_high = _interval(
        bootstrap_means,
        confidence_level=confidence_level,
        alternative=alternative,
    )
    standard_error = (
        float(bootstrap_means.std(ddof=1)) if n_resamples > 1 else float("nan")
    )

    if test_method == "t":
        # Deterministic: test_rng is deliberately left unconsumed. Seeds are
        # derived per hypothesis, so that cannot shift any other comparison.
        # The interval above is still resampled, so this does not make the
        # call RNG-free.
        p_value = _t_test_p(d, difference, alternative=alternative, metric=metric)
    else:
        if test_method == "randomization":
            null_statistics = _randomization_means(
                d,
                n_resamples=n_resamples,
                rng=test_rng,
                resample_batch_size=resample_batch_size,
                progress=advance,
        )
        else:
            # Resampling the centered differences is an exact shift of the
            # ordinary bootstrap, so the replicates above already contain the
            # null statistic.
            null_statistics = bootstrap_means - difference

        p_value = _monte_carlo_p(null_statistics, difference, alternative=alternative)

    if progress is not None:
        # Land on a whole hypothesis whatever the chunk arithmetic did.
        progress(max(0.0, 1.0 - given))

    if n_nonzero == 0:
        # Not the low-count case. The two models scored every user identically,
        # so difference 0, interval [0, 0] and p 1 are exactly right rather
        # than degraded, and saying "few observations" would misdescribe them.
        warnings.warn(
            f"{metric!r}: every one of the {n_units} units has a mean paired "
            f"difference of exactly zero, so there is nothing to resample. "
            f"The difference, "
            f"interval and p-value are exact, not estimated.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif n_nonzero < MIN_NONZERO_SAMPLES:
        warnings.warn(
            f"{metric!r}: only {n_nonzero} of {n_units} units have a nonzero "
            f"paired difference, so the empirical difference distribution is "
            f"highly discrete and the percentile interval lands on few distinct "
            f"values. Interpret it cautiously. The estimate itself still uses "
            f"all {n_units} units.",
            RuntimeWarning,
            stacklevel=3,
        )

    return PairwiseComparison(
        metric=metric,
        baseline=baseline_name,
        candidate=candidate_name,
        n_samples=n_samples,
        n_units=n_units,
        n_nonzero=n_nonzero,
        baseline_mean=baseline_mean,
        candidate_mean=candidate_mean,
        difference=difference,
        relative_difference=relative_difference,
        bootstrap_standard_error=standard_error,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence_level=float(confidence_level),
        p_value=p_value,
        adjusted_p_value=p_value,
        significant=p_value <= 1.0 - confidence_level,
        alternative=alternative,
        test_method=test_method,
        interval_method="percentile",
        n_resamples=int(n_resamples),
        random_state=random_state,
    )


def compare_pair(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    metric: str,
    baseline_name: str = "baseline",
    candidate_name: str = "candidate",
    confidence_level: float = 0.95,
    n_resamples: int = 9_999,
    alternative: Alternative = "two-sided",
    test_method: TestMethod = "randomization",
    random_state: int | None = 0,
    resample_batch_size: int = 64,
    show_progress: bool = False,
) -> PairwiseComparison:
    """Compare one candidate against one baseline on one metric.

    The difference is ``candidate - baseline``, so positive values favour the
    candidate. No multiplicity correction is applied: a single comparison is a
    single hypothesis, and ``adjusted_p_value`` equals ``p_value``. Use
    :func:`compare_models` when testing more than one hypothesis together.
    """
    _validate_common(
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        alternative=alternative,
        test_method=test_method,
        resample_batch_size=resample_batch_size,
    )
    x, y, units = _paired_values(
        baseline,
        candidate,
        metric=metric,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
    )
    interval_rng, test_rng = _hypothesis_streams(
        _base_entropy(random_state),
        metric=metric,
        baseline_name=baseline_name,
        candidate_name=candidate_name,
    )
    with _progress(show_progress, 1, f"comparing {metric}") as advance:
        return _compare_arrays(
            x,
            y,
            metric=metric,
            baseline_name=baseline_name,
            candidate_name=candidate_name,
            confidence_level=confidence_level,
            n_resamples=n_resamples,
            alternative=alternative,
            test_method=test_method,
            interval_rng=interval_rng,
            test_rng=test_rng,
            random_state=random_state,
            resample_batch_size=resample_batch_size,
            units=units,
            progress=advance,
            )


def compare_models(
    results: Mapping[str, EvaluationResult],
    *,
    metrics: str | Sequence[str],
    reference: str | None = None,
    confidence_level: float = 0.95,
    n_resamples: int = 9_999,
    alternative: Alternative = "two-sided",
    correction: Correction = "holm",
    test_method: TestMethod = "randomization",
    random_state: int | None = 0,
    resample_batch_size: int = 64,
    show_progress: bool = False,
) -> ComparisonReport:
    """Compare several models across one or more metrics in a single family.

    With ``reference`` set, every other model is compared against it. Without
    it, every unordered pair is compared in mapping insertion order, with the
    earlier model as baseline.

    The correction spans every pair and metric produced by the call, so calling
    this once with three metrics is not the same as calling it three times: the
    family is what the call generates.

    ``show_progress`` draws a bar when tqdm is installed. Cost is linear in
    units and in the number of hypotheses -- roughly a second per hypothesis
    per 25,000 units at the default resample count -- so a large evaluation
    compared across several metrics and models can run for minutes. The bar
    counts hypotheses but advances within each one, so it still moves when a
    call produces only a single very slow comparison.

    Holm is the default because these hypotheses are dependent: they are
    computed over overlapping users, and several metrics on one pair of models
    measure closely related things. Holm controls the family-wise error rate
    under arbitrary dependence. Procedures that assume independence or positive
    dependence are not offered for that reason.

    Each hypothesis draws its own resamples, seeded from its metric and its
    pair of model names. Adding a metric, reordering ``metrics``, or reordering
    ``results`` therefore cannot change a raw comparison that was already in
    the report. Adjusted p-values still move, because the family changed.
    """
    _validate_common(
        confidence_level=confidence_level,
        n_resamples=n_resamples,
        alternative=alternative,
        test_method=test_method,
        resample_batch_size=resample_batch_size,
    )
    if correction not in {"holm", "bonferroni", None}:
        raise ValueError(f"unknown correction: {correction!r}")

    names = list(results)
    if len(names) != len(set(names)):
        raise ValueError("model names must be unique")
    if len(names) < 2:
        raise ValueError("compare_models needs at least two models")
    if reference is not None and reference not in results:
        raise ValueError(f"reference {reference!r} is not among the models")

    metric_names = [metrics] if isinstance(metrics, str) else list(metrics)
    if not metric_names:
        raise ValueError("metrics must contain at least one metric name")
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("metrics must be unique")

    if reference is None:
        if alternative != "two-sided":
            # Without a reference, direction comes from mapping insertion order,
            # which is a cosmetic detail for a two-sided test and the entire
            # hypothesis for a one-sided one: reordering the dict would silently
            # test the opposite claim.
            raise ValueError(
                f"alternative={alternative!r} is directional, so the comparison "
                "must say which model is the baseline. Pass reference=..., or "
                "use alternative='two-sided'."
            )
        pairs = [
            (names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
    else:
        pairs = [(reference, name) for name in names if name != reference]

    # Seeds are derived per hypothesis rather than drawn from a running
    # stream, so a comparison is a function of its own identity and not of its
    # position in the report. Resolve the entropy once here: with
    # random_state=None every hypothesis must share one nondeterministic draw,
    # not make its own.
    base_entropy = _base_entropy(random_state)
    comparisons: list[PairwiseComparison] = []
    total = len(metric_names) * len(pairs)
    with _progress(show_progress, total, "comparing") as advance:
        for metric in metric_names:
            for baseline_name, candidate_name in pairs:
                interval_rng, test_rng = _hypothesis_streams(
                    base_entropy,
                    metric=metric,
                    baseline_name=baseline_name,
                    candidate_name=candidate_name,
                )
                x, y, units = _paired_values(
                    results[baseline_name],
                    results[candidate_name],
                    metric=metric,
                    baseline_name=baseline_name,
                    candidate_name=candidate_name,
                )
                comparisons.append(
                    _compare_arrays(
                        x,
                        y,
                        metric=metric,
                        baseline_name=baseline_name,
                        candidate_name=candidate_name,
                        confidence_level=confidence_level,
                        n_resamples=n_resamples,
                        alternative=alternative,
                        test_method=test_method,
                        interval_rng=interval_rng,
                        test_rng=test_rng,
                        random_state=random_state,
                        resample_batch_size=resample_batch_size,
                        units=units,
                        progress=advance,
                    )
                )

    alpha = 1.0 - confidence_level
    adjusted = _adjust(
        np.array([c.p_value for c in comparisons], dtype=np.float64),
        correction,
    )
    comparisons = [
        _with_adjusted(comparison, value, alpha)
        for comparison, value in zip(comparisons, adjusted)
    ]

    return ComparisonReport(
        comparisons=tuple(comparisons),
        metrics=tuple(metric_names),
        model_names=tuple(names),
        reference=reference,
        correction=correction,
        confidence_level=float(confidence_level),
        alternative=alternative,
        test_method=test_method,
        n_resamples=int(n_resamples),
        random_state=random_state,
    )
