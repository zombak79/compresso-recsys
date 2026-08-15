"""Paired statistical comparison of recommender evaluations.

Two models evaluated on the same users can be compared far more precisely than
their aggregate means suggest, because most of the variation between users is
shared. Everything here works on the paired per-user difference

.. math::

    d_u = m_u^{(b)} - m_u^{(a)},

and never resamples the two models independently.

Two procedures, each doing the job it is best at:

* **Effect size** — a paired bootstrap over users gives a confidence interval
  for the mean difference. This is the primary output; report it.
* **Hypothesis test** — a paired sign-flip randomization test gives the
  p-value. Its null is *paired label exchangeability*: swapping which model
  produced which score, independently for each user, leaves the joint
  distribution unchanged. Under it the sign of every paired difference is
  arbitrary, so the test is exact up to Monte Carlo error. That assumes no
  parametric family, which is not the same as assuming nothing — exchangeability
  is a real assumption, and it requires that users are the independent units
  being resampled. It is the default in the information-retrieval evaluation
  literature.

``test_method="t"`` runs a paired t-test instead, as a one-sample test on the
same differences. Smucker, Allan and Carterette found the two agree closely on
retrieval data, so it is available as a familiar cross-check and for the
occasions when a p-value below the ``1 / (n_resamples + 1)`` floor is wanted.
The randomization test remains the default: it is exact under exchangeability
where the t-test is asymptotic, and it stays valid where a heavily tied,
skewed difference distribution strains the normal approximation.

Every procedure works on one difference per independent unit. When a protocol
gives a user several evaluation rows, that user is first reduced to the mean of
their rows, so one user is one observation however many rows they produced.
``n_samples`` counts rows, ``n_units`` counts users, and they are equal whenever
each row is its own unit.

Ranking differences are dominated by exact ties: for most users both models
return the same items and the difference is zero. Every comparison therefore
reports ``n_nonzero`` and ``tie_rate`` over units.

Tied users are not spare. They carry no *sign* information — flipping the sign
of a zero changes nothing, so ``n_nonzero`` alone governs the combinatorial
support of the randomization test. But they are part of the empirical
population, and the mean difference and the paired bootstrap interval are
computed over all ``n_units`` of them. Thirty users who all differ by +1 and
ten thousand users of whom thirty differ by +1 share an ``n_nonzero`` and
describe entirely different systems.

Inference here is conditional on the fitted models. It answers whether an
advantage is stable across resampled users, not whether it survives retraining
with a different seed. Report seed variation separately.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from scipy.stats import ttest_1samp

from compresso_recsys.evaluation import EvaluationResult

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters for typing
    import pandas as pd

__all__ = [
    "ComparisonReport",
    "PairwiseComparison",
    "compare_models",
    "compare_pair",
]

Alternative = Literal["two-sided", "greater", "less"]
Correction = Literal["holm", "bonferroni"] | None
TestMethod = Literal["randomization", "bootstrap", "t"]

#: Chunks of random draws are bounded by element count rather than by replicate
#: count, because a chunk is ``batch * n`` wide. Bounding replicates alone would
#: allocate gigabytes once ``n`` reaches the millions.
MAX_CHUNK_ELEMENTS = 8_000_000

#: Below this many untied users the empirical difference distribution is too
#: discrete for a percentile interval to be read literally, and the comparison
#: warns. This bounds the *shape* of the resampled distribution, not the amount
#: of data behind the estimate, which is always ``n_units``.
MIN_NONZERO_SAMPLES = 30

_ZERO_TOLERANCE = 1e-12

_FRAME_COLUMNS = (
    "metric",
    "baseline",
    "candidate",
    "n_samples",
    "n_units",
    "n_nonzero",
    "tie_rate",
    "baseline_mean",
    "candidate_mean",
    "difference",
    "relative_difference",
    "ci_low",
    "ci_high",
    "confidence_level",
    "bootstrap_standard_error",
    "p_value",
    "adjusted_p_value",
    "significant",
    "direction",
    "alternative",
    "test_method",
    "interval_method",
    "n_resamples",
    "random_state",
)


@dataclass(frozen=True)
class PairwiseComparison:
    """One model-versus-model hypothesis for one metric.

    ``difference`` is always ``candidate - baseline``, so positive values favour
    the candidate.
    """

    metric: str
    baseline: str
    candidate: str
    n_samples: int
    n_units: int
    n_nonzero: int
    baseline_mean: float
    candidate_mean: float
    difference: float
    relative_difference: float | None
    bootstrap_standard_error: float
    ci_low: float
    ci_high: float
    confidence_level: float
    p_value: float
    adjusted_p_value: float
    significant: bool
    alternative: Alternative
    test_method: TestMethod
    interval_method: str
    n_resamples: int
    random_state: int | None

    @property
    def tie_rate(self) -> float:
        """Fraction of evaluated users the two models scored identically.

        High tie rates are normal for ranking metrics at small cutoffs and are
        not a defect. They say the two models agree for that share of the
        population, which is itself a finding, and they are why ``n_nonzero``
        is reported: it bounds how discrete the randomization test's null
        distribution can be, while the estimate still rests on every unit.
        """
        if self.n_units == 0:
            return 0.0
        return 1.0 - self.n_nonzero / self.n_units

    @property
    def direction(self) -> str:
        """``'better'``, ``'worse'`` or ``'inconclusive'``."""
        if not self.significant:
            return "inconclusive"
        return "better" if self.difference > 0 else "worse"

    def to_dict(self) -> dict[str, Any]:
        """Return this comparison as a flat dictionary, including ``direction``."""
        return {column: getattr(self, column) for column in _FRAME_COLUMNS}


@dataclass(frozen=True)
class ComparisonReport:
    """Every hypothesis produced by one :func:`compare_models` call.

    The multiple-testing correction applies across the whole report, so a
    report is the unit of analysis rather than any single comparison in it.
    """

    comparisons: tuple[PairwiseComparison, ...]
    metrics: tuple[str, ...]
    model_names: tuple[str, ...]
    reference: str | None
    correction: Correction
    confidence_level: float
    alternative: Alternative
    test_method: TestMethod
    n_resamples: int
    random_state: int | None

    def __len__(self) -> int:
        return len(self.comparisons)

    def __iter__(self):
        return iter(self.comparisons)

    def to_frame(self) -> "pd.DataFrame":
        """Return one row per hypothesis with a fixed column order."""
        import pandas as pd

        return pd.DataFrame(
            [comparison.to_dict() for comparison in self.comparisons],
            columns=list(_FRAME_COLUMNS),
        )


def _base_entropy(random_state: int | None) -> int:
    """Entropy every hypothesis in one call derives its seeds from.

    ``random_state=None`` asks for a nondeterministic run. Feeding it straight
    into the derivation below would hash the string ``"None"`` into a fixed
    value and silently make the call reproducible, so draw fresh operating
    system entropy once here instead. Hypotheses stay order-invariant within
    the call, and the call stays nondeterministic across runs.
    """
    if random_state is None:
        return int(np.random.SeedSequence().entropy)
    return int(random_state)


def _hypothesis_streams(
    base_entropy: int,
    *,
    metric: str,
    baseline_name: str,
    candidate_name: str,
) -> tuple[np.random.Generator, np.random.Generator]:
    """Independent generators for the interval and for the test.

    Seeds are derived from the identity of the hypothesis rather than taken
    from a position in a shared stream. A comparison therefore draws the same
    resamples no matter what else the report contains, or in what order:
    adding a metric or reordering the model mapping cannot perturb a result
    that was already there.

    Model names are sorted, so reversing a pair reuses its draws. The reported
    difference and interval then mirror exactly rather than picking up
    unrelated resampling noise. Orientation itself still follows insertion
    order; only the seed is canonical.

    Each component is length-prefixed before hashing, so no combination of
    names and metrics can collide by running into its neighbour. blake2b
    rather than :func:`hash`: string hashing is salted per process, and a seed
    that changed between runs would be worse than the ordering it fixes.

    The two generators are spawned from that seed rather than drawn in turn.
    Sharing one would make the confidence interval depend on ``test_method``,
    since the randomization test consumes draws the bootstrap test does not.
    """
    digest = hashlib.blake2b(digest_size=32)
    for part in (str(base_entropy), metric, *sorted((baseline_name, candidate_name))):
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    seed = int.from_bytes(digest.digest(), "big")
    interval, test = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(interval), np.random.default_rng(test)


def _validate_common(
    *,
    confidence_level: float,
    n_resamples: int,
    alternative: str,
    test_method: str,
    resample_batch_size: int,
) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError(f"unknown alternative: {alternative!r}")
    if test_method not in {"randomization", "bootstrap", "t"}:
        raise ValueError(f"unknown test_method: {test_method!r}")
    if resample_batch_size < 1:
        raise ValueError("resample_batch_size must be >= 1")


def _paired_values(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    metric: str,
    baseline_name: str,
    candidate_name: str,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, int] | None]:
    """Return aligned per-user arrays and how their rows group into units.

    Refuses anything that is not paired. The third element groups rows that
    share an identifier, or is ``None`` when every row is its own unit.
    """
    for name, result in ((baseline_name, baseline), (candidate_name, candidate)):
        if not isinstance(result, EvaluationResult):
            raise TypeError(f"{name} must be an EvaluationResult")
        if result.per_user is None:
            raise ValueError(
                f"{name} was evaluated with collect_per_user=False, so it holds no "
                "per-user values; paired comparison needs them"
            )
        if metric not in result.per_user:
            available = ", ".join(sorted(result.per_user))
            raise KeyError(f"{name} has no metric {metric!r}; available: {available}")

    left, right = baseline.sample_ids, candidate.sample_ids
    assert left is not None and right is not None  # implied by per_user
    if left.shape[0] != right.shape[0] or not np.array_equal(left, right):
        raise ValueError(
            f"{baseline_name} and {candidate_name} were not evaluated on the same "
            "samples in the same order. Paired analysis compares each evaluation "
            "unit against itself, so sample_ids must match exactly, including "
            "order. Re-evaluate both models on identical rows rather than "
            "reordering or intersecting after the fact."
        )

    # Matching identifiers say the same users were scored. They cannot say the
    # users were scored against the same relevant items, which is the other half
    # of what pairing assumes and the half a positional identifier hides
    # completely: two evaluations on unrelated datasets both number their rows
    # from zero.
    left_print = baseline.target_fingerprint
    right_print = candidate.target_fingerprint
    if left_print is None or right_print is None:
        warnings.warn(
            f"{baseline_name} or {candidate_name} carries no target fingerprint, "
            "so the comparison cannot confirm both models were scored against "
            "the same relevant items. Results built by hand rather than by an "
            "evaluator are unverifiable this way; check the pairing yourself.",
            RuntimeWarning,
            stacklevel=3,
        )
    elif left_print != right_print:
        raise ValueError(
            f"{baseline_name} and {candidate_name} were evaluated against "
            "different target matrices. Their sample_ids match, so this would "
            "otherwise have paired users who were scored on different relevant "
            "items. Re-evaluate both models against the same targets."
        )

    x = np.asarray(baseline.per_user[metric], dtype=np.float64)
    y = np.asarray(candidate.per_user[metric], dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(
            f"per-user arrays for {metric!r} differ in length: "
            f"{x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[0] < 2:
        raise ValueError(
            f"paired comparison needs at least 2 evaluable samples, got {x.shape[0]}"
        )
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"per-user values for {metric!r} contain non-finite entries")
    return x, y, _unit_codes(left)


def _effective_batch(requested: int, n: int) -> int:
    """Bound a chunk by total elements, not by replicate count."""
    return max(1, min(int(requested), MAX_CHUNK_ELEMENTS // max(int(n), 1)))


def _bootstrap_means(
    d: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    resample_batch_size: int,
) -> np.ndarray:
    """Mean of ``d`` over ``n_resamples`` resamples of its rows, with replacement."""
    n = d.shape[0]
    out = np.empty(n_resamples, dtype=np.float64)
    step = _effective_batch(resample_batch_size, n)
    for start in range(0, n_resamples, step):
        size = min(step, n_resamples - start)
        indices = rng.integers(0, n, size=(size, n))
        out[start : start + size] = d[indices].mean(axis=1)
    return out


def _randomization_means(
    d: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    resample_batch_size: int,
) -> np.ndarray:
    """Mean of ``d`` under ``n_resamples`` uniform sign assignments."""
    n = d.shape[0]
    out = np.empty(n_resamples, dtype=np.float64)
    step = _effective_batch(resample_batch_size, n)
    for start in range(0, n_resamples, step):
        size = min(step, n_resamples - start)
        # int8 signs cost an eighth of float64 and give identical products.
        signs = rng.integers(0, 2, size=(size, n), dtype=np.int8) * 2 - 1
        out[start : start + size] = (signs * d).mean(axis=1)
    return out


def _unit_codes(sample_ids: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Group rows by identifier, or ``None`` when every row is its own unit.

    Repeated identifiers mean one evaluation unit produced several rows.
    :func:`compresso_recsys.retrieval.build_eval_holdout` does exactly that at
    its default ``eval_draws=5``: each user is split into fold-in and scored
    parts five times, so 2,500 users produce 12,500 rows. Those rows are not
    independent, and resampling them as though they were understates the
    interval by the square root of the design effect -- on GoodBooks, an
    interval 27 to 44 percent too narrow.

    Returning ``None`` when every row is its own unit lets the ordinary
    row-level paths run unchanged, so results for the common case are
    bit-for-bit what they were before repeated rows were handled at all.

    The statistics literature calls this cluster sampling, and the references
    use that word. It is avoided here because :mod:`compresso.clustering` means
    something entirely unrelated -- grouping items into cluster graphs -- and
    one of the two had to give.
    """
    codes, inverse = np.unique(sample_ids, return_inverse=True)
    n_units = int(codes.shape[0])
    if n_units == sample_ids.shape[0]:
        return None
    return inverse.astype(np.int64, copy=False), n_units


def _unit_sums(d: np.ndarray, codes: np.ndarray, n_units: int) -> np.ndarray:
    """Total paired difference per unit."""
    return np.bincount(codes, weights=d, minlength=n_units).astype(np.float64)


def _monte_carlo_p(
    null_statistics: np.ndarray,
    observed: float,
    *,
    alternative: Alternative,
) -> float:
    """Finite-sample Monte Carlo p-value, never zero and never above one."""
    if alternative == "two-sided":
        extreme = np.abs(null_statistics) >= abs(observed)
    elif alternative == "greater":
        extreme = null_statistics >= observed
    else:
        extreme = null_statistics <= observed
    return float((1 + int(extreme.sum())) / (null_statistics.shape[0] + 1))


def _t_test_p(
    d: np.ndarray,
    difference: float,
    *,
    alternative: Alternative,
    metric: str,
) -> float:
    """Paired t-test, as a one-sample test on the paired differences.

    One-sample on ``d`` rather than ``ttest_rel(y, x)``. The two are
    mathematically identical, but the bootstrap and the randomization test both
    consume the same ``d``, and letting this derive its own would mean any
    precision divergence surfaced as the three methods disagreeing about
    statistics rather than about floating point. ``d`` holds one value per
    independent unit, so when a user owns several rows this is already their
    mean and all three methods test the same estimand.

    Unlike the resampled tests this has no Monte Carlo floor, so it can report
    p-values far below ``1 / (n_resamples + 1)``. Treat those with the caution
    any far-tail normal approximation deserves: the Berry-Esseen bound on the
    error of the approximation is governed by the number of *untied* units, and
    is loose. Agreement with the randomization test is reassuring; disagreement
    means one of the two approximations is strained, and which one is a question
    about the tie rate and the skew of the nonzero differences rather than about
    the resample count.
    """
    if np.ptp(d) == 0:
        # Zero sample variance: the t statistic is 0/0 or x/0, and scipy
        # returns nan or exactly zero. All differences equal means there is
        # nothing to estimate a standard error from.
        if difference == 0.0:
            return 1.0
        raise ValueError(
            f"{metric!r}: every paired difference is identical, so the t "
            f"statistic is undefined -- there is no sample variance to divide "
            f"by. scipy would return exactly 0.0, which is a verdict the test "
            f"cannot support. Use test_method='randomization', which is exact "
            f"here and reports its resolution floor."
        )
    return float(ttest_1samp(d, 0.0, alternative=alternative).pvalue)


def _interval(
    bootstrap_means: np.ndarray,
    *,
    confidence_level: float,
    alternative: Alternative,
) -> tuple[float, float]:
    """Percentile interval oriented to match the alternative.

    A one-sided test beside a two-sided interval can report a significant
    result next to an interval containing zero, so the orientation follows.
    """
    alpha = 1.0 - confidence_level
    if alternative == "two-sided":
        low, high = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
        return float(low), float(high)
    if alternative == "greater":
        return float(np.quantile(bootstrap_means, alpha)), float("inf")
    return float("-inf"), float(np.quantile(bootstrap_means, 1 - alpha))


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

    bootstrap_means = _bootstrap_means(
        d,
        n_resamples=n_resamples,
        rng=interval_rng,
        resample_batch_size=resample_batch_size,
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
            )
        else:
            # Resampling the centered differences is an exact shift of the
            # ordinary bootstrap, so the replicates above already contain the
            # null statistic.
            null_statistics = bootstrap_means - difference

        p_value = _monte_carlo_p(null_statistics, difference, alternative=alternative)

    if n_nonzero == 0:
        # Not the low-count case. The two models scored every user identically,
        # so difference 0, interval [0, 0] and p 1 are exactly right rather
        # than degraded, and saying "few observations" would misdescribe them.
        warnings.warn(
            f"{metric!r}: the two models scored all {n_units} units "
            f"identically, so there is nothing to resample. The difference, "
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
) -> ComparisonReport:
    """Compare several models across one or more metrics in a single family.

    With ``reference`` set, every other model is compared against it. Without
    it, every unordered pair is compared in mapping insertion order, with the
    earlier model as baseline.

    The correction spans every pair and metric produced by the call, so calling
    this once with three metrics is not the same as calling it three times: the
    family is what the call generates.

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
