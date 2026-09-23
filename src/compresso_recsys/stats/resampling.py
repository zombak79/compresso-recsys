"""The two resampling procedures, and the parametric cross-check.

A paired bootstrap over units gives the interval; a paired sign-flip
randomization gives the p-value. Neither resamples the two models
independently. :func:`_t_test_p` is the familiar alternative null.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

import numpy as np
from scipy.stats import ttest_1samp

from compresso_recsys.stats.results import Alternative


MAX_CHUNK_ELEMENTS = 8_000_000


MIN_NONZERO_SAMPLES = 30


_ZERO_TOLERANCE = 1e-12


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


def _effective_batch(requested: int, n: int) -> int:
    """Bound a chunk by total elements, not by replicate count."""
    return max(1, min(int(requested), MAX_CHUNK_ELEMENTS // max(int(n), 1)))


def _bootstrap_means(
    d: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    resample_batch_size: int,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Mean of ``d`` over ``n_resamples`` resamples of its rows, with replacement."""
    n = d.shape[0]
    out = np.empty(n_resamples, dtype=np.float64)
    step = _effective_batch(resample_batch_size, n)
    for start in range(0, n_resamples, step):
        size = min(step, n_resamples - start)
        indices = rng.integers(0, n, size=(size, n))
        out[start : start + size] = d[indices].mean(axis=1)
        if progress is not None:
            progress(size / n_resamples)
    return out


def _randomization_means(
    d: np.ndarray,
    *,
    n_resamples: int,
    rng: np.random.Generator,
    resample_batch_size: int,
    progress: Callable[[float], None] | None = None,
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
        if progress is not None:
            progress(size / n_resamples)
    return out


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
    is loose.

    Note also that the two tests do not share a null. This one asks whether the
    population mean difference is zero; the randomization test asks whether the
    two model labels are exchangeable within each user, which additionally
    implies the differences are symmetric about zero. Exchangeability is the
    stronger assumption, so a disagreement between them can reflect the nulls
    differing rather than an approximation being strained -- a skewed difference
    distribution centred on zero satisfies one and not the other.
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
