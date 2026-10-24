"""Evaluation rows in, one paired difference per independent unit out.

Every procedure downstream assumes the units it resamples are independent, so
a user contributing several rows is reduced to their mean before anything
else happens.
"""

from __future__ import annotations

import warnings

import numpy as np

from compresso_recsys.evaluation import EvaluationResult


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
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError(f"per-user values for {metric!r} contain non-finite entries")
    units = _unit_codes(left)
    n_units = x.shape[0] if units is None else units[1]
    if n_units < 2:
        raise ValueError(
            "paired comparison needs at least 2 independent units, got "
            f"{n_units} from {x.shape[0]} evaluable samples"
        )
    return x, y, units


def _unit_codes(sample_ids: np.ndarray) -> tuple[np.ndarray, int] | None:
    """Group rows by identifier, or ``None`` when every row is its own unit.

    Repeated identifiers mean one evaluation unit produced several rows.
    :func:`compresso_recsys.retrieval.build_eval_holdout` does exactly that with
    an explicit ``eval_draws=5``: each user is split into fold-in and scored
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
