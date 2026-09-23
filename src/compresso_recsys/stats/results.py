"""What a comparison returns, and the vocabulary its options are written in."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - pandas is imported lazily in to_frame
    import pandas as pd




Alternative = Literal["two-sided", "greater", "less"]


Correction = Literal["holm", "bonferroni"] | None


TestMethod = Literal["randomization", "bootstrap", "t"]


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
        """Fraction of units whose **mean** paired difference is exactly zero.

        With one row per user that means the two models scored them
        identically. With several, it means those rows cancelled: a user who
        gained on one draw and lost the same amount on another is tied here
        even though no single row was.

        High tie rates are normal for ranking metrics at small cutoffs and are
        not a defect. They say the two models come out level for that share of
        the population, which is itself a finding, and they are why
        ``n_nonzero`` is reported: it bounds how discrete the randomization
        test's null distribution can be, since flipping the sign of a zero
        changes nothing. The estimate still rests on every unit.
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
