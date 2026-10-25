"""Compact tables of a ComparisonReport: a DataFrame summary, LaTeX and Markdown.

``ComparisonReport.to_frame()`` keeps every column of every hypothesis. The
helpers here pick the ones a reader needs (means, the paired difference, its
interval, the corrected p-value and its direction), orient every row as
"candidate versus reference", and render them without any templating
dependency.

**p-value floor.** A Monte Carlo test with ``B`` resamples cannot report a raw
p below ``1/(B+1)``. When a raw p sits on that floor its corrected value is only
an upper bound (Holm multiplies the smallest p by the family size), so tables
print it as ``<= m/(B+1)`` rounded up, never as the bare number. A t-test has no
such floor.
"""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

    from compresso_recsys.stats.results import ComparisonReport, PairwiseComparison

Layout = Literal["long", "compact"]
FloorStyle = Literal["bound", "raw"]
MarkerStyle = Literal["triangle", "dagger"]

_METRIC_LABELS = {
    "calibrated_recall": "R@{k}",
    "recall": "Recall@{k}",
    "ndcg": "nDCG@{k}",
    "precision": "P@{k}",
    "hit_rate": "HR@{k}",
    "mrr": "MRR@{k}",
    "map": "MAP@{k}",
}
_TEST_NAMES = {
    "randomization": "paired randomization (sign-flip) test",
    "bootstrap": "paired bootstrap test",
    "t": "paired $t$-test",
}
_CORRECTION_NAMES = {"holm": "Holm", "bonferroni": "Bonferroni", None: "no"}
_LATEX_SPECIAL = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
    "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


@dataclass(frozen=True)
class _Row:
    """One hypothesis oriented as ``candidate - baseline``."""

    metric: str
    baseline: str
    candidate: str
    baseline_mean: float
    candidate_mean: float
    difference: float
    ci_low: float
    ci_high: float
    p_value: float
    adjusted_p_value: float
    significant: bool
    n_resamples: int
    test_method: str
    floored: bool

    @property
    def direction(self) -> str:
        if not self.significant:
            return "inconclusive"
        return "better" if self.difference > 0 else "worse"


def _floored(comparison: "PairwiseComparison") -> bool:
    if comparison.test_method == "t" or comparison.n_resamples < 1:
        return False
    return comparison.p_value <= (1.0 + 1e-9) / (comparison.n_resamples + 1)


def _rows(report: "ComparisonReport", reference: str | None) -> list[_Row]:
    rows = []
    for c in report.comparisons:
        floored = _floored(c)
        if reference is None or c.baseline == reference:
            rows.append(_Row(c.metric, c.baseline, c.candidate, c.baseline_mean, c.candidate_mean,
                             c.difference, c.ci_low, c.ci_high, c.p_value, c.adjusted_p_value,
                             c.significant, c.n_resamples, c.test_method, floored))
        elif c.candidate == reference:  # tested the other way round: flip it
            rows.append(_Row(c.metric, c.candidate, c.baseline, c.candidate_mean, c.baseline_mean,
                             -c.difference, -c.ci_high, -c.ci_low, c.p_value, c.adjusted_p_value,
                             c.significant, c.n_resamples, c.test_method, floored))
    if reference is not None and not rows:
        raise ValueError(f"no comparison in this report involves the reference {reference!r}")
    return rows


def _select(rows: list[_Row], metrics: Iterable[str] | None,
            models: Iterable[str] | None) -> tuple[list[_Row], list[str], list[str]]:
    metric_order = list(metrics) if metrics is not None else list(dict.fromkeys(r.metric for r in rows))
    model_order = list(models) if models is not None else list(dict.fromkeys(r.candidate for r in rows))
    for name in metric_order:
        if name not in {r.metric for r in rows}:
            raise ValueError(f"metric {name!r} is not in this report")
    kept = [r for r in rows if r.metric in metric_order and r.candidate in model_order]
    return kept, metric_order, model_order


def _warn_if_all_floored(rows: list[_Row]) -> None:
    if rows and all(r.floored for r in rows):
        b = rows[0].n_resamples
        warnings.warn(
            f"every p-value in this table sits on the Monte Carlo floor 1/(B+1) with B={b}; "
            "the corrected values are upper bounds. Increase n_resamples to resolve them.",
            stacklevel=3,
        )


def _resolve_reference(report: "ComparisonReport", reference: str | None) -> str | None:
    return report.reference if reference is None else reference


def summary(report: "ComparisonReport", *, reference: str | None = None,
            metrics: Iterable[str] | None = None, models: Iterable[str] | None = None) -> "pd.DataFrame":
    import pandas as pd

    reference = _resolve_reference(report, reference)
    rows, metric_order, model_order = _select(_rows(report, reference), metrics, models)
    columns = ["metric", "baseline", "candidate", "baseline_mean", "candidate_mean", "difference",
               "ci_low", "ci_high", "adjusted_p_value", "significant", "direction", "p_floored"]
    rank = {(m, n): (metric_order.index(m), model_order.index(n)) for m in metric_order for n in model_order}
    rows = sorted(rows, key=lambda r: rank[(r.metric, r.candidate)])
    return pd.DataFrame(
        [[r.metric, r.baseline, r.candidate, r.baseline_mean, r.candidate_mean, r.difference,
          r.ci_low, r.ci_high, r.adjusted_p_value, r.significant, r.direction, r.floored] for r in rows],
        columns=columns,
    )


# ---------------------------------------------------------------- formatting helpers


def _escape(text: str) -> str:
    return "".join(_LATEX_SPECIAL.get(ch, ch) for ch in str(text))


def metric_label(key: str, labels: Mapping[str, str] | None = None) -> str:
    """'calibrated_recall@20' -> 'R@20'; unknown keys are returned unchanged."""
    if labels and key in labels:
        return labels[key]
    match = re.fullmatch(r"([a-z_]+)@(\d+)", key)
    if match and match.group(1) in _METRIC_LABELS:
        return _METRIC_LABELS[match.group(1)].format(k=match.group(2))
    return key


def _ceil_one_digit(value: float) -> tuple[int, int]:
    """Mantissa and exponent of ``value`` with one significant digit, rounded up."""
    exponent = math.floor(math.log10(value))
    mantissa = math.ceil(round(value / 10**exponent, 9))
    if mantissa == 10:
        mantissa, exponent = 1, exponent + 1
    return mantissa, exponent


def _num(value: float, digits: int, *, signed: bool = False, latex: bool = True) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "--"
    # A nonzero value that rounds to zero would print as "-0.000" or "+0.000",
    # a sign with no visible magnitude. Show up to two more digits instead, so
    # -0.0003 reads as -0.0003; only values below that print as zero.
    shown = digits
    while value and round(value, shown) == 0 and shown < digits + 2:
        shown += 1
    text = f"{value:+.{shown}f}" if signed else f"{value:.{shown}f}"
    return text.replace("-", "$-$") if (signed and latex) else text


def _p_cell(row: _Row, *, latex: bool, floor_style: FloorStyle, digits: int = 3) -> str:
    if row.floored:
        value = 1.0 / (row.n_resamples + 1) if floor_style == "raw" else row.adjusted_p_value
        m, e = _ceil_one_digit(value)
        if latex:
            sign = "<" if floor_style == "raw" else r"\le"
            return rf"${sign} {m}{{\cdot}}10^{{{e}}}$"
        return f"{'<' if floor_style == 'raw' else '≤'} {m}e{e}"
    if row.adjusted_p_value < 10 ** (-digits):
        m, e = _ceil_one_digit(row.adjusted_p_value)
        return rf"${m}{{\cdot}}10^{{{e}}}$" if latex else f"{m}e{e}"
    return f"{row.adjusted_p_value:.{digits}f}"


def _marker(row: _Row, style: MarkerStyle, *, latex: bool) -> str:
    if not row.significant:
        return ""
    better = row.difference > 0
    if not latex:
        return " ▲" if better else " ▼"
    if style == "dagger":
        return r"$^{\dagger}$" if better else r"$^{\ddagger}$"
    return r"$^{\blacktriangle}$" if better else r"$^{\blacktriangledown}$"


def _caption(report: "ComparisonReport", rows: list[_Row], reference: str | None,
             floor_style: FloorStyle, marker: MarkerStyle) -> str:
    level = int(round(report.confidence_level * 100))
    test = _TEST_NAMES.get(report.test_method, _escape(report.test_method))
    family = len(report.comparisons)
    up, down = ((r"$^{\dagger}$", r"$^{\ddagger}$") if marker == "dagger"
                else (r"$^{\blacktriangle}$", r"$^{\blacktriangledown}$"))
    b = f"{report.n_resamples:,}".replace(",", "{,}")
    parts = []
    if any(r.metric.startswith("calibrated_recall@") for r in rows):
        parts.append(r"R@$k$: calibrated recall, hits$/\min(k,|R_u|)$")
    parts.append(f"{level}\\% paired bootstrap CIs on the per-user difference")
    parts.append(test if report.test_method == "t" else f"{test} with $B={b}$")
    parts.append(f"{_CORRECTION_NAMES.get(report.correction, _escape(str(report.correction)))} correction "
                 f"over {family} hypothes{'is' if family == 1 else 'es'}")
    parts.append(f"{up}/{down}: significantly better/worse"
                 + (f" than {_escape(reference)}" if reference else ""))
    if any(r.floored for r in rows):
        parts.append(r"$<$: unadjusted p at the Monte Carlo floor $1/(B{+}1)$" if floor_style == "raw"
                     else r"$\le$: raw p at the Monte Carlo floor $1/(B{+}1)$, so the corrected value is an upper bound")
    return "; ".join(parts) + "."


def to_latex(report: "ComparisonReport", *, layout: Layout = "long", reference: str | None = None,
             metrics: Iterable[str] | None = None, models: Iterable[str] | None = None,
             metric_labels: Mapping[str, str] | None = None, digits: int = 4,
             floor_style: FloorStyle = "bound", marker: MarkerStyle = "triangle",
             caption: str | None = None, label: str | None = None, table_env: bool = True) -> str:
    if layout not in ("long", "compact"):
        raise ValueError(f"layout must be 'long' or 'compact', got {layout!r}")
    if floor_style not in ("bound", "raw"):
        raise ValueError(f"floor_style must be 'bound' or 'raw', got {floor_style!r}")
    reference = _resolve_reference(report, reference)
    rows, metric_order, model_order = _select(_rows(report, reference), metrics, models)
    _warn_if_all_floored(rows)
    labels = {m: _escape(metric_label(m, metric_labels)) for m in metric_order}
    body = (_latex_long if layout == "long" else _latex_compact)(
        rows, metric_order, model_order, reference, labels, digits, floor_style, marker)
    note = _caption(report, rows, reference, floor_style, marker)
    if not table_env:
        return body + "\n% " + note
    out = [r"\begin{table}", r"\centering", rf"\caption{{{caption + ' ' if caption else ''}{note}}}"]
    if label:
        out.append(rf"\label{{{label}}}")
    out += [body, r"\end{table}"]
    return "\n".join(out)


def _latex_long(rows, metric_order, model_order, reference, labels, digits, floor_style, marker):
    multi_baseline = len({r.baseline for r in rows}) > 1
    head = (["Model"] + (["vs."] if multi_baseline else [])
            + ["Metric", "Mean", r"$\Delta$", r"95\% CI ($\Delta$)", r"$p_{\mathrm{adj}}$"])
    lines = [rf"\begin{{tabular}}{{{'l' + ('l' if multi_baseline else '') + 'lrrcr'}}}", r"\toprule",
             " & ".join(head) + r" \\", r"\midrule"]
    if reference and not multi_baseline:
        for i, m in enumerate(metric_order):
            mean = next((r.baseline_mean for r in rows if r.metric == m), None)
            if mean is not None:
                lines.append(" & ".join([_escape(reference) if i == 0 else "", labels[m],
                                         _num(mean, digits), "--", "--", "--"]) + r" \\")
        lines.append(r"\midrule")
    for i, model in enumerate(model_order):
        mine = {r.metric: r for r in rows if r.candidate == model}
        for j, m in enumerate(x for x in metric_order if x in mine):
            r = mine[m]
            cells = [_escape(model) if j == 0 else ""]
            if multi_baseline:
                cells.append(_escape(r.baseline) if j == 0 else "")
            cells += [labels[m], _num(r.candidate_mean, digits) + _marker(r, marker, latex=True),
                      _num(r.difference, digits, signed=True),
                      f"[{_num(r.ci_low, digits, signed=True)}, {_num(r.ci_high, digits, signed=True)}]",
                      _p_cell(r, latex=True, floor_style=floor_style)]
            lines.append(" & ".join(cells) + r" \\")
        if i < len(model_order) - 1:
            lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def _latex_compact(rows, metric_order, model_order, reference, labels, digits, floor_style, marker):
    means, cells = {}, {}
    for r in rows:
        means[(r.candidate, r.metric)] = r.candidate_mean
        means[(r.baseline, r.metric)] = r.baseline_mean
        cells[(r.candidate, r.metric)] = (_num(r.candidate_mean, digits), _marker(r, marker, latex=True))
    order = ([reference] if reference else []) + [m for m in model_order if m != reference]
    best = {m: max((means[(n, m)] for n in order if (n, m) in means), default=None) for m in metric_order}
    lines = [rf"\begin{{tabular}}{{l{'r' * len(metric_order)}}}", r"\toprule",
             " & ".join(["Model"] + [labels[m] for m in metric_order]) + r" \\", r"\midrule"]
    for name in order:
        out = [_escape(name)]
        for m in metric_order:
            if (name, m) not in means:
                out.append("--")
                continue
            text, mark = cells.get((name, m), (_num(means[(name, m)], digits), ""))
            if best[m] is not None and means[(name, m)] == best[m]:
                text = rf"\textbf{{{text}}}"
            out.append(text + mark)
        lines.append(" & ".join(out) + r" \\")
        if name == reference:
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def to_markdown(report: "ComparisonReport", *, reference: str | None = None,
                metrics: Iterable[str] | None = None, models: Iterable[str] | None = None,
                digits: int = 4, floor_style: FloorStyle = "bound") -> str:
    reference = _resolve_reference(report, reference)
    rows, metric_order, model_order = _select(_rows(report, reference), metrics, models)
    _warn_if_all_floored(rows)
    lines = ["| Model | vs. | Metric | Mean | Δ | 95% CI (Δ) | p (adj.) |",
             "|---|---|---|---:|---:|:---:|---:|"]
    for model in model_order:
        mine = {r.metric: r for r in rows if r.candidate == model}
        for m in (x for x in metric_order if x in mine):
            r = mine[m]
            lines.append(
                f"| {model} | {r.baseline} | {metric_label(m)} | "
                f"{_num(r.candidate_mean, digits, latex=False)}{_marker(r, 'triangle', latex=False)} | "
                f"{_num(r.difference, digits, signed=True, latex=False)} | "
                f"[{_num(r.ci_low, digits, signed=True, latex=False)}, "
                f"{_num(r.ci_high, digits, signed=True, latex=False)}] | "
                f"{_p_cell(r, latex=False, floor_style=floor_style)} |")
    corr = _CORRECTION_NAMES.get(report.correction, str(report.correction))
    lines.append("")
    lines.append(f"{int(round(report.confidence_level * 100))}% paired bootstrap CIs; "
                 f"{_TEST_NAMES.get(report.test_method, report.test_method).replace('$', '')}"
                 f"{'' if report.test_method == 't' else f' (B={report.n_resamples})'}; "
                 f"{corr} correction over {len(report.comparisons)} hypotheses; ▲/▼ significantly better/worse"
                 + (f" than {reference}" if reference else "")
                 + ("; ≤: p at the Monte Carlo floor, corrected value is an upper bound"
                    if any(r.floored for r in rows) and floor_style == "bound" else "") + ".")
    return "\n".join(lines)
