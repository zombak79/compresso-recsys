from __future__ import annotations

import sys
import warnings

import pytest

from compresso_recsys.stats import ComparisonReport, PairwiseComparison


def _cmp(metric, baseline, candidate, b_mean, c_mean, low, high, p, p_adj,
         significant=True, *, n_resamples=9_999, test_method="randomization"):
    return PairwiseComparison(
        metric=metric, baseline=baseline, candidate=candidate, n_samples=100, n_units=100,
        n_nonzero=90, baseline_mean=b_mean, candidate_mean=c_mean, difference=c_mean - b_mean,
        relative_difference=(c_mean - b_mean) / b_mean, bootstrap_standard_error=0.01,
        ci_low=low, ci_high=high, confidence_level=0.95, p_value=p, adjusted_p_value=p_adj,
        significant=significant, alternative="two-sided", test_method=test_method,
        interval_method="percentile", n_resamples=n_resamples, random_state=0)


def _report(comparisons, *, reference="EASE", test_method="randomization"):
    return ComparisonReport(
        comparisons=tuple(comparisons), metrics=tuple(dict.fromkeys(c.metric for c in comparisons)),
        model_names=("EASE", "ItemKNN", "Pop"), reference=reference, correction="holm",
        confidence_level=0.95, alternative="two-sided", test_method=test_method,
        n_resamples=comparisons[0].n_resamples, random_state=0)


@pytest.fixture
def report():
    # The editor's ML-1M run: every raw p on the floor 1e-4, Holm over 4 -> 4e-4.
    return _report([
        _cmp("ndcg@100", "EASE", "Pop", 0.4303, 0.2300, -0.2096, -0.1909, 1e-4, 4e-4),
        _cmp("ndcg@100", "EASE", "ItemKNN", 0.4303, 0.3285, -0.1095, -0.0943, 1e-4, 4e-4),
        _cmp("calibrated_recall@20", "EASE", "Pop", 0.3570, 0.1768, -0.1921, -0.1682, 1e-4, 4e-4),
        _cmp("calibrated_recall@20", "EASE", "ItemKNN", 0.3570, 0.2573, -0.1103, -0.0891, 1e-4, 4e-4),
    ])


def test_summary_has_the_reader_columns_in_requested_order(report):
    frame = report.summary(models=["ItemKNN", "Pop"], metrics=["ndcg@100", "calibrated_recall@20"])
    assert list(frame.columns) == ["metric", "baseline", "candidate", "baseline_mean", "candidate_mean",
                                   "difference", "ci_low", "ci_high", "adjusted_p_value", "significant",
                                   "direction", "p_floored"]
    assert frame[["metric", "candidate"]].values.tolist() == [
        ["ndcg@100", "ItemKNN"], ["ndcg@100", "Pop"],
        ["calibrated_recall@20", "ItemKNN"], ["calibrated_recall@20", "Pop"]]
    assert frame["direction"].tolist() == ["worse"] * 4
    assert frame["p_floored"].all()


def test_pairs_tested_the_other_way_round_are_flipped():
    flipped = _report([_cmp("ndcg@100", "Pop", "EASE", 0.2300, 0.4303, 0.1909, 0.2096, 0.01, 0.01)])
    row = flipped.summary(reference="EASE").iloc[0]
    assert (row.baseline, row.candidate) == ("EASE", "Pop")
    assert row.difference == pytest.approx(0.2300 - 0.4303)
    assert (row.ci_low, row.ci_high) == (-0.2096, -0.1909)


def test_unknown_reference_or_metric_raises(report):
    with pytest.raises(ValueError, match="reference"):
        report.summary(reference="SASRec")
    with pytest.raises(ValueError, match="metric"):
        report.summary(metrics=["mrr@10"])


def test_long_latex_matches_the_paper_layout(report):
    with pytest.warns(UserWarning, match="Monte Carlo floor"):
        tex = report.to_latex(models=["ItemKNN", "Pop"], label="tab:x")
    assert tex.startswith("\\begin{table}\n\\centering\n\\caption{R@$k$: calibrated recall")
    assert "Holm correction over 4 hypotheses" in tex
    assert "paired randomization (sign-flip) test with $B=9{,}999$" in tex
    assert "EASE & nDCG@100 & 0.4303 & -- & -- & -- \\\\" in tex
    assert ("ItemKNN & nDCG@100 & 0.3285$^{\\blacktriangledown}$ & $-$0.1018 & "
            "[$-$0.1095, $-$0.0943] & $\\le 4{\\cdot}10^{-4}$ \\\\") in tex
    assert "\\label{tab:x}" in tex and tex.endswith("\\end{table}")


def test_compact_latex_bolds_the_best_and_marks_significance(report):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tex = report.to_latex(layout="compact", models=["ItemKNN", "Pop"], marker="dagger")
    assert "Model & nDCG@100 & R@20 \\\\" in tex
    assert "EASE & \\textbf{0.4303} & \\textbf{0.3570} \\\\" in tex
    assert "Pop & 0.2300$^{\\ddagger}$ & 0.1768$^{\\ddagger}$ \\\\" in tex


def test_floor_bounds_round_up_and_raw_style_reports_the_floor():
    report = _report([_cmp("ndcg@100", "EASE", "Pop", 0.4, 0.3, -0.2, -0.1, 1e-4, 3.2e-4),
                      _cmp("ndcg@100", "EASE", "ItemKNN", 0.4, 0.35, -0.1, 0.01, 0.2, 0.2, False)])
    tex = report.to_latex(table_env=False)
    assert "$\\le 4{\\cdot}10^{-4}$" in tex  # 3.2e-4 rounds UP: it is an upper bound
    assert "& 0.200 \\\\" in tex             # not floored, printed as is
    raw = report.to_latex(table_env=False, floor_style="raw")
    assert "$< 1{\\cdot}10^{-4}$" in raw


def test_t_test_p_values_are_never_floored():
    report = _report([_cmp("ndcg@100", "EASE", "Pop", 0.4, 0.3, -0.2, -0.1, 1e-6, 2e-6,
                           test_method="t")], test_method="t")
    assert not report.summary()["p_floored"].any()
    tex = report.to_latex(table_env=False)
    assert "paired $t$-test" in tex and "$B=" not in tex


def test_markdown_has_the_long_layout_and_a_legend(report):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        md = report.to_markdown(models=["Pop"], metrics=["ndcg@100"])
    lines = md.splitlines()
    assert lines[0] == "| Model | vs. | Metric | Mean | Δ | 95% CI (Δ) | p (adj.) |"
    assert lines[2] == "| Pop | EASE | nDCG@100 | 0.2300 ▼ | -0.2003 | [-0.2096, -0.1909] | ≤ 4e-4 |"
    assert "Holm correction over 4 hypotheses" in lines[-1]


def test_latex_and_markdown_need_no_templating_dependency(report, monkeypatch):
    monkeypatch.setitem(sys.modules, "jinja2", None)  # import jinja2 would now fail
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert report.to_latex(table_env=False)
        assert report.to_markdown()
