"""Scoring ranked predictions against held-out targets.

An evaluation is three separable decisions, and this package keeps them apart.
:mod:`~compresso_recsys.evaluation.arrays` canonicalises whatever the caller
passed -- indices, CSR, SRP tensors -- so nothing downstream re-checks a dtype.
:mod:`~compresso_recsys.evaluation.matching` decides which predictions were
hits, choosing between a dense take and a sorted search by candidate-space size.
:mod:`~compresso_recsys.evaluation.results` holds what comes back, including the
target fingerprint that makes two runs comparable.
:class:`~compresso_recsys.evaluation.RankingEvaluator` drives the three.

:func:`evaluate_recommender` is the entry point most callers want; it takes a
fitted model and a holdout, and is overloaded on whether you pass sequences or
a matrix.
"""

from compresso_recsys.evaluation.results import EvaluationResult
from compresso_recsys.evaluation.evaluator import (
    RankingEvaluator,
    evaluate_ranked_predictions,
    evaluate_recommender,
)

__all__ = [
    "EvaluationResult",
    "RankingEvaluator",
    "evaluate_ranked_predictions",
    "evaluate_recommender",
]
