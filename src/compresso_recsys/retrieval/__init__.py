"""Evaluation holdouts, and scoring item embeddings against them.

Two halves that meet only at the end. The holdout builders decide which
interactions a model is allowed to see and which it is scored on -- by a random
draw per user in :mod:`~compresso_recsys.retrieval.holdout`, or by order in
:mod:`~compresso_recsys.retrieval.sequential`, where leave-last-out and the
temporal cut need no seed because they sample nothing.
:mod:`~compresso_recsys.retrieval.embeddings` then ranks a fixed item-embedding
matrix against whichever holdout you built, which is how a checkpoint is
evaluated without fitting anything.
"""

from compresso_recsys.retrieval.holdout import (
    build_eval_holdout,
    build_item_cold_holdout,
)
from compresso_recsys.retrieval.sequential import (
    LEAVE_LAST_OUT_MIN_HISTORY,
    LEAVE_LAST_OUT_STAGES,
    build_leave_last_out_holdout,
    build_temporal_holdout,
    leave_last_out_histories,
    leave_last_out_stage_slices,
)
from compresso_recsys.retrieval.embeddings import (
    evaluate_item_embeddings,
    evaluate_item_embeddings_with_holdout,
)

__all__ = [
    "LEAVE_LAST_OUT_MIN_HISTORY",
    "LEAVE_LAST_OUT_STAGES",
    "build_eval_holdout",
    "build_item_cold_holdout",
    "build_leave_last_out_holdout",
    "build_temporal_holdout",
    "evaluate_item_embeddings",
    "evaluate_item_embeddings_with_holdout",
    "leave_last_out_histories",
    "leave_last_out_stage_slices",
]
