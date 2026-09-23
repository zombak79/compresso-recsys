"""Scoring item embeddings against a holdout, without fitting a model.

Ranks by inner product in chunks, so a catalog that does not fit in memory as
a dense score matrix still evaluates.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, List

import numpy as np
import pandas as pd
import torch

from compresso import SRPTensor
from compresso_recsys.evaluation import RankingEvaluator
from compresso_recsys.evaluation.arrays import _indices_to_csr
from compresso_recsys.metrics import CalibratedRecall, NDCG, RankingMetric
from compresso_recsys.retrieval.holdout import build_eval_holdout


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


def _iter_topk_predictions(
    e: torch.Tensor,
    source_indices: List[np.ndarray],
    k: int,
    *,
    batch_size: int = 512,
    show_progress: bool = False,
    desc: str = "evaluate top-k",
) -> Iterator[tuple[int, int, SRPTensor]]:
    """Yield batched vectorized top-k retrieval results.

    ELSA-forward scoring:
      scores_u = relu((x_u @ e) @ e.T - x_u), where x_u is sparse source
      interaction vector over item ids.
    """
    n_items = e.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > n_items:
        raise ValueError(
            f"k ({k}) cannot exceed the number of items ({n_items})"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    starts = range(0, len(source_indices), batch_size)
    for start in _progress(starts, enabled=show_progress, desc=desc):
        batch = source_indices[start : start + batch_size]
        b = len(batch)

        # Flatten variable-length source item lists into one index tensor.
        lengths = [len(x) for x in batch]
        flat_src = np.concatenate(batch, axis=0)
        flat_src_t = torch.from_numpy(flat_src).long().to(e.device)

        # Owner row id for each flattened source index.
        owners = torch.repeat_interleave(
            torch.arange(b, device=e.device, dtype=torch.long),
            torch.tensor(lengths, device=e.device, dtype=torch.long),
        )

        # Build sparse-like dense batch x over items.
        x = torch.zeros((b, n_items), device=e.device, dtype=e.dtype)
        x[owners, flat_src_t] = 1.0
        x_a = x @ e
        scores = torch.relu((x_a @ e.T) - x)

        # Mask seen source items.
        scores[owners, flat_src_t] = -torch.inf

        topk_vals, topk_idx = torch.topk(scores, k, dim=1, largest=True, sorted=True)
        yield (
            start,
            start + b,
            SRPTensor(
                cols=topk_idx,
                vals=topk_vals,
                shape=(b, n_items),
                validate=False,
            ),
        )


def _default_metrics(k: int) -> list[RankingMetric]:
    return [CalibratedRecall(k), NDCG(k)]


def evaluate_item_embeddings(
    *,
    train_item_ids: pd.Index,
    item_embeddings: np.ndarray,
    eval_interactions: pd.DataFrame,
    k: int = 100,
    eval_holdout_frac: float = 0.2,
    min_user_support: int = 5,
    random_state: int = 42,
    eval_draws: int = 1,
    score_batch_size: int = 512,
    metrics: Sequence[RankingMetric] | None = None,
    debug: bool = False,
    debug_users: int = 5,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Evaluate item embeddings with torch top-k retrieval.

    - User profile: sum of source-item embeddings.
    - Scores: dot(profile, item_embedding).
    - Seen source items are masked.
    """
    if item_embeddings.shape[0] != len(train_item_ids):
        raise ValueError(
            f"Embeddings rows ({item_embeddings.shape[0]}) must match number of train items ({len(train_item_ids)})."
        )

    holdout = build_eval_holdout(
        train_item_ids=train_item_ids,
        eval_interactions=eval_interactions,
        min_user_support=min_user_support,
        random_state=random_state,
        eval_draws=eval_draws,
        eval_holdout_frac=eval_holdout_frac,
    )
    return evaluate_item_embeddings_with_holdout(
        item_embeddings=item_embeddings,
        source_indices=holdout["source_indices"],  # type: ignore[arg-type]
        target_indices=holdout["target_indices"],  # type: ignore[arg-type]
        k=k,
        score_batch_size=score_batch_size,
        metrics=metrics,
        debug=debug,
        debug_users=debug_users,
        show_progress=show_progress,
    )


def evaluate_item_embeddings_with_holdout(
    *,
    item_embeddings: np.ndarray,
    source_indices: list[np.ndarray],
    target_indices: list[np.ndarray],
    k: int = 100,
    score_batch_size: int = 512,
    metrics: Sequence[RankingMetric] | None = None,
    debug: bool = False,
    debug_users: int = 5,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Evaluate item embeddings against a precomputed source/target holdout.

    Predictions are generated and evaluated one batch at a time. Supplying
    ``metrics`` allows multiple cutoffs to reuse the same ranked predictions
    and target-hit tensor.
    """
    if len(source_indices) != len(target_indices):
        raise ValueError("source_indices and target_indices must have same length")
    if item_embeddings.ndim != 2:
        raise ValueError("item_embeddings must be a 2D array")
    if k < 1:
        raise ValueError("k must be >= 1")
    if k > item_embeddings.shape[0]:
        raise ValueError(
            f"k ({k}) cannot exceed the number of items ({item_embeddings.shape[0]})"
        )
    if score_batch_size < 1:
        raise ValueError("score_batch_size must be >= 1")

    with np.errstate(over="ignore", invalid="ignore"):
        converted_embeddings = item_embeddings.astype(np.float32)
    if not np.isfinite(converted_embeddings).all():
        raise ValueError(
            "item_embeddings must contain only finite values after "
            "conversion to float32"
        )

    e = torch.from_numpy(converted_embeddings)
    e = torch.nn.functional.normalize(e, dim=-1)
    targets = _indices_to_csr(target_indices, n_items=item_embeddings.shape[0])
    evaluator = RankingEvaluator(
        list(metrics) if metrics is not None else _default_metrics(k),
        validate_predictions=False,
        debug=debug,
        debug_users=debug_users,
    )
    if evaluator.required_k > k:
        raise ValueError(
            f"metrics require top-{evaluator.required_k}, but retrieval was configured for top-{k}"
        )
    for start, end, predictions in _iter_topk_predictions(
        e,
        source_indices,
        k=k,
        batch_size=score_batch_size,
        show_progress=show_progress,
        desc=f"evaluate@{k}",
    ):
        evaluator.update(predictions, targets[start:end])
    return evaluator.compute()
