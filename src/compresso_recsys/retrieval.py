from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, vstack

from compresso import SRPTensor
from compresso_recsys.evaluation import RankingEvaluator, _indices_to_csr
from compresso_recsys.metrics import CalibratedRecall, NDCG, RankingMetric


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


def _filter_users_by_support(df: pd.DataFrame, min_user_support: int) -> pd.DataFrame:
    if min_user_support <= 1:
        return df
    counts = df.groupby("user_id")["item_id"].nunique()
    keep_users = counts[counts >= min_user_support].index
    return df[df["user_id"].isin(keep_users)].copy()


def _build_user_holdout(
    df: pd.DataFrame,
    *,
    holdout_frac: float = 0.2,
    min_items_per_user: int = 2,
    random_state: int = 42,
) -> Dict[str, Tuple[List[str], List[str]]]:
    """Split each user's interacted items into (source, target) item lists."""
    rng = np.random.default_rng(random_state)
    out: Dict[str, Tuple[List[str], List[str]]] = {}

    for user_id, g in df.groupby("user_id"):
        items = list(pd.unique(g["item_id"].astype(str)))
        if len(items) < min_items_per_user:
            continue

        n_target = max(1, int(np.ceil(len(items) * holdout_frac)))
        n_target = min(n_target, len(items) - 1)

        perm = rng.permutation(len(items))
        tgt_idx = set(perm[:n_target].tolist())

        target = [items[i] for i in range(len(items)) if i in tgt_idx]
        source = [items[i] for i in range(len(items)) if i not in tgt_idx]
        if source and target:
            out[str(user_id)] = (source, target)

    return out


def _get_random_indices_exact(row: csr_matrix, frac: float = 0.2, part: int = 0):
    """Exact copy of compressed_elsa behavior.

    Note: In the source project, `frac` is effectively ignored in selection size
    and they always use 0.2 internally. We intentionally mirror that behavior.
    """
    a = row.indices
    pick = int(np.ceil(len(a) * 0.2))
    if part == 0:
        if pick <= 0:
            return np.array([], dtype=np.int64)
        return np.random.choice(a, pick, replace=False if pick <= len(a) else True)
    q = []
    for i in range(int(1 / 0.2)):
        q.append(a[i * pick : i * pick + pick])
    return q[part]


def _get_src_target_fold_exact(x_val: csr_matrix, fold: int = 0):
    """Faithful reproduction of compressed_elsa get_src_target_fold."""
    xs = []
    xvs = []

    x_val_src = x_val.copy()
    for i in range(x_val_src.shape[0]):
        ind = _get_random_indices_exact(x_val_src[i], 1)
        x_val_src[i, ind] = 0
    xs.append(x_val_src)
    xvs.append(x_val)

    if fold != 1:
        x_val_src = x_val.copy()
        for i in range(x_val_src.shape[0]):
            ind = _get_random_indices_exact(x_val_src[i], 2)
            x_val_src[i, ind] = 0
        xs.append(x_val_src)
        xvs.append(x_val)

        x_val_src = x_val.copy()
        for i in range(x_val_src.shape[0]):
            ind = _get_random_indices_exact(x_val_src[i], 3)
            x_val_src[i, ind] = 0
        xs.append(x_val_src)
        xvs.append(x_val)

        x_val_src = x_val.copy()
        for i in range(x_val_src.shape[0]):
            ind = _get_random_indices_exact(x_val_src[i], 4)
            x_val_src[i, ind] = 0
        xs.append(x_val_src)
        xvs.append(x_val)

        x_val_src = x_val.copy()
        for i in range(x_val_src.shape[0]):
            ind = _get_random_indices_exact(x_val_src[i], 5)
            x_val_src[i, ind] = 0
        xs.append(x_val_src)
        xvs.append(x_val)

    x_val_src = vstack(xs).tocsr()
    x_val_stacked = vstack(xvs).tocsr()

    x_val_src.eliminate_zeros()
    x_val_targets = (x_val_stacked - x_val_src).tocsr()

    return x_val_src, x_val_targets


def _prepare_eval_users(
    *,
    train_item_ids: pd.Index,
    eval_interactions: pd.DataFrame,
    holdout_frac: float,
    min_items_per_user: int,
    min_user_support: int,
    random_state: int,
):
    item_ids = np.array(train_item_ids.astype(str))
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    df = eval_interactions.copy()
    df["item_id"] = df["item_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)

    df = df[df["item_id"].isin(item_to_idx.keys())]
    df = _filter_users_by_support(df, min_user_support=min_user_support)

    user_split = _build_user_holdout(
        df,
        holdout_frac=holdout_frac,
        min_items_per_user=min_items_per_user,
        random_state=random_state,
    )

    return item_ids, item_to_idx, user_split


def _prepare_eval_from_fold_protocol(
    *,
    train_item_ids: pd.Index,
    eval_interactions: pd.DataFrame,
    min_user_support: int,
    eval_fold: int = 0,
):
    item_ids = np.array(train_item_ids.astype(str))
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    df = eval_interactions.copy()
    df["item_id"] = df["item_id"].astype(str)
    df["user_id"] = df["user_id"].astype(str)
    df = df[df["item_id"].isin(item_to_idx.keys())]
    df = _filter_users_by_support(df, min_user_support=min_user_support)

    users = np.array(sorted(df["user_id"].unique()))
    u_codes = pd.Categorical(df["user_id"], categories=users).codes
    i_codes = pd.Categorical(df["item_id"], categories=item_ids).codes
    vals = np.ones(len(df), dtype=np.float32)
    x_val = csr_matrix((vals, (u_codes, i_codes)), shape=(len(users), len(item_ids)), dtype=np.float32)

    x_src, x_tgt = _get_src_target_fold_exact(x_val, fold=eval_fold)

    source_indices = [x_src[i].indices.astype(np.int64, copy=False) for i in range(x_src.shape[0])]
    target_sets = [set(x_tgt[i].indices.tolist()) for i in range(x_tgt.shape[0])]
    repeats = int(x_src.shape[0] // max(1, len(users)))
    eval_user_ids = np.tile(users.astype(str), repeats) if len(users) else np.array([], dtype=str)

    return source_indices, target_sets, eval_user_ids


def build_eval_holdout(
    *,
    train_item_ids: pd.Index | np.ndarray,
    eval_interactions: pd.DataFrame,
    min_user_support: int = 5,
    random_state: int = 42,
    eval_fold: int = 0,
) -> dict[str, object]:
    """Build fixed eval holdout (source/target) using compressed_elsa fold protocol.

    eval_fold:
      - 0: stacked 5-fold behavior (paper default in compressed_elsa)
      - 1: single fold
    """
    if isinstance(train_item_ids, pd.Index):
        item_ids = np.array(train_item_ids.astype(str))
    else:
        item_ids = np.asarray(train_item_ids).astype(str)

    np.random.seed(random_state)
    source_indices, target_sets, eval_user_ids = _prepare_eval_from_fold_protocol(
        train_item_ids=pd.Index(item_ids),
        eval_interactions=eval_interactions,
        min_user_support=min_user_support,
        eval_fold=eval_fold,
    )
    target_indices = [np.array(sorted(list(s)), dtype=np.int64) for s in target_sets]
    return {
        "item_ids": item_ids,
        "source_indices": source_indices,
        "target_indices": target_indices,
        "user_ids": eval_user_ids,
    }


def build_item_cold_holdout(
    *,
    item_ids: pd.Index | np.ndarray,
    interactions: pd.DataFrame,
    source_item_ids: set[str] | list[str] | np.ndarray,
    target_item_ids: set[str] | list[str] | np.ndarray,
    min_source_items: int = 1,
    min_target_items: int = 1,
) -> dict[str, object]:
    """Build source=train-item and target=cold-item holdout for overlapping users."""
    if isinstance(item_ids, pd.Index):
        item_ids_arr = np.array(item_ids.astype(str))
    else:
        item_ids_arr = np.asarray(item_ids).astype(str)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids_arr)}
    source_items = set(np.asarray(list(source_item_ids)).astype(str))
    target_items = set(np.asarray(list(target_item_ids)).astype(str))

    df = interactions.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df[df["item_id"].isin(item_to_idx)]

    source_indices: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    user_ids: list[str] = []
    for _, g in df.groupby("user_id"):
        src = sorted({item_to_idx[item] for item in g["item_id"] if item in source_items})
        tgt = sorted({item_to_idx[item] for item in g["item_id"] if item in target_items})
        if len(src) >= min_source_items and len(tgt) >= min_target_items:
            source_indices.append(np.asarray(src, dtype=np.int64))
            target_indices.append(np.asarray(tgt, dtype=np.int64))
            user_ids.append(str(g["user_id"].iloc[0]))

    return {
        "item_ids": item_ids_arr,
        "source_indices": source_indices,
        "target_indices": target_indices,
        "user_ids": np.asarray(user_ids, dtype=str),
    }


def build_leave_last_out_holdout(
    *,
    item_ids: pd.Index | np.ndarray,
    interactions: pd.DataFrame,
    min_source_items: int = 1,
    min_target_items: int = 1,
) -> dict[str, object]:
    """Build per-user source/target by holding out each user's latest interaction."""
    if isinstance(item_ids, pd.Index):
        item_ids_arr = np.array(item_ids.astype(str))
    else:
        item_ids_arr = np.asarray(item_ids).astype(str)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids_arr)}

    df = interactions.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df[df["item_id"].isin(item_to_idx)]
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        raise ValueError("leave_last_out split requires non-empty timestamp values")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    source_indices: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    user_ids: list[str] = []
    for _, g in df.sort_values("timestamp").groupby("user_id", sort=False):
        items = g["item_id"].tolist()
        if len(items) < min_source_items + min_target_items:
            continue
        target_item = items[-1]
        source = sorted({item_to_idx[item] for item in items[:-1]})
        target = [item_to_idx[target_item]]
        if len(source) >= min_source_items:
            source_indices.append(np.asarray(source, dtype=np.int64))
            target_indices.append(np.asarray(target, dtype=np.int64))
            user_ids.append(str(g["user_id"].iloc[0]))

    return {
        "item_ids": item_ids_arr,
        "source_indices": source_indices,
        "target_indices": target_indices,
        "user_ids": np.asarray(user_ids, dtype=str),
    }


def build_temporal_holdout(
    *,
    item_ids: pd.Index | np.ndarray,
    interactions: pd.DataFrame,
    test_frac: float = 0.1,
    min_source_items: int = 1,
    min_target_items: int = 1,
) -> dict[str, object]:
    """Build source/target using a global timestamp cutoff."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be in (0, 1)")
    if isinstance(item_ids, pd.Index):
        item_ids_arr = np.array(item_ids.astype(str))
    else:
        item_ids_arr = np.asarray(item_ids).astype(str)
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids_arr)}

    df = interactions.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    df = df[df["item_id"].isin(item_to_idx)]
    if "timestamp" not in df.columns or df["timestamp"].isna().all():
        raise ValueError("temporal split requires non-empty timestamp values")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    cutoff = df["timestamp"].quantile(1.0 - test_frac)

    source_indices: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    user_ids: list[str] = []
    for _, g in df.groupby("user_id"):
        src = sorted({item_to_idx[item] for item in g.loc[g["timestamp"] <= cutoff, "item_id"]})
        tgt = sorted({item_to_idx[item] for item in g.loc[g["timestamp"] > cutoff, "item_id"]})
        if len(src) >= min_source_items and len(tgt) >= min_target_items:
            source_indices.append(np.asarray(src, dtype=np.int64))
            target_indices.append(np.asarray(tgt, dtype=np.int64))
            user_ids.append(str(g["user_id"].iloc[0]))

    return {
        "item_ids": item_ids_arr,
        "source_indices": source_indices,
        "target_indices": target_indices,
        "user_ids": np.asarray(user_ids, dtype=str),
        "timestamp_cutoff": float(cutoff),
    }


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
    holdout_frac: float = 0.2,
    min_items_per_user: int = 2,
    min_user_support: int = 5,
    random_state: int = 42,
    eval_fold: int = 0,
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
        eval_fold=eval_fold,
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

    e = torch.from_numpy(item_embeddings.astype(np.float32))
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
