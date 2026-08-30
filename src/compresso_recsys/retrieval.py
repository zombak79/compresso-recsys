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


def _sample_holdout_indices(row: csr_matrix, holdout_frac: float = 0.2):
    """Draw the fraction of a user's items to score against.

    The complement is the fold-in history the model sees. Liang et al. (2018)
    describe the protocol as choosing 80% of each held-out user's click history
    to learn a user representation from and reporting metrics on the remaining
    20%; ``holdout_frac`` is that remaining share.

    Rounded up, so a user always contributes at least one target as long as they
    have any items at all.
    """
    items = row.indices
    pick = int(np.ceil(len(items) * holdout_frac))
    if pick <= 0:
        return np.array([], dtype=np.int64)
    return np.random.choice(items, pick, replace=pick > len(items))


def _build_eval_draws(
    x_val: csr_matrix,
    draws: int = 5,
    holdout_frac: float = 0.2,
):
    """Stack ``draws`` independent source/target splits of the same users.

    Each draw holds out a fresh random ``holdout_frac`` of every user's items,
    so a user contributes one row per draw. The draws are independent samples
    rather than a partition, so they overlap and more of them is always
    possible -- there is no ``1 / holdout_frac`` ceiling.

    More draws buy precision rather than sample size. Averaging a user's score
    over several holdout samples removes the noise of which items happened to be
    held out, while leaving the variation between users alone. Measured on
    GoodBooks, five draws give confidence intervals 35 to 42 percent narrower
    than one, for the same users. What they do not give is more independent
    observations: the rows of one user are correlated, and
    :mod:`compresso_recsys.stats` resamples users rather than rows for exactly
    that reason.
    """
    sources, targets = [], []
    for _ in range(draws):
        source = x_val.copy()
        for i in range(source.shape[0]):
            source[i, _sample_holdout_indices(source[i], holdout_frac)] = 0
        sources.append(source)
        targets.append(x_val)

    stacked_source = vstack(sources).tocsr()
    stacked_full = vstack(targets).tocsr()
    stacked_source.eliminate_zeros()
    return stacked_source, (stacked_full - stacked_source).tocsr()


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
    eval_draws: int = 5,
    eval_holdout_frac: float = 0.2,
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

    x_src, x_tgt = _build_eval_draws(x_val, eval_draws, eval_holdout_frac)

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
    eval_draws: int = 5,
    eval_holdout_frac: float = 0.2,
) -> dict[str, object]:
    """Build a fixed source/target holdout for strongly generalized evaluation.

    Each held-out user's items are split into a fold-in history the model sees
    and a held-out share it is scored against, following Liang et al. (2018).
    ``eval_holdout_frac`` is the scored share; their description sets it to 0.2.

    ``eval_draws`` repeats that split independently, stacking one row per user
    per draw. The ELSA line of papers uses five. More draws sharpen each user's
    score by averaging over which items happened to be held out; they do not add
    independent observations, so paired comparison groups the rows back together
    by user.
    """
    if eval_draws < 1:
        raise ValueError(f"eval_draws must be >= 1, got {eval_draws!r}")
    if not 0.0 < eval_holdout_frac < 1.0:
        raise ValueError(
            f"eval_holdout_frac must be strictly between 0 and 1, "
            f"got {eval_holdout_frac!r}"
        )
    if isinstance(train_item_ids, pd.Index):
        item_ids = np.array(train_item_ids.astype(str))
    else:
        item_ids = np.asarray(train_item_ids).astype(str)

    np.random.seed(random_state)
    source_indices, target_sets, eval_user_ids = _prepare_eval_from_fold_protocol(
        train_item_ids=pd.Index(item_ids),
        eval_interactions=eval_interactions,
        min_user_support=min_user_support,
        eval_draws=eval_draws,
        eval_holdout_frac=eval_holdout_frac,
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


#: Stages of the leave-last-out protocol, oldest target first.
LEAVE_LAST_OUT_STAGES = ("train", "val", "test")

#: A user needs one source item plus one target for each of the three stages.
LEAVE_LAST_OUT_MIN_HISTORY = 4


def leave_last_out_histories(
    *,
    item_ids: pd.Index | np.ndarray,
    interactions: pd.DataFrame,
    min_history: int = LEAVE_LAST_OUT_MIN_HISTORY,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Per-user chronological catalog indices, oldest first, duplicates kept.

    The ordering is the whole point: every stage below is a prefix of it, and
    collapsing to a set here would make the sequential views of §9 impossible to
    recover. Users with fewer than ``min_history`` interactions are dropped,
    since the protocol needs one source item and three targets.
    """
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

    histories: list[np.ndarray] = []
    user_ids: list[str] = []
    for user_id, group in df.sort_values("timestamp", kind="stable").groupby(
        "user_id", sort=True
    ):
        if len(group) < min_history:
            continue
        histories.append(
            np.fromiter(
                (item_to_idx[item] for item in group["item_id"]),
                dtype=np.int64,
                count=len(group),
            )
        )
        user_ids.append(str(user_id))

    return histories, np.asarray(user_ids, dtype=str)


def leave_last_out_stage_slices(history: np.ndarray, stage: str) -> tuple[np.ndarray, np.ndarray]:
    """Source and target catalog indices for one stage of one user's history.

    For ``[1, 2, 3, 4, 5, 6, 7]``::

        train   source [1,2,3,4]      target [5]
        val     source [1,2,3,4,5]    target [6]
        test    source [1,2,3,4,5,6]  target [7]

    Items 6 and 7 are withheld from training; item 5 is not — it is the target of
    the training pair, and ``x_train`` is the union of that pair. So two items
    per user are held out of training, matching the sequential literature.
    """
    if stage == "train":
        return history[:-3], history[-3:-2]
    if stage == "val":
        return history[:-2], history[-2:-1]
    if stage == "test":
        return history[:-1], history[-1:]
    raise ValueError(f"unknown leave_last_out stage: {stage!r}")


def build_leave_last_out_holdout(
    *,
    item_ids: pd.Index | np.ndarray,
    interactions: pd.DataFrame,
    stage: str = "test",
    min_history: int = LEAVE_LAST_OUT_MIN_HISTORY,
) -> dict[str, object]:
    """Build one stage of the leave-last-out holdout.

    Each user's chronologically last interaction is the test target, the one
    before it the validation target, and the one before that the training
    target. Sources are the corresponding prefixes, so each stage's source is
    the previous stage's source plus the previous stage's target.

    Nothing is removed from the catalog. An item is absent from training only
    when every one of its occurrences happens to fall in a held-out tail, which
    is a property of the data rather than something this function imposes.
    """
    if stage not in LEAVE_LAST_OUT_STAGES:
        raise ValueError(
            f"stage must be one of {LEAVE_LAST_OUT_STAGES}, got {stage!r}"
        )
    if min_history < LEAVE_LAST_OUT_MIN_HISTORY:
        raise ValueError(
            f"min_history must be >= {LEAVE_LAST_OUT_MIN_HISTORY} so every stage "
            f"has a non-empty source and target, got {min_history!r}"
        )

    if isinstance(item_ids, pd.Index):
        item_ids_arr = np.array(item_ids.astype(str))
    else:
        item_ids_arr = np.asarray(item_ids).astype(str)

    histories, user_ids = leave_last_out_histories(
        item_ids=item_ids_arr,
        interactions=interactions,
        min_history=min_history,
    )
    source_indices: list[np.ndarray] = []
    target_indices: list[np.ndarray] = []
    for history in histories:
        source, target = leave_last_out_stage_slices(history, stage)
        # The matrix view is a set of items; order and duplicates belong to the
        # sequence view, which reads the same histories.
        source_indices.append(np.unique(source))
        target_indices.append(np.unique(target))

    return {
        "item_ids": item_ids_arr,
        "source_indices": source_indices,
        "target_indices": target_indices,
        "user_ids": user_ids,
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
    eval_draws: int = 5,
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
