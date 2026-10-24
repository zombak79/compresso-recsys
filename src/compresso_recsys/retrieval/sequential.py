"""Holdouts defined by order rather than by a draw.

Leave-last-out holds out each user's final interactions; the temporal split
cuts every user at the same wall-clock boundary. Neither samples, so neither
takes a random seed.
"""

from __future__ import annotations


import numpy as np
import pandas as pd



LEAVE_LAST_OUT_STAGES = ("train", "val", "test")


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
