"""The time-ordered split, and the code turning timestamps into stages.

The largest of the split strategies, and the only one whose stages depend on a
period rather than on a partition of rows or columns.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.checkpoint import (
    _indices_to_csr,
    save_recsys_split,
    update_checkpoint,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.builder.progress import _CheckpointProgress


def _csr_row_indices(matrix: csr_matrix) -> list[np.ndarray]:
    """Per-row column indices as read-only views into ``matrix.indices``.

    The split returned by the builders keeps ``matrix`` alongside this list, so
    copying every row would duplicate the whole index buffer while the original
    stays alive: 80 MB of the 200 MB these lists cost at a million users with
    twenty interactions each. Slices share that buffer instead.

    The views are marked read-only because they alias the matrix, and writing
    through one would silently corrupt the other. Consumers do not need to
    write: both ``_indices_to_csr`` and ``_as_obj_array`` convert to int64,
    and the retrieval helpers concatenate, each producing fresh writable arrays.
    """
    indices = matrix.indices
    indptr = matrix.indptr
    rows: list[np.ndarray] = []
    for row in range(matrix.shape[0]):
        view = indices[indptr[row] : indptr[row + 1]]
        view.flags.writeable = False
        rows.append(view)
    return rows


def _filter_temporal_pair(
    source: csr_matrix,
    target: csr_matrix,
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    inherited_items: int,
    min_user_support: int,
    item_min_support: int,
    min_source_items: int,
    min_target_items: int,
    stage: str,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, dict[str, int]]:
    if source.shape != target.shape:
        raise ValueError(f"{stage} source and target shapes must match")
    if source.shape != (len(user_ids), len(item_ids)):
        raise ValueError(f"{stage} matrix shape does not match its IDs")

    initial_users = int(source.shape[0])
    initial_items = int(source.shape[1])
    initial_new_items = initial_items - int(inherited_items)
    initial_source_interactions = int(source.nnz)
    initial_target_interactions = int(target.nnz)
    iterations = 0
    while True:
        iterations += 1
        combined = source.maximum(target)
        row_keep = (
            (source.getnnz(axis=1) >= min_source_items)
            & (target.getnnz(axis=1) >= min_target_items)
            & (combined.getnnz(axis=1) >= min_user_support)
        )
        if not bool(row_keep.any()):
            raise ValueError(
                f"{stage} temporal window has no users after support filtering"
            )
        rows_changed = not bool(row_keep.all())
        if rows_changed:
            source = source[row_keep].tocsr()
            target = target[row_keep].tocsr()
            user_ids = user_ids[row_keep]
            combined = source.maximum(target)

        item_support = np.asarray(combined.getnnz(axis=0)).ravel()
        column_keep = np.ones(source.shape[1], dtype=bool)
        column_keep[inherited_items:] = (
            item_support[inherited_items:] >= item_min_support
        )
        columns_changed = not bool(column_keep.all())
        if columns_changed:
            source = source[:, column_keep].tocsr()
            target = target[:, column_keep].tocsr()
            item_ids = item_ids[column_keep]

        if not rows_changed and not columns_changed:
            break

    stats = {
        "initial_users": initial_users,
        "users": int(source.shape[0]),
        "initial_items": initial_items,
        "items": int(source.shape[1]),
        "inherited_items": int(inherited_items),
        "initial_new_items": int(initial_new_items),
        "new_items": int(source.shape[1] - inherited_items),
        "initial_source_interactions": initial_source_interactions,
        "initial_target_interactions": initial_target_interactions,
        "source_interactions": int(source.nnz),
        "target_interactions": int(target.nnz),
        "support_iterations": int(iterations),
    }
    return source, target, user_ids, item_ids, stats


def _sequences_from_temporal_codes(
    *,
    event_mask: np.ndarray,
    global_user_codes: np.ndarray,
    global_item_codes: np.ndarray,
    timestamps: np.ndarray,
    user_lookup: np.ndarray,
    item_lookup: np.ndarray,
    n_rows: int,
    n_items: int,
) -> ItemSequences:
    """Chronological histories for the events a mask selects.

    The matrix twin of this drops order and merges duplicates; both read the same
    masked events, so the two views describe the same interactions rather than
    two things that happen to look alike.

    Sorting is by ``(row, timestamp)`` with a stable kind, so events sharing a
    timestamp keep the order the source data gave them rather than an arbitrary
    one.
    """
    if n_rows == 0:
        return ItemSequences.from_rows([], n_items=n_items)

    rows = user_lookup[global_user_codes[event_mask]]
    cols = item_lookup[global_item_codes[event_mask]]
    times = timestamps[event_mask]
    keep = (rows >= 0) & (cols >= 0)
    rows, cols, times = rows[keep], cols[keep], times[keep]

    order = np.lexsort((times, rows))
    rows, cols = rows[order], cols[order]

    counts = np.bincount(rows, minlength=n_rows)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    return ItemSequences(values=cols, indptr=indptr, n_items=n_items)


def _matrix_from_temporal_codes(
    *,
    event_mask: np.ndarray,
    global_user_codes: np.ndarray,
    global_item_codes: np.ndarray,
    values: np.ndarray,
    user_lookup: np.ndarray,
    item_lookup: np.ndarray,
    shape: tuple[int, int],
) -> csr_matrix:
    if not bool(event_mask.any()) or shape[0] == 0 or shape[1] == 0:
        return csr_matrix(shape, dtype=np.float32)

    rows = user_lookup[global_user_codes[event_mask]]
    cols = item_lookup[global_item_codes[event_mask]]
    valid = (rows >= 0) & (cols >= 0)
    matrix = csr_matrix(
        (values[event_mask][valid], (rows[valid], cols[valid])),
        shape=shape,
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _temporal_user_upper_bound(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    global_user_codes: np.ndarray,
    n_users: int,
    min_user_support: int,
    min_source_items: int,
    min_target_items: int,
) -> tuple[np.ndarray, int]:
    """Reject users that cannot meet support before allocating tall CSRs.

    Event counts are an upper bound on distinct nonzero item counts. Keeping a
    user here does not guarantee eligibility, but rejecting one is always safe;
    the exact fixed-point filter still runs on the resulting sparse matrices.
    """
    source_counts = np.bincount(
        global_user_codes[source_mask], minlength=n_users
    )
    target_counts = np.bincount(
        global_user_codes[target_mask], minlength=n_users
    )
    keep = source_counts >= min_source_items
    keep &= target_counts >= min_target_items
    source_counts += target_counts
    initial_users = int(np.count_nonzero(source_counts > 0))
    keep &= source_counts >= min_user_support
    return np.flatnonzero(keep).astype(np.int64, copy=False), initial_users


def _timestamps_in_seconds(values: pd.Series) -> np.ndarray:
    # Parquet-backed/Pandas copy-on-write arrays may be read-only. Unit
    # conversion must also never mutate the caller's original timestamps.
    timestamps = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(timestamps)
    if not bool(finite.any()):
        raise ValueError("temporal split requires non-empty timestamp values")
    magnitude = float(np.max(np.abs(timestamps[finite])))
    if magnitude >= 1e17:
        timestamps /= 1e9
    elif magnitude >= 1e14:
        timestamps /= 1e6
    elif magnitude >= 1e11:
        timestamps /= 1e3
    return timestamps


def _build_temporal_stage(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    global_user_codes: np.ndarray,
    global_item_codes: np.ndarray,
    global_user_ids: np.ndarray,
    global_item_ids: np.ndarray,
    values: np.ndarray,
    timestamps: np.ndarray,
    inherited_item_codes: np.ndarray,
    args,
    stage: str,
) -> dict[str, object]:
    observed_item_codes = np.unique(
        global_item_codes[source_mask | target_mask]
    )
    inherited_item_codes = np.asarray(inherited_item_codes, dtype=np.int64)
    new_item_codes = np.setdiff1d(
        observed_item_codes,
        inherited_item_codes,
        assume_unique=True,
    )
    item_codes = np.concatenate((inherited_item_codes, new_item_codes))

    user_codes, initial_users = _temporal_user_upper_bound(
        source_mask=source_mask,
        target_mask=target_mask,
        global_user_codes=global_user_codes,
        n_users=len(global_user_ids),
        min_user_support=args.min_user_support,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    if len(user_codes) == 0:
        raise ValueError(
            f"{stage} temporal window has no users after support filtering"
        )

    user_lookup = np.full(len(global_user_ids), -1, dtype=np.int64)
    user_lookup[user_codes] = np.arange(len(user_codes), dtype=np.int64)
    eligible_users = np.zeros(len(global_user_ids), dtype=bool)
    eligible_users[user_codes] = True
    matrix_source_mask = source_mask & eligible_users[global_user_codes]
    matrix_target_mask = target_mask & eligible_users[global_user_codes]
    item_lookup = np.full(len(global_item_ids), -1, dtype=np.int64)
    item_lookup[item_codes] = np.arange(len(item_codes), dtype=np.int64)
    shape = (len(user_codes), len(item_codes))
    source = _matrix_from_temporal_codes(
        event_mask=matrix_source_mask,
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        values=values,
        user_lookup=user_lookup,
        item_lookup=item_lookup,
        shape=shape,
    )
    target = _matrix_from_temporal_codes(
        event_mask=matrix_target_mask,
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        values=values,
        user_lookup=user_lookup,
        item_lookup=item_lookup,
        shape=shape,
    )
    user_ids = global_user_ids[user_codes]
    item_ids = global_item_ids[item_codes]
    source, target, user_ids, item_ids, stats = _filter_temporal_pair(
        source,
        target,
        user_ids=user_ids,
        item_ids=item_ids,
        inherited_items=len(inherited_item_codes),
        min_user_support=args.min_user_support,
        item_min_support=args.item_min_support,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
        stage=stage,
    )
    stats["initial_users"] = initial_users
    stats["prefiltered_users"] = int(len(user_codes))
    retained_item_codes = pd.Index(global_item_ids).get_indexer(item_ids)

    # _filter_temporal_pair drops users and items, so the lookups built above no
    # longer describe the returned matrices. Rebuild them from what survived, or
    # the sequence rows would address a row space the matrices no longer have.
    retained_user_codes = pd.Index(global_user_ids).get_indexer(user_ids)
    final_user_lookup = np.full(len(global_user_ids), -1, dtype=np.int64)
    final_user_lookup[retained_user_codes] = np.arange(len(user_ids), dtype=np.int64)
    final_item_lookup = np.full(len(global_item_ids), -1, dtype=np.int64)
    final_item_lookup[retained_item_codes] = np.arange(len(item_ids), dtype=np.int64)

    def _stage_sequences(mask: np.ndarray) -> ItemSequences:
        return _sequences_from_temporal_codes(
            event_mask=mask,
            global_user_codes=global_user_codes,
            global_item_codes=global_item_codes,
            timestamps=timestamps,
            user_lookup=final_user_lookup,
            item_lookup=final_item_lookup,
            n_rows=len(user_ids),
            n_items=len(item_ids),
        )

    return {
        "source": source,
        "target": target,
        "source_sequences": _stage_sequences(source_mask),
        # The stage's whole window, source and target together. Each stage is
        # filtered independently, so a window sequence taken from a later stage
        # would address a different row and column space than this stage's
        # matrices.
        "window_sequences": _stage_sequences(source_mask | target_mask),
        "user_ids": user_ids,
        "item_ids": item_ids,
        "item_codes": retained_item_codes.astype(np.int64, copy=False),
        "stats": stats,
    }


def _build_temporal_split(args, proc_df, progress: _CheckpointProgress | None = None):
    if "timestamp" not in proc_df.columns:
        raise ValueError("temporal split requires a timestamp column")
    timestamps = _timestamps_in_seconds(proc_df["timestamp"])
    finite = np.isfinite(timestamps)
    if not bool(finite.any()):
        raise ValueError("temporal split requires non-empty timestamp values")
    timestamps = timestamps[finite]
    values = proc_df.loc[finite, "value"].to_numpy(dtype=np.float32)
    global_user_codes, global_user_ids = pd.factorize(
        proc_df.loc[finite, "user_id"], sort=True
    )
    global_item_codes, global_item_ids = pd.factorize(
        proc_df.loc[finite, "item_id"], sort=True
    )
    global_user_codes = global_user_codes.astype(np.int64, copy=False)
    global_item_codes = global_item_codes.astype(np.int64, copy=False)
    global_user_ids = np.asarray(global_user_ids, dtype=object)
    global_item_ids = np.asarray(global_item_ids, dtype=object)

    period_seconds = float(args.temporal_period_hours) * 60.0 * 60.0
    timestamp_min = float(timestamps.min())
    timestamp_max = float(timestamps.max())
    train_target_start = timestamp_max - 3.0 * period_seconds
    validation_target_start = timestamp_max - 2.0 * period_seconds
    test_target_start = timestamp_max - period_seconds
    if train_target_start <= timestamp_min:
        span_hours = (timestamp_max - timestamp_min) / 3600.0
        raise ValueError(
            "temporal_period_hours requires three target windows shorter than "
            f"the available {span_hours:.3f}-hour timestamp span"
        )

    if progress is not None:
        progress.detail("Building temporal split: train")
    train_stage = _build_temporal_stage(
        source_mask=timestamps < train_target_start,
        target_mask=(timestamps >= train_target_start)
        & (timestamps < validation_target_start),
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        global_user_ids=global_user_ids,
        global_item_ids=global_item_ids,
        values=values,
        timestamps=timestamps,
        inherited_item_codes=np.asarray([], dtype=np.int64),
        args=args,
        stage="train",
    )
    if progress is not None:
        progress.detail("Building temporal split: validation")
    val_stage = _build_temporal_stage(
        source_mask=timestamps < validation_target_start,
        target_mask=(timestamps >= validation_target_start)
        & (timestamps < test_target_start),
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        global_user_ids=global_user_ids,
        global_item_ids=global_item_ids,
        values=values,
        timestamps=timestamps,
        inherited_item_codes=train_stage["item_codes"],
        args=args,
        stage="validation",
    )
    if progress is not None:
        progress.detail("Building temporal split: test")
    test_stage = _build_temporal_stage(
        source_mask=timestamps < test_target_start,
        target_mask=timestamps >= test_target_start,
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        global_user_ids=global_user_ids,
        global_item_ids=global_item_ids,
        values=values,
        timestamps=timestamps,
        inherited_item_codes=val_stage["item_codes"],
        args=args,
        stage="test",
    )

    train_item_ids = np.asarray(train_stage["item_ids"]).astype(str)
    val_item_ids = np.asarray(val_stage["item_ids"]).astype(str)
    test_item_ids = np.asarray(test_stage["item_ids"]).astype(str)
    train_source = train_stage["source"]
    train_target = train_stage["target"]
    val_source = val_stage["source"]
    val_target = val_stage["target"]
    test_source = test_stage["source"]
    test_target = test_stage["target"]
    x_train = train_source.maximum(train_target).tocsr()

    # The training window in order. For both chronological modes the validation
    # source is that same window, so these two agree exactly, mirroring x_train
    # and val_source_matrix on the matrix side.
    sequences = {
        # x_train is the train stage's window, so its sequence must come from the
        # same stage: val_stage covers the same events but in its own filtered
        # row and column space.
        "x_train_sequences": train_stage["window_sequences"],
        "train_source_sequences": train_stage["source_sequences"],
        "val_source_sequences": val_stage["source_sequences"],
        "test_source_sequences": test_stage["source_sequences"],
    }

    train_count = len(train_item_ids)
    val_count = len(val_item_ids)
    test_count = len(test_item_ids)
    return {
        "item_ids": test_item_ids,
        "train_item_ids": train_item_ids,
        "val_item_ids": val_item_ids,
        "test_item_ids": test_item_ids,
        "x_train": x_train,
        "train_source_matrix": train_source,
        "train_target_matrix": train_target,
        **sequences,
        "val_source_matrix": val_source,
        "val_target_matrix": val_target,
        "test_source_matrix": test_source,
        "test_target_matrix": test_target,
        "val_holdout": {
            "source_indices": _csr_row_indices(val_source),
            "target_indices": _csr_row_indices(val_target),
            "user_ids": val_stage["user_ids"],
        },
        "test_holdout": {
            "source_indices": _csr_row_indices(test_source),
            "target_indices": _csr_row_indices(test_target),
            "user_ids": test_stage["user_ids"],
        },
        "warm_item_indices": np.arange(train_count, dtype=np.int64),
        "val_cold_item_indices": np.arange(train_count, val_count, dtype=np.int64),
        "test_cold_item_indices": np.arange(val_count, test_count, dtype=np.int64),
        "train_user_ids": train_stage["user_ids"],
        "val_user_ids": val_stage["user_ids"],
        "test_user_ids": test_stage["user_ids"],
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": False,
            "has_stage_item_spaces": True,
            "is_temporal": True,
            "is_future_blind": True,
            "leakage_note": (
                "Temporal targets follow expanding histories. Item support may "
                "use a complete target window only to define benchmark eligibility."
            ),
            "temporal_period_hours": float(args.temporal_period_hours),
            "timestamp_unit": "unix_seconds",
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
            "train_target_start": train_target_start,
            "validation_target_start": validation_target_start,
            "test_target_start": test_target_start,
            "train_stage": train_stage["stats"],
            "validation_stage": val_stage["stats"],
            "test_stage": test_stage["stats"],
            "val_cold_items": int(val_count - train_count),
            "test_new_items": int(test_count - val_count),
            "test_model_cold_items": int(test_count - train_count),
        },
    }
