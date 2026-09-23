"""Splits defined by what is held out: official, user, item, leave-last-out."""

from __future__ import annotations


import numpy as np

from compresso_recsys.checkpoint import (
    _indices_to_csr,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.retrieval import (
    LEAVE_LAST_OUT_MIN_HISTORY,
    LEAVE_LAST_OUT_STAGES,
    build_eval_holdout,
    build_item_cold_holdout,
    leave_last_out_histories,
    leave_last_out_stage_slices,
)
from compresso_recsys.builder.features import (
    _to_sparse_matrix_for_items_with_users,
)


def _split_item_ids_random(item_ids: np.ndarray, *, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_ids = np.asarray(item_ids).astype(str)
    n_items = len(item_ids)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_items)

    n_val = args.val_items if args.val_items is not None else int(np.ceil(n_items * args.item_val_frac))
    n_test = args.test_items if args.test_items is not None else int(np.ceil(n_items * args.item_test_frac))
    n_val = max(0, int(n_val))
    n_test = max(0, int(n_test))
    if n_val + n_test >= n_items:
        raise ValueError("Cold val/test items must leave at least one train item")

    val_idx = np.sort(perm[:n_val])
    test_idx = np.sort(perm[n_val : n_val + n_test])
    train_idx = np.sort(perm[n_val + n_test :])
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def _build_official_split(args, ds, proc_df):
    """DBbook's supplied test boundary, with validation carved from train only."""
    train = proc_df.drop_duplicates(["user_id", "item_id"]).copy()
    original_train, users, item_ids = ds.to_sparse_matrix(train)
    users, item_ids = np.asarray(users).astype(str), np.asarray(item_ids).astype(str)
    validation = build_eval_holdout(
        train_item_ids=item_ids, eval_interactions=train,
        min_user_support=max(2, args.min_source_items + args.min_target_items),
        random_state=args.seed, eval_draws=1, eval_holdout_frac=args.eval_holdout_frac,
    )
    eligible = [row for row, (source, target) in enumerate(zip(
        validation["source_indices"], validation["target_indices"]
    )) if len(source) >= args.min_source_items and len(target) >= args.min_target_items]
    validation["source_indices"] = [validation["source_indices"][row] for row in eligible]
    validation["target_indices"] = [validation["target_indices"][row] for row in eligible]
    validation["user_ids"] = np.asarray(validation["user_ids"])[eligible]
    if not len(validation["user_ids"]):
        raise ValueError("Official split has no eligible validation users")
    # The model must not see validation targets in its training matrix.
    x_train = original_train.tolil()
    user_rows = {key: row for row, key in enumerate(users)}
    for key, targets in zip(validation["user_ids"], validation["target_indices"]):
        x_train[user_rows[str(key)], targets] = 0
    x_train = x_train.tocsr()
    x_train.eliminate_zeros()
    test = ds.get_official_split()["test"]
    if args.min_value_to_keep is not None:
        test = test[test.value >= args.min_value_to_keep]
    test = test.drop_duplicates(["user_id", "item_id"])
    before = len(test)
    test = test[test.user_id.isin(users) & test.item_id.isin(item_ids)]
    excluded = before - len(test)
    item_rows = {key: row for row, key in enumerate(item_ids)}
    test_users, source_indices, target_indices = [], [], []
    for key, group in test.groupby("user_id", sort=True):
        source = original_train[user_rows[str(key)]].indices.astype(np.int64)
        target = np.array([item_rows[str(item)] for item in group.item_id], dtype=np.int64)
        if np.intersect1d(source, target).size:
            raise ValueError("Official DBbook train/test contain overlapping user-item pairs")
        if len(source) < args.min_source_items or len(target) < args.min_target_items:
            excluded += len(target)
            continue
        test_users.append(str(key))
        source_indices.append(source)
        target_indices.append(target)
    if not test_users:
        raise ValueError("Official split has no eligible test users")
    return {
        "item_ids": item_ids, "x_train": x_train,
        "train_source_matrix": x_train, "train_target_matrix": x_train,
        "train_user_ids": users,
        "val_user_ids": np.asarray(validation["user_ids"]).astype(str),
        "test_user_ids": np.asarray(test_users),
        "val_holdout": validation,
        "test_holdout": {"source_indices": source_indices, "target_indices": target_indices,
                         "user_ids": np.asarray(test_users)},
        "extra_metadata": {
            "has_user_partitions": False, "has_item_partitions": False,
            "is_temporal": False, "is_future_blind": False,
            "official_test_excluded_interactions": excluded,
            "effective_eval_draws": 1,
            "leakage_note": "Supplied DBbook test boundary. Validation withheld from train. "
                            "Test histories use full supplied train; model is not refit. "
                            "Test users/items outside the training vocabulary are excluded.",
        },
    }


def _build_user_split(args, ds, proc_df):
    split = ds.split_users_strong_generalization(
        val_users=args.val_users,
        test_users=args.test_users,
        min_user_support=1,
        random_state=args.seed,
        interactions=proc_df,
    )
    x_train, train_user_index, item_ids = ds.to_sparse_matrix(split.train)
    val_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.val,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_draws=args.eval_draws,
        eval_holdout_frac=args.eval_holdout_frac,
    )
    test_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.test,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_draws=args.eval_draws,
        eval_holdout_frac=args.eval_holdout_frac,
    )
    catalog_item_ids = test_holdout["item_ids"]
    return {
        "item_ids": catalog_item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": val_holdout,
        "test_holdout": test_holdout,
        # Every item is present while training and no later phase introduces new
        # ones, so training spans the catalog and validation/test add nothing.
        # Written out explicitly instead of left as None so that every split mode
        # stores all three partitions and none of them has to be inferred.
        "warm_item_indices": np.arange(len(catalog_item_ids), dtype=np.int64),
        "val_cold_item_indices": np.array([], dtype=np.int64),
        "test_cold_item_indices": np.array([], dtype=np.int64),
        "train_user_ids": np.asarray(train_user_index).astype(str),
        "val_user_ids": np.asarray(sorted(split.val["user_id"].astype(str).unique())),
        "test_user_ids": np.asarray(sorted(split.test["user_id"].astype(str).unique())),
        "extra_metadata": {
            "has_user_partitions": True,
            "has_item_partitions": False,
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": "Random user split; timestamps are not used to prevent future-to-past leakage.",
        },
    }


def _build_item_split(args, proc_df):
    item_ids = np.array(sorted(proc_df["item_id"].astype(str).unique()))
    train_idx, val_idx, test_idx = _split_item_ids_random(item_ids, args=args)
    train_items = set(item_ids[train_idx].tolist())
    val_items = set(item_ids[val_idx].tolist())
    test_items = set(item_ids[test_idx].tolist())
    train_df = proc_df[proc_df["item_id"].astype(str).isin(train_items)].copy()
    x_train, train_user_ids = _to_sparse_matrix_for_items_with_users(train_df, item_ids)
    val_holdout = build_item_cold_holdout(
        item_ids=item_ids,
        interactions=proc_df,
        source_item_ids=train_items,
        target_item_ids=val_items,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    test_holdout = build_item_cold_holdout(
        item_ids=item_ids,
        interactions=proc_df,
        source_item_ids=train_items,
        target_item_ids=test_items,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": val_holdout,
        "test_holdout": test_holdout,
        "warm_item_indices": train_idx,
        "val_cold_item_indices": val_idx,
        "test_cold_item_indices": test_idx,
        "train_user_ids": train_user_ids,
        "val_user_ids": None,
        "test_user_ids": None,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": True,
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": "Random item split; timestamps are not used to prevent future-to-past leakage.",
            "item_val_frac": args.item_val_frac,
            "item_test_frac": args.item_test_frac,
            "val_items": int(len(val_idx)),
            "test_items": int(len(test_idx)),
        },
    }


def _build_leave_last_out_split(args, proc_df):
    """Chronological per-user holdout with the catalog left intact.

    Each user's last interaction is the test target, the one before it the
    validation target, and the one before that the training target. Sources are
    the corresponding prefixes.

    Nothing is stripped from training. Item partitions are *observed* rather than
    imposed: an item lands in the validation or test partition only when every
    one of its occurrences happens to fall in a held-out tail, which on dense
    data means the partitions come out empty and on sparse data means they hold
    the genuinely new items.
    """
    # Every support argument has to reach the protocol or be refused outright.
    # min_target_items cannot: each stage holds out exactly one item, so a
    # request for more is a request this split cannot fill.
    if int(args.min_target_items) > 1:
        raise ValueError(
            "leave_last_out holds out exactly one item per stage, so "
            f"min_target_items must be 1, got {args.min_target_items}"
        )
    # The floor of four is structural: one source item plus three stage targets.
    # Above it, min_user_support drops short users outright, and min_source_items
    # lengthens the training source, which costs three more interactions.
    # min_user_support is resolved from the dataset spec by the main build
    # path, so it is still None when a split builder is called directly.
    user_support = 0 if args.min_user_support is None else int(args.min_user_support)
    min_history = max(
        LEAVE_LAST_OUT_MIN_HISTORY,
        user_support,
        int(args.min_source_items) + 3,
    )

    provisional_item_ids = np.array(
        sorted(proc_df["item_id"].astype(str).unique())
    )
    histories, user_ids = leave_last_out_histories(
        item_ids=provisional_item_ids,
        interactions=proc_df,
        min_history=min_history,
    )
    if len(user_ids) == 0:
        raise ValueError(
            f"leave_last_out needs users with at least {min_history} "
            "interactions; none qualified"
        )

    # Only retained, timestamp-valid histories define this checkpoint. Compacting
    # here removes permanently empty columns contributed by discarded users or
    # invalid events while preserving the provisional catalog's sorted order.
    used_item_indices = np.unique(np.concatenate(histories))
    item_ids = provisional_item_ids[used_item_indices]
    old_to_new = np.full(len(provisional_item_ids), -1, dtype=np.int64)
    old_to_new[used_item_indices] = np.arange(len(used_item_indices), dtype=np.int64)
    histories = [old_to_new[history] for history in histories]

    stages: dict[str, dict[str, list[np.ndarray]]] = {}
    ordered_sources: dict[str, list[np.ndarray]] = {}
    for stage in LEAVE_LAST_OUT_STAGES:
        sources, targets, in_order = [], [], []
        for history in histories:
            source, target = leave_last_out_stage_slices(history, stage)
            # Two views of the same events, taken in one pass: the matrix wants a
            # set, the sequence wants the order. Deriving one from the other later
            # is impossible in the direction that matters.
            sources.append(np.unique(source))
            targets.append(np.unique(target))
            in_order.append(source)
        stages[stage] = {"source_indices": sources, "target_indices": targets}
        ordered_sources[stage] = in_order

    n_items = len(item_ids)
    train_source = _indices_to_csr(stages["train"]["source_indices"], n_cols=n_items)
    train_target = _indices_to_csr(stages["train"]["target_indices"], n_cols=n_items)
    # The same relationship temporal uses: the training window is the pair's
    # union, and a symmetric model trains on that.
    x_train = train_source.maximum(train_target).tocsr()
    # Items first seen in each phase, exactly as the temporal stages compute it.
    def _observed(stage: str) -> np.ndarray:
        rows = stages[stage]["source_indices"] + stages[stage]["target_indices"]
        return np.unique(np.concatenate(rows)) if rows else np.array([], dtype=np.int64)

    warm_item_indices = _observed("train")
    val_cold_item_indices = np.setdiff1d(_observed("val"), warm_item_indices)
    test_cold_item_indices = np.setdiff1d(
        _observed("test"), np.union1d(warm_item_indices, val_cold_item_indices)
    )

    # The training window in order, which is what a sequential model trains on:
    # it shifts internally, so handing over only the source would discard the
    # last transition the matrix pair encodes explicitly.
    train_window = [history[:-2] for history in histories]
    sequences = {
        "x_train_sequences": ItemSequences.from_rows(train_window, n_items=n_items),
        "train_source_sequences": ItemSequences.from_rows(
            ordered_sources["train"], n_items=n_items
        ),
        "val_source_sequences": ItemSequences.from_rows(
            ordered_sources["val"], n_items=n_items
        ),
        "test_source_sequences": ItemSequences.from_rows(
            ordered_sources["test"], n_items=n_items
        ),
    }

    holdouts = {
        stage: {
            "item_ids": item_ids,
            "source_indices": stages[stage]["source_indices"],
            "target_indices": stages[stage]["target_indices"],
            "user_ids": user_ids,
        }
        for stage in LEAVE_LAST_OUT_STAGES
    }

    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": train_source,
        "train_target_matrix": train_target,
        **sequences,
        "val_holdout": holdouts["val"],
        "test_holdout": holdouts["test"],
        "train_holdout": holdouts["train"],
        "warm_item_indices": warm_item_indices,
        "val_cold_item_indices": val_cold_item_indices,
        "test_cold_item_indices": test_cold_item_indices,
        "train_user_ids": user_ids,
        "val_user_ids": user_ids,
        "test_user_ids": user_ids,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": bool(val_cold_item_indices.size or test_cold_item_indices.size),
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": (
                "Leave-last-out is chronological within each user but not "
                "globally future-blind: another user's training interactions may "
                "post-date this user's test target."
            ),
            "min_history": int(min_history),
            "eligible_users": int(len(user_ids)),
            "new_val_items": int(val_cold_item_indices.size),
            "new_test_items": int(test_cold_item_indices.size),
        },
    }
