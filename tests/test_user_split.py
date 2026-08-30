from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from compresso_recsys.builder import (
    _build_args,
    _build_leave_last_out_split,
    _build_user_split,
)
from compresso_recsys.checkpoint import load_recsys_split, save_recsys_split
from compresso_recsys.datasets.base import RecSysDataset

N_ITEMS = 6


def _user_args(**overrides):
    values = {
        "dataset": "ml1m",
        "split_mode": "user_split",
        "val_users": 2,
        "test_users": 2,
        "min_user_support": 2,
        "seed": 0,
    }
    values.update(overrides)
    return _build_args(**values)


def _interactions() -> pd.DataFrame:
    """Ten users over one shared item universe, so no phase adds items."""
    items = [f"i{index}" for index in range(N_ITEMS)]
    rows = [
        {"user_id": f"u{user}", "item_id": item, "value": 1.0, "timestamp": 1_000 + user}
        for user in range(10)
        for item in items
    ]
    return pd.DataFrame(rows)


def _user_split():
    return _build_user_split(_user_args(), RecSysDataset(), _interactions())


def test_user_split_stores_every_item_partition_explicitly():
    """Regression: train used to be None while val/test were empty arrays."""
    split = _user_split()

    for key in ("warm_item_indices", "val_cold_item_indices", "test_cold_item_indices"):
        assert split[key] is not None, f"{key} must be stored explicitly"
        assert np.asarray(split[key]).dtype == np.int64

    n_items = len(split["item_ids"])
    assert n_items == N_ITEMS
    # Training spans the catalog; no later phase introduces new items.
    assert split["warm_item_indices"].tolist() == list(range(n_items))
    assert split["val_cold_item_indices"].tolist() == []
    assert split["test_cold_item_indices"].tolist() == []
    assert split["extra_metadata"]["has_item_partitions"] is False


def test_user_split_train_partition_indexes_the_catalog():
    """The partition is usable as a row selector into item-aligned arrays."""
    split = _user_split()
    item_ids = np.asarray(split["item_ids"])
    features = np.arange(len(item_ids) * 3, dtype=np.float32).reshape(len(item_ids), 3)

    rows = split["warm_item_indices"]
    assert item_ids[rows].tolist() == item_ids.tolist()
    assert features[rows].shape == features.shape
    assert split["x_train"].shape[1] == len(item_ids)


def test_user_split_item_indices_survive_a_checkpoint_round_trip(tmp_path):
    split = _user_split()
    holdout = split["test_holdout"]
    val_holdout = split["val_holdout"]

    save_recsys_split(
        tmp_path,
        item_ids=np.asarray(split["item_ids"]),
        x_train=split["x_train"],
        val_source_indices=val_holdout["source_indices"],
        val_target_indices=val_holdout["target_indices"],
        test_source_indices=holdout["source_indices"],
        test_target_indices=holdout["target_indices"],
        warm_item_indices=split["warm_item_indices"],
        val_cold_item_indices=split["val_cold_item_indices"],
        test_cold_item_indices=split["test_cold_item_indices"],
    )
    loaded = load_recsys_split(tmp_path)

    n_items = len(split["item_ids"])
    assert loaded["warm_item_indices"].tolist() == list(range(n_items))
    assert loaded["val_cold_item_indices"].tolist() == []
    assert loaded["test_cold_item_indices"].tolist() == []
    # Every phase scores the same catalog, which is what the *_item_ids report.
    assert loaded["train_item_ids"].tolist() == loaded["item_ids"].tolist()
    assert loaded["val_item_ids"].tolist() == loaded["item_ids"].tolist()
    assert loaded["test_item_ids"].tolist() == loaded["item_ids"].tolist()


def test_missing_train_partition_still_loads_as_the_whole_catalog(tmp_path):
    """Checkpoints written before the partitions were explicit must still load."""
    matrix = csr_matrix(np.ones((1, 2), dtype=np.float32))

    save_recsys_split(
        tmp_path,
        item_ids=np.asarray(["A", "B"]),
        x_train=matrix,
        val_source_indices=[np.asarray([0])],
        val_target_indices=[np.asarray([1])],
        test_source_indices=[np.asarray([0])],
        test_target_indices=[np.asarray([1])],
        warm_item_indices=None,
        val_cold_item_indices=None,
        test_cold_item_indices=None,
    )
    loaded = load_recsys_split(tmp_path)

    assert loaded["warm_item_indices"].tolist() == [0, 1]
    assert loaded["val_cold_item_indices"].tolist() == []
    assert loaded["test_cold_item_indices"].tolist() == []


# --------------------------------------------------------------------------
# eval_draws and eval_holdout_frac
# --------------------------------------------------------------------------


def _toy_matrix(rows=40, cols=25, density=0.4, seed=7):
    import numpy as np
    from scipy.sparse import csr_matrix

    rng = np.random.default_rng(seed)
    return csr_matrix((rng.random((rows, cols)) < density).astype("float32"))


def test_eval_draws_stacks_one_row_per_user_per_draw():
    from compresso_recsys.retrieval import _build_eval_draws

    x = _toy_matrix()

    for draws in (1, 3, 5, 7):
        np.random.seed(42)
        source, targets = _build_eval_draws(x, draws)
        assert source.shape[0] == x.shape[0] * draws
        assert targets.shape[0] == x.shape[0] * draws


def test_eval_draws_are_independent_samples_not_a_partition():
    """They overlap by design, so the draw count has no 1/frac ceiling."""
    from compresso_recsys.retrieval import _build_eval_draws

    x = _toy_matrix()
    n = x.shape[0]
    np.random.seed(42)
    _, targets = _build_eval_draws(x, 5)

    held = [set(targets.indices[targets.indptr[b * n] : targets.indptr[b * n + 1]].tolist())
            for b in range(5)]
    sizes = sum(len(h) for h in held)

    assert len(set(map(frozenset, held))) > 1        # the draws differ
    assert len(set().union(*held)) < sizes           # and they overlap


def test_eval_holdout_frac_is_honoured():
    """It used to be accepted and ignored, always holding out 20%."""
    from compresso_recsys.retrieval import _sample_holdout_indices

    row = csr_matrix(np.ones((1, 50), dtype=np.float32))

    for frac, expected in ((0.1, 5), (0.2, 10), (0.5, 25), (0.8, 40)):
        np.random.seed(0)
        assert len(_sample_holdout_indices(row, frac)) == expected


def test_a_user_always_contributes_at_least_one_target():
    from compresso_recsys.retrieval import _sample_holdout_indices

    row = csr_matrix(np.ones((1, 3), dtype=np.float32))

    np.random.seed(0)
    assert len(_sample_holdout_indices(row, 0.01)) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"eval_draws": 0}, "eval_draws must be >= 1"),
        ({"eval_draws": -1}, "eval_draws must be >= 1"),
        ({"eval_holdout_frac": 0.0}, "strictly between 0 and 1"),
        ({"eval_holdout_frac": 1.0}, "strictly between 0 and 1"),
    ],
)
def test_invalid_draw_settings_are_rejected(kwargs, message):
    from compresso_recsys.retrieval import build_eval_holdout

    with pytest.raises(ValueError, match=message):
        build_eval_holdout(
            train_item_ids=pd.Index(["a", "b"]),
            eval_interactions=pd.DataFrame({"user_id": ["u"], "item_id": ["a"]}),
            **kwargs,
        )


# --------------------------------------------------------------------------
# leave_last_out: the chronological protocol
# --------------------------------------------------------------------------


def _events(histories: dict[str, list[str]]) -> pd.DataFrame:
    """Event-level frame with strictly increasing timestamps per user."""
    rows = []
    for user, items in histories.items():
        for t, item in enumerate(items):
            rows.append({"user_id": user, "item_id": item, "value": 1.0, "timestamp": 1000 + t})
    return pd.DataFrame(rows)


def _llo_args(**over):
    return _build_args(dataset="goodbooks", split_mode="leave_last_out", **over)


def test_leave_last_out_holds_out_the_last_two_interactions():
    """Positions n and n-1 leave training; n-2 is the training target."""
    from compresso_recsys.retrieval import leave_last_out_stage_slices

    history = np.arange(10)
    train_s, train_t = leave_last_out_stage_slices(history, "train")
    val_s, val_t = leave_last_out_stage_slices(history, "val")
    test_s, test_t = leave_last_out_stage_slices(history, "test")

    assert test_t.tolist() == [9]
    assert val_t.tolist() == [8]
    assert train_t.tolist() == [7]
    # Each source is the previous stage's source plus its target.
    assert train_s.tolist() == list(range(7))
    assert val_s.tolist() == list(range(8))
    assert test_s.tolist() == list(range(9))
    # Two items are withheld from training; the training target is not.
    assert set(np.union1d(train_s, train_t).tolist()) == set(range(8))


def test_leave_last_out_keeps_the_whole_catalog():
    """The bug this replaces stripped every target item from training."""
    df = _events({f"u{i}": [f"i{(i + j) % 6}" for j in range(5)] for i in range(12)})
    payload = _build_leave_last_out_split(_llo_args(), df)

    catalog = set(payload["item_ids"].tolist())
    trained = set(payload["item_ids"][payload["x_train"].indices].tolist())
    assert trained == catalog, "training must still see every item"
    assert payload["x_train"].shape[1] == len(catalog)


def test_leave_last_out_val_and_test_are_distinct():
    """They were literally the same object, so tuning on val was tuning on test."""
    df = _events({f"u{i}": [f"i{j}" for j in range(6)] for i in range(5)})
    payload = _build_leave_last_out_split(_llo_args(), df)

    val, test = payload["val_holdout"], payload["test_holdout"]
    assert val is not test
    for v, t in zip(val["target_indices"], test["target_indices"]):
        assert v.tolist() != t.tolist()
    # And the test source is exactly one item longer than the validation source.
    for vs, ts in zip(val["source_indices"], test["source_indices"]):
        assert len(ts) == len(vs) + 1


def test_leave_last_out_training_never_sees_the_held_out_items():
    df = _events({f"u{i}": [f"i{j}" for j in range(6)] for i in range(5)})
    payload = _build_leave_last_out_split(_llo_args(), df)

    x_train = payload["x_train"].tolil()
    for row, (v, t) in enumerate(
        zip(payload["val_holdout"]["target_indices"],
            payload["test_holdout"]["target_indices"])
    ):
        assert x_train[row, int(v[0])] == 0, "validation target leaked into training"
        assert x_train[row, int(t[0])] == 0, "test target leaked into training"


def test_x_train_is_the_union_of_the_training_pair():
    df = _events({f"u{i}": [f"i{j}" for j in range(7)] for i in range(4)})
    payload = _build_leave_last_out_split(_llo_args(), df)

    union = payload["train_source_matrix"].maximum(payload["train_target_matrix"])
    assert (payload["x_train"] != union.tocsr()).nnz == 0
    # And the pair is a genuine partition, not two copies of x_train.
    assert (payload["train_source_matrix"] != payload["x_train"]).nnz > 0


def test_leave_last_out_requires_four_interactions():
    short = _events({"u0": ["a", "b", "c"], "u1": ["a", "b", "c", "d"]})
    payload = _build_leave_last_out_split(_llo_args(), short)

    assert payload["train_user_ids"].tolist() == ["u1"], "3 interactions is too few"

    with pytest.raises(ValueError, match="at least 4 interactions"):
        _build_leave_last_out_split(_llo_args(), _events({"u0": ["a", "b", "c"]}))


def test_leave_last_out_catalog_contains_only_items_from_eligible_histories():
    eligible = _events({"eligible": ["a", "b", "c", "d", "e"]})
    short = _events({"short": ["short-only"]})
    invalid = pd.DataFrame(
        [
            {
                "user_id": "eligible",
                "item_id": "invalid-time-only",
                "value": 1.0,
                "timestamp": "not-a-timestamp",
            }
        ]
    )

    payload = _build_leave_last_out_split(
        _llo_args(),
        pd.concat([eligible, short, invalid], ignore_index=True),
    )

    assert payload["item_ids"].tolist() == ["a", "b", "c", "d", "e"]
    assert payload["x_train"].shape[1] == 5
    partitioned = np.concatenate(
        [
            payload["warm_item_indices"],
            payload["val_cold_item_indices"],
            payload["test_cold_item_indices"],
        ]
    )
    assert np.sort(partitioned).tolist() == list(range(5))


def test_item_partitions_are_observed_not_imposed():
    """Overlapping histories yield empty partitions; a tail-only item lands in one.

    The histories must be *staggered*. Give every user the same order and the
    final items are everyone's tail, so they never appear in any training prefix
    and are correctly reported as new — which is the protocol working, not a
    dense case.
    """
    dense = _events({f"u{i}": [f"i{(i + j) % 6}" for j in range(5)] for i in range(12)})
    payload = _build_leave_last_out_split(_llo_args(), dense)
    assert payload["val_cold_item_indices"].size == 0
    assert payload["test_cold_item_indices"].size == 0

    # "rare" appears once, as u0's final interaction, so nothing trains on it.
    # "e" is also a test target, but u2 sees it early enough to train on, which
    # is exactly the distinction the partitions are supposed to make.
    sparse = _events({
        "u0": ["a", "b", "c", "d", "rare"],
        "u1": ["a", "b", "c", "d", "e"],
        "u2": ["e", "a", "b", "c", "d"],
    })
    payload = _build_leave_last_out_split(_llo_args(), sparse)
    items = payload["item_ids"]
    assert items[payload["test_cold_item_indices"]].tolist() == ["rare"]
    # "d" first appears as a validation target, so it belongs to that phase.
    assert items[payload["val_cold_item_indices"]].tolist() == ["d"]


def test_leave_last_out_orders_by_timestamp_not_by_row_order():
    shuffled = pd.DataFrame([
        {"user_id": "u0", "item_id": "d", "value": 1.0, "timestamp": 40},
        {"user_id": "u0", "item_id": "b", "value": 1.0, "timestamp": 20},
        {"user_id": "u0", "item_id": "e", "value": 1.0, "timestamp": 50},
        {"user_id": "u0", "item_id": "a", "value": 1.0, "timestamp": 10},
        {"user_id": "u0", "item_id": "c", "value": 1.0, "timestamp": 30},
    ])
    payload = _build_leave_last_out_split(_llo_args(), shuffled)

    items = payload["item_ids"]
    assert items[payload["test_holdout"]["target_indices"][0]].tolist() == ["e"]
    assert items[payload["val_holdout"]["target_indices"][0]].tolist() == ["d"]


@pytest.mark.parametrize("mode", ["user_split", "leave_last_out", "temporal"])
def test_x_train_is_the_union_of_the_training_pair_in_every_split_mode(mode):
    """The one relationship every split mode owes the checkpoint.

    Chronological modes partition genuinely; ``user_split`` sets both keys to
    ``x_train`` and satisfies it trivially. Either way a model reading the pair
    and a model reading ``x_train`` must see the same interactions.
    """
    if mode == "user_split":
        payload = _user_split()
    elif mode == "leave_last_out":
        df = _events({f"u{i}": [f"i{(i + j) % 7}" for j in range(6)] for i in range(10)})
        payload = _build_leave_last_out_split(_llo_args(), df)
    else:
        from test_temporal_split import _temporal_args, _timeline
        from compresso_recsys.builder import _build_temporal_split

        payload = _build_temporal_split(_temporal_args(), _timeline())

    union = payload["train_source_matrix"].maximum(payload["train_target_matrix"])
    assert (payload["x_train"].tocsr() != union.tocsr()).nnz == 0, mode


def test_only_chronological_modes_partition_the_training_data():
    """user_split has no boundary to divide on, so both keys are x_train."""
    payload = _user_split()
    assert (payload["train_source_matrix"] != payload["x_train"]).nnz == 0
    assert (payload["train_target_matrix"] != payload["x_train"]).nnz == 0

    df = _events({f"u{i}": [f"i{(i + j) % 7}" for j in range(6)] for i in range(10)})
    chrono = _build_leave_last_out_split(_llo_args(), df)
    assert (chrono["train_source_matrix"] != chrono["x_train"]).nnz > 0


def test_save_refuses_a_training_pair_that_disagrees_with_x_train():
    """A future split mode cannot quietly partition its training data wrongly."""
    import tempfile
    from compresso_recsys.checkpoint import save_recsys_split

    n = 4
    full = csr_matrix(np.eye(n, dtype=np.float32))
    half = csr_matrix((n, n), dtype=np.float32)          # loses every interaction

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError, match="union of train_source_matrix"):
            save_recsys_split(
                tmp,
                item_ids=np.array([f"i{i}" for i in range(n)]),
                x_train=full,
                train_source_matrix=half,
                train_target_matrix=half,
                val_source_indices=[np.array([0])] * n,
                val_target_indices=[np.array([1])] * n,
                test_source_indices=[np.array([0])] * n,
                test_target_indices=[np.array([1])] * n,
            )


# --------------------------------------------------------------------------
# leave_last_out honours the support arguments, or refuses them
# --------------------------------------------------------------------------


def _llo_events(n_events: int, n_users: int = 6) -> pd.DataFrame:
    return pd.DataFrame([
        {"user_id": f"u{u}", "item_id": f"i{(u + t) % 12}", "value": 1.0,
         "timestamp": 100 + t}
        for u in range(n_users)
        for t in range(n_events)
    ])


@pytest.mark.parametrize(
    ("overrides", "expected_min_history"),
    [
        ({}, 4),                              # structural floor
        ({"min_user_support": 10}, 10),        # drop short users outright
        ({"min_source_items": 5}, 8),          # a longer source costs three targets
        ({"min_user_support": 6, "min_source_items": 9}, 12),   # the stricter wins
    ],
)
def test_support_arguments_reach_the_minimum_history(overrides, expected_min_history):
    """They were silently ignored once; the floor must be derived, not hardcoded."""
    payload = _build_leave_last_out_split(_llo_args(**overrides), _llo_events(14))

    assert payload["extra_metadata"]["min_history"] == expected_min_history


def test_min_user_support_actually_drops_users():
    mixed = pd.concat([_llo_events(12, n_users=3),
                       _llo_events(5, n_users=1).assign(user_id="short")])

    payload = _build_leave_last_out_split(
        _llo_args(min_user_support=10), mixed
    )

    assert "short" not in payload["train_user_ids"].tolist()
    assert payload["extra_metadata"]["eligible_users"] == 3


def test_min_target_items_above_one_is_refused_not_ignored():
    """Each stage holds out exactly one item, so more cannot be delivered."""
    with pytest.raises(ValueError, match="exactly one item per stage"):
        _build_leave_last_out_split(_llo_args(min_target_items=2), _llo_events(14))


def test_a_checkpoint_written_before_the_rename_still_reads_correctly(tmp_path):
    """The fallback exists to prevent a confident wrong answer, not for politeness.

    A missing warm file defaults to the whole catalog and a missing cold file to
    nothing, so reading only the new names would report every item warm on an
    older checkpoint — and for ``leave_last_out`` and ``item_split``, where all
    three phases share one catalog, nothing downstream could notice.
    """
    from compresso_recsys.checkpoint import SPLIT_DIR, load_recsys_split

    item_ids = np.asarray(["a", "b", "c", "d"])
    matrix = csr_matrix(np.ones((2, 4), dtype=np.float32))
    save_recsys_split(
        tmp_path,
        item_ids=item_ids,
        x_train=matrix,
        val_source_indices=[np.asarray([0]), np.asarray([1])],
        val_target_indices=[np.asarray([1]), np.asarray([2])],
        test_source_indices=[np.asarray([0]), np.asarray([1])],
        test_target_indices=[np.asarray([2]), np.asarray([3])],
        warm_item_indices=np.asarray([0, 1]),
        val_cold_item_indices=np.asarray([2]),
        test_cold_item_indices=np.asarray([3]),
    )

    # Rewind the files to what they used to be called.
    data = tmp_path / SPLIT_DIR
    for new, old in (
        ("warm_item_indices", "train_item_indices"),
        ("val_cold_item_indices", "val_item_indices"),
        ("test_cold_item_indices", "test_item_indices"),
    ):
        (data / f"{new}.npy").rename(data / f"{old}.npy")

    loaded = load_recsys_split(tmp_path)

    assert loaded["warm_item_indices"].tolist() == [0, 1]
    assert loaded["val_cold_item_indices"].tolist() == [2]
    assert loaded["test_cold_item_indices"].tolist() == [3]


def test_a_new_name_wins_over_a_stale_old_one(tmp_path):
    """Both present means the old file is a leftover, not the truth."""
    from compresso_recsys.checkpoint import SPLIT_DIR, load_recsys_split

    matrix = csr_matrix(np.ones((2, 4), dtype=np.float32))
    save_recsys_split(
        tmp_path,
        item_ids=np.asarray(["a", "b", "c", "d"]),
        x_train=matrix,
        val_source_indices=[np.asarray([0]), np.asarray([1])],
        val_target_indices=[np.asarray([1]), np.asarray([2])],
        test_source_indices=[np.asarray([0]), np.asarray([1])],
        test_target_indices=[np.asarray([2]), np.asarray([3])],
        warm_item_indices=np.asarray([0, 1]),
        val_cold_item_indices=np.asarray([2]),
        test_cold_item_indices=np.asarray([3]),
    )
    np.save(tmp_path / SPLIT_DIR / "train_item_indices.npy", np.asarray([9, 9, 9]))

    assert load_recsys_split(tmp_path)["warm_item_indices"].tolist() == [0, 1]
