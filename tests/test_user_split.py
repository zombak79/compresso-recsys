from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.builder import _build_args, _build_user_split
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

    for key in ("train_item_indices", "val_item_indices", "test_item_indices"):
        assert split[key] is not None, f"{key} must be stored explicitly"
        assert np.asarray(split[key]).dtype == np.int64

    n_items = len(split["item_ids"])
    assert n_items == N_ITEMS
    # Training spans the catalog; no later phase introduces new items.
    assert split["train_item_indices"].tolist() == list(range(n_items))
    assert split["val_item_indices"].tolist() == []
    assert split["test_item_indices"].tolist() == []
    assert split["extra_metadata"]["has_item_partitions"] is False


def test_user_split_train_partition_indexes_the_catalog():
    """The partition is usable as a row selector into item-aligned arrays."""
    split = _user_split()
    item_ids = np.asarray(split["item_ids"])
    features = np.arange(len(item_ids) * 3, dtype=np.float32).reshape(len(item_ids), 3)

    rows = split["train_item_indices"]
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
        train_item_indices=split["train_item_indices"],
        val_item_indices=split["val_item_indices"],
        test_item_indices=split["test_item_indices"],
    )
    loaded = load_recsys_split(tmp_path)

    n_items = len(split["item_ids"])
    assert loaded["train_item_indices"].tolist() == list(range(n_items))
    assert loaded["val_item_indices"].tolist() == []
    assert loaded["test_item_indices"].tolist() == []
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
        train_item_indices=None,
        val_item_indices=None,
        test_item_indices=None,
    )
    loaded = load_recsys_split(tmp_path)

    assert loaded["train_item_indices"].tolist() == [0, 1]
    assert loaded["val_item_indices"].tolist() == []
    assert loaded["test_item_indices"].tolist() == []
