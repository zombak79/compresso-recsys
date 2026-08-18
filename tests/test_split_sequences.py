"""Sequences and matrices are two views of one event stream.

The structure itself is covered in ``test_sequences.py``. These tests are about
the builders producing both views from the same events, in the same row and
column space, for the modes that have a chronological order to preserve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compresso_recsys.builder import (
    _build_args,
    _build_leave_last_out_split,
    _build_temporal_split,
    _build_user_split,
)
from compresso_recsys.datasets.base import RecSysDataset
from compresso_recsys.sequences import ItemSequences

SEQUENCE_KEYS = (
    "x_train_sequences",
    "train_source_sequences",
    "val_source_sequences",
    "test_source_sequences",
)


def _events(order: list[str], n_users: int = 8, stagger: bool = True) -> pd.DataFrame:
    """One event per row, timestamps strictly increasing within each user."""
    rows = []
    for user in range(n_users):
        items = order[user:] + order[:user] if stagger else order
        for step, item in enumerate(items):
            rows.append(
                {
                    "user_id": f"u{user}",
                    "item_id": item,
                    "value": 1.0,
                    "timestamp": 1_000_000 + step * 90_000,
                }
            )
    return pd.DataFrame(rows)


def _llo(df: pd.DataFrame):
    return _build_leave_last_out_split(
        _build_args(dataset="goodbooks", split_mode="leave_last_out"), df
    )


def _temporal(df: pd.DataFrame):
    args = _build_args(
        dataset="goodbooks",
        split_mode="temporal",
        temporal_period_hours=50,
        min_user_support=1,
        item_min_support=1,
    )
    return _build_temporal_split(args, df)


# --------------------------------------------------------------------------
# both chronological modes carry all four sequence fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["leave_last_out", "temporal"])
def test_chronological_modes_produce_every_sequence_field(mode):
    df = _events([f"i{j}" for j in range(9)])
    payload = _llo(df) if mode == "leave_last_out" else _temporal(df)

    for key in SEQUENCE_KEYS:
        assert key in payload, key
        assert isinstance(payload[key], ItemSequences), key


def test_non_chronological_modes_produce_none():
    """user_split has no ordering to preserve, so pretending otherwise is worse."""
    payload = _build_user_split(
        _build_args(
            dataset="ml1m",
            split_mode="user_split",
            val_users=2,
            test_users=2,
            min_user_support=2,
            seed=0,
        ),
        RecSysDataset(),
        _events([f"i{j}" for j in range(6)], n_users=10, stagger=False),
    )

    assert not any(key in payload for key in SEQUENCE_KEYS)


# --------------------------------------------------------------------------
# the two views describe the same events
# --------------------------------------------------------------------------


def test_leave_last_out_sequences_match_their_matrices():
    df = _events([f"i{j}" for j in range(9)])
    payload = _llo(df)

    x_train = payload["x_train"].tocsr()
    sequences = payload["x_train_sequences"]

    assert sequences.n_rows == x_train.shape[0]
    assert sequences.n_items == x_train.shape[1]
    for row in range(sequences.n_rows):
        assert set(sequences.row(row).tolist()) == set(x_train[row].indices.tolist())


@pytest.mark.parametrize(
    ("matrix_key", "sequence_key"),
    [
        ("x_train", "x_train_sequences"),
        ("val_source_matrix", "val_source_sequences"),
        ("test_source_matrix", "test_source_sequences"),
    ],
)
def test_temporal_sequences_come_from_the_same_stage_as_their_matrix(
    matrix_key, sequence_key
):
    """Each temporal stage is filtered independently.

    A window sequence borrowed from a neighbouring stage covers the same events
    but addresses a different row and column space, so the shapes silently
    disagree. This is the check for that.
    """
    payload = _temporal(_events([f"i{j}" for j in range(9)]))

    matrix = payload[matrix_key].tocsr()
    sequences = payload[sequence_key]

    assert sequences.n_rows == matrix.shape[0], sequence_key
    assert sequences.n_items == matrix.shape[1], sequence_key
    for row in range(sequences.n_rows):
        assert set(sequences.row(row).tolist()) == set(matrix[row].indices.tolist())


# --------------------------------------------------------------------------
# what the sequence view carries that the matrix cannot
# --------------------------------------------------------------------------


def test_sequences_are_chronological_not_index_sorted():
    """Item indices run against the clock, so the two orders cannot coincide."""
    df = _events(["i9", "i7", "i5", "i3", "i1", "i0", "i2", "i4", "i6"], stagger=False)
    payload = _temporal(df)

    row = payload["test_source_sequences"].row(0)
    item_ids = np.asarray(payload["test_item_ids"])

    assert row.size > 1
    assert not np.all(np.diff(row) >= 0), "index-sorted, so the order was lost"
    assert item_ids[row].tolist()[:3] == ["i9", "i7", "i5"]


def test_duplicate_interactions_survive_in_the_sequence_view():
    """A CSR row merges a repeat; the history keeps it.

    ``leave_last_out`` rather than ``temporal`` here: it needs no window
    arithmetic, so the fixture can be exactly the repetition under test.
    """
    rows = []
    for user in range(4):
        for step, item in enumerate(["a", "b", "a", "c", "a", "d"]):
            rows.append(
                {"user_id": f"u{user}", "item_id": item, "value": 1.0,
                 "timestamp": 1_000 + step}
            )
    payload = _llo(pd.DataFrame(rows))

    sequences = payload["x_train_sequences"]
    history = sequences.row(0).tolist()
    item_ids = np.asarray(payload["item_ids"])

    # Training window is the first four events: a, b, a, c.
    assert item_ids[history].tolist() == ["a", "b", "a", "c"]
    assert len(history) > len(set(history)), "duplicates were merged away"

    # The matrix view of the same events keeps one column per item.
    matrix_row = payload["train_source_matrix"].maximum(
        payload["train_target_matrix"]
    ).tocsr()[0]
    assert matrix_row.indices.size == len(set(history))


# --------------------------------------------------------------------------
# the stage relationships the protocol promises
# --------------------------------------------------------------------------


def test_leave_last_out_stage_sources_grow_by_one_each_time():
    df = _events([f"i{j}" for j in range(9)])
    payload = _llo(df)

    train = payload["train_source_sequences"]
    val = payload["val_source_sequences"]
    test = payload["test_source_sequences"]

    for row in range(train.n_rows):
        t, v, s = train.row(row), val.row(row), test.row(row)
        assert v.tolist() == t.tolist() + [v[-1]]
        assert s.tolist() == v.tolist() + [s[-1]]


def test_leave_last_out_train_window_equals_the_validation_source():
    """Validation scores a model on exactly what it was allowed to train on."""
    payload = _llo(_events([f"i{j}" for j in range(9)]))

    assert (
        payload["x_train_sequences"].values.tolist()
        == payload["val_source_sequences"].values.tolist()
    )
    # And the matrix side already agrees the same way.
    assert (
        payload["x_train"].tocsr()
        != payload["train_source_matrix"].maximum(payload["train_target_matrix"]).tocsr()
    ).nnz == 0


def test_sequences_never_contain_the_held_out_targets():
    df = _events([f"i{j}" for j in range(9)])
    payload = _llo(df)

    train_window = payload["x_train_sequences"]
    for row, (val_target, test_target) in enumerate(
        zip(
            payload["val_holdout"]["target_indices"],
            payload["test_holdout"]["target_indices"],
        )
    ):
        history = set(train_window.row(row).tolist())
        assert int(val_target[0]) not in history
        assert int(test_target[0]) not in history
