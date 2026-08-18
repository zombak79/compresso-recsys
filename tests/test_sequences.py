from __future__ import annotations

import numpy as np
import pytest

from compresso_recsys.sequences import (
    ItemSequences,
    load_item_sequences,
    save_item_sequences,
)


def _seqs(rows, n_items=10):
    return ItemSequences.from_rows(rows, n_items=n_items)


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def test_from_rows_round_trips_every_history():
    rows = [[5, 8, 5, 2], [9], [], [4, 4, 4]]

    s = _seqs(rows)

    assert s.n_rows == len(s) == 4
    assert s.values.tolist() == [5, 8, 5, 2, 9, 4, 4, 4]
    assert s.indptr.tolist() == [0, 4, 5, 5, 8]
    for i, expected in enumerate(rows):
        assert s.row(i).tolist() == expected


def test_order_and_duplicates_survive():
    """The two things a CSR row cannot express."""
    s = _seqs([[3, 1, 3, 1, 3]])

    assert s.row(0).tolist() == [3, 1, 3, 1, 3]
    # A set would give {1, 3}; a CSR row would give [1, 3] sorted.
    assert s.row(0).tolist() != sorted(set(s.row(0).tolist()))


def test_empty_rows_are_representable():
    """A user with no history is a real prediction case."""
    s = _seqs([[], [7], []])

    assert s.n_rows == 3
    assert s.row(0).size == 0
    assert s.row_lengths.tolist() == [0, 1, 0]


def test_no_rows_at_all():
    s = _seqs([])

    assert s.n_rows == 0
    assert s.values.size == 0
    assert s.indptr.tolist() == [0]


def test_dtype_is_fixed_regardless_of_input():
    s = ItemSequences(
        values=np.array([1, 2], dtype=np.int16),
        indptr=np.array([0, 2], dtype=np.int32),
        n_items=5,
    )

    assert s.values.dtype == np.int64
    assert s.indptr.dtype == np.int64


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "indptr", "n_items", "message"),
    [
        ([1, 2], [1, 2], 5, "must start at 0"),
        ([1, 2], [0, 2, 1], 5, "non-decreasing"),
        ([1, 2], [0, 1], 5, "every value must belong to exactly one row"),
        ([1, 2], [0, 3], 5, "every value must belong to exactly one row"),
        ([1, 9], [0, 2], 5, r"catalog indices in \[0, 5\)"),
        ([-1, 1], [0, 2], 5, r"catalog indices in \[0, 5\)"),
        ([1], [], 5, "length >= 1"),
    ],
)
def test_invalid_structures_are_refused(values, indptr, n_items, message):
    with pytest.raises(ValueError, match=message):
        ItemSequences(
            values=np.asarray(values, dtype=np.int64),
            indptr=np.asarray(indptr, dtype=np.int64),
            n_items=n_items,
        )


def test_buffers_are_read_only():
    """Frozen protects the bindings; the arrays need freezing too."""
    s = _seqs([[1, 2, 3]])

    with pytest.raises(ValueError, match="read-only"):
        s.values[0] = 9
    with pytest.raises(ValueError, match="read-only"):
        s.indptr[0] = 9


def test_constructing_does_not_freeze_the_callers_array():
    values = np.array([1, 2, 3])
    indptr = np.array([0, 3])

    ItemSequences(values=values, indptr=indptr, n_items=5)

    values[0] = 7  # must not raise
    assert values[0] == 7


# --------------------------------------------------------------------------
# slicing
# --------------------------------------------------------------------------


def test_take_rows_rebases_indptr():
    s = _seqs([[1], [2, 3], [4, 5, 6], [7]])

    middle = s.take_rows(1, 3)

    assert middle.n_rows == 2
    assert middle.indptr.tolist() == [0, 2, 5]
    assert middle.row(0).tolist() == [2, 3]
    assert middle.row(1).tolist() == [4, 5, 6]
    assert middle.n_items == s.n_items


def test_row_lengths_survive_slicing_where_indptr_does_not():
    """Why anything identifying sequences must hash lengths, not offsets."""
    s = _seqs([[1], [2, 3], [4, 5, 6]])

    tail = s.take_rows(1, 3)

    assert tail.row_lengths.tolist() == s.row_lengths[1:].tolist()
    assert tail.indptr.tolist() != s.indptr[1:].tolist()


def test_slicing_matches_take_rows_and_clamps():
    s = _seqs([[1], [2], [3]])

    assert s[1:3].values.tolist() == s.take_rows(1, 3).values.tolist()
    assert s[:].n_rows == 3
    assert s[5:9].n_rows == 0
    assert s.take_rows(-4, 99).n_rows == 3


def test_slicing_batches_reassemble_the_whole():
    rows = [[i, i + 1] for i in range(7)]
    s = _seqs(rows, n_items=20)

    for size in (1, 2, 3, 7, 100):
        batched = [
            s.take_rows(start, start + size).row(i).tolist()
            for start in range(0, s.n_rows, size)
            for i in range(s.take_rows(start, start + size).n_rows)
        ]
        assert batched == rows, size


@pytest.mark.parametrize("step", [2, -1])
def test_slicing_refuses_a_step(step):
    with pytest.raises(ValueError, match="does not support a step"):
        _seqs([[1], [2], [3]])[::step]


def test_single_row_access_needs_row_not_getitem():
    s = _seqs([[1], [2]])

    with pytest.raises(TypeError, match="slicing only"):
        s[0]
    with pytest.raises(IndexError, match="out of range"):
        s.row(5)
    assert s.row(-1).tolist() == [2]


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------


def test_round_trips_through_a_file(tmp_path):
    s = _seqs([[5, 8, 5], [], [2, 2]], n_items=9)
    path = tmp_path / "seqs.npz"

    save_item_sequences(path, s)
    loaded = load_item_sequences(path)

    assert loaded.values.tolist() == s.values.tolist()
    assert loaded.indptr.tolist() == s.indptr.tolist()
    assert loaded.n_items == s.n_items
    assert loaded.n_rows == s.n_rows


def test_empty_round_trips(tmp_path):
    s = _seqs([], n_items=4)
    path = tmp_path / "empty.npz"

    save_item_sequences(path, s)
    loaded = load_item_sequences(path)

    assert loaded.n_rows == 0
    assert loaded.n_items == 4
