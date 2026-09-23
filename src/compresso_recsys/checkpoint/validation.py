"""Cross-stage consistency checks, run before a split is written.

A later stage must nest inside the catalog of the one before it, and a
sequence entry must agree with the matrix beside it. Both are cheap to check
here and very expensive to discover after training.
"""

from __future__ import annotations


import numpy as np
from scipy.sparse import csr_matrix

from compresso_recsys.sequences import (
    ItemSequences,
)


def _check_stage_catalogs_nest(
    *,
    train_item_ids: np.ndarray,
    val_item_ids: np.ndarray,
    test_item_ids: np.ndarray,
) -> None:
    """Every stage catalog must extend the previous one by appending.

    A warm item therefore keeps its column index in every later stage, which is
    what lets a model fitted on the training catalog read a later stage's
    indices directly: below its own item count is one of its items, at or above
    is one it has never seen. ``temporal`` grows the catalog window by window and
    the other modes hold it fixed, so this already held everywhere -- but nothing
    enforced it, and a mode that re-sorted item IDs per stage would silently
    change what an index means between stages rather than failing.
    """
    for earlier_name, earlier, later_name, later in (
        ("train_item_ids", train_item_ids, "val_item_ids", val_item_ids),
        ("val_item_ids", val_item_ids, "test_item_ids", test_item_ids),
    ):
        if earlier.size > later.size:
            raise ValueError(
                f"{later_name} has {later.size} items but {earlier_name} has "
                f"{earlier.size}; a stage catalog may only grow"
            )
        if not np.array_equal(earlier, later[: earlier.size]):
            disagreement = int(np.flatnonzero(earlier != later[: earlier.size])[0])
            raise ValueError(
                f"{later_name} must extend {earlier_name} by appending, but they "
                f"differ at index {disagreement}: {earlier[disagreement]!r} "
                f"versus {later[disagreement]!r}. Stage catalogs that reorder "
                "make a column index mean different items in different stages"
            )


def _check_sequence_matches_sibling(
    sequences: ItemSequences,
    sibling: csr_matrix | list[np.ndarray],
    name: str,
    sibling_name: str,
) -> None:
    """A sequence and the view beside it must describe the same events.

    Sharing a column space is not enough: two views built from different filter
    passes can agree on their shape and disagree on their contents, which trains
    a sequential model and a matrix model on different data while every shape
    check passes. Order and repeats are the sequence view's whole purpose, so the
    comparison is per row and set-wise -- the matrix view cannot express either.
    """
    if isinstance(sibling, csr_matrix):
        rows = sibling.shape[0]
        member_sets = (
            set(sibling.indices[sibling.indptr[i] : sibling.indptr[i + 1]].tolist())
            for i in range(rows)
        )
    else:
        rows = len(sibling)
        member_sets = (np.asarray(entry).tolist() for entry in sibling)

    if rows != sequences.n_rows:
        raise ValueError(
            f"{name} has {sequences.n_rows} rows but {sibling_name} has {rows}; "
            "the two views must address the same rows"
        )
    for row, members in enumerate(member_sets):
        if set(sequences.row(row).tolist()) != set(members):
            raise ValueError(
                f"{name} and {sibling_name} disagree on row {row}; the two views "
                "must describe the same events"
            )
