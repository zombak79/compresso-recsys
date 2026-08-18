"""Chronological interaction histories, in a form no model has opinions about.

A CSR matrix answers "which items did this user interact with". It cannot answer
"in what order", because a set has none, and it cannot say an item recurred.
Sequential models need both, so the checkpoint has to keep them, which means
building them from event-level data before anything collapses events into a
matrix.

:class:`ItemSequences` is deliberately the smallest structure that carries that
information and nothing else. It holds catalog indices in chronological order,
with duplicates preserved, and no padding, no ``MASK`` or ``PAD`` or ``BOS``
token, no maximum length and no truncation.

That austerity is the point. Tokenisation is a modelling decision and models
disagree about it: SASRec offsets indices and samples negatives, a masked model
inserts ``MASK``, an RNN pads to the batch maximum, a transformer truncates to a
context window. Baking any of those into the checkpoint would make it wrong for
the others, so all of them happen inside trainers and none of them here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "ItemSequences",
    "load_item_sequences",
    "save_item_sequences",
]

#: Row offsets and values are both stored at fixed width, so a checkpoint means
#: the same thing across platforms and regardless of catalog size.
_INDEX_DTYPE = np.int64


def _owned(array: np.ndarray, source: object) -> np.ndarray:
    """Return ``array`` guaranteed not to alias ``source``.

    ``np.ascontiguousarray`` hands back the input untouched when it already has
    the requested dtype and layout, so freezing the result would reach out and
    freeze the caller's own array. The same helper exists in
    :mod:`compresso_recsys.evaluation` for the same reason; it is duplicated
    rather than imported because evaluation depends on this module, not the
    other way round.
    """
    return array.copy() if array is source else array


@dataclass(frozen=True)
class ItemSequences:
    """Per-row chronological item histories, oldest first.

    ``values`` holds catalog indices for every row concatenated, and ``indptr``
    marks where each row begins, exactly as CSR does — row ``i`` is
    ``values[indptr[i]:indptr[i + 1]]``. Unlike CSR there is no data array,
    because a history has no weights, and no column sorting, because the order
    *is* the information.

    Rows may be empty. A user with no history is a real prediction case, and
    refusing to represent one would only push the special case into every caller.

    Rows carry no identity. ``csr_matrix`` does not name its rows either, and
    evaluation aligns ``sample_ids`` positionally against whichever source it was
    given; carrying identifiers on one source type and not the other would mean
    two alignment stories instead of one.
    """

    values: np.ndarray
    indptr: np.ndarray
    n_items: int

    def __post_init__(self) -> None:
        values = _owned(
            np.ascontiguousarray(self.values, dtype=_INDEX_DTYPE), self.values
        )
        indptr = _owned(
            np.ascontiguousarray(self.indptr, dtype=_INDEX_DTYPE), self.indptr
        )

        if indptr.ndim != 1 or indptr.size < 1:
            raise ValueError("indptr must be a one-dimensional array of length >= 1")
        if values.ndim != 1:
            raise ValueError("values must be one-dimensional")
        if int(indptr[0]) != 0:
            raise ValueError(f"indptr must start at 0, got {int(indptr[0])}")
        if np.any(np.diff(indptr) < 0):
            raise ValueError("indptr must be non-decreasing")
        if int(indptr[-1]) != values.size:
            raise ValueError(
                f"indptr ends at {int(indptr[-1])} but values holds {values.size} "
                "entries; every value must belong to exactly one row"
            )

        n_items = int(self.n_items)
        if n_items < 0:
            raise ValueError("n_items must be >= 0")
        if values.size:
            if int(values.min()) < 0 or int(values.max()) >= n_items:
                raise ValueError(
                    f"values must be catalog indices in [0, {n_items}), got range "
                    f"[{int(values.min())}, {int(values.max())}]"
                )

        # Frozen protects the attribute bindings, not the buffers behind them.
        # Freezing the buffers too means a caller cannot reorder a history after
        # the fact and silently change what a model was trained on.
        values.setflags(write=False)
        indptr.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "indptr", indptr)
        object.__setattr__(self, "n_items", n_items)

    @classmethod
    def from_rows(
        cls,
        rows: list[np.ndarray] | list[list[int]],
        *,
        n_items: int,
    ) -> "ItemSequences":
        """Build from one array per row, in row order."""
        arrays = [np.asarray(row, dtype=_INDEX_DTYPE).ravel() for row in rows]
        lengths = np.fromiter((a.size for a in arrays), dtype=_INDEX_DTYPE,
                              count=len(arrays))
        indptr = np.concatenate(([0], np.cumsum(lengths))).astype(_INDEX_DTYPE)
        values = (
            np.concatenate(arrays).astype(_INDEX_DTYPE, copy=False)
            if arrays
            else np.empty(0, dtype=_INDEX_DTYPE)
        )
        return cls(values=values, indptr=indptr, n_items=n_items)

    @property
    def n_rows(self) -> int:
        """Number of histories."""
        return int(self.indptr.size - 1)

    def __len__(self) -> int:
        return self.n_rows

    @property
    def row_lengths(self) -> np.ndarray:
        """Interactions per row.

        Also the slicing-invariant description of the structure: ``indptr`` is
        rebased by :meth:`take_rows`, so anything identifying these sequences
        must hash lengths rather than offsets.
        """
        return np.diff(self.indptr)

    def row(self, index: int) -> np.ndarray:
        """One history, oldest first, as a read-only view."""
        if not -self.n_rows <= index < self.n_rows:
            raise IndexError(
                f"row index {index} out of range for {self.n_rows} rows"
            )
        if index < 0:
            index += self.n_rows
        return self.values[self.indptr[index] : self.indptr[index + 1]]

    def take_rows(self, start: int, stop: int) -> "ItemSequences":
        """Rows ``start:stop`` as a new instance, with ``indptr`` rebased.

        Half-open, as Python slicing is. The rebasing is why ``indptr`` cannot
        identify a slice: every slice starts at zero regardless of where it came
        from.
        """
        start = max(0, min(int(start), self.n_rows))
        stop = max(start, min(int(stop), self.n_rows))
        offsets = self.indptr[start : stop + 1]
        return ItemSequences(
            values=self.values[self.indptr[start] : self.indptr[stop]],
            indptr=offsets - offsets[0],
            n_items=self.n_items,
        )

    def __getitem__(self, key: slice) -> "ItemSequences":
        if not isinstance(key, slice):
            raise TypeError(
                "ItemSequences supports slicing only; use .row(i) for one history"
            )
        if key.step not in (None, 1):
            raise ValueError("ItemSequences slicing does not support a step")
        start, stop, _ = key.indices(self.n_rows)
        return self.take_rows(start, stop)

    def __repr__(self) -> str:
        return (
            f"ItemSequences(n_rows={self.n_rows}, n_items={self.n_items}, "
            f"n_events={self.values.size})"
        )


def save_item_sequences(path: str | Path, sequences: ItemSequences) -> None:
    """Write sequences to a single ``.npz``."""
    np.savez(
        Path(path),
        values=sequences.values,
        indptr=sequences.indptr,
        n_items=np.asarray(sequences.n_items, dtype=_INDEX_DTYPE),
    )


def load_item_sequences(path: str | Path) -> ItemSequences:
    """Read sequences written by :func:`save_item_sequences`."""
    with np.load(Path(path), allow_pickle=False) as handle:
        return ItemSequences(
            values=handle["values"],
            indptr=handle["indptr"],
            n_items=int(handle["n_items"]),
        )
