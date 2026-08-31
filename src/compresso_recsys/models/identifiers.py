"""Stable item identities at the production boundary of a recommender."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, isspmatrix_csr

__all__ = ["ItemVocabulary", "Recommendations"]


def _item_id_values(
    item_ids: Sequence[Hashable] | np.ndarray,
    *,
    name: str,
    allow_empty: bool,
    require_unique: bool,
) -> np.ndarray:
    if isinstance(item_ids, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence of item IDs")
    try:
        values = list(item_ids)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a one-dimensional sequence of item IDs"
        ) from error

    ids = np.empty(len(values), dtype=object)
    ids[:] = values
    if not allow_empty and ids.size == 0:
        raise ValueError(f"{name} must contain at least one item")

    seen: dict[Hashable, int] = {}
    for position, item_id in enumerate(ids.tolist()):
        try:
            hash(item_id)
        except TypeError as error:
            raise TypeError(
                f"{name} entry at position {position} is not hashable"
            ) from error
        try:
            missing = bool(pd.isna(item_id))
        except (TypeError, ValueError):
            missing = False
        if item_id is None or missing:
            raise ValueError(f"{name} must not contain missing IDs")
        if require_unique and item_id in seen:
            raise ValueError(
                f"{name} must not contain duplicate IDs: {item_id!r}"
            )
        seen[item_id] = position
    return ids


def canonical_item_ids(
    item_ids: Sequence[Hashable] | np.ndarray,
    *,
    expected_rows: int | None = None,
    expected_rows_name: str = "item_features",
    expected_rows_unit: str = "rows",
    name: str = "item_ids",
) -> np.ndarray:
    """Validate one complete, unique item catalog."""
    ids = _item_id_values(
        item_ids,
        name=name,
        allow_empty=False,
        require_unique=True,
    )
    if expected_rows is not None and ids.size != expected_rows:
        raise ValueError(
            f"{name} has {ids.size} entries, but {expected_rows_name} has "
            f"{expected_rows} {expected_rows_unit}"
        )
    ids.setflags(write=False)
    return ids


@dataclass(frozen=True)
class ItemVocabulary:
    """Immutable mapping between stable item IDs and catalog rows."""

    item_ids: np.ndarray
    id_to_row: Mapping[Hashable, int]

    @classmethod
    def from_ids(
        cls,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        name: str = "item_ids",
    ) -> "ItemVocabulary":
        ids = canonical_item_ids(item_ids, name=name)
        frozen_ids = ids.copy()
        frozen_ids.setflags(write=False)
        return cls(
            item_ids=frozen_ids,
            id_to_row=MappingProxyType(
                {item_id: row for row, item_id in enumerate(frozen_ids.tolist())}
            ),
        )

    @classmethod
    def positional(cls, n_items: int) -> "ItemVocabulary":
        """Build the default integer identity for an unnamed catalog."""
        if int(n_items) < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        return cls.from_ids(np.arange(int(n_items), dtype=np.int64))

    @property
    def n_items(self) -> int:
        """Number of item IDs in the vocabulary."""
        return int(self.item_ids.size)

    def rows_for(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        name: str = "item_ids",
    ) -> np.ndarray:
        """Resolve IDs to rows, preserving duplicates and request order."""
        ids = _item_id_values(
            item_ids,
            name=name,
            allow_empty=True,
            require_unique=False,
        )
        rows = np.empty(ids.size, dtype=np.int64)
        for position, item_id in enumerate(ids.tolist()):
            try:
                rows[position] = self.id_to_row[item_id]
            except KeyError as error:
                raise ValueError(
                    f"{name} contains unknown item ID {item_id!r}"
                ) from error
        return rows

    def align_csr(
        self,
        matrix: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        name: str = "source",
    ) -> csr_matrix:
        """Select and reorder sparse columns to match this vocabulary."""
        if not isspmatrix_csr(matrix):
            raise TypeError(f"{name} must be a scipy.sparse.csr_matrix")
        if item_ids is self.item_ids and matrix.shape[1] == self.n_items:
            return matrix
        external_ids = canonical_item_ids(
            item_ids,
            expected_rows=matrix.shape[1],
            expected_rows_name=name,
            expected_rows_unit="columns",
        )
        if np.array_equal(external_ids, self.item_ids):
            return matrix
        external_to_column = {
            item_id: column for column, item_id in enumerate(external_ids.tolist())
        }
        missing = [
            item_id
            for item_id in self.item_ids.tolist()
            if item_id not in external_to_column
        ]
        if missing:
            raise ValueError(
                f"{name} item_ids is missing fitted source item ID: {missing[0]!r}"
            )
        columns = np.fromiter(
            (external_to_column[item_id] for item_id in self.item_ids.tolist()),
            dtype=np.int64,
            count=self.n_items,
        )
        return matrix[:, columns].tocsr()


@dataclass(frozen=True)
class Recommendations:
    """Batch of ranked stable item IDs and their scores."""

    item_ids: np.ndarray
    scores: np.ndarray
    valid_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        item_ids = np.array(self.item_ids, dtype=object, copy=True)
        scores = np.array(self.scores, copy=True)
        if item_ids.ndim != 2 or scores.ndim != 2:
            raise ValueError("recommendation item_ids and scores must be 2D")
        if item_ids.shape != scores.shape:
            raise ValueError(
                "recommendation item_ids and scores must have the same shape"
            )
        if not np.issubdtype(scores.dtype, np.number) or np.iscomplexobj(scores):
            raise TypeError("recommendation scores must be real numeric values")
        valid_mask = (
            np.ones(item_ids.shape, dtype=bool)
            if self.valid_mask is None
            else np.array(self.valid_mask, dtype=bool, copy=True)
        )
        if valid_mask.shape != item_ids.shape:
            raise ValueError(
                "recommendation valid_mask must have the same shape as item_ids"
            )
        item_ids.setflags(write=False)
        scores.setflags(write=False)
        valid_mask.setflags(write=False)
        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "valid_mask", valid_mask)

    @property
    def valid_counts(self) -> np.ndarray:
        """Number of real recommendations in each batch row."""
        counts = np.asarray(self.valid_mask.sum(axis=1), dtype=np.int64)
        counts.setflags(write=False)
        return counts

    def to_dicts(self) -> list[dict[Hashable, float]]:
        """Return one rank-ordered mapping per row, omitting padded positions."""
        return [
            {
                item_id: float(score)
                for item_id, score, valid in zip(
                    ids.tolist(),
                    values.tolist(),
                    mask.tolist(),
                )
                if valid
            }
            for ids, values, mask in zip(
                self.item_ids,
                self.scores,
                self.valid_mask,
            )
        ]
