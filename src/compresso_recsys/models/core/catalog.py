"""The candidate catalog: what a cold-start model is allowed to recommend.

:class:`CandidateCatalog` is fixed at fit time -- a snapshot of the items, the
feature space and the identifiers a model was trained against. The growable
counterpart is :class:`~compresso_recsys.models.core.mutable_catalog.MutableCandidateCatalog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Hashable,
    Literal,
    Mapping,
    Sequence,
)

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models.core.identifiers import (
    ItemVocabulary,
    canonical_item_ids,
)
from compresso_recsys.models.core.features import (
    _freeze_features,
)


__all__ = [
    "CandidateCatalog",
    "CandidateConflict",
    "CandidateSelection",
]

CandidateConflict = Literal["error", "replace", "ignore"]


_NOT_INSTALLED = (
    "no candidate catalog is installed: the model has not been fitted, or "
    "install() was never called on the catalog"
)


@dataclass(frozen=True, init=False)
class CandidateCatalog:
    """Immutable snapshot of a feature-based candidate catalog."""

    item_ids: np.ndarray
    item_features: csr_matrix | np.ndarray
    _metadata: pd.DataFrame | None = field(repr=False)
    feature_space_id: str | None
    version: int
    id_to_row: Mapping[Hashable, int]

    def __init__(
        self,
        *,
        item_ids: np.ndarray,
        item_features: csr_matrix | np.ndarray,
        metadata: pd.DataFrame | None,
        feature_space_id: str | None,
        version: int,
        id_to_row: Mapping[Hashable, int],
    ) -> None:
        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "item_features", item_features)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "feature_space_id", feature_space_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "id_to_row", id_to_row)

    @property
    def n_items(self) -> int:
        """Number of candidates in this snapshot."""
        return int(self.item_ids.size)

    @property
    def metadata(self) -> pd.DataFrame | None:
        """Return a defensive copy of metadata aligned with candidate rows."""
        return None if self._metadata is None else self._metadata.copy(deep=True)

    def rows_for(self, item_ids: Sequence[Hashable]) -> np.ndarray:
        """Resolve stable item IDs to candidate rows in request order."""
        ids = canonical_item_ids(item_ids, name="candidate_ids")
        rows = np.empty(ids.size, dtype=np.int64)
        for position, item_id in enumerate(ids.tolist()):
            try:
                rows[position] = self.id_to_row[item_id]
            except KeyError as error:
                raise KeyError(f"unknown candidate item ID: {item_id!r}") from error
        return rows

    def ids_for(self, rows: np.ndarray | torch.Tensor) -> np.ndarray:
        """Resolve candidate row indices to stable item IDs."""
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().numpy()
        row_array = np.asarray(rows)
        if not np.issubdtype(row_array.dtype, np.integer):
            raise TypeError("candidate rows must contain integers")
        if row_array.size and (
            int(row_array.min()) < 0 or int(row_array.max()) >= self.n_items
        ):
            raise IndexError("candidate row is out of bounds")
        return self.item_ids[row_array]


@dataclass(frozen=True)
class CandidateSelection:
    catalog: CandidateCatalog
    rows: np.ndarray
    features: csr_matrix | np.ndarray
    source_to_candidate: np.ndarray
    candidate_to_local: np.ndarray


def _make_catalog(
    *,
    item_ids: np.ndarray,
    item_features: csr_matrix | np.ndarray,
    metadata: pd.DataFrame | None,
    feature_space_id: str | None,
    version: int,
) -> CandidateCatalog:
    vocabulary = ItemVocabulary.from_ids(item_ids)
    return CandidateCatalog(
        item_ids=vocabulary.item_ids,
        item_features=_freeze_features(item_features),
        metadata=None if metadata is None else metadata.copy(deep=True),
        feature_space_id=feature_space_id,
        version=int(version),
        id_to_row=vocabulary.id_to_row,
    )
