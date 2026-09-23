from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.sparse import csr_matrix, isspmatrix_csr


def canonical_csr(matrix: csr_matrix, *, name: str) -> csr_matrix:
    """Validate and canonicalize a finite SciPy CSR matrix."""
    if not isspmatrix_csr(matrix):
        raise TypeError(f"{name} must be a scipy.sparse.csr_matrix")
    needs_copy = not matrix.has_canonical_format or bool(np.any(matrix.data == 0))
    out = matrix.copy() if needs_copy else matrix
    if needs_copy:
        out.sum_duplicates()
        out.eliminate_zeros()
        out.sort_indices()
    if not np.all(np.isfinite(out.data)):
        raise ValueError(f"{name} values must be finite")
    return out


def canonical_train_item_indices(
    train_item_indices: np.ndarray | Sequence[int] | None,
    *,
    n_items: int,
) -> np.ndarray:
    """Validate a unique subset of item rows, defaulting to every row."""
    if train_item_indices is None:
        return np.arange(n_items, dtype=np.int64)
    indices = np.asarray(train_item_indices)
    if indices.ndim != 1:
        raise ValueError("train_item_indices must be one-dimensional")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("train_item_indices must contain integers")
    indices = indices.astype(np.int64, copy=False)
    if indices.size < 1:
        raise ValueError("train_item_indices must contain at least one item")
    if np.any(indices < 0) or np.any(indices >= n_items):
        raise ValueError(f"train_item_indices must be in [0, {n_items - 1}]")
    if np.unique(indices).size != indices.size:
        raise ValueError("train_item_indices must not contain duplicates")
    return indices
