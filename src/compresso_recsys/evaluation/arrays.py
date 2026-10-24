"""Turning whatever a caller passes in into the arrays the evaluator needs.

Indices, CSR matrices and SRP tensors all arrive here and leave canonical,
contiguous and owned, so nothing downstream has to re-check a dtype.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr

from compresso import SRPTensor
from compresso_recsys.sequences import ItemSequences


def _owned(array: np.ndarray, source: Any) -> np.ndarray:
    """Return ``array`` guaranteed not to alias ``source``.

    ``np.ascontiguousarray`` and ``np.asarray`` hand back the input untouched
    when it already has the requested layout, so storing the result would alias
    an array the caller still holds. Marking that read-only, as the per-user
    values are, would reach out and freeze the caller's own array.
    """
    return array.copy() if array is source else array


@dataclass(frozen=True)
class _CsrRowBatches:
    """Adapts a CSR source to the row-batching contract sequences already meet.

    Evaluation only ever asks a source two things: how many rows it has, and give
    me rows ``start:stop``. :class:`~compresso_recsys.sequences.ItemSequences`
    answers both natively; ``csr_matrix`` answers them under different names. One
    adapter here keeps the batching loop written once, so a third source type
    means teaching :func:`_as_row_batches` rather than editing the loop.
    """

    matrix: csr_matrix

    @property
    def n_rows(self) -> int:
        return int(self.matrix.shape[0])

    def take_rows(self, start: int, stop: int) -> csr_matrix:
        return self.matrix[start:stop]


def _as_row_batches(source: Any) -> Any:
    """Return ``source`` in a form that can report and slice its own rows."""
    if isinstance(source, ItemSequences):
        return source
    if isspmatrix_csr(source):
        return _CsrRowBatches(source)
    raise TypeError(
        "source must be a scipy.sparse.csr_matrix or an ItemSequences, got "
        f"{type(source).__name__}"
    )


def _canonical_csr(targets: csr_matrix) -> csr_matrix:
    if not isspmatrix_csr(targets):
        raise TypeError("targets must be a scipy.sparse.csr_matrix")
    needs_copy = not targets.has_canonical_format or bool(np.any(targets.data == 0))
    out = targets.copy() if needs_copy else targets
    if needs_copy:
        out.sum_duplicates()
        out.eliminate_zeros()
        out.sort_indices()
    if out.indices.size and (out.indices.min() < 0 or out.indices.max() >= out.shape[1]):
        raise ValueError("target item indices are out of bounds")
    return out


def _indices_to_csr(rows: Sequence[np.ndarray], *, n_items: int) -> csr_matrix:
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64, count=len(rows))
    indptr = np.empty(len(rows) + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(lengths, out=indptr[1:])
    indices = (
        np.concatenate([np.asarray(row, dtype=np.int64) for row in rows])
        if int(indptr[-1]) > 0
        else np.empty(0, dtype=np.int64)
    )
    data = np.ones(len(indices), dtype=np.float32)
    targets = csr_matrix((data, indices, indptr), shape=(len(rows), int(n_items)))
    targets.sum_duplicates()
    targets.eliminate_zeros()
    targets.sort_indices()
    return targets


def _canonical_sample_ids(
    sample_ids: Sequence[Any] | np.ndarray | None,
    *,
    n_rows: int,
) -> np.ndarray | None:
    """Validate caller-supplied identifiers against the input row count."""
    if sample_ids is None:
        return None
    ids = np.asarray(sample_ids)
    if ids.ndim != 1:
        raise ValueError("sample_ids must be one-dimensional")
    if ids.shape[0] != n_rows:
        raise ValueError(
            f"sample_ids has {ids.shape[0]} values, expected one per input row ({n_rows})"
        )
    return ids


def _slice_srp_rows(predictions: SRPTensor, start: int, end: int) -> SRPTensor:
    return SRPTensor(
        cols=predictions.cols[start:end],
        vals=predictions.vals[start:end],
        shape=(end - start, predictions.cols_total),
        validate=False,
    )


def _target_tensors(
    targets: csr_matrix,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    counts = torch.from_numpy(np.diff(targets.indptr).astype(np.int64, copy=False)).to(device)
    rows = torch.repeat_interleave(
        torch.arange(targets.shape[0], dtype=torch.long, device=device),
        counts,
    )
    columns = torch.from_numpy(targets.indices.astype(np.int64, copy=False)).to(device)
    return rows, columns, counts
