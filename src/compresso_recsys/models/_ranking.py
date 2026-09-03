from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor


def validate_candidate_topk(
    source: csr_matrix,
    candidate_rows: np.ndarray,
    *,
    k: int,
    exclude_seen: bool,
) -> None:
    candidate_count = int(candidate_rows.size)
    if not 1 <= int(k) <= candidate_count:
        raise ValueError(f"k must be in [1, {candidate_count}], got {k}")
    if not exclude_seen:
        return

    selected = np.zeros(source.shape[1], dtype=bool)
    selected[candidate_rows] = True
    seen_counts = np.diff(source.indptr)
    seen_rows = np.repeat(
        np.arange(source.shape[0], dtype=np.int64),
        seen_counts,
    )
    selected_seen = selected[source.indices]
    selected_seen_counts = np.bincount(
        seen_rows[selected_seen],
        minlength=source.shape[0],
    )
    available_counts = candidate_count - selected_seen_counts
    if available_counts.size and np.any(available_counts < int(k)):
        row = int(np.flatnonzero(available_counts < int(k))[0])
        raise ValueError(
            f"source row {row} has only {available_counts[row]} unseen items, "
            f"fewer than k={k}"
        )


def mask_seen_numpy(
    scores: np.ndarray,
    source: csr_matrix,
    candidate_rows: np.ndarray,
) -> None:
    candidate_to_local = np.full(source.shape[1], -1, dtype=np.int64)
    candidate_to_local[candidate_rows] = np.arange(candidate_rows.size)
    seen_counts = np.diff(source.indptr)
    seen_rows = np.repeat(
        np.arange(source.shape[0], dtype=np.int64),
        seen_counts,
    )
    seen_local = candidate_to_local[source.indices]
    in_selection = seen_local >= 0
    scores[seen_rows[in_selection], seen_local[in_selection]] = -np.inf


def rank_numpy_scores(
    scores: np.ndarray,
    *,
    candidate_rows: np.ndarray,
    shape: tuple[int, int],
    k: int,
) -> SRPTensor:
    scores = np.asarray(scores)
    if scores.shape != (shape[0], candidate_rows.size):
        raise ValueError(
            "scores must have shape "
            f"{(shape[0], candidate_rows.size)}, got {scores.shape}"
        )
    if shape[0] == 0:
        value_dtype = torch.from_numpy(np.empty(0, dtype=scores.dtype)).dtype
        return SRPTensor(
            cols=torch.empty((0, k), dtype=torch.long),
            vals=torch.empty((0, k), dtype=value_dtype),
            shape=shape,
        )
    local = SRPTensor.from_dense(
        torch.from_numpy(scores),
        k=int(k),
        score_mode="raw",
    )
    global_rows = torch.from_numpy(candidate_rows).to(local.cols.device)
    return SRPTensor(
        cols=global_rows[local.cols],
        vals=local.vals,
        shape=shape,
    )
