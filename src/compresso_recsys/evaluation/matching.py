"""Deciding which predicted items were hits, and how to look that up.

A dense boolean take is fastest when the candidate space is small; a sorted
search wins once it is not. :data:`MatchBackend` selects between them.
"""

from __future__ import annotations

from typing import Literal

import torch
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation.arrays import _target_tensors


MatchBackend = Literal["auto", "dense", "searchsorted"]


def _match_dense(
    prediction_columns: torch.Tensor,
    targets: csr_matrix,
    *,
    n_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, target_columns, counts = _target_tensors(targets, device=prediction_columns.device)
    target_mask = torch.zeros(
        (targets.shape[0], n_items),
        dtype=torch.bool,
        device=prediction_columns.device,
    )
    if target_columns.numel() > 0:
        target_mask[rows, target_columns] = True
    return target_mask.gather(dim=1, index=prediction_columns), counts


def _match_searchsorted(
    prediction_columns: torch.Tensor,
    targets: csr_matrix,
    *,
    n_items: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, target_columns, counts = _target_tensors(targets, device=prediction_columns.device)
    if target_columns.numel() == 0:
        return torch.zeros_like(prediction_columns, dtype=torch.bool), counts

    target_keys = rows * int(n_items) + target_columns
    prediction_rows = torch.arange(
        targets.shape[0],
        dtype=torch.long,
        device=prediction_columns.device,
    )[:, None]
    prediction_keys = (prediction_rows * int(n_items) + prediction_columns).reshape(-1)
    positions = torch.searchsorted(target_keys, prediction_keys)
    valid_positions = positions < target_keys.numel()
    safe_positions = positions.clamp_max(target_keys.numel() - 1)
    matches = valid_positions & (target_keys[safe_positions] == prediction_keys)
    return matches.reshape_as(prediction_columns), counts


def _match_predictions(
    prediction_columns: torch.Tensor,
    targets: csr_matrix,
    *,
    n_items: int,
    backend: MatchBackend,
    max_dense_cells: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    selected_backend = backend
    if backend == "auto":
        selected_backend = (
            "dense"
            if targets.shape[0] * n_items <= max_dense_cells
            else "searchsorted"
        )
    if selected_backend == "dense":
        return _match_dense(prediction_columns, targets, n_items=n_items)
    if selected_backend == "searchsorted":
        return _match_searchsorted(prediction_columns, targets, n_items=n_items)
    raise ValueError(f"unknown match_backend: {backend!r}")
