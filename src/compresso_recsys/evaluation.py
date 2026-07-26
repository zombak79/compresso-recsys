from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr

from compresso import SRPTensor
from compresso_recsys.metrics import CalibratedRecall, NDCG, RankingBatch, RankingMetric

MatchBackend = Literal["auto", "dense", "searchsorted"]

__all__ = [
    "RankingEvaluator",
    "evaluate_ranked_predictions",
]


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


class RankingEvaluator:
    """Stream ranked SRP predictions against variable-length CSR targets.

    The evaluator matches each prediction batch to its target CSR rows once
    and sends the resulting :class:`~compresso_recsys.metrics.RankingBatch` to
    every metric. ``auto`` matching uses a dense boolean target mask for small
    batches and composite-key ``torch.searchsorted`` matching for larger item
    spaces.
    """

    def __init__(
        self,
        metrics: Sequence[RankingMetric],
        *,
        match_backend: MatchBackend = "auto",
        max_dense_cells: int = 20_000_000,
        validate_predictions: bool = True,
        debug: bool = False,
        debug_users: int = 5,
    ) -> None:
        if not metrics:
            raise ValueError("metrics must contain at least one RankingMetric")
        if match_backend not in {"auto", "dense", "searchsorted"}:
            raise ValueError(f"unknown match_backend: {match_backend!r}")
        if max_dense_cells < 1:
            raise ValueError("max_dense_cells must be >= 1")
        if debug_users < 0:
            raise ValueError("debug_users must be >= 0")

        self.metrics = list(metrics)
        self.match_backend = match_backend
        self.max_dense_cells = int(max_dense_cells)
        self.validate_predictions = bool(validate_predictions)
        self.debug = bool(debug)
        self.debug_users = int(debug_users)
        self.required_k = max(metric.required_k for metric in self.metrics)

        keys = [key for metric in self.metrics for key in metric.result_keys]
        if len(keys) != len(set(keys)):
            raise ValueError("metrics must produce unique result keys")
        self.reset()

    def reset(self) -> None:
        for metric in self.metrics:
            metric.reset()
        self._n_eval_users = 0
        self._rows_seen = 0
        self._debug_rows: list[dict[str, Any]] = []

    def _validate(self, predictions: SRPTensor, targets: csr_matrix) -> None:
        if predictions.rows != targets.shape[0]:
            raise ValueError(
                f"prediction rows ({predictions.rows}) must match target rows ({targets.shape[0]})"
            )
        if predictions.cols_total != targets.shape[1]:
            raise ValueError(
                f"prediction items ({predictions.cols_total}) must match target items ({targets.shape[1]})"
            )
        if predictions.k < self.required_k:
            raise ValueError(
                f"predictions contain top-{predictions.k}, but metrics require top-{self.required_k}"
            )
        if predictions.cols.device != predictions.vals.device:
            raise ValueError("prediction columns and scores must be on the same device")
        ranked_columns = predictions.cols[:, : self.required_k]
        if ranked_columns.numel() == 0:
            return
        if (
            int(ranked_columns.min().item()) < 0
            or int(ranked_columns.max().item()) >= predictions.cols_total
        ):
            raise ValueError("prediction item indices are out of bounds")
        sorted_columns = ranked_columns.sort(dim=1).values
        if self.required_k > 1 and bool((sorted_columns[:, 1:] == sorted_columns[:, :-1]).any()):
            raise ValueError("predictions must not contain duplicate items within a row")
        ranked_values = predictions.vals[:, : self.required_k]
        if bool(torch.isnan(ranked_values).any()):
            raise ValueError("prediction scores must not contain NaN")
        if self.required_k > 1 and bool((ranked_values[:, 1:] > ranked_values[:, :-1]).any()):
            raise ValueError("prediction scores must be ordered from highest to lowest")

    def _collect_debug(self, batch: RankingBatch) -> None:
        if not self.debug or self._rows_seen >= self.debug_users:
            return
        local_limit = min(batch.predictions.rows, self.debug_users - self._rows_seen)
        hits = batch.hits[:local_limit, : self.required_k].detach().cpu()
        target_counts = batch.target_counts[:local_limit].detach().cpu()
        discounts = torch.reciprocal(
            torch.log2(torch.arange(2, self.required_k + 2, dtype=torch.float64))
        )
        ideal_curve = discounts.cumsum(dim=0)

        for row in range(local_limit):
            n_true = int(target_counts[row].item())
            if n_true == 0:
                continue
            hit_ranks = (torch.nonzero(hits[row], as_tuple=False).flatten() + 1).tolist()
            dcg = float(discounts[hits[row]].sum().item())
            ideal_len = min(self.required_k, n_true)
            idcg = float(ideal_curve[ideal_len - 1].item())
            self._debug_rows.append(
                {
                    "user_row": self._rows_seen + row,
                    "n_true": n_true,
                    "n_hits_topk": len(hit_ranks),
                    "first_hit_rank": hit_ranks[0] if hit_ranks else None,
                    "hit_ranks": hit_ranks,
                    "dcg": dcg,
                    "idcg": idcg,
                    "ndcg": dcg / idcg if idcg > 0 else 0.0,
                }
            )

    def update(self, predictions: SRPTensor, targets: csr_matrix) -> None:
        targets = _canonical_csr(targets)
        if self.validate_predictions:
            self._validate(predictions, targets)
        else:
            if predictions.rows != targets.shape[0] or predictions.cols_total != targets.shape[1]:
                raise ValueError("prediction and target shapes must match")
            if predictions.k < self.required_k:
                raise ValueError(
                    f"predictions contain top-{predictions.k}, but metrics require top-{self.required_k}"
                )

        ranked_columns = predictions.cols[:, : self.required_k]
        hits, target_counts = _match_predictions(
            ranked_columns,
            targets,
            n_items=predictions.cols_total,
            backend=self.match_backend,
            max_dense_cells=self.max_dense_cells,
        )
        batch = RankingBatch(
            predictions=predictions,
            hits=hits,
            target_counts=target_counts,
        )
        for metric in self.metrics:
            metric.update(batch)
        self._collect_debug(batch)
        self._n_eval_users += int((target_counts > 0).sum().item())
        self._rows_seen += predictions.rows

    def compute(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for metric in self.metrics:
            out.update(metric.compute())
        out["n_eval_users"] = float(self._n_eval_users)
        if self.debug:
            out["debug"] = list(self._debug_rows)
        return out


def evaluate_ranked_predictions(
    *,
    predictions: SRPTensor,
    targets: csr_matrix,
    metrics: Sequence[RankingMetric] | None = None,
    batch_size: int = 4096,
    match_backend: MatchBackend = "auto",
    max_dense_cells: int = 20_000_000,
    validate_predictions: bool = True,
    debug: bool = False,
    debug_users: int = 5,
) -> dict[str, Any]:
    """Evaluate ranked top-k SRP predictions against binary CSR targets.

    Prediction columns must be unique within each row and ordered by
    descending prediction score. Target values are interpreted as binary
    relevance. Rows without nonzero targets are excluded from metric means.

    When ``metrics`` is omitted, calibrated recall and nDCG are calculated at
    the full prediction width. Use :class:`RankingEvaluator` directly when
    predictions are generated one batch at a time.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    targets = _canonical_csr(targets)
    if predictions.rows != targets.shape[0]:
        raise ValueError(
            f"prediction rows ({predictions.rows}) must match target rows ({targets.shape[0]})"
        )
    if predictions.cols_total != targets.shape[1]:
        raise ValueError(
            f"prediction items ({predictions.cols_total}) must match target items ({targets.shape[1]})"
        )
    resolved_metrics = (
        list(metrics)
        if metrics is not None
        else [CalibratedRecall(predictions.k), NDCG(predictions.k)]
    )
    evaluator = RankingEvaluator(
        resolved_metrics,
        match_backend=match_backend,
        max_dense_cells=max_dense_cells,
        validate_predictions=validate_predictions,
        debug=debug,
        debug_users=debug_users,
    )
    if predictions.k < evaluator.required_k:
        raise ValueError(
            f"predictions contain top-{predictions.k}, "
            f"but metrics require top-{evaluator.required_k}"
        )
    if predictions.cols.device != predictions.vals.device:
        raise ValueError("prediction columns and scores must be on the same device")
    for start in range(0, predictions.rows, batch_size):
        end = min(start + batch_size, predictions.rows)
        evaluator.update(
            _slice_srp_rows(predictions, start, end),
            targets[start:end],
        )
    return evaluator.compute()
