"""The evaluator itself, and the published ways to call it."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any, overload

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys._reporting import _Reporter, _format_duration
from compresso_recsys.metrics import CalibratedRecall, NDCG, RankingBatch, RankingMetric
from compresso_recsys.models import Recommender, SequentialRecommender
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.evaluation.arrays import (
    _as_row_batches,
    _canonical_csr,
    _canonical_sample_ids,
    _slice_srp_rows,
)
from compresso_recsys.evaluation.matching import MatchBackend, _match_predictions
from compresso_recsys.evaluation.results import EvaluationResult, _TargetFingerprint


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
        collect_per_user: bool = True,
        metadata: Mapping[str, Any] | None = None,
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
        self.collect_per_user = bool(collect_per_user)
        self.metadata = dict(metadata) if metadata is not None else {}
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
        self._n_scored_rows = 0
        self._rows_seen = 0
        self._debug_rows: list[dict[str, Any]] = []
        self._value_chunks: dict[str, list[np.ndarray]] = {}
        self._id_chunks: list[np.ndarray] = []
        self._fingerprint = _TargetFingerprint()

    def _validate_metric_values(
        self,
        metric: RankingMetric,
        values: Any,
        *,
        rows: int,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Check the per-row tensor a metric returned under the update contract."""
        name = type(metric).__name__
        if not isinstance(values, torch.Tensor):
            raise TypeError(
                f"{name}.update must return a torch.Tensor of per-row values when "
                f"collect_per_user is enabled, got {type(values).__name__}"
            )
        if values.ndim != 2:
            raise ValueError(f"{name}.update must return a 2D tensor, got {values.ndim}D")
        expected = len(metric.result_keys)
        if values.shape != (rows, expected):
            raise ValueError(
                f"{name}.update must return shape ({rows}, {expected}), "
                f"got {tuple(values.shape)}"
            )
        if not values.dtype.is_floating_point:
            raise ValueError(f"{name}.update must return a floating-point tensor")
        if bool(valid.any()) and not bool(torch.isfinite(values[valid]).all()):
            raise ValueError(f"{name}.update returned non-finite values for evaluable rows")
        return values

    def _collect(
        self,
        metric: RankingMetric,
        values: torch.Tensor,
        valid: torch.Tensor,
    ) -> None:
        """Retain the evaluable rows of one metric's per-row values."""
        # Host first, then cast: see the note in _MeanAtCutoffsMetric.update.
        kept = values[valid].detach().cpu().to(torch.float32)
        array = kept.numpy()
        for column, key in enumerate(metric.result_keys):
            self._value_chunks.setdefault(key, []).append(
                np.ascontiguousarray(array[:, column])
            )

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

    def update(
        self,
        predictions: SRPTensor,
        targets: csr_matrix,
        *,
        sample_ids: Sequence[Any] | np.ndarray | None = None,
    ) -> None:
        targets = _canonical_csr(targets)
        self._fingerprint.update(targets)
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
        rows = predictions.rows
        valid = target_counts > 0

        if sample_ids is None:
            batch_ids = np.arange(self._rows_seen, self._rows_seen + rows)
        else:
            batch_ids = np.asarray(sample_ids)
            if batch_ids.ndim != 1:
                raise ValueError("sample_ids must be one-dimensional")
            if batch_ids.shape[0] != rows:
                raise ValueError(
                    f"sample_ids has {batch_ids.shape[0]} values, "
                    f"expected one per prediction row ({rows})"
                )

        for metric in self.metrics:
            values = metric.update(batch)
            if self.collect_per_user:
                values = self._validate_metric_values(
                    metric, values, rows=rows, valid=valid
                )
                self._collect(metric, values, valid)

        if self.collect_per_user:
            self._id_chunks.append(batch_ids[valid.detach().cpu().numpy()])

        self._collect_debug(batch)
        self._n_scored_rows += int(valid.sum().item())
        self._rows_seen += rows

    def compute(self) -> EvaluationResult:
        metrics: dict[str, float] = {}
        for metric in self.metrics:
            metrics.update(metric.compute())

        per_user: dict[str, np.ndarray] | None = None
        sample_ids: np.ndarray | None = None
        if self.collect_per_user:
            per_user = {
                key: (
                    np.concatenate(self._value_chunks[key])
                    if self._value_chunks.get(key)
                    else np.empty(0, dtype=np.float32)
                )
                for key in metrics
            }
            sample_ids = (
                np.concatenate(self._id_chunks)
                if self._id_chunks
                else np.empty(0, dtype=np.int64)
            )

        return EvaluationResult(
            metrics=metrics,
            per_user=per_user,
            sample_ids=sample_ids,
            n_rows=self._rows_seen,
            n_scored_rows=self._n_scored_rows,
            required_k=self.required_k,
            metadata=dict(self.metadata),
            debug_rows=tuple(self._debug_rows) if self.debug else None,
            target_fingerprint=self._fingerprint.digest(),
        )


def evaluate_ranked_predictions(
    *,
    predictions: SRPTensor,
    targets: csr_matrix,
    metrics: Sequence[RankingMetric] | None = None,
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    collect_per_user: bool = True,
    metadata: Mapping[str, Any] | None = None,
    batch_size: int = 4096,
    match_backend: MatchBackend = "auto",
    max_dense_cells: int = 20_000_000,
    validate_predictions: bool = True,
    debug: bool = False,
    debug_users: int = 5,
) -> EvaluationResult:
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
        collect_per_user=collect_per_user,
        metadata=metadata,
        debug=debug,
        debug_users=debug_users,
    )
    resolved_ids = _canonical_sample_ids(sample_ids, n_rows=predictions.rows)
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
            sample_ids=None if resolved_ids is None else resolved_ids[start:end],
        )
    return evaluator.compute()


@overload
def evaluate_recommender(
    model: Recommender,
    *,
    source: csr_matrix,
    targets: csr_matrix,
    metrics: Sequence[RankingMetric],
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    collect_per_user: bool = True,
    metadata: Mapping[str, Any] | None = None,
    batch_size: int = 1024,
    match_backend: MatchBackend = "auto",
    max_dense_cells: int = 20_000_000,
    validate_predictions: bool = True,
    debug: bool = False,
    debug_users: int = 5,
    show_progress: bool = False,
    logger: Any | None = None,
    log_every_n_steps: int = 1000,
) -> EvaluationResult: ...


@overload
def evaluate_recommender(
    model: SequentialRecommender,
    *,
    source: ItemSequences,
    targets: csr_matrix,
    metrics: Sequence[RankingMetric],
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    collect_per_user: bool = True,
    metadata: Mapping[str, Any] | None = None,
    batch_size: int = 1024,
    match_backend: MatchBackend = "auto",
    max_dense_cells: int = 20_000_000,
    validate_predictions: bool = True,
    debug: bool = False,
    debug_users: int = 5,
    show_progress: bool = False,
    logger: Any | None = None,
    log_every_n_steps: int = 1000,
) -> EvaluationResult: ...


def evaluate_recommender(
    model: Recommender | SequentialRecommender,
    *,
    source: csr_matrix | ItemSequences,
    targets: csr_matrix,
    metrics: Sequence[RankingMetric],
    sample_ids: Sequence[Any] | np.ndarray | None = None,
    collect_per_user: bool = True,
    metadata: Mapping[str, Any] | None = None,
    batch_size: int = 1024,
    match_backend: MatchBackend = "auto",
    max_dense_cells: int = 20_000_000,
    validate_predictions: bool = True,
    debug: bool = False,
    debug_users: int = 5,
    show_progress: bool = False,
    logger: Any | None = None,
    log_every_n_steps: int = 1000,
) -> EvaluationResult:
    """Evaluate a recommender without retaining predictions between batches.

    The largest metric cutoff determines the ``k`` passed to the model's
    ``predict_on_batch`` method. Source and target rows are sliced together,
    and each prediction batch is immediately sent to :class:`RankingEvaluator`.
    Source and target column counts may differ: source columns describe the
    model's history vocabulary, while target columns describe its candidates.

    ``source`` may be a ``csr_matrix`` of interactions or an
    :class:`~compresso_recsys.sequences.ItemSequences` of chronological
    histories, matching whichever the model reads. Only the row count has to
    agree with ``targets``; nothing downstream of ``predict_on_batch`` knows or
    cares which was given, which is why sequential and matrix models can be
    compared against each other with no statistics-side changes.
    """
    if not isinstance(model, (Recommender, SequentialRecommender)):
        raise TypeError("model must implement predict_on_batch(source, *, k)")
    batches = _as_row_batches(source)
    targets = _canonical_csr(targets)
    if batches.n_rows != targets.shape[0]:
        raise ValueError(
            f"source rows ({batches.n_rows}) must match target rows "
            f"({targets.shape[0]})"
        )
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    evaluator = RankingEvaluator(
        metrics,
        match_backend=match_backend,
        max_dense_cells=max_dense_cells,
        validate_predictions=validate_predictions,
        collect_per_user=collect_per_user,
        metadata=metadata,
        debug=debug,
        debug_users=debug_users,
    )
    resolved_ids = _canonical_sample_ids(sample_ids, n_rows=batches.n_rows)
    starts = range(0, batches.n_rows, batch_size)
    reporter = _Reporter(
        logger,
        show_progress,
        "evaluation",
        log_every_n_steps,
    )
    steps = len(starts)
    started = time.monotonic()
    reporter.log(
        f"evaluate recommender@{evaluator.required_k} started: "
        f"{batches.n_rows} rows | {steps} batches of {batch_size}"
    )
    for step, start in enumerate(
        reporter.wrap(
            starts,
            total=steps,
            desc=f"evaluate recommender@{evaluator.required_k}",
        ),
        start=1,
    ):
        end = min(start + batch_size, batches.n_rows)
        predictions = model.predict_on_batch(
            batches.take_rows(start, end),
            k=evaluator.required_k,
        )
        evaluator.update(
            predictions,
            targets[start:end],
            sample_ids=None if resolved_ids is None else resolved_ids[start:end],
        )
        log_steps = reporter.log_every_n_steps
        if log_steps and step % log_steps == 0:
            reporter.step(
                f"evaluate recommender@{evaluator.required_k} step {step}/{steps}",
                step,
                steps,
                started,
            )
    result = evaluator.compute()
    reporter.log(
        f"evaluate recommender@{evaluator.required_k} finished: "
        f"{_format_duration(time.monotonic() - started)} total | "
        f"{batches.n_rows} rows"
    )
    return result
