from __future__ import annotations

import hashlib
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix, isspmatrix_csr

from compresso import SRPTensor
from compresso_recsys.metrics import CalibratedRecall, NDCG, RankingBatch, RankingMetric
from compresso_recsys.models import Recommender

MatchBackend = Literal["auto", "dense", "searchsorted"]

__all__ = [
    "EvaluationResult",
    "RankingEvaluator",
    "evaluate_ranked_predictions",
    "evaluate_recommender",
]


def _owned(array: np.ndarray, source: Any) -> np.ndarray:
    """Return ``array`` guaranteed not to alias ``source``.

    ``np.ascontiguousarray`` and ``np.asarray`` hand back the input untouched
    when it already has the requested layout, so storing the result would alias
    an array the caller still holds. Marking that read-only, as the per-user
    values are, would reach out and freeze the caller's own array.
    """
    return array.copy() if array is source else array


class _TargetFingerprint:
    """Canonical, batch-size independent digest of the target matrix.

    Two evaluations can only be paired when they scored the same users against
    the same relevant items. Identifiers catch a different user set; this
    catches the same identifiers against different targets, which no identifier
    check can see.

    The evaluator receives targets one batch at a time and never holds the
    whole matrix, so the digest has to be accumulated. Three properties make
    the result independent of how the rows were divided:

    * Row counts and column indices go into two separate streams. Interleaving
      them per batch would put the bytes in a different order for a different
      ``batch_size``, while each stream on its own is simply the same sequence
      of rows however it was chunked, and blake2b digests a stream in pieces
      exactly as it digests it whole.
    * Neither stream contains ``indptr``, whose values are rebased to zero in
      every slice. Row lengths carry the same information and survive slicing.
    * Both are cast to fixed width and byte order, so the same logical matrix
      agrees across platforms and across int32 and int64 index dtypes.

    Values are not hashed. Targets are binary relevance and only nonzero
    locations matter, so two matrices that differ solely in stored values are
    genuinely the same evaluation.

    Canonical form -- sorted, deduplicated, no stored zeros -- is a
    precondition, supplied by :func:`_canonical_csr` before every update.
    """

    __slots__ = ("_row_lengths", "_indices", "_n_items", "_n_rows")

    def __init__(self) -> None:
        self._row_lengths = hashlib.blake2b(digest_size=16)
        self._indices = hashlib.blake2b(digest_size=16)
        self._n_items: int | None = None
        self._n_rows = 0

    def update(self, targets: csr_matrix) -> None:
        """Fold one canonical batch in, in global row order."""
        if self._n_items is None:
            self._n_items = int(targets.shape[1])
        self._row_lengths.update(np.diff(targets.indptr).astype("<u8").tobytes())
        self._indices.update(targets.indices.astype("<i8", copy=False).tobytes())
        self._n_rows += int(targets.shape[0])

    def digest(self) -> str:
        """Hex digest binding both streams to the matrix shape."""
        final = hashlib.blake2b(digest_size=16)
        final.update(int(self._n_items or 0).to_bytes(8, "big"))
        final.update(self._n_rows.to_bytes(8, "big"))
        final.update(self._row_lengths.digest())
        final.update(self._indices.digest())
        return final.hexdigest()


@dataclass(eq=False)
class EvaluationResult(Mapping[str, Any]):
    """Aggregate metrics plus the per-user observations behind them.

    The mapping view carries the aggregates and ``n_eval_users``, so existing
    code that treats an evaluation as a dictionary keeps working::

        result["ndcg@20"]
        dict(result)

    Per-user values, sample identifiers and metadata are attributes rather than
    mapping keys, because they are large and because a caller reaching for them
    is doing something other than reading a headline number.

    ``per_user`` and ``sample_ids`` are what make paired statistical comparison
    possible: two evaluations can only be compared when they refer to the same
    evaluation units in the same order.
    """

    metrics: dict[str, float]
    per_user: dict[str, np.ndarray] | None
    sample_ids: np.ndarray | None
    n_rows: int
    n_eval_users: int
    required_k: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # ``None`` means debug collection was off; an empty tuple means it was on
    # and produced nothing. The mapping exposes ``"debug"`` in the second case
    # but not the first, so a caller that asked for debug always finds the key.
    debug_rows: tuple[dict[str, Any], ...] | None = None
    # Identifies the targets these metrics were computed against, so paired
    # comparison can refuse two results that scored the same users on different
    # relevant items. ``None`` for results built by hand rather than by an
    # evaluator; comparison warns rather than failing in that case.
    target_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for key, value in self.metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"aggregate metric {key!r} is not finite: {value!r}")
        self.n_rows = int(self.n_rows)
        self.n_eval_users = int(self.n_eval_users)
        self.required_k = int(self.required_k)
        if self.n_rows < 0:
            raise ValueError("n_rows must be >= 0")
        if not 0 <= self.n_eval_users <= self.n_rows:
            raise ValueError(
                f"n_eval_users ({self.n_eval_users}) must be in [0, n_rows={self.n_rows}]"
            )
        if self.required_k < 1:
            raise ValueError("required_k must be >= 1")

        if self.per_user is None:
            if self.sample_ids is not None:
                raise ValueError("sample_ids requires per_user values")
            return

        if set(self.per_user) != set(self.metrics):
            missing = sorted(set(self.metrics) - set(self.per_user))
            extra = sorted(set(self.per_user) - set(self.metrics))
            raise ValueError(
                "per_user keys must match metric keys; "
                f"missing={missing}, unexpected={extra}"
            )
        cleaned: dict[str, np.ndarray] = {}
        for key, values in self.per_user.items():
            array = _owned(np.ascontiguousarray(values, dtype=np.float32), values)
            if array.ndim != 1:
                raise ValueError(f"per_user[{key!r}] must be one-dimensional")
            if array.shape[0] != self.n_eval_users:
                raise ValueError(
                    f"per_user[{key!r}] has {array.shape[0]} values, "
                    f"expected n_eval_users={self.n_eval_users}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"per_user[{key!r}] contains non-finite values")
            array.setflags(write=False)
            cleaned[key] = array
        self.per_user = cleaned

        if self.sample_ids is None:
            raise ValueError("per_user values require sample_ids")
        ids = _owned(np.asarray(self.sample_ids), self.sample_ids)
        if ids.ndim != 1:
            raise ValueError("sample_ids must be one-dimensional")
        if ids.shape[0] != self.n_eval_users:
            raise ValueError(
                f"sample_ids has {ids.shape[0]} values, "
                f"expected n_eval_users={self.n_eval_users}"
            )
        # Repeated identifiers are legitimate here: the stacked-fold protocol
        # in :func:`compresso_recsys.retrieval.build_eval_holdout` evaluates
        # each user in several folds, so one user owns several rows. Evaluating
        # them is fine; resampling them as independent units is not, which is a
        # question for paired comparison rather than for this constructor.
        #
        # As with the per-user values above: frozen so a later mutation cannot
        # silently invalidate the pairing this result was matched on.
        ids.setflags(write=False)
        self.sample_ids = ids

    def _mapping_view(self) -> dict[str, Any]:
        view: dict[str, Any] = dict(self.metrics)
        view["n_eval_users"] = self.n_eval_users
        if self.debug_rows is not None:
            view["debug"] = list(self.debug_rows)
        return view

    def __getitem__(self, key: str) -> Any:
        return self._mapping_view()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping_view())

    def __len__(self) -> int:
        return len(self._mapping_view())

    def __repr__(self) -> str:
        collected = "none" if self.per_user is None else f"{len(self.per_user)} keys"
        return (
            f"EvaluationResult(metrics={self.metrics!r}, "
            f"n_eval_users={self.n_eval_users}, n_rows={self.n_rows}, "
            f"required_k={self.required_k}, per_user={collected})"
        )

    @property
    def has_per_user(self) -> bool:
        """Whether per-user observations were collected."""
        return self.per_user is not None

    def to_dict(self, *, include_debug: bool = True) -> dict[str, Any]:
        """Return the mapping view as a plain ``dict``.

        Use this where an actual ``dict`` is required, such as JSON
        serialization. Per-user values are deliberately excluded.
        """
        view = self._mapping_view()
        if not include_debug:
            view.pop("debug", None)
        return view


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


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


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
        self._n_eval_users = 0
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
        self._n_eval_users += int(valid.sum().item())
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
            n_eval_users=self._n_eval_users,
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
) -> EvaluationResult:
    """Evaluate a recommender without retaining predictions between batches.

    The largest metric cutoff determines the ``k`` passed to the model's
    ``predict_on_batch`` method. Source and target rows are sliced together,
    and each prediction batch is immediately sent to :class:`RankingEvaluator`.
    Source and target column counts may differ: source columns describe the
    model's history vocabulary, while target columns describe its candidates.
    """
    if not isinstance(model, Recommender):
        raise TypeError("model must implement predict_on_batch(source, *, k)")
    if not isspmatrix_csr(source):
        raise TypeError("source must be a scipy.sparse.csr_matrix")
    targets = _canonical_csr(targets)
    if source.shape[0] != targets.shape[0]:
        raise ValueError(
            f"source rows ({source.shape[0]}) must match target rows "
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
    resolved_ids = _canonical_sample_ids(sample_ids, n_rows=source.shape[0])
    starts = range(0, source.shape[0], batch_size)
    for start in _progress(
        starts,
        enabled=show_progress,
        desc=f"evaluate recommender@{evaluator.required_k}",
    ):
        end = min(start + batch_size, source.shape[0])
        predictions = model.predict_on_batch(
            source[start:end],
            k=evaluator.required_k,
        )
        evaluator.update(
            predictions,
            targets[start:end],
            sample_ids=None if resolved_ids is None else resolved_ids[start:end],
        )
    return evaluator.compute()
