from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from compresso import SRPTensor

__all__ = [
    "CalibratedRecall",
    "NDCG",
    "RankingBatch",
    "RankingMetric",
]


def _normalize_cutoffs(cutoffs: int | Sequence[int]) -> tuple[int, ...]:
    values = (int(cutoffs),) if isinstance(cutoffs, int) else tuple(int(k) for k in cutoffs)
    if not values or any(k < 1 for k in values):
        raise ValueError("cutoffs must contain at least one positive integer")
    return tuple(sorted(set(values)))


@dataclass(frozen=True)
class RankingBatch:
    """Shared vectorized inputs for ranking metrics.

    ``hits[row, rank]`` records whether the item at that prediction rank is
    relevant. ``target_counts`` contains the number of relevant items per row.
    Metric implementations can therefore stay independent of CSR matching.
    """

    predictions: SRPTensor
    hits: torch.Tensor
    target_counts: torch.Tensor

    def __post_init__(self) -> None:
        if self.hits.dtype != torch.bool or self.hits.ndim != 2:
            raise ValueError("hits must be a 2D boolean tensor")
        if self.target_counts.dtype != torch.long or self.target_counts.ndim != 1:
            raise ValueError("target_counts must be a 1D torch.long tensor")
        if self.hits.shape[0] != self.predictions.rows:
            raise ValueError("hits rows must match prediction rows")
        if self.target_counts.shape[0] != self.predictions.rows:
            raise ValueError("target_counts rows must match prediction rows")
        if self.hits.shape[1] > self.predictions.k:
            raise ValueError("hits cannot contain more ranks than predictions")
        if (
            self.hits.device != self.predictions.device
            or self.target_counts.device != self.predictions.device
        ):
            raise ValueError("ranking batch tensors must be on the prediction device")


class RankingMetric(ABC):
    """Abstract streaming metric over ranked recommendation batches."""

    @property
    @abstractmethod
    def required_k(self) -> int:
        """Largest recommendation rank needed by this metric."""

    @property
    @abstractmethod
    def result_keys(self) -> tuple[str, ...]:
        """Metric keys returned by :meth:`compute`."""

    @abstractmethod
    def reset(self) -> None:
        """Clear accumulated metric state."""

    @abstractmethod
    def update(self, batch: RankingBatch) -> None:
        """Accumulate one ranking batch."""

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return aggregated metric values."""


class _MeanAtCutoffsMetric(RankingMetric):
    result_prefix: str

    def __init__(self, cutoffs: int | Sequence[int]) -> None:
        self.cutoffs = _normalize_cutoffs(cutoffs)
        self.reset()

    @property
    def required_k(self) -> int:
        return self.cutoffs[-1]

    @property
    def result_keys(self) -> tuple[str, ...]:
        return tuple(f"{self.result_prefix}@{k}" for k in self.cutoffs)

    def reset(self) -> None:
        self._sums = torch.zeros(len(self.cutoffs), dtype=torch.float64)
        self._count = 0

    def update(self, batch: RankingBatch) -> None:
        if batch.hits.shape[1] < self.required_k:
            raise ValueError(
                f"{type(self).__name__} requires predictions through rank {self.required_k}, "
                f"got {batch.hits.shape[1]}"
            )
        valid = batch.target_counts > 0
        if not bool(valid.any()):
            return
        values = self._batch_values(batch)
        self._sums += values[valid].detach().to(device="cpu", dtype=torch.float64).sum(dim=0)
        self._count += int(valid.sum().item())

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {key: 0.0 for key in self.result_keys}
        means = self._sums / self._count
        return {key: float(value) for key, value in zip(self.result_keys, means.tolist())}

    @abstractmethod
    def _batch_values(self, batch: RankingBatch) -> torch.Tensor:
        """Return per-row values with shape ``(rows, len(cutoffs))``."""


class CalibratedRecall(_MeanAtCutoffsMetric):
    """Recall normalized by ``min(k, number of relevant targets)``."""

    result_prefix = "recall"

    def _batch_values(self, batch: RankingBatch) -> torch.Tensor:
        cutoff_indices = torch.tensor(
            [k - 1 for k in self.cutoffs],
            dtype=torch.long,
            device=batch.hits.device,
        )
        cumulative_hits = batch.hits[:, : self.required_k].to(torch.float32).cumsum(dim=1)
        hits_at_k = cumulative_hits.index_select(dim=1, index=cutoff_indices)
        cutoffs = torch.tensor(self.cutoffs, dtype=torch.long, device=batch.hits.device)
        denominators = torch.minimum(batch.target_counts[:, None], cutoffs[None, :]).clamp_min(1)
        return hits_at_k / denominators


class NDCG(_MeanAtCutoffsMetric):
    """Binary-relevance normalized discounted cumulative gain."""

    result_prefix = "ndcg"

    def _batch_values(self, batch: RankingBatch) -> torch.Tensor:
        ranks = torch.arange(
            2,
            self.required_k + 2,
            dtype=torch.float32,
            device=batch.hits.device,
        )
        discounts = torch.reciprocal(torch.log2(ranks))
        discounted_hits = batch.hits[:, : self.required_k].to(torch.float32) * discounts
        dcg_curve = discounted_hits.cumsum(dim=1)
        cutoff_indices = torch.tensor(
            [k - 1 for k in self.cutoffs],
            dtype=torch.long,
            device=batch.hits.device,
        )
        dcg = dcg_curve.index_select(dim=1, index=cutoff_indices)

        cutoffs = torch.tensor(self.cutoffs, dtype=torch.long, device=batch.hits.device)
        ideal_lengths = torch.minimum(batch.target_counts[:, None], cutoffs[None, :])
        ideal_curve = discounts.cumsum(dim=0)
        idcg = ideal_curve[(ideal_lengths - 1).clamp_min(0)]
        return torch.where(ideal_lengths > 0, dcg / idcg, torch.zeros_like(dcg))
