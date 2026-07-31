from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class InteractionBatch:
    """Compact interaction batch with source-prefix output candidates."""

    x: torch.Tensor
    sources: torch.Tensor
    candidates: torch.Tensor | None


def normalized_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Mean squared distance between row-normalized predictions and targets."""
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    return (predictions - targets).square().sum(dim=-1).mean()


def dense_training_target(
    x: torch.Tensor,
    *,
    sources: torch.Tensor,
    candidates: torch.Tensor | None,
    input_dim: int,
) -> torch.Tensor:
    """Expand compact source interactions into the selected output space."""
    if candidates is None:
        target = x.new_zeros((x.shape[0], input_dim))
        target[:, sources] = x
        return target
    if candidates.numel() < x.shape[1]:
        raise ValueError("the candidate prefix must contain every source item")
    return torch.nn.functional.pad(x, (0, candidates.numel() - x.shape[1]))


class InteractionBatchSampler:
    """Batched sparse interactions with optional output candidate sampling."""

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
        max_output: int | None,
        seed: int,
    ) -> None:
        self.interactions = interactions
        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.max_output = max_output
        self.user_indices = np.arange(interactions.shape[0], dtype=np.int64)
        self.item_indices = np.arange(interactions.shape[1], dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(self.interactions.shape[0] / self.batch_size)

    def __getitem__(self, batch_index: int) -> InteractionBatch:
        start = int(batch_index) * self.batch_size
        end = min(start + self.batch_size, self.interactions.shape[0])
        if start < 0 or start >= self.interactions.shape[0]:
            raise IndexError(batch_index)

        matrix = self.interactions[self.user_indices[start:end]]
        source_columns = np.flatnonzero(
            np.asarray(matrix.getnnz(axis=0)).ravel()
        ).astype(np.int64, copy=False)
        if self.max_output is None:
            candidate_columns = None
        else:
            negative_mask = np.ones(matrix.shape[1], dtype=bool)
            negative_mask[source_columns] = False
            negative_pool = self.item_indices[negative_mask]
            n_negatives = min(
                len(negative_pool),
                max(0, int(self.max_output) - len(source_columns)),
            )
            if n_negatives == len(negative_pool):
                negative_columns = negative_pool
            elif n_negatives > 0:
                negative_columns = self.rng.choice(
                    negative_pool,
                    size=n_negatives,
                    replace=False,
                    shuffle=False,
                )
            else:
                negative_columns = np.empty(0, dtype=np.int64)
            candidate_columns = np.concatenate((source_columns, negative_columns))

        row_indices = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int64),
            np.diff(matrix.indptr),
        )
        source_lookup = np.empty(matrix.shape[1], dtype=np.int64)
        source_lookup[source_columns] = np.arange(
            len(source_columns),
            dtype=np.int64,
        )
        source_local_columns = source_lookup[matrix.indices]
        values = matrix.data.astype(np.float32, copy=False)
        x = self._sparse_tensor(
            row_indices,
            source_local_columns,
            values,
            shape=(matrix.shape[0], len(source_columns)),
        )
        if candidate_columns is None:
            sources = torch.from_numpy(source_columns).to(self.device)
            candidates = None
        else:
            candidates = torch.from_numpy(candidate_columns).to(self.device)
            sources = candidates[: len(source_columns)]
        return InteractionBatch(x=x, sources=sources, candidates=candidates)

    def _sparse_tensor(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
        values: np.ndarray,
        *,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        indices = torch.from_numpy(np.vstack((rows, columns)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse invariant checks are implicitly disabled.*",
                category=UserWarning,
            )
            tensor = torch.sparse_coo_tensor(
                indices,
                torch.from_numpy(values),
                shape,
                is_coalesced=True,
                check_invariants=False,
            )
            return tensor.to(self.device)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.user_indices)
