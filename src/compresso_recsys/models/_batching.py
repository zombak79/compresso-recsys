from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix

from compresso_recsys.models._validation import canonical_csr


@dataclass(frozen=True)
class InteractionBatch:
    """Compact interaction batch with source-prefix output candidates.

    ``x`` is a sparse COO tensor whose columns correspond positionally to the
    global item rows in ``sources``. ``candidates`` is either ``None`` for the
    full output catalog or a global-row tensor beginning with ``sources``.
    """

    x: torch.Tensor
    sources: torch.Tensor
    candidates: torch.Tensor | None


@dataclass(frozen=True)
class SymmetricInteractionBatch:
    """Two disjoint views of the same user histories."""

    x: torch.Tensor
    y: torch.Tensor
    sources: torch.Tensor
    candidates: torch.Tensor | None


@dataclass(frozen=True)
class LeaveOneOutInteractionBatch:
    """Virtual leave-one-out rows derived from an interaction CSR matrix."""

    x: torch.Tensor
    sources: torch.Tensor
    candidates: torch.Tensor | None
    target_positions: torch.Tensor
    user_rows: torch.Tensor
    target_items: torch.Tensor


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
    """Batched sparse interactions with optional output candidate sampling.

    Every batch packs the union of its active source items into ``batch.x``.
    ``batch.sources`` maps those local columns back to global item rows. When
    ``max_output`` is set, ``batch.candidates`` starts with that exact source
    prefix and appends sampled items absent from the complete batch. The limit
    is therefore soft when a batch already contains more active source items.
    ``None`` leaves output selection to the model and denotes the full catalog.
    """

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        device: torch.device | str,
        batch_size: int,
        shuffle: bool,
        max_output: int | None,
        seed: int,
    ) -> None:
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, (int, np.integer))
            or batch_size < 1
        ):
            raise ValueError("batch_size must be an integer >= 1")
        if not isinstance(shuffle, (bool, np.bool_)):
            raise TypeError("shuffle must be a bool")
        if max_output is not None and (
            isinstance(max_output, bool)
            or not isinstance(max_output, (int, np.integer))
            or max_output < 1
        ):
            raise ValueError("max_output must be an integer >= 1 or None")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, (int, np.integer))
            or seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")

        self.interactions = interactions
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.max_output = None if max_output is None else int(max_output)
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


class SymmetricInteractionBatchSampler(InteractionBatchSampler):
    """Random non-empty history splits for symmetric cross-reconstruction."""

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
        max_output: int | None,
        seed: int,
        split_probability: float,
    ) -> None:
        super().__init__(
            interactions,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            max_output=max_output,
            seed=seed,
        )
        self.shuffle = bool(shuffle)
        self.split_probability = float(split_probability)
        self.user_indices = np.flatnonzero(
            np.diff(interactions.indptr) >= 2
        ).astype(np.int64, copy=False)
        if self.user_indices.size == 0:
            raise ValueError(
                "symmetric interaction splitting requires at least one user "
                "with two interactions"
            )
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(self.user_indices.size / self.batch_size)

    def __getitem__(self, batch_index: int) -> SymmetricInteractionBatch:
        start = int(batch_index) * self.batch_size
        end = min(start + self.batch_size, self.user_indices.size)
        if start < 0 or start >= self.user_indices.size:
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
            candidate_columns = np.concatenate(
                (source_columns, negative_columns)
            )

        row_indices = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int64),
            np.diff(matrix.indptr),
        )
        source_lookup = np.empty(matrix.shape[1], dtype=np.int64)
        source_lookup[source_columns] = np.arange(
            source_columns.size,
            dtype=np.int64,
        )
        local_columns = source_lookup[matrix.indices]
        in_x = self.rng.random(matrix.nnz) < self.split_probability
        for row in range(matrix.shape[0]):
            row_start, row_end = matrix.indptr[row : row + 2]
            if bool(in_x[row_start:row_end].all()):
                position = int(self.rng.integers(row_start, row_end))
                in_x[position] = False
            elif not bool(in_x[row_start:row_end].any()):
                position = int(self.rng.integers(row_start, row_end))
                in_x[position] = True

        values = matrix.data.astype(np.float32, copy=False)
        shape = (matrix.shape[0], source_columns.size)
        x = self._sparse_tensor(
            row_indices[in_x],
            local_columns[in_x],
            values[in_x],
            shape=shape,
        )
        y = self._sparse_tensor(
            row_indices[~in_x],
            local_columns[~in_x],
            values[~in_x],
            shape=shape,
        )
        if candidate_columns is None:
            sources = torch.from_numpy(source_columns).to(self.device)
            candidates = None
        else:
            candidates = torch.from_numpy(candidate_columns).to(self.device)
            sources = candidates[: source_columns.size]
        return SymmetricInteractionBatch(
            x=x,
            y=y,
            sources=sources,
            candidates=candidates,
        )


class LeaveOneOutInteractionBatchSampler(InteractionBatchSampler):
    """Visit every eligible observed interaction as a held-out target."""

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
        max_output: int | None,
        seed: int,
        batch_order: str = "round_robin",
    ) -> None:
        super().__init__(
            interactions,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            max_output=max_output,
            seed=seed,
        )
        self.shuffle = bool(shuffle)
        if batch_order not in {"round_robin", "grouped"}:
            raise ValueError(
                "batch_order must be 'round_robin' or 'grouped'"
            )
        self.batch_order = batch_order
        self.user_degrees = np.diff(interactions.indptr).astype(
            np.int64,
            copy=False,
        )
        self.eligible_users = np.flatnonzero(
            self.user_degrees >= 2
        ).astype(np.int64, copy=False)
        if self.eligible_users.size == 0:
            raise ValueError(
                "leave-one-out training requires at least one user with two "
                "interactions"
            )
        self.event_indices = np.empty(0, dtype=np.int64)
        self.batch_indptr = np.empty(0, dtype=np.int64)
        self._rebuild_epoch_order(randomize=self.shuffle)

    def _events_for_users(self, users: np.ndarray) -> np.ndarray:
        counts = np.diff(self.interactions.indptr)[users].astype(
            np.int64,
            copy=False,
        )
        repeated_starts = np.repeat(
            self.interactions.indptr[users].astype(np.int64, copy=False),
            counts,
        )
        group_starts = np.repeat(np.cumsum(counts) - counts, counts)
        within_group = np.arange(int(counts.sum()), dtype=np.int64) - group_starts
        return repeated_starts + within_group

    def __len__(self) -> int:
        return max(0, self.batch_indptr.size - 1)

    @property
    def n_examples(self) -> int:
        """Number of observed interactions visited in one complete epoch."""
        return int(self.event_indices.size)

    def _fixed_batch_indptr(self, n_examples: int) -> np.ndarray:
        starts = np.arange(0, n_examples, self.batch_size, dtype=np.int64)
        return np.append(starts, np.int64(n_examples))

    def _round_robin_order(
        self,
        *,
        randomize: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        per_user_events = np.arange(self.interactions.nnz, dtype=np.int64)
        if randomize:
            for user in self.eligible_users:
                start, end = self.interactions.indptr[user : user + 2]
                self.rng.shuffle(per_user_events[start:end])

        n_examples = int(self.user_degrees[self.eligible_users].sum())
        event_indices = np.empty(n_examples, dtype=np.int64)
        batch_ends = [0]
        cursor = 0
        max_degree = int(self.user_degrees[self.eligible_users].max())
        # Batch each interaction depth separately so users cannot repeat.
        for depth in range(max_degree):
            active_users = self.eligible_users[
                self.user_degrees[self.eligible_users] > depth
            ].copy()
            if randomize:
                self.rng.shuffle(active_users)
            round_events = per_user_events[
                self.interactions.indptr[active_users] + depth
            ]
            round_start = cursor
            cursor += active_users.size
            event_indices[round_start:cursor] = round_events
            for local_start in range(0, active_users.size, self.batch_size):
                batch_ends.append(
                    round_start
                    + min(local_start + self.batch_size, active_users.size)
                )
        return event_indices, np.asarray(batch_ends, dtype=np.int64)

    def _rebuild_epoch_order(self, *, randomize: bool) -> None:
        if self.batch_order == "round_robin":
            self.event_indices, self.batch_indptr = self._round_robin_order(
                randomize=randomize,
            )
            return

        users = self.eligible_users.copy()
        if randomize:
            self.rng.shuffle(users)
        self.event_indices = self._events_for_users(users)
        self.batch_indptr = self._fixed_batch_indptr(self.event_indices.size)

    def __getitem__(self, batch_index: int) -> LeaveOneOutInteractionBatch:
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError(batch_index)
        start, end = self.batch_indptr[batch_index : batch_index + 2]

        events = self.event_indices[start:end]
        user_rows = np.searchsorted(
            self.interactions.indptr,
            events,
            side="right",
        ) - 1
        target_items = self.interactions.indices[events]
        matrix = self.interactions[user_rows]
        row_indices = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int64),
            np.diff(matrix.indptr),
        )
        keep = matrix.indices != target_items[row_indices]
        source_global_columns = matrix.indices[keep]
        source_columns = np.unique(source_global_columns).astype(
            np.int64,
            copy=False,
        )
        source_lookup = np.empty(matrix.shape[1], dtype=np.int64)
        source_lookup[source_columns] = np.arange(
            source_columns.size,
            dtype=np.int64,
        )
        source_local_columns = source_lookup[source_global_columns]
        x = self._sparse_tensor(
            row_indices[keep],
            source_local_columns,
            matrix.data[keep].astype(np.float32, copy=False),
            shape=(matrix.shape[0], source_columns.size),
        )

        if self.max_output is None:
            candidate_columns = None
            target_positions = target_items.astype(np.int64, copy=False)
        else:
            unique_targets = np.unique(target_items)
            target_only = unique_targets[
                ~np.isin(unique_targets, source_columns, assume_unique=True)
            ]
            mandatory = np.concatenate((source_columns, target_only))
            negative_mask = np.ones(matrix.shape[1], dtype=bool)
            negative_mask[mandatory] = False
            negative_pool = self.item_indices[negative_mask]
            n_negatives = min(
                negative_pool.size,
                max(0, int(self.max_output) - mandatory.size),
            )
            if n_negatives == negative_pool.size:
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
            candidate_columns = np.concatenate((mandatory, negative_columns))
            candidate_lookup = np.empty(matrix.shape[1], dtype=np.int64)
            candidate_lookup[candidate_columns] = np.arange(
                candidate_columns.size,
                dtype=np.int64,
            )
            target_positions = candidate_lookup[target_items]

        sources = torch.from_numpy(source_columns).to(self.device)
        candidates = (
            None
            if candidate_columns is None
            else torch.from_numpy(candidate_columns).to(self.device)
        )
        return LeaveOneOutInteractionBatch(
            x=x,
            sources=sources,
            candidates=candidates,
            target_positions=torch.from_numpy(target_positions).to(self.device),
            user_rows=torch.from_numpy(user_rows).to(self.device),
            target_items=torch.from_numpy(target_items).to(self.device),
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rebuild_epoch_order(randomize=True)
