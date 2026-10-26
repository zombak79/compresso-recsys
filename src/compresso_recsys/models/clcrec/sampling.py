"""Negative sampling against a user's observed items."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix


class NegativeSampler:
    """Draw items a user has not interacted with, uniformly over the columns.

    Membership is tested against the sorted ``row * n_items + column`` keys of
    the interactions, so a whole batch is checked with one ``searchsorted``.
    Colliding draws are redrawn up to ``max_attempts`` times; whatever still
    collides after that (only possible for users who interacted with nearly
    every item) is kept and counted in :attr:`residual_collisions`.
    """

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        max_attempts: int,
    ) -> None:
        self.n_items = int(interactions.shape[1])
        rows = np.repeat(
            np.arange(interactions.shape[0], dtype=np.int64),
            np.diff(interactions.indptr),
        )
        keys = rows * self.n_items + interactions.indices.astype(np.int64)
        self._keys = np.sort(keys)
        self.max_attempts = int(max_attempts)
        self.residual_collisions = 0

    def _observed(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        if self._keys.size == 0:
            return np.zeros(items.shape, dtype=bool)
        keys = users * self.n_items + items
        positions = np.searchsorted(self._keys, keys)
        positions = np.minimum(positions, self._keys.size - 1)
        return self._keys[positions] == keys

    def sample(
        self,
        users: np.ndarray,
        n_negatives: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return ``(len(users), n_negatives)`` item columns unseen by each user."""
        users = np.asarray(users, dtype=np.int64)[:, None]
        items = rng.integers(0, self.n_items, size=(users.shape[0], n_negatives))
        collisions = self._observed(users, items)
        for _ in range(self.max_attempts):
            if not collisions.any():
                break
            redraw = rng.integers(0, self.n_items, size=int(collisions.sum()))
            items[collisions] = redraw
            collisions = self._observed(users, items)
        self.residual_collisions += int(collisions.sum())
        return items


def sample_other_items(
    anchors: np.ndarray,
    n_negatives: int,
    n_items: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``n_negatives`` items per anchor uniformly from the other items."""
    anchors = np.asarray(anchors, dtype=np.int64)[:, None]
    items = rng.integers(0, n_items - 1, size=(anchors.shape[0], n_negatives))
    return items + (items >= anchors)
