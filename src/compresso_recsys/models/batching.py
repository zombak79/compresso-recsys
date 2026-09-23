"""Public interaction batching utilities for custom recommender training."""

from .core.batching import (
    InteractionBatch,
    InteractionBatchSampler,
    dense_training_target,
)

__all__ = [
    "InteractionBatch",
    "InteractionBatchSampler",
    "dense_training_target",
]
