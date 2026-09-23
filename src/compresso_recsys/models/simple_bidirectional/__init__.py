"""A small bidirectional Transformer for sequence-to-set recommendation.

Unlike :mod:`simple_gpt`, which learns one next-token target at every sequence
position, this model reads a complete history into a ``CLS`` representation and
scores one unordered set of catalog items per row.  That makes it the sequential
model that can consume the target matrix produced by temporal and asymmetric
interaction splits::

    ItemSequences
      -> SequenceBatcher.encode                  tokens, padding mask
      -> [CLS] + item and position embeddings
      -> N x bidirectional, padding-aware blocks
      -> final CLS state
      -> Linear(d_model, n_items)
      -> multinomial cross-entropy against a target set

``fit(..., targets=None)`` reconstructs the set of items in each source history.
Passing a CSR target matrix instead trains the mapping from the source history to
that explicit set.  The distinction is persisted because it controls prediction:
source items remain eligible after explicit-target training, where a source item
may legitimately also be a target.
"""


# Re-exported: the flat module carried these, and
# tests/test_simple_rnn.py asserts the sequential trainers share one
# schedule implementation by identity off the module.

from compresso_recsys.models.simple_bidirectional.config import (
    OptimizerName,
    SimpleBidirectionalTransformerConfig,
)
from compresso_recsys.models.simple_bidirectional.model import (
    BidirectionalBlock,
    BidirectionalSelfAttention,
    SimpleBidirectionalTransformer,
)
from compresso_recsys.models.simple_bidirectional.trainer import (
    SimpleBidirectionalTransformerTrainer,
)


__all__ = [
    "SimpleBidirectionalTransformer",
    "SimpleBidirectionalTransformerConfig",
    "SimpleBidirectionalTransformerTrainer",
]
