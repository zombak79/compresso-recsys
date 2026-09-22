"""A recurrent next-item recommender — the smallest honest sequential baseline.

One training example per user: read the history left to right and predict the
next item at every position. That is the GRU4Rec objective, and it is the
cheapest thing that actually uses order, which makes it the baseline a
transformer has to beat before its extra machinery has earned anything.

The architecture is deliberately unremarkable::

    ItemSequences
      -> SequenceBatcher.encode        tokens (rows, length), mask
      -> Embedding(vocab, dim, padding_idx=pad_id)
      -> GRU or LSTM                   states (rows, length, hidden)
      -> Linear(hidden, n_items)       one score per catalog item
      -> cross entropy against the history shifted one step left

Two details carry all the risk, and both are pushed into
:class:`~compresso_recsys.models.sequence_batching.SequenceBatcher`.

**Reading the final state.** With right padding, the last *column* is padding
for every row shorter than the batch maximum, so scoring from ``states[:, -1]``
would score most users from a pad embedding. Prediction goes through
:meth:`~compresso_recsys.models.sequence_batching.SequenceBatcher.gather_final`,
which reads each row's own last real position.

**Truncation is not exclusion.** The batcher's ``max_length`` bounds what the
encoder reads, not what the model may recommend: ``exclude_seen`` masks the whole
history, including the part truncation dropped -- and, since a history may span a
wider catalog than the model was fitted on, including nothing it could not have
scored anyway.

Training uses a fixed epoch budget and rebuilds the model on every ``fit`` call;
early stopping and incremental training are not implemented. Tied embeddings
and sampled softmax are also absent.
"""


# Re-exported: the flat module carried these, and
# tests/test_simple_rnn.py asserts the sequential trainers share one
# schedule implementation by identity off the module.
from compresso_recsys.models.core.schedule import (  # noqa: F401
    LRSchedule,
    build_scheduler,
    check_schedule,
)

from compresso_recsys.models.simple_rnn.config import (
    OptimizerName,
    RNNType,
    SimpleRNNConfig,
)
from compresso_recsys.models.simple_rnn.model import (
    SimpleRNN,
)
from compresso_recsys.models.simple_rnn.trainer import (
    SimpleRNNTrainer,
)

__all__ = ["SimpleRNN", "SimpleRNNConfig", "SimpleRNNTrainer"]
