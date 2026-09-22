"""A causal transformer recommender, and the example a sequential model should be.

`SimpleRNN` reads a history one step at a time and carries what it has seen in a
state vector. This reads the whole history at once and lets each position attend
to every earlier one, which is the only architectural difference that matters:
the objective, the vocabulary, the padding and the evaluation are identical.

The architecture is nanoGPT with two recommendation-shaped adjustments::

    ItemSequences
      -> SequenceBatcher.encode          tokens (rows, W), mask
      -> [CLS] + Embedding(vocab, d)     (rows, W + 1, d)
      -> + learned absolute positions
      -> N x pre-LN causal blocks
      -> LayerNorm
      -> Linear(d, n_items)              one score per catalog item
      -> cross entropy against the *unshifted* tokens

**Why the targets are unshifted.** A `CLS` prefix occupies position 0, so
``states[:, i]`` has read `CLS` plus ``tokens[:, :i]`` and therefore predicts
``tokens[:, i]``. The next-item shift stops being arithmetic in the trainer and
becomes a property of the input, which also means every position is a training
example rather than every position but the first — `CLS` buys back one example
per user compared with `SimpleRNN`.

**Why `CLS` is a parameter and not a token.** It could have been a vocabulary
entry, and that would be simpler. It is an `nn.Parameter` so it can be
*conditioned*: a user embedding or a global feature can be added into position 0
per row, which a vocabulary lookup cannot express. Nothing in this library has
user features yet, so today it is a bare learned prefix doing the job `BOS` would
do — including giving an empty history a defined input instead of the state after
reading one pad.

**Why there is no attention mask.** The batcher always pads on the right, so a
causal mask already excludes it: a real token at position ``i`` attends only to
``<= i``, all of which are real. Pad positions do compute garbage and nothing
reads it — the loss is masked and prediction reads each row's last real
position.

The output head is tied to the input embedding by default (``tie_embeddings``),
which halves the parameters.

Training uses a fixed epoch budget and rebuilds the model on every ``fit`` call;
early stopping and incremental training are not implemented. Sampled softmax, a
logit temperature, and pooling other than "read the last real position" are also
absent.
"""


# Re-exported: the flat module carried these, and
# tests/test_simple_rnn.py asserts the sequential trainers share one
# schedule implementation by identity off the module.
from compresso_recsys.models.core.schedule import (  # noqa: F401
    LRSchedule,
    build_scheduler,
    check_schedule,
)

from compresso_recsys.models.simple_gpt.config import (
    OptimizerName,
    SimpleGPTConfig,
    TransformerConfig,
)
from compresso_recsys.models.simple_gpt.model import (
    Block,
    CausalSelfAttention,
    LayerNorm,
    MLP,
    SimpleGPT,
)
from compresso_recsys.models.simple_gpt.trainer import (
    SimpleGPTTrainer,
)

__all__ = [
    "SimpleGPT",
    "SimpleGPTConfig",
    "SimpleGPTTrainer",
    "TransformerConfig",
]
