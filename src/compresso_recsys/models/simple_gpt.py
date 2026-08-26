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

**Why there is no attention mask.** Padding is on the right, so a causal mask
already excludes it: a real token at position ``i`` attends only to ``<= i``, all
of which are real. Pad positions do compute garbage and nothing reads it — the
loss is masked and prediction reads each row's last real position. That is why
:class:`SimpleGPTTrainer` refuses a left-padding batcher rather than quietly
building a key-padding mask.

Deliberately absent: tied embeddings, learning-rate schedules, early stopping,
sampled softmax, and any pooling other than "read the last real position". Each is
a separate claim that deserves measuring on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["SimpleGPT", "TransformerConfig"]


@dataclass(frozen=True)
class TransformerConfig:
    """The backbone, separated from the recommendation concerns around it.

    A transformer has one width. Unlike :class:`SimpleRNNConfig`, which lets
    ``embedding_dim`` and ``hidden_dim`` differ, the residual stream forces the
    embedding, the attention and the output of every block to share ``d_model``
    — and ``n_heads`` must divide it, since each head takes an equal slice.

    ``bias`` turns off the additive terms in the linear projections and the layer
    norms together. Off by default, following nanoGPT: it is slightly faster and
    marginally better, and having one flag rather than three keeps the
    combinations that were never tested from being expressible.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "n_layers"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={self.d_model} "
                f"and n_heads={self.n_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        """Width of each attention head."""
        return self.d_model // self.n_heads


class LayerNorm(nn.Module):
    """Layer norm with an optional bias, which :class:`torch.nn.LayerNorm` lacks."""

    def __init__(self, ndim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head self attention where a position may only read earlier ones.

    No mask is built or accepted. ``is_causal=True`` is the whole story, and it
    is sufficient *because* the batcher pads on the right — see the module
    docstring. Passing an additive mask as well would be legal but would give up
    the fused attention kernels for nothing.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows, length, _ = x.shape
        query, key, value = self.attn(x).split(self.d_model, dim=2)
        shape = (rows, length, self.n_heads, self.d_model // self.n_heads)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(rows, length, self.d_model)
        )
        return self.resid_dropout(self.proj(attended))


class MLP(nn.Module):
    """The position-wise feed-forward half of a block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.activation = nn.GELU()
        self.down = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.activation(self.up(x))))


class Block(nn.Module):
    """Pre-norm transformer block: norm before each sublayer, residual around it."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleGPT(nn.Module):
    """Embeddings, a `CLS` prefix, causal blocks, and a linear head.

    The head scores ``n_items`` rather than ``vocab_size``: a special token is
    never a prediction target, so an output column for one could only ever learn
    to be wrong — and it would let a misaligned objective score plausibly instead
    of raising.

    :meth:`forward` returns states and :meth:`score` turns states into logits,
    kept separate because prediction needs logits at one position per row.
    Scoring first would materialise ``rows x length x n_items``, which on a real
    catalog is where the memory goes.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        n_items: int,
        max_positions: int,
        pad_id: int,
        config: TransformerConfig,
    ) -> None:
        super().__init__()
        if max_positions < 2:
            raise ValueError(
                "max_positions must be >= 2: one slot for CLS and at least one "
                f"for an item, got {max_positions}"
            )
        self.config = config
        self.max_positions = int(max_positions)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_positions, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)
        self.head_dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.d_model, n_items)

        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        # nn.Embedding zeroes padding_idx at construction and the initialisation
        # above overwrote it. Re-zero explicitly: padding_idx keeps the gradient
        # zero, so whatever sits there at the start stays there for good.
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """States for `CLS` and every token, shape ``(rows, length + 1, d_model)``.

        ``states[:, i]`` has read `CLS` and ``tokens[:, :i]``, so it is the state
        from which ``tokens[:, i]`` should be predicted.
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (rows, length), got {tuple(tokens.shape)}")
        rows, length = tokens.shape
        if length + 1 > self.max_positions:
            raise ValueError(
                f"a history of {length} items needs {length + 1} positions "
                f"including CLS, but this model was built for {self.max_positions}"
            )

        prefix = self.cls_token.expand(rows, 1, -1)
        hidden = torch.cat([prefix, self.embedding(tokens)], dim=1)
        positions = torch.arange(length + 1, device=tokens.device)
        hidden = self.embed_dropout(hidden + self.position(positions))
        for block in self.blocks:
            hidden = block(hidden)
        return self.ln_f(hidden)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Catalog logits for the given states, one score per item."""
        return self.head(self.head_dropout(states))
