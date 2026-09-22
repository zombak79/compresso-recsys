"""The architecture: modules holding the learned parameters."""

from __future__ import annotations

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn



from compresso_recsys.models.simple_gpt.config import (
    TransformerConfig,
)


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
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if max_positions < 2:
            raise ValueError(
                "max_positions must be >= 2: one slot for CLS and at least one "
                f"for an item, got {max_positions}"
            )
        if n_items > vocab_size:
            raise ValueError(
                f"n_items ({n_items}) cannot exceed vocab_size ({vocab_size}): "
                "the head scores a subset of the vocabulary"
            )
        self.config = config
        self.tie_embeddings = bool(tie_embeddings)
        # Items occupy the LAST n_items rows of the vocabulary, so this is the
        # tokenizer's n_reserved -- the front-loaded convention the trainer's
        # objective already relies on when it decodes targets. Deriving it keeps
        # the module from carrying a second copy that could disagree.
        self.item_offset = int(vocab_size) - int(n_items)
        self.max_positions = int(max_positions)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_positions, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)
        self.head_dropout = nn.Dropout(config.dropout)
        # A tied head keeps its bias: tying is a claim about the weight, and
        # dropping the bias with it would confound two changes in one flag.
        if self.tie_embeddings:
            self.head = None
            self.head_bias = nn.Parameter(torch.zeros(n_items))
        else:
            self.head = nn.Linear(config.d_model, n_items)
            self.head_bias = None

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        # GPT-2's scaled init for the projections that write into the residual
        # stream. Without it the stream's variance grows with depth, since each
        # of the 2 * n_layers residual adds contributes at full scale. nanoGPT
        # matches c_proj by name; matching the modules directly cannot rot.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
        # nn.Embedding zeroes padding_idx at construction and the initialisation
        # above overwrote it. Re-zero explicitly: padding_idx keeps the gradient
        # zero, so whatever sits there at the start stays there for good.
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 initialisation: every weight normal(0, 0.02), biases zero.

        PyTorch's default for ``nn.Linear`` is uniform over
        ``+/-1/sqrt(fan_in)``, which for ``d_model=128`` is roughly 2.5x wider
        than this. Leaving it there is a silent departure from the architecture
        this model claims to be, and it interacts with a tied head: the output
        weight would start at one scale and the input embedding at another.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        """Catalog logits for the given states, one score per item.

        When tied, the weight is a *slice* of the embedding rather than its own
        parameter: ``pad`` and ``unk`` sit below ``item_offset`` and so stay out
        of the head, which is what we want anyway — neither is ever a target.
        Autograd carries the output-side gradient back into the item rows, so a
        tied embedding is trained from both directions.
        """
        hidden = self.head_dropout(states)
        if self.head is not None:
            return self.head(hidden)
        return F.linear(hidden, self.embedding.weight[self.item_offset :], self.head_bias)
