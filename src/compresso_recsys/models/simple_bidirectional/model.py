"""The architecture: modules holding the learned parameters."""

from __future__ import annotations

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


from ..simple_gpt import LayerNorm, MLP, TransformerConfig


class BidirectionalSelfAttention(nn.Module):
    """Multi-head self-attention whose real positions may read one another."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attend to every non-padding key in the row.

        ``mask`` is true for ``CLS`` and real history positions.  Padding
        queries may compute disposable states, but padding is never visible as a
        key to a real position at any layer.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must be (rows, length, dim), got {tuple(x.shape)}"
            )
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match x rows and length "
                f"{tuple(x.shape[:2])}"
            )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have boolean dtype")

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
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(rows, length, self.d_model)
        )
        return self.resid_dropout(self.proj(attended))


class BidirectionalBlock(nn.Module):
    """Pre-normalized bidirectional attention followed by an MLP."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleBidirectionalTransformer(nn.Module):
    """Item embeddings, bidirectional blocks, and a catalog-scoring head."""

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
                f"n_items ({n_items}) cannot exceed vocab_size ({vocab_size})"
            )
        self.config = config
        self.tie_embeddings = bool(tie_embeddings)
        self.item_offset = int(vocab_size) - int(n_items)
        self.max_positions = int(max_positions)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_positions, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            BidirectionalBlock(config) for _ in range(config.n_layers)
        )
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)
        self.head_dropout = nn.Dropout(config.dropout)
        if self.tie_embeddings:
            self.head = None
            self.head_bias = nn.Parameter(torch.zeros(n_items))
        else:
            self.head = nn.Linear(config.d_model, n_items)
            self.head_bias = None

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return states for ``CLS`` and each token."""
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must be (rows, length), got {tuple(tokens.shape)}"
            )
        if mask.shape != tokens.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match tokens shape "
                f"{tuple(tokens.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have boolean dtype")
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
        full_mask = torch.cat(
            [torch.ones((rows, 1), dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        for block in self.blocks:
            hidden = block(hidden, full_mask)
        return self.ln_f(hidden)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Turn one or more hidden states into catalog logits."""
        hidden = self.head_dropout(states)
        if self.head is not None:
            return self.head(hidden)
        return F.linear(
            hidden,
            self.embedding.weight[self.item_offset :],
            self.head_bias,
        )
