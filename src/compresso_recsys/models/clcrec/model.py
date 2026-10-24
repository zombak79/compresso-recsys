"""The CLCRec network: ID tables, the content encoder and the two density ratios."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from compresso_recsys.models.clcrec.config import Activation


_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


class ContentEncoder(nn.Module):
    """``f = W2 φ(W1 x + b1) + b2``, the one-hidden-layer encoder of Eq. 11."""

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        embedding_dim: int,
        activation: Activation,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(n_features, hidden_dim)
        self.activation = _ACTIVATIONS[activation]()
        self.output = nn.Linear(hidden_dim, embedding_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(self.activation(self.hidden(features)))


class CLCRec(nn.Module):
    """User and item ID embeddings (the MF encoder of Eq. 9) plus Eq. 11.

    ``item_emb`` has rows only for training items; items without one are
    represented by :attr:`encoder` alone.
    """

    def __init__(
        self,
        *,
        n_users: int,
        n_items: int,
        n_features: int,
        embedding_dim: int,
        hidden_dim: int,
        activation: Activation,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.user_emb = nn.Embedding(n_users, embedding_dim)
        self.item_emb = nn.Embedding(n_items, embedding_dim)
        self.encoder = ContentEncoder(
            n_features,
            hidden_dim,
            embedding_dim,
            activation,
        )
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        """Xavier initialization (paper §3.1.4), zero biases."""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        for layer in (self.encoder.hidden, self.encoder.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    @staticmethod
    def ui_logits(
        users: torch.Tensor,
        items: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """``log g(z_u, v) = z_u·v / τ`` (Eq. 10) for ``(B, d)`` users, ``(B, S, d)`` items."""
        return torch.einsum("bd,bsd->bs", users, items) / temperature

    @staticmethod
    def re_logits(
        items: torch.Tensor,
        features: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """``log h(z_i, f_j) = cos(z_i, f_j) / τ`` (Eq. 12) for ``(B, d)``, ``(B, S, d)``."""
        return (
            torch.einsum(
                "bd,bsd->bs",
                F.normalize(items, dim=-1),
                F.normalize(features, dim=-1),
            )
            / temperature
        )
