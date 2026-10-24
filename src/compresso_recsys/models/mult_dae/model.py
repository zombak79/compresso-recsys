"""The architecture: modules holding the learned parameters."""

from __future__ import annotations


import torch
from torch import nn
from torch.nn import functional as F



class MultDAE(nn.Module):
    """The deterministic ``n_items -> latent -> n_items`` Mult-DAE network."""

    def __init__(self, n_items: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = int(n_items)
        self.input_dropout = nn.Dropout(float(dropout))
        self.encoder = nn.Linear(self.n_items, int(latent_dim))
        self.decoder = nn.Linear(int(latent_dim), self.n_items)

    def forward(self, interactions: torch.Tensor) -> torch.Tensor:
        """Return one unnormalized multinomial score per catalog item."""
        if interactions.ndim != 2 or interactions.shape[1] != self.n_items:
            raise ValueError(
                "interactions must have shape (rows, "
                f"{self.n_items}), got {tuple(interactions.shape)}"
            )
        normalized = F.normalize(interactions, p=2, dim=1)
        hidden = torch.tanh(self.encoder(self.input_dropout(normalized)))
        return self.decoder(hidden)
