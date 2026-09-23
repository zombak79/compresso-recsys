"""The architecture: modules holding the learned parameters."""

from __future__ import annotations


import torch
from torch import nn
from torch.nn import functional as F



class MultVAE(nn.Module):
    """Symmetric multinomial VAE with a Gaussian latent representation."""

    def __init__(
        self,
        n_items: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = int(n_items)
        self.input_dropout = nn.Dropout(float(dropout))
        self.encoder = nn.Linear(self.n_items, int(hidden_dim))
        self.mean = nn.Linear(int(hidden_dim), int(latent_dim))
        self.log_variance = nn.Linear(int(hidden_dim), int(latent_dim))
        self.decoder_hidden = nn.Linear(int(latent_dim), int(hidden_dim))
        self.decoder = nn.Linear(int(hidden_dim), self.n_items)

    def encode(self, interactions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log variance for interaction rows."""
        if interactions.ndim != 2 or interactions.shape[1] != self.n_items:
            raise ValueError(
                "interactions must have shape (rows, "
                f"{self.n_items}), got {tuple(interactions.shape)}"
            )
        normalized = F.normalize(interactions, p=2, dim=1)
        hidden = torch.tanh(self.encoder(self.input_dropout(normalized)))
        return self.mean(hidden), self.log_variance(hidden)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent rows to unnormalized multinomial item scores."""
        return self.decoder(torch.tanh(self.decoder_hidden(latent)))

    def forward(
        self,
        interactions: torch.Tensor,
        *,
        sample: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return item logits, posterior mean, and posterior log variance.

        Sampling defaults to the module's training mode. Evaluation therefore
        uses the posterior mean and produces deterministic rankings.
        """
        mean, log_variance = self.encode(interactions)
        should_sample = self.training if sample is None else bool(sample)
        if should_sample:
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mean + standard_deviation * torch.randn_like(mean)
        else:
            latent = mean
        return self.decode(latent), mean, log_variance
