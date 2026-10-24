"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)


@dataclass
class MultVAEConfig:
    """Configuration for :class:`MultVAETrainer`.

    ``kl_cap`` is the maximum coefficient on KL divergence.
    ``kl_anneal_steps`` is the denominator in ``updates / kl_anneal_steps``;
    the coefficient is clipped at ``kl_cap``. It therefore reaches the cap
    after ``kl_cap * kl_anneal_steps`` updates. Set the step count to zero to
    use ``kl_cap`` from the first update.
    ``preload_training_data=True`` caches the dense interaction matrix on the
    training device by default. Set it to ``False`` to stream CSR minibatches
    when the complete dense matrix does not fit.
    """

    latent_dim: int = 200
    hidden_dim: int = 600
    dropout: float = 0.5
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    kl_cap: float = 0.2
    kl_anneal_steps: int = 200_000
    preload_training_data: bool = True
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "MultVAE"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in ("latent_dim", "hidden_dim", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if isinstance(self.kl_anneal_steps, (bool, np.bool_)) or not isinstance(
            self.kl_anneal_steps, (int, np.integer)
        ):
            raise TypeError("kl_anneal_steps must be an integer")
        if self.kl_anneal_steps < 0:
            raise ValueError("kl_anneal_steps must be >= 0")
        if not np.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not np.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(
                "weight_decay must be finite and >= 0, got "
                f"{self.weight_decay}"
            )
        if not np.isfinite(self.kl_cap) or self.kl_cap < 0.0:
            raise ValueError(f"kl_cap must be finite and >= 0, got {self.kl_cap}")
        if not isinstance(self.preload_training_data, (bool, np.bool_)):
            raise TypeError("preload_training_data must be a bool")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        torch.device(self.device)
