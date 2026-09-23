"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)


@dataclass
class MultDAEConfig:
    """Configuration for :class:`MultDAETrainer`.

    ``latent_dim`` is the deterministic bottleneck width. ``dropout`` corrupts
    normalized interaction vectors during training only, as in Mult-DAE.
    ``l2_reg`` is the coefficient on the squared L2 norm of the encoder and
    decoder weight matrices; biases are not regularized. The default matches
    the original implementation's ``0.01 / 500`` setting.
    ``preload_training_data=True`` caches the dense interaction matrix on the
    training device by default. Set it to ``False`` to stream CSR minibatches
    when the complete dense matrix does not fit.
    """

    latent_dim: int = 200
    dropout: float = 0.5
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    l2_reg: float = 0.01 / 500
    preload_training_data: bool = True
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "MultDAE"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in ("latent_dim", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if not np.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not np.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")
        if not np.isfinite(self.l2_reg) or self.l2_reg < 0.0:
            raise ValueError(
                "l2_reg must be finite and >= 0, got "
                f"{self.l2_reg}"
            )
        if not isinstance(self.preload_training_data, (bool, np.bool_)):
            raise TypeError("preload_training_data must be a bool")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        torch.device(self.device)
