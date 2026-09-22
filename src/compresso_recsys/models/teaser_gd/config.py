"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Hashable, Literal, Sequence

import numpy as np
import torch

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
    _format_duration,
    _validate_log_every_n_steps,
)
from compresso_recsys.models.core.batching import (
    InteractionBatchSampler,
    dense_training_target,
    normalized_mse,
)


OptimizerName = Literal["NAdam", "AdamW"]


TEASERGDLoss = Literal["normalized_mse", "teaser"]


EncoderInit = Literal["xavier", "features"]


@dataclass(frozen=True)
class TEASERGDConfig:
    """Configuration for gradient-trained TEASER.

    ``loss="normalized_mse"`` preserves the ELSA-style objective, while
    ``loss="teaser"`` uses the original TEASER Frobenius reconstruction and
    regularization scale. ``max_output`` uses the same source-prefix candidate
    sampling as ELSA. In TEASER mode, sampled negatives are importance-weighted
    to estimate full-output reconstruction.
    ``coefficient_regularization_samples`` controls a Monte Carlo estimate of
    the effective coefficient norm; zero disables that term.
    """

    batch_size: int = 1024
    max_output: int | None = None
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    l2_coefficients: float = 0.05
    l2_encoder: float = 0.05
    coefficient_regularization_samples: int = 4096
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    use_relu: bool = True
    include_popularity: bool = True
    optimizer: OptimizerName = "NAdam"
    loss: TEASERGDLoss = "normalized_mse"
    encoder_init: EncoderInit = "xavier"
    normalize_encoder: bool = False
    diagonal_scale: float = 1.0
    log_prefix: str = "TEASERGD"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in ("batch_size", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_output is not None and (
            isinstance(self.max_output, bool)
            or not isinstance(self.max_output, int)
            or self.max_output < 1
        ):
            raise ValueError("max_output must be >= 1 or None")
        if (
            isinstance(self.coefficient_regularization_samples, bool)
            or not isinstance(self.coefficient_regularization_samples, int)
            or self.coefficient_regularization_samples < 0
        ):
            raise ValueError("coefficient_regularization_samples must be >= 0")
        for name in (
            "lr",
            "weight_decay",
            "l2_coefficients",
            "l2_encoder",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                relation = "> 0" if name == "lr" else ">= 0"
                raise ValueError(f"{name} must be finite and {relation}")
        if (
            isinstance(self.diagonal_scale, bool)
            or not np.isfinite(self.diagonal_scale)
            or not 0 <= self.diagonal_scale <= 1
        ):
            raise ValueError("diagonal_scale must be finite and in [0, 1]")
        if self.optimizer not in {"NAdam", "AdamW"}:
            raise ValueError("optimizer must be 'NAdam' or 'AdamW'")
        if self.loss not in {"normalized_mse", "teaser"}:
            raise ValueError("loss must be 'normalized_mse' or 'teaser'")
        if self.encoder_init not in {"xavier", "features"}:
            raise ValueError("encoder_init must be 'xavier' or 'features'")
        for name in (
            "decay",
            "compile",
            "show_progress",
            "use_relu",
            "normalize_encoder",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if not isinstance(self.include_popularity, bool):
            raise ValueError("include_popularity must be a bool")
