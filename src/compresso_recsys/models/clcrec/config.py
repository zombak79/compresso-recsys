"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)


Activation = Literal["relu", "leaky_relu", "tanh"]


Inference = Literal["hybrid", "content"]


UnknownUsers = Literal["error", "fold_in"]


@dataclass(frozen=True)
class CLCRecConfig:
    """Configuration for CLCRec (Wei et al., ACM MM 2021), MF encoder variant.

    Values the paper states: ``embedding_dim`` (64, §3.1.4), Xavier
    initialization and Adam, the ``lr`` and ``l2_weight`` grids
    {1e-4, 1e-3, 1e-2, 1e-1}, ``n_negatives`` (128 in the U-I ablation) and the
    early-stopping protocol (recall@10, patience 10). ``re_weight`` is the
    paper's λ (Eq. 13) and ``hybrid_prob`` its ρ (Fig. 4); both are swept over
    [0, 1] in the paper without a single reported optimum.

    Values the paper does not report, chosen here and to be tuned on validation:
    ``temperature`` (τ, shared by both terms), ``hidden_dim`` and
    ``activation`` (the hidden layer and φ of Eq. 11), and ``batch_size``.

    ``re_n_negatives=None`` reuses the U-I negatives for the R-E term, as
    Fig. 1 draws one set of negatives for both; an integer draws a separate set.
    ``inference="hybrid"`` scores items with an ID row by that row and the rest
    by their content representation; ``"content"`` scores every item by content.
    ``unknown_users="fold_in"`` is an extension that is not in the paper.
    """

    embedding_dim: int = 64
    hidden_dim: int = 256
    activation: Activation = "leaky_relu"
    n_negatives: int = 128
    re_n_negatives: int | None = None
    temperature: float = 0.2
    re_weight: float = 0.5
    hybrid_prob: float = 0.5
    l2_weight: float = 1e-3
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    max_negative_attempts: int = 10
    inference: Inference = "hybrid"
    unknown_users: UnknownUsers = "error"
    fold_in_steps: int = 50
    fold_in_lr: float = 1e-2
    encoder_batch_size: int = 8192
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "CLCRec"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in (
            "embedding_dim",
            "hidden_dim",
            "n_negatives",
            "batch_size",
            "epochs",
            "max_negative_attempts",
            "encoder_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        if self.re_n_negatives is not None and (
            isinstance(self.re_n_negatives, bool)
            or not isinstance(self.re_n_negatives, int)
            or self.re_n_negatives < 1
        ):
            raise ValueError("re_n_negatives must be an integer >= 1 or None")
        if (
            isinstance(self.fold_in_steps, bool)
            or not isinstance(self.fold_in_steps, int)
            or self.fold_in_steps < 0
        ):
            raise ValueError("fold_in_steps must be an integer >= 0")
        for name in ("temperature", "lr", "fold_in_lr"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        value = float(self.l2_weight)
        if not np.isfinite(value) or value < 0:
            raise ValueError("l2_weight must be finite and >= 0")
        for name in ("re_weight", "hybrid_prob"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.activation not in {"relu", "leaky_relu", "tanh"}:
            raise ValueError("activation must be 'relu', 'leaky_relu', or 'tanh'")
        if self.inference not in {"hybrid", "content"}:
            raise ValueError("inference must be 'hybrid' or 'content'")
        if self.unknown_users not in {"error", "fold_in"}:
            raise ValueError("unknown_users must be 'error' or 'fold_in'")
        if not isinstance(self.show_progress, bool):
            raise ValueError("show_progress must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
