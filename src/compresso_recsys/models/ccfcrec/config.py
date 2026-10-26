"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)


Inference = Literal["content", "hybrid"]


@dataclass(frozen=True)
class CCFCRecConfig:
    """Configuration for CCFCRec (Zhou et al., WWW 2023).

    Values the paper states (§4.1.4): ``batch_size`` 1024, ``temperature`` τ
    0.1, Adam, and per dataset ``embedding_dim`` 128 or 256, ``lr`` 5e-6 or
    1e-4, ``contrastive_weight`` λ 0.5 or 0.6, ``n_positive_items`` 10 or 5 and
    ``n_negative_items`` 40. The defaults take the learning rate and positive
    count of the Amazon-VG column and the embedding size and λ of the ML-20M
    column; tune them on validation.

    Values the paper does not report: ``hidden_dim`` (the hidden layer of the
    two-layer encoder ``g_c``) and the scale of the ``‖Θ‖`` term of Eq. 9,
    implemented as Adam weight decay ``weight_decay``.

    ``max_negative_attempts`` bounds the rejection sampling of negative users
    for the BPR terms; contrastive negative items are drawn exactly.

    ``inference="content"`` follows the paper and scores every item by its
    content embedding. ``"hybrid"`` is an extension that scores items with
    training interactions by their co-occurrence embedding instead.
    """

    embedding_dim: int = 128
    hidden_dim: int = 256
    temperature: float = 0.1
    contrastive_weight: float = 0.5
    n_positive_items: int = 5
    n_negative_items: int = 40
    lr: float = 1e-4
    weight_decay: float = 0.0
    batch_size: int = 1024
    epochs: int = 100
    max_negative_attempts: int = 10
    inference: Inference = "content"
    encoder_batch_size: int = 8192
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "CCFCRec"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in (
            "embedding_dim",
            "hidden_dim",
            "n_positive_items",
            "n_negative_items",
            "batch_size",
            "epochs",
            "max_negative_attempts",
            "encoder_batch_size",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer >= 1")
        for name in ("temperature", "lr"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        for name in ("contrastive_weight", "weight_decay"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and >= 0")
        if self.inference not in {"content", "hybrid"}:
            raise ValueError("inference must be 'content' or 'hybrid'")
        if not isinstance(self.show_progress, bool):
            raise ValueError("show_progress must be a bool")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
