"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)
from compresso_recsys.models.core.batching import (
    dense_training_target,
    normalized_mse,
)


OptimizerName = Literal["NAdam", "AdamW"]


CompressionScoreMode = Literal["abs", "raw", "relu"]


SparseFinetuneBackend = Literal["dense", "coo"]


SparseInferenceBackend = Literal["csr", "dense"]


_dense_training_target = dense_training_target


_normalized_mse = normalized_mse


@dataclass(frozen=True)
class ELSACompressionConfig:
    """Lottery-ticket compression settings for :class:`ELSATrainer`.

    Mask-search stages advance only when the proposed mask remains below
    ``change_threshold`` for ``stability_window`` mask updates. Once the final
    ticket is found, it is converted to an :class:`compresso.SRPParam` and its
    values are trained for ``ELSAConfig.epochs``. ``max_epochs_per_stage`` can
    force an unstable stage to accept its latest proposed mask; ``None`` leaves
    stability search unlimited. ``sparse_finetune_backend="dense"`` densifies
    only the selected SRP rows and uses dense matrix multiplication, while
    ``"coo"`` preserves sparse factors for lower-memory fine-tuning.
    ``sparse_inference_backend`` selects cached CSR or dense full-catalog
    scoring and can be overridden by each prediction call.
    """

    k_target: int
    k_schedule: tuple[int, ...] | None = None
    num_stages: int = 10
    stability_window: int = 5
    change_threshold: float = 0.01
    mask_update_interval: int = 10
    max_epochs_per_stage: int | None = None
    score_mode: CompressionScoreMode = "abs"
    ste_alpha: float = 1.0
    sparse_finetune_backend: SparseFinetuneBackend = "dense"
    sparse_inference_backend: SparseInferenceBackend = "csr"

    def __post_init__(self) -> None:
        if self.k_target < 1:
            raise ValueError("k_target must be >= 1")
        if self.k_schedule is not None:
            if not self.k_schedule:
                raise ValueError("k_schedule must not be empty")
            if any(k < 1 for k in self.k_schedule):
                raise ValueError("every k_schedule value must be >= 1")
            if any(
                current < following
                for current, following in zip(
                    self.k_schedule,
                    self.k_schedule[1:],
                )
            ):
                raise ValueError("k_schedule must be non-increasing")
            if self.k_schedule[-1] != self.k_target:
                raise ValueError("the last k_schedule value must equal k_target")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if self.stability_window < 1:
            raise ValueError("stability_window must be >= 1")
        if not np.isfinite(self.change_threshold) or self.change_threshold < 0:
            raise ValueError("change_threshold must be finite and >= 0")
        if self.mask_update_interval < 1:
            raise ValueError("mask_update_interval must be >= 1")
        if self.max_epochs_per_stage is not None and self.max_epochs_per_stage < 1:
            raise ValueError("max_epochs_per_stage must be >= 1 or None")
        if self.score_mode not in {"abs", "raw", "relu"}:
            raise ValueError("score_mode must be 'abs', 'raw', or 'relu'")
        if not np.isfinite(self.ste_alpha) or not 0 <= self.ste_alpha <= 1:
            raise ValueError("ste_alpha must be finite and in [0, 1]")
        if self.sparse_finetune_backend not in {"dense", "coo"}:
            raise ValueError("sparse_finetune_backend must be 'dense' or 'coo'")
        if self.sparse_inference_backend not in {"csr", "dense"}:
            raise ValueError("sparse_inference_backend must be 'csr' or 'dense'")


@dataclass(frozen=True)
class ELSAConfig:
    """Configuration for :class:`ELSATrainer`.

    ``max_output`` limits the number of output candidates used by a training
    batch. Every item with a positive interaction in the batch is retained,
    and the remaining budget is sampled without replacement from items absent
    from the entire batch. Consequently, a batch with more positive columns
    than ``max_output`` exceeds the requested limit rather than dropping
    positive targets. ``None`` evaluates the full item output during training.
    """

    latent_dim: int = 1024
    batch_size: int = 1024
    max_output: int | None = None
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    use_relu: bool = True
    optimizer: OptimizerName = "NAdam"
    compression: ELSACompressionConfig | None = None
    log_prefix: str = "ELSA"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_output is not None and self.max_output < 1:
            raise ValueError("max_output must be >= 1 or None")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and > 0")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and >= 0")
        if self.optimizer not in {"NAdam", "AdamW"}:
            raise ValueError("optimizer must be 'NAdam' or 'AdamW'")
        if self.compression is not None:
            if self.compile:
                raise ValueError(
                    "torch.compile is not supported during compressed ELSA "
                    "mask search"
                )
            if self.compression.k_target > self.latent_dim:
                raise ValueError("compression.k_target must be <= latent_dim")
            if self.compression.k_schedule is not None:
                if self.compression.k_schedule[0] != self.latent_dim:
                    raise ValueError(
                        "compression.k_schedule must start with latent_dim"
                    )
                if any(k > self.latent_dim for k in self.compression.k_schedule):
                    raise ValueError(
                        "compression.k_schedule values must be <= latent_dim"
                    )
