"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, Mapping, Sequence

import torch

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
    _validate_log_every_n_steps,
)

from ..core.schedule import LRSchedule, build_scheduler, check_schedule
from ..simple_gpt import LayerNorm, MLP, TransformerConfig


OptimizerName = Literal["NAdam", "AdamW"]


@dataclass
class SimpleBidirectionalTransformerConfig:
    """Architecture and training settings for the bidirectional trainer."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    tie_embeddings: bool = True
    lr_schedule: LRSchedule = "cosine"
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    unk_dropout: float = 0.05
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "NAdam"
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "SimpleBidirectionalTransformer"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if not 0.0 <= self.unk_dropout < 1.0:
            raise ValueError(
                f"unk_dropout must be in [0, 1), got {self.unk_dropout}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        check_schedule(self.lr_schedule, self.warmup_fraction, self.min_lr_ratio)
