"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)

from ..core.schedule import LRSchedule, check_schedule


RNNType = Literal["gru", "lstm"]


OptimizerName = Literal["NAdam", "AdamW"]


@dataclass
class SimpleRNNConfig:
    """Configuration for :class:`SimpleRNNTrainer`.

    ``dropout`` is applied to the states before scoring, and additionally
    between recurrent layers when ``num_layers > 1``. A single-layer RNN has no
    between-layer position to apply it, which is PyTorch's own behaviour rather
    than a choice made here.

    ``unk_dropout`` replaces that fraction of *input* positions with the
    tokenizer's ``unk`` token, teaching the model to read a history containing an
    item it cannot identify. It defaults to a non-zero rate because otherwise
    ``unk`` is never trained at all: the training vocabulary *is* the training
    window, so an out-of-catalog item cannot occur until evaluation, and its
    embedding would still sit at its initialisation when a quarter of a temporal
    test history turns out to need it.

    The right rate tracks the out-of-catalog share the model will actually face,
    which is a property of the split rather than of the model: near zero under
    ``leave_last_out``, and far higher on a late ``temporal`` stage. It is
    ignored when the tokenizer has no ``unk`` to substitute.
    """

    rnn_type: RNNType = "gru"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    unk_dropout: float = 0.05
    lr_schedule: LRSchedule = "constant"
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "NAdam"
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "SimpleRNN"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        check_schedule(self.lr_schedule, self.warmup_fraction, self.min_lr_ratio)
        if self.rnn_type not in ("gru", "lstm"):
            raise ValueError(
                f"rnn_type must be 'gru' or 'lstm', got {self.rnn_type!r}"
            )
        for name in ("embedding_dim", "hidden_dim", "num_layers", "batch_size"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.unk_dropout < 1.0:
            raise ValueError(
                f"unk_dropout must be in [0, 1), got {self.unk_dropout}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
