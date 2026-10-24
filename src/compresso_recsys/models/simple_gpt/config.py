"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch


from compresso_recsys._reporting import (
    _validate_log_every_n_steps,
)

from ..core.schedule import LRSchedule, check_schedule


OptimizerName = Literal["NAdam", "AdamW"]


@dataclass(frozen=True)
class TransformerConfig:
    """The backbone, separated from the recommendation concerns around it.

    A transformer has one width. Unlike :class:`SimpleRNNConfig`, which lets
    ``embedding_dim`` and ``hidden_dim`` differ, the residual stream forces the
    embedding, the attention and the output of every block to share ``d_model``
    — and ``n_heads`` must divide it, since each head takes an equal slice.

    ``bias`` turns off the additive terms in the linear projections and the layer
    norms together. Off by default, following nanoGPT: it is slightly faster and
    marginally better, and having one flag rather than three keeps the
    combinations that were never tested from being expressible.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "n_layers"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={self.d_model} "
                f"and n_heads={self.n_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        """Width of each attention head."""
        return self.d_model // self.n_heads


@dataclass
class SimpleGPTConfig:
    """Configuration for :class:`SimpleGPTTrainer`.

    ``transformer`` carries the backbone; everything else is about training it.
    The context window is deliberately *not* a field — it belongs to the batcher,
    because it describes what the encoder reads rather than the shape of the
    network, and duplicating it is how the two drift apart. ``rstar`` carries it
    in both places and needs a runtime check to keep them equal.

    ``tie_embeddings`` scores with the input embedding's item rows instead of a
    separate head, halving the parameters. It is on by default; set it ``False``
    to use an independent output projection.

    Tying can change convergence as well as parameter count. ``nn.Linear`` initialises around
    ``+/-1/sqrt(d_model)`` while the embedding starts at ``std=0.02``, so a tied
    head begins with a flatter softmax. Compare variants at independently
    validated budgets rather than assuming their training curves match.

    ``unk_dropout`` replaces that fraction of input positions with the
    tokenizer's ``unk`` token. Non-zero by default because otherwise ``unk`` is
    never trained at all: the training vocabulary *is* the training window, so an
    out-of-catalog item cannot occur until evaluation, and its embedding would
    still sit at initialisation when a quarter of a temporal test history needs
    it. Match the rate to the out-of-catalog share you expect — near zero under
    ``leave_last_out``, far higher on a late ``temporal`` stage.
    """

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
    log_prefix: str = "SimpleGPT"
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
