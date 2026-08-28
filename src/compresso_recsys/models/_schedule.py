"""Learning-rate schedules shared by the sequential trainers.

Both `SimpleRNN` and `SimpleGPT` want the same warmup-then-decay curve, and it
is enough code that duplicating it the way ``OptimizerName`` is duplicated would
be a worse trade: a schedule that disagrees between two models is a difference
nobody intended and nobody would notice.

Measured in optimizer *steps* rather than epochs, so the shape does not move when
the batch size does.
"""

from __future__ import annotations

import math
from typing import Literal

import torch

LRSchedule = Literal["constant", "cosine"]


def check_schedule(
    schedule: str, warmup_fraction: float, min_lr_ratio: float
) -> None:
    """Validate the three fields a config needs to describe a schedule."""
    if schedule not in ("constant", "cosine"):
        raise ValueError(
            f"lr_schedule must be 'constant' or 'cosine', got {schedule!r}"
        )
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError(
            f"warmup_fraction must be in [0, 1), got {warmup_fraction}"
        )
    if not 0.0 < min_lr_ratio <= 1.0:
        raise ValueError(f"min_lr_ratio must be in (0, 1], got {min_lr_ratio}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: LRSchedule,
    total_steps: int,
    warmup_fraction: float,
    min_lr_ratio: float,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Linear warmup then cosine decay, or ``None`` for a flat rate.

    Warmup exists because the earliest steps are the ones most able to wreck a
    transformer: attention has learned nothing, so gradients are large and badly
    aimed. Cosine decay then spends the end of the run refining rather than
    bouncing. Neither is expressible through the optimizer alone, which is why
    they arrive as one option rather than two.
    """
    if schedule == "constant":
        return None
    warmup = int(warmup_fraction * total_steps)

    def factor(step: int) -> float:
        if step < warmup:
            # Step 0 would otherwise train at exactly zero and waste a step.
            return (step + 1) / (warmup + 1)
        if total_steps <= warmup:
            return 1.0
        progress = (step - warmup) / (total_steps - warmup)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
