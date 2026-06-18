from __future__ import annotations

from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader


def hf_to_torch_dataloader(
    hf_dataset,
    *,
    batch_size: int = 256,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
):
    """Thin bridge from HuggingFace Dataset to torch DataLoader."""

    def _default_collate(rows):
        keys = rows[0].keys()
        out = {k: [r[k] for r in rows] for k in keys}
        return out

    return DataLoader(
        hf_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn or _default_collate,
    )

