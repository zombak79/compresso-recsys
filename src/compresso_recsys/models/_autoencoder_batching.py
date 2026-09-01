from __future__ import annotations

import numpy as np
import torch
from scipy.sparse import csr_matrix


def prepare_dense_training_data(
    interactions: csr_matrix,
    *,
    device: torch.device,
    preload: bool,
) -> torch.Tensor | None:
    """Materialize training rows once when requested or safe on CUDA."""
    if not preload:
        return None

    if device.type == "cuda":
        dense_bytes = (
            int(interactions.shape[0])
            * int(interactions.shape[1])
            * np.dtype(np.float32).itemsize
        )
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device)
        except (AssertionError, RuntimeError):
            free_bytes = 0
        if dense_bytes > free_bytes // 2:
            return None

    dense = interactions.toarray().astype(np.float32, copy=False)
    try:
        return torch.from_numpy(dense).to(device)
    except torch.OutOfMemoryError:
        return None


def dense_training_batch(
    interactions: csr_matrix,
    selected: np.ndarray,
    *,
    device: torch.device,
    preloaded: torch.Tensor | None,
) -> torch.Tensor:
    """Fetch selected dense rows from device storage or a CSR matrix."""
    if preloaded is not None:
        rows = torch.from_numpy(selected).to(device=device, dtype=torch.long)
        return preloaded.index_select(0, rows)
    dense = interactions[selected].toarray().astype(np.float32, copy=False)
    return torch.from_numpy(dense).to(device)
