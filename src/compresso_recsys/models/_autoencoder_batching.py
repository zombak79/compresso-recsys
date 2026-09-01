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
    """Materialize training rows once when requested."""
    if not preload:
        return None

    dense = interactions.toarray().astype(np.float32, copy=False)
    try:
        return torch.from_numpy(dense).to(device)
    except torch.OutOfMemoryError as error:
        raise torch.OutOfMemoryError(
            "the dense autoencoder training matrix does not fit on "
            f"{device}; set preload_training_data=False to stream CSR "
            "minibatches instead"
        ) from error


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
