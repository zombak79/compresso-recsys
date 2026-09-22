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

    try:
        # Cast while the matrix is still sparse. Densifying float64 first would
        # temporarily require both the float64 and float32 dense matrices.
        dense = interactions.astype(np.float32, copy=False).toarray()
    except MemoryError as error:
        raise MemoryError(
            "the dense autoencoder training matrix does not fit in host "
            "memory; set preload_training_data=False to stream CSR "
            "minibatches instead"
        ) from error

    device_oom_message = (
        "the dense autoencoder training matrix does not fit on "
        f"{device}; set preload_training_data=False to stream CSR "
        "minibatches instead"
    )
    try:
        return torch.from_numpy(dense).to(device)
    except torch.OutOfMemoryError as error:
        raise torch.OutOfMemoryError(
            device_oom_message
        ) from error
    except RuntimeError as error:
        # MPS currently reports allocator exhaustion as a plain RuntimeError.
        # Do not disguise unrelated backend failures as memory errors.
        if device.type != "mps" or "out of memory" not in str(error).lower():
            raise
        raise torch.OutOfMemoryError(device_oom_message) from error


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
    dense = interactions[selected].astype(np.float32, copy=False).toarray()
    return torch.from_numpy(dense).to(device)
