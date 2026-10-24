"""Published settings for closed-form TEASER, as validated dataclasses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np



TEASERDataType = Literal["float32", "float64"]


@dataclass(frozen=True)
class TEASERConfig:
    """Configuration for the reference ADMM implementation of TEASER.

    Parameters
    ----------
    l2_coefficients:
        L2 regularization on the diagonal-free item coefficient matrix.
    l2_encoder:
        L2 regularization on the learned item-to-feature encoder.
    rho:
        Positive ADMM penalty parameter.
    max_iterations:
        Number of fixed ADMM iterations. The reference implementation uses 10.
    include_popularity:
        Append normalized training-item popularity as an additional feature.
    dtype:
        Numerical precision used by fitting and prediction. ``float64`` matches
        the reference implementation.
    """

    l2_coefficients: float = 0.05
    l2_encoder: float = 0.05
    rho: float = 0.05
    max_iterations: int = 10
    include_popularity: bool = False
    dtype: TEASERDataType = "float64"

    def __post_init__(self) -> None:
        for name in ("l2_coefficients", "l2_encoder", "rho"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be >= 1")
        if not isinstance(self.include_popularity, bool):
            raise ValueError("include_popularity must be a bool")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
