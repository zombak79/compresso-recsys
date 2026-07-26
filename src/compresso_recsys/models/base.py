from __future__ import annotations

from typing import Protocol, runtime_checkable

from scipy.sparse import csr_matrix

from compresso import SRPTensor

__all__ = ["Recommender"]


@runtime_checkable
class Recommender(Protocol):
    """A fitted recommender that produces ranked predictions for one batch."""

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
    ) -> SRPTensor:
        """Return top-``k`` ranked item predictions for ``source``."""

