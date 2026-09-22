"""The contracts a recommender implements, with no implementation behind them.

Each is a runtime-checkable Protocol describing one capability: being saveable,
carrying identifiers, scoring a matrix, or scoring a history. A model satisfies
them structurally, so nothing here has to be inherited to be honoured.
"""

from __future__ import annotations


from pathlib import Path
from typing import (
    Any,
    ClassVar,
    Hashable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    TypeVar,
    runtime_checkable,
)

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models.core.identifiers import ItemVocabulary, Recommendations


@runtime_checkable
class PersistableRecommender(Protocol):
    """A fitted recommender with the package model-checkpoint API."""

    def to(
        self,
        device: str | torch.device,
    ) -> "PersistableRecommender": ...

    def save(
        self,
        path: str | Path,
        *,
        include_optimizer: bool = False,
    ) -> None: ...

    def save_to_checkpoint(
        self,
        checkpoint_path: str | Path,
        name: str,
        *,
        include_optimizer: bool = False,
    ) -> None: ...

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> "PersistableRecommender": ...

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        name: str,
        *,
        device: str | torch.device = "cpu",
        load_optimizer: bool = False,
    ) -> "PersistableRecommender": ...


@runtime_checkable
class IdentifiedRecommender(Protocol):
    """A recommender accepting histories and filters as stable item IDs."""

    def recommend(
        self,
        histories: Sequence[Sequence[Hashable]],
        *,
        k: int = 100,
        exclude_seen: bool = False,
        allowlist: Sequence[Hashable] | np.ndarray | None = None,
        blocklist: Sequence[Hashable] | np.ndarray | None = None,
        on_insufficient: Literal["truncate", "raise"] = "truncate",
    ) -> Recommendations: ...


@runtime_checkable
class Recommender(Protocol):
    """A fitted recommender that produces ranked predictions for one batch."""

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return top-``k`` predictions, optionally excluding source items."""


@runtime_checkable
class SequentialRecommender(Protocol):
    """A fitted recommender that ranks from chronological histories.

    The same contract as :class:`Recommender` with a different source type. Kept
    structural, like its sibling, so a model satisfies it by having the method
    rather than by inheriting anything.
    """

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return top-``k`` predictions, optionally excluding source items."""
