"""The recommender contract for models that consume named feature matrices."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Sequence
from typing import Literal

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.models.core.catalog import (
    CandidateConflict,
)
from compresso_recsys.models.core.cold_start import BaseColdStartRecommender
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
)
from compresso_recsys.models.core.multimodal.features import MultiModalItemFeatures
from compresso_recsys.models.core.multimodal.catalog import (
    MultiModalCandidateCatalog,
    MutableMultiModalCandidateCatalog,
)


class BaseMultiModalRecommender(BaseColdStartRecommender):
    """Optional CSR-history base with a default named-matrix candidate catalog.

    Implement ``fit``, ``is_fitted`` and ``predict_on_batch``. Fitting installs
    the source IDs and stored matrices with ``self.candidates.install(...)``.
    Recommendation, history alignment and prediction batching are inherited.

    Initial installation, rebuilding and updates must produce the same stored
    representation using the same fitted transformation. Override candidate
    methods if a model must encode incoming matrices first. Other storage and
    scoring designs are equally valid; no fusion rule is imposed by this base.

    The default persistence hooks save/load the catalog. Overrides saving model
    state should call ``super()`` as well, or manage catalog persistence themselves.
    The model still supplies its checkpoint construction/configuration hooks.

    Mapping-specific overrides intentionally specialize the existing runtime
    interface; the matrix base and its annotations are not made generic here.
    """

    candidates: MutableMultiModalCandidateCatalog  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.candidates = MutableMultiModalCandidateCatalog(
            on_publish=self._on_catalog_published
        )

    @abstractmethod
    def fit(  # type: ignore[override]
        self,
        interactions: csr_matrix,
        item_features: MultiModalItemFeatures,
        **kwargs,
    ) -> BaseMultiModalRecommender:
        """Fit the model and install its initial multimodal candidate catalog."""

    def _on_catalog_published(self, catalog: MultiModalCandidateCatalog) -> None:  # type: ignore[override]
        """Invalidate caches under the catalog lock; raising does not roll back.

        Do not wait for another thread that reads or mutates this catalog: it
        needs the same lock. Workers can instead use the immutable ``catalog``
        argument directly without reacquiring the owner's lock.
        """

    def build_candidates(  # type: ignore[override]
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> MultiModalCandidateCatalog:
        """Replace candidates using already prepared stored representations."""
        return self.candidates.build(
            item_ids=item_ids,
            item_features=item_features,
            metadata=metadata,
            feature_space_id=feature_space_id,
        )

    def update_candidates(  # type: ignore[override]
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: MultiModalItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> MultiModalCandidateCatalog:
        """Append or replace candidates using already prepared representations."""
        return self.candidates.update(
            item_ids=item_ids,
            item_features=item_features,
            metadata=metadata,
            on_conflict=on_conflict,
            feature_space_id=feature_space_id,
        )

    def remove_candidates(  # type: ignore[override]
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> MultiModalCandidateCatalog:
        return self.candidates.remove(item_ids, missing=missing)

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        self.candidates._save_checkpoint(writer)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        self.candidates._load_checkpoint(reader)
