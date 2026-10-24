"""Opt-in cold-start infrastructure for named item-feature matrices.

The catalog preserves modalities; it does not choose a fusion rule or run an
encoder. Models may override registration and persistence for other stored
representations without changing the single-matrix cold-start family.

Split by role, mirroring the single-matrix family one level up: the feature
canonicalisation, the catalog that holds it, and the recommender contract.
"""

from compresso_recsys.models.core.multimodal.features import MultiModalItemFeatures
from compresso_recsys.models.core.multimodal.catalog import (
    MultiModalCandidateCatalog,
    MultiModalCandidateSelection,
    MutableMultiModalCandidateCatalog,
)
from compresso_recsys.models.core.multimodal.base import BaseMultiModalRecommender

__all__ = [
    "BaseMultiModalRecommender",
    "MultiModalCandidateCatalog",
    "MultiModalCandidateSelection",
    "MultiModalItemFeatures",
    "MutableMultiModalCandidateCatalog",
]
