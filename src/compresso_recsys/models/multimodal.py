"""Multimodal recommender base and catalogs, kept at their pre-0.4.0 import path.

The implementation lives in :mod:`compresso_recsys.models.core.multimodal`;
this module keeps ``from compresso_recsys.models.multimodal import ...`` working.
"""

from .core.multimodal import (
    BaseMultiModalRecommender,
    MultiModalCandidateCatalog,
    MultiModalCandidateSelection,
    MultiModalItemFeatures,
    MutableMultiModalCandidateCatalog,
)

__all__ = [
    "BaseMultiModalRecommender",
    "MultiModalCandidateCatalog",
    "MultiModalCandidateSelection",
    "MultiModalItemFeatures",
    "MutableMultiModalCandidateCatalog",
]
