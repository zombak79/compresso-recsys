"""Cold-start bases and catalogs, kept at their pre-0.4.0 import path.

The implementations moved to :mod:`compresso_recsys.models.core` (``cold_start``,
``catalog``, ``mutable_catalog``, ``adapters``, ``identifiers``); this module
keeps ``from compresso_recsys.models.cold_start import ...`` working.
"""

from .core.adapters import WarmCatalogAdapter
from .core.catalog import CandidateCatalog
from .core.cold_start import BaseColdStartRecommender, ColdStartRecommender
from .core.identifiers import ItemVocabulary
from .core.mutable_catalog import MutableCandidateCatalog

__all__ = [
    "BaseColdStartRecommender",
    "CandidateCatalog",
    "ColdStartRecommender",
    "ItemVocabulary",
    "MutableCandidateCatalog",
    "WarmCatalogAdapter",
]
