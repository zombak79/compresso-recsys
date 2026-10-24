"""Public item identifiers, kept at their pre-0.4.0 import path.

The implementation lives in :mod:`compresso_recsys.models.core.identifiers`;
this module keeps ``from compresso_recsys.models.identifiers import ...`` working.
"""

from .core.identifiers import ItemVocabulary, Recommendations

__all__ = [
    "ItemVocabulary",
    "Recommendations",
]
