"""TEASER solved in closed form, over a fixed candidate catalog.

The gradient-descent sibling, which trades the exact solve for sampled outputs
and a growable catalog, lives in :mod:`compresso_recsys.models.teaser_gd`.
"""

from compresso_recsys.models.teaser.config import TEASERConfig, TEASERDataType
from compresso_recsys.models.teaser.model import TEASER

__all__ = [
    "TEASER",
    "TEASERConfig",
    "TEASERDataType",
]
