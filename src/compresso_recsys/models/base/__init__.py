"""Recommender contracts and the base classes implementing them.

Formerly one module; split by role, with the same names exported. The protocols
carry no implementation, and the three bases form a chain -- identified, then
persistable, then either matrix-shaped or sequence-shaped.
"""

from compresso_recsys.models.base.protocols import (
    IdentifiedRecommender,
    PersistableRecommender,
    Recommender,
    SequentialRecommender,
)
# The private helpers are re-exported because callers reached them through
# the flat module: cold_start asks for _accepts_reporting_keywords.
from compresso_recsys.models.base.identified import (
    BaseIdentifiedRecommender,
    _MODEL_NAME,
    _MODELS_DIR,
    _accepts_reporting_keywords,
    _embedded_model_path,
    _unwrapped_module,
)
from compresso_recsys.models.base.persistable import (
    BasePersistableRecommender,
    _PersistableT,
)
from compresso_recsys.models.base.collaborative import BaseCollaborativeRecommender
from compresso_recsys.models.base.sequential import BaseSequentialRecommender

__all__ = [
    "BasePersistableRecommender",
    "BaseIdentifiedRecommender",
    "BaseCollaborativeRecommender",
    "BaseSequentialRecommender",
    "IdentifiedRecommender",
    "PersistableRecommender",
    "Recommender",
    "SequentialRecommender",
]
