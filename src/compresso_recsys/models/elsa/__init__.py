"""Scalable linear shallow autoencoder with normalized item embeddings."""

from compresso_recsys.models.elsa.config import (
    CompressionScoreMode,
    ELSACompressionConfig,
    ELSAConfig,
    OptimizerName,
    SparseFinetuneBackend,
    SparseInferenceBackend,
)
from compresso_recsys.models.elsa.model import (
    CompressedELSA,
    ELSA,
)
from compresso_recsys.models.elsa.trainer import (
    ELSATrainer,
)


# Private helpers too: the flat module exposed every top-level name,
# and tests reach for some of them by module path.
from compresso_recsys.models.elsa.config import (  # noqa: F401
    _dense_training_target,
    _normalized_mse,
)
from compresso_recsys.models.elsa.model import (  # noqa: F401
    _normalize_srp,
    _score_candidates,
    _score_sparse_candidates,
    _srp_to_coo,
)
from compresso_recsys.models.elsa.trainer import (  # noqa: F401
    _ELSAInteractionDataset,
)

__all__ = [
    "CompressedELSA",
    "ELSA",
    "ELSACompressionConfig",
    "ELSAConfig",
    "ELSATrainer",
]
