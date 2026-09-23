"""Turn matrix-based cold-start models into multimodal models by concatenation."""

from compresso_recsys.models.mm_concat.config import MMConcatWrapperConfig
from compresso_recsys.models.mm_concat.model import MMConcatWrapper

__all__ = [
    "MMConcatWrapper",
    "MMConcatWrapperConfig",
]
