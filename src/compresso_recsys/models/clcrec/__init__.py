"""CLCRec: contrastive learning for cold-start recommendation (Wei et al., MM 2021).

Implemented from the paper; no third-party code consulted.
"""

from compresso_recsys.models.clcrec.config import CLCRecConfig
from compresso_recsys.models.clcrec.model import CLCRec
from compresso_recsys.models.clcrec.trainer import CLCRecTrainer

__all__ = ["CLCRec", "CLCRecConfig", "CLCRecTrainer"]
