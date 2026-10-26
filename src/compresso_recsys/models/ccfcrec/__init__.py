"""CCFCRec: contrastive collaborative filtering for cold-start items (Zhou et al., WWW 2023).

Implemented from the paper; no third-party code consulted.
"""

from compresso_recsys.models.ccfcrec.config import CCFCRecConfig
from compresso_recsys.models.ccfcrec.model import CCFCRec
from compresso_recsys.models.ccfcrec.trainer import CCFCRecTrainer

__all__ = ["CCFCRec", "CCFCRecConfig", "CCFCRecTrainer"]
