"""TEASER fitted by gradient descent, over cold candidate catalogs.

The closed-form sibling lives in :mod:`compresso_recsys.models.teaser`.
"""

from compresso_recsys.models.teaser_gd.config import (
    EncoderInit as EncoderInit,
    OptimizerName as OptimizerName,
    TEASERGDConfig,
    TEASERGDLoss as TEASERGDLoss,
)
from compresso_recsys.models.teaser_gd.model import (
    TEASERGD,
)
from compresso_recsys.models.teaser_gd.trainer import (
    TEASERGDTrainer,
)


__all__ = ["TEASERGD", "TEASERGDConfig", "TEASERGDTrainer"]
