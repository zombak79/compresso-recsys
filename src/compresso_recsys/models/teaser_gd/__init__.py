"""teaser_gd"""

from compresso_recsys.models.teaser_gd.config import (
    EncoderInit,
    OptimizerName,
    TEASERGDConfig,
    TEASERGDLoss,
)
from compresso_recsys.models.teaser_gd.model import (
    TEASERGD,
)
from compresso_recsys.models.teaser_gd.trainer import (
    TEASERGDTrainer,
)


# Private helpers too: the flat module exposed every top-level name,
# and tests reach for some of them by module path.
from compresso_recsys.models.teaser_gd.model import (  # noqa: F401
    _score_feature_rows,
)
from compresso_recsys.models.teaser_gd.trainer import (  # noqa: F401
    _dense_feature_rows,
    _feature_tensor,
    _initialize_encoder_from_features,
    _teaser_reconstruction_loss,
)

__all__ = ["TEASERGD", "TEASERGDConfig", "TEASERGDTrainer"]
