"""Variational autoencoder for multinomial implicit collaborative filtering."""

from compresso_recsys.models.mult_vae.config import (
    MultVAEConfig,
)
from compresso_recsys.models.mult_vae.model import (
    MultVAE,
)
from compresso_recsys.models.mult_vae.trainer import (
    MultVAETrainer,
)

__all__ = ["MultVAE", "MultVAEConfig", "MultVAETrainer"]
