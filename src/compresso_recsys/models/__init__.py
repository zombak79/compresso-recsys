"""Collaborative-filtering models."""

from .base import Recommender
from .ease import EASE, EASEConfig
from .elsa import (
    CompressedELSA,
    ELSA,
    ELSACompressionConfig,
    ELSAConfig,
    ELSATrainer,
)

__all__ = [
    "CompressedELSA",
    "EASE",
    "EASEConfig",
    "ELSA",
    "ELSACompressionConfig",
    "ELSAConfig",
    "ELSATrainer",
    "Recommender",
]
