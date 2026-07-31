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
from .teaser import CandidateCatalog, TEASER, TEASERConfig

__all__ = [
    "CompressedELSA",
    "CandidateCatalog",
    "EASE",
    "EASEConfig",
    "ELSA",
    "ELSACompressionConfig",
    "ELSAConfig",
    "ELSATrainer",
    "Recommender",
    "TEASER",
    "TEASERConfig",
]
