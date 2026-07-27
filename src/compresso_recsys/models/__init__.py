"""Collaborative-filtering models."""

from .base import Recommender
from .ease import EASE, EASEConfig
from .elsa import ELSA, ELSAConfig, ELSATrainer

__all__ = [
    "EASE",
    "EASEConfig",
    "ELSA",
    "ELSAConfig",
    "ELSATrainer",
    "Recommender",
]
