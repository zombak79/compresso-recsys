"""Collaborative-filtering models."""

from .base import Recommender
from .ease import EASE, EASEConfig

__all__ = [
    "EASE",
    "EASEConfig",
    "Recommender",
]

