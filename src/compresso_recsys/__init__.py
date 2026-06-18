"""Recommender-system companion package for Compresso."""

from .checkpoint import (
    COMPRESSED_ELSA_DIR,
    ELSA_DIR,
    SAE_DIR,
    SBERT_DIR,
    SBERT_SAE_DIR,
    load_recsys_split,
    read_checkpoint,
    save_recsys_split,
    update_checkpoint,
)
from .datasets import Goodbooks, MovieLens1M, MovieLens20M, RecSysDataset
from .models import CompressedELSA, TorchELSA, fit_compressed_elsa, fit_elsa, fit_sae_on_embeddings

__all__ = [
    "COMPRESSED_ELSA_DIR",
    "ELSA_DIR",
    "SAE_DIR",
    "SBERT_DIR",
    "SBERT_SAE_DIR",
    "CompressedELSA",
    "Goodbooks",
    "MovieLens1M",
    "MovieLens20M",
    "RecSysDataset",
    "TorchELSA",
    "fit_compressed_elsa",
    "fit_elsa",
    "fit_sae_on_embeddings",
    "load_recsys_split",
    "read_checkpoint",
    "save_recsys_split",
    "update_checkpoint",
]
