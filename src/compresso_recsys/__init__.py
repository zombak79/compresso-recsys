"""Recommender-system companion package for Compresso."""

from .checkpoint import (
    load_cluster_graph_stage,
    load_json,
    load_manifest,
    load_recsys_split,
    read_checkpoint,
    save_cluster_graph_stage,
    save_json,
    save_manifest,
    save_recsys_split,
    update_checkpoint,
    update_stage_manifest,
)
from .builder import build_recsys_checkpoint
from .datasets import AmazonReviews2023, Goodbooks, MovieLens1M, MovieLens20M, RecSysDataset, SplitBundle
from .datasets import Steam, NetflixPrize, TasteProfile, Gowalla
from .datasets import DBbook, LastFM2K
from .datasets import RetailRocket, Music4AllOnion, OTTO, Yambda
from .embeddings import save_item_embeddings, load_item_embeddings, list_item_embeddings
from .multimodal import enrich_multimodal_checkpoint
from .persistence import ModelCheckpointReader, ModelCheckpointWriter
from .sequences import ItemSequences, load_item_sequences, save_item_sequences

__all__ = [
    "DBbook",
    "LastFM2K",
    "save_item_embeddings",
    "load_item_embeddings",
    "list_item_embeddings",
    "enrich_multimodal_checkpoint",
    "AmazonReviews2023",
    "Steam",
    "NetflixPrize",
    "TasteProfile",
    "Gowalla",
    "build_recsys_checkpoint",
    "Goodbooks",
    "RetailRocket",
    "Music4AllOnion",
    "OTTO",
    "Yambda",
    "ItemSequences",
    "MovieLens1M",
    "MovieLens20M",
    "ModelCheckpointReader",
    "ModelCheckpointWriter",
    "RecSysDataset",
    "SplitBundle",
    "load_cluster_graph_stage",
    "load_item_sequences",
    "load_json",
    "load_manifest",
    "load_recsys_split",
    "read_checkpoint",
    "save_cluster_graph_stage",
    "save_item_sequences",
    "save_json",
    "save_manifest",
    "save_recsys_split",
    "update_checkpoint",
    "update_stage_manifest",
]
