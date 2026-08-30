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
from .sequences import ItemSequences, load_item_sequences, save_item_sequences

__all__ = [
    "AmazonReviews2023",
    "build_recsys_checkpoint",
    "Goodbooks",
    "ItemSequences",
    "MovieLens1M",
    "MovieLens20M",
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
