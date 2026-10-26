"""Recommender-system companion package for Compresso.

Every public name is importable from the top level::

    import compresso_recsys as cr
    model = cr.EASE().fit(split["x_train"])
    result = cr.evaluate_recommender(model, source=..., targets=..., metrics=[cr.NDCG(100)])

Names are resolved lazily (PEP 562): ``import compresso_recsys`` loads no
submodule, and each name imports its own module on first use. A script that
only builds datasets therefore never loads the models, and one that only needs
``cr.read_checkpoint`` never loads the evaluator.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Public name -> submodule that defines it.
_EXPORTS: dict[str, str] = {}
for _module, _names in {
    "checkpoint": (
        "CheckpointFormatError",
        "load_cluster_graph_stage",
        "load_provenance",
        "load_json",
        "load_manifest",
        "load_recsys_split",
        "read_checkpoint",
        "save_cluster_graph_stage",
        "save_json",
        "save_manifest",
        "save_recsys_split",
        "update_checkpoint",
        "update_stage_manifest",
    ),
    "builder": ("build_recsys_checkpoint",),
    "datasets": (
        "AmazonReviews2023",
        "Goodbooks",
        "MovieLens1M",
        "MovieLens20M",
        "RecSysDataset",
        "SplitBundle",
        "Steam",
        "NetflixPrize",
        "TasteProfile",
        "Gowalla",
        "DBbook",
        "LastFM2K",
        "RetailRocket",
        "Music4AllOnion",
        "OTTO",
        "Yambda",
    ),
    "embeddings": ("save_item_embeddings", "load_item_embeddings", "list_item_embeddings"),
    "multimodal": ("enrich_multimodal_checkpoint",),
    "persistence": ("ModelCheckpointReader", "ModelCheckpointWriter"),
    "sequences": ("ItemSequences", "load_item_sequences", "save_item_sequences"),
    "evaluation": ("evaluate_recommender",),
    "stats": ("adjust_p_values", "compare_models", "compare_pair"),
    "training": ("EarlyStopping", "Validation"),
    "metrics": ("CalibratedRecall", "HitRate", "MAP", "MRR", "NDCG", "Precision", "Recall"),
    "models": (
        "PopularityBaseline",
        "PopularityBaselineConfig",
        "RandomBaseline",
        "RandomBaselineConfig",
        "EASE",
        "EASEConfig",
        "ItemKNNRecommender",
        "ItemKNNConfig",
        "UserKNNRecommender",
        "UserKNNConfig",
        "ELSATrainer",
        "ELSAConfig",
        "ELSACompressionConfig",
        "MultDAETrainer",
        "MultDAEConfig",
        "MultVAETrainer",
        "MultVAEConfig",
        "SASRecTrainer",
        "SASRecConfig",
        "Bert4RecTrainer",
        "Bert4RecConfig",
        "SimpleGPTTrainer",
        "SimpleGPTConfig",
        "SimpleRNNTrainer",
        "SimpleRNNConfig",
        "SimpleBidirectionalTransformerTrainer",
        "SimpleBidirectionalTransformerConfig",
        "ContentRecommender",
        "ContentRecommenderConfig",
        "TEASER",
        "TEASERConfig",
        "TEASERGDTrainer",
        "TEASERGDConfig",
        "CLCRecTrainer",
        "CLCRecConfig",
        "MMConcatWrapper",
        "MMConcatWrapperConfig",
    ),
}.items():
    for _name in _names:
        _EXPORTS[_name] = _module
del _module, _names, _name

# Submodules reachable as attributes (``cr.models``) without an explicit import.
_SUBMODULES = frozenset(
    {
        "builder",
        "checkpoint",
        "datasets",
        "embeddings",
        "evaluation",
        "metrics",
        "models",
        "multimodal",
        "persistence",
        "retrieval",
        "sequences",
        "stats",
        "training",
    }
)

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is not None:
        value = getattr(importlib.import_module(f".{module}", __name__), name)
        globals()[name] = value  # cache: later lookups skip __getattr__
        return value
    if name in _SUBMODULES:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)


if TYPE_CHECKING:  # pragma: no cover - static analysers and IDEs see the real names
    from .builder import build_recsys_checkpoint as build_recsys_checkpoint
    from .checkpoint import (
        CheckpointFormatError as CheckpointFormatError,
        load_cluster_graph_stage as load_cluster_graph_stage,
        load_provenance as load_provenance,
        load_json as load_json,
        load_manifest as load_manifest,
        load_recsys_split as load_recsys_split,
        read_checkpoint as read_checkpoint,
        save_cluster_graph_stage as save_cluster_graph_stage,
        save_json as save_json,
        save_manifest as save_manifest,
        save_recsys_split as save_recsys_split,
        update_checkpoint as update_checkpoint,
        update_stage_manifest as update_stage_manifest,
    )
    from .datasets import (
        AmazonReviews2023 as AmazonReviews2023,
        DBbook as DBbook,
        Goodbooks as Goodbooks,
        Gowalla as Gowalla,
        LastFM2K as LastFM2K,
        MovieLens1M as MovieLens1M,
        MovieLens20M as MovieLens20M,
        Music4AllOnion as Music4AllOnion,
        NetflixPrize as NetflixPrize,
        OTTO as OTTO,
        RecSysDataset as RecSysDataset,
        RetailRocket as RetailRocket,
        SplitBundle as SplitBundle,
        Steam as Steam,
        TasteProfile as TasteProfile,
        Yambda as Yambda,
    )
    from .embeddings import (
        list_item_embeddings as list_item_embeddings,
        load_item_embeddings as load_item_embeddings,
        save_item_embeddings as save_item_embeddings,
    )
    from .evaluation import evaluate_recommender as evaluate_recommender
    from .metrics import (
        MAP as MAP,
        MRR as MRR,
        NDCG as NDCG,
        CalibratedRecall as CalibratedRecall,
        HitRate as HitRate,
        Precision as Precision,
        Recall as Recall,
    )
    from .models import (
        EASE as EASE,
        TEASER as TEASER,
        Bert4RecConfig as Bert4RecConfig,
        Bert4RecTrainer as Bert4RecTrainer,
        CLCRecConfig as CLCRecConfig,
        CLCRecTrainer as CLCRecTrainer,
        ContentRecommender as ContentRecommender,
        ContentRecommenderConfig as ContentRecommenderConfig,
        EASEConfig as EASEConfig,
        ELSACompressionConfig as ELSACompressionConfig,
        ELSAConfig as ELSAConfig,
        ELSATrainer as ELSATrainer,
        ItemKNNConfig as ItemKNNConfig,
        ItemKNNRecommender as ItemKNNRecommender,
        MMConcatWrapper as MMConcatWrapper,
        MMConcatWrapperConfig as MMConcatWrapperConfig,
        MultDAEConfig as MultDAEConfig,
        MultDAETrainer as MultDAETrainer,
        MultVAEConfig as MultVAEConfig,
        MultVAETrainer as MultVAETrainer,
        PopularityBaseline as PopularityBaseline,
        PopularityBaselineConfig as PopularityBaselineConfig,
        RandomBaseline as RandomBaseline,
        RandomBaselineConfig as RandomBaselineConfig,
        SASRecConfig as SASRecConfig,
        SASRecTrainer as SASRecTrainer,
        SimpleBidirectionalTransformerConfig as SimpleBidirectionalTransformerConfig,
        SimpleBidirectionalTransformerTrainer as SimpleBidirectionalTransformerTrainer,
        SimpleGPTConfig as SimpleGPTConfig,
        SimpleGPTTrainer as SimpleGPTTrainer,
        SimpleRNNConfig as SimpleRNNConfig,
        SimpleRNNTrainer as SimpleRNNTrainer,
        TEASERConfig as TEASERConfig,
        TEASERGDConfig as TEASERGDConfig,
        TEASERGDTrainer as TEASERGDTrainer,
        UserKNNConfig as UserKNNConfig,
        UserKNNRecommender as UserKNNRecommender,
    )
    from .multimodal import enrich_multimodal_checkpoint as enrich_multimodal_checkpoint
    from .persistence import (
        ModelCheckpointReader as ModelCheckpointReader,
        ModelCheckpointWriter as ModelCheckpointWriter,
    )
    from .sequences import (
        ItemSequences as ItemSequences,
        load_item_sequences as load_item_sequences,
        save_item_sequences as save_item_sequences,
    )
    from .stats import (
        adjust_p_values as adjust_p_values,
        compare_models as compare_models,
        compare_pair as compare_pair,
    )
    from .training import EarlyStopping as EarlyStopping, Validation as Validation
