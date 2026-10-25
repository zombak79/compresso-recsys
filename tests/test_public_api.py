from __future__ import annotations

import compresso_recsys as cr
import compresso_recsys.checkpoint as checkpoint
import compresso_recsys.datasets as datasets
import compresso_recsys.evaluation as evaluation
import compresso_recsys.metrics as metrics
import compresso_recsys.models as models
import compresso_recsys.models.batching as batching
import compresso_recsys.persistence as persistence
import compresso_recsys.stats as stats
import compresso_recsys.training as training


def test_top_level_public_api_is_intentional():
    expected = {
        "DBbook",
        "LastFM2K",
        "save_item_embeddings",
        "load_item_embeddings",
        "list_item_embeddings",
        "enrich_multimodal_checkpoint",
        "Steam",
        "NetflixPrize",
        "TasteProfile",
        "Gowalla",
        "RetailRocket",
        "Music4AllOnion",
        "OTTO",
        "Yambda",
        "AmazonReviews2023",
        "build_recsys_checkpoint",
        "Goodbooks",
        "ItemSequences",
        "MovieLens1M",
        "MovieLens20M",
        "ModelCheckpointReader",
        "ModelCheckpointWriter",
        "RecSysDataset",
        "SplitBundle",
        "CheckpointFormatError",
        "load_cluster_graph_stage",
        "load_provenance",
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
        # 0.4.0: the evaluation, comparison and model entry points, so a script
        # needs one import.
        "evaluate_recommender",
        "adjust_p_values",
        "compare_models",
        "compare_pair",
        "EarlyStopping",
        "Validation",
        "CalibratedRecall", "HitRate", "MAP", "MRR", "NDCG", "Precision", "Recall",
        "PopularityBaseline", "PopularityBaselineConfig", "RandomBaseline", "RandomBaselineConfig",
        "EASE", "EASEConfig", "ItemKNNRecommender", "ItemKNNConfig",
        "UserKNNRecommender", "UserKNNConfig",
        "ELSATrainer", "ELSAConfig", "ELSACompressionConfig",
        "MultDAETrainer", "MultDAEConfig", "MultVAETrainer", "MultVAEConfig",
        "SASRecTrainer", "SASRecConfig", "Bert4RecTrainer", "Bert4RecConfig",
        "SimpleGPTTrainer", "SimpleGPTConfig", "SimpleRNNTrainer", "SimpleRNNConfig",
        "SimpleBidirectionalTransformerTrainer", "SimpleBidirectionalTransformerConfig",
        "ContentRecommender", "ContentRecommenderConfig", "TEASER", "TEASERConfig",
        "TEASERGDTrainer", "TEASERGDConfig", "CLCRecTrainer", "CLCRecConfig",
        "MMConcatWrapper", "MMConcatWrapperConfig",
    }

    assert set(cr.__all__) == expected
    for name in expected:
        assert hasattr(cr, name)


def test_top_level_names_are_the_submodule_objects():
    assert cr.EASE is models.EASE
    assert cr.compare_models is stats.compare_models
    assert cr.evaluate_recommender is evaluation.evaluate_recommender
    assert cr.NDCG is metrics.NDCG
    assert cr.read_checkpoint is checkpoint.read_checkpoint
    assert cr.models is models  # submodules are reachable as attributes too
    assert set(cr.__all__) <= set(dir(cr))


def test_unknown_top_level_names_raise_attribute_error():
    import pytest

    with pytest.raises(AttributeError, match="has no attribute 'no_such_name'"):
        cr.no_such_name  # noqa: B018


def test_importing_the_package_loads_no_submodule():
    import subprocess
    import sys

    code = (
        "import sys, compresso_recsys as cr\n"
        "loaded = sorted(m for m in sys.modules if m.startswith('compresso_recsys.'))\n"
        "print(loaded, 'torch' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[] False"


def test_model_and_retrieval_helpers_are_not_top_level_exports():
    hidden = {
        "CLUSTERING_DIR",
        "COMPRESSED_ELSA_DIR",
        "CompressedELSA",
        "ELSA_DIR",
        "SAE_DIR",
        "SBERT_DIR",
        "SBERT_SAE_DIR",
        "TorchELSA",
        "build_eval_holdout",
        "build_item_cold_holdout",
        "build_leave_last_out_holdout",
        "build_temporal_holdout",
        "evaluate_item_embeddings",
        "evaluate_item_embeddings_with_holdout",
        "fit_compressed_elsa",
        "fit_elsa",
        "fit_sae_on_embeddings",
        "hf_to_torch_dataloader",
    }

    for name in hidden:
        assert not hasattr(cr, name)


def test_submodule_public_apis_are_intentional():
    expected_by_module = {
        checkpoint: {
            "CheckpointFormatError",
            "load_provenance",
            "update_checkpoint",
            "read_checkpoint",
            "load_manifest",
            "save_manifest",
            "update_stage_manifest",
            "save_json",
            "load_json",
            "save_recsys_split",
            "load_recsys_split",
            "save_cluster_graph_stage",
            "load_cluster_graph_stage",
        },
        datasets: {
            "DBbook",
            "LastFM2K",
            "Steam",
            "NetflixPrize",
            "TasteProfile",
            "Gowalla",
            "RetailRocket",
            "Music4AllOnion",
            "OTTO",
            "Yambda",
            "SplitBundle",
            "RecSysDataset",
            "MovieLens1M",
            "MovieLens20M",
            "Goodbooks",
            "AmazonReviews2023",
        },
        evaluation: {
            "EvaluationResult",
            "RankingEvaluator",
            "evaluate_ranked_predictions",
            "evaluate_recommender",
        },
        stats: {
            "ComparisonReport",
            "PairwiseComparison",
            "adjust_p_values",
            "compare_models",
            "compare_pair",
        },
        training: {"EarlyStopping", "Validation"},
        metrics: {
            "CalibratedRecall",
            "HitRate",
            "MAP",
            "MRR",
            "MeanAtCutoffsMetric",
            "NDCG",
            "Precision",
            "Recall",
            "RankingBatch",
            "RankingMetric",
        },
        batching: {
            "InteractionBatch",
            "InteractionBatchSampler",
            "dense_training_target",
            "normalized_mse",
        },
        persistence: {
            "MODEL_CHECKPOINT_FORMAT",
            "MODEL_CHECKPOINT_VERSION",
            "ModelCheckpointReader",
            "ModelCheckpointWriter",
        },
        models: {
            "BaseColdStartRecommender",
            "BaseMultiModalRecommender",
            "BaseCollaborativeRecommender",
            "BaseIdentifiedRecommender",
            "BasePersistableRecommender",
            "BaseSequentialRecommender",
            "Bert4Rec",
            "Bert4RecConfig",
            "Bert4RecTrainer",
            "CandidateCatalog",
            "ColdStartRecommender",
            "CompressedELSA",
            "ContentRecommender",
            "ContentRecommenderConfig",
            "EASE",
            "EASEConfig",
            "ELSA",
            "ELSACompressionConfig",
            "ELSAConfig",
            "ELSATrainer",
            "ItemTokenizer",
            "ItemVocabulary",
            "IdentifiedRecommender",
            "MutableCandidateCatalog",
            "MMConcatWrapper",
            "MMConcatWrapperConfig",
            "MultiModalCandidateCatalog",
            "MultiModalCandidateSelection",
            "MultiModalItemFeatures",
            "MutableMultiModalCandidateCatalog",
            "InteractionBatch",
            "InteractionBatchSampler",
            "CLCRec",
            "CLCRecConfig",
            "CLCRecTrainer",
            "ItemKNNConfig",
            "ItemKNNRecommender",
            "dense_training_target",
            "MultDAE",
            "MultDAEConfig",
            "MultDAETrainer",
            "MultVAE",
            "MultVAEConfig",
            "MultVAETrainer",
            "PopularityBaseline",
            "PopularityBaselineConfig",
            "PersistableRecommender",
            "RandomBaseline",
            "RandomBaselineConfig",
            "UserKNNConfig",
            "UserKNNRecommender",
            "Recommender",
            "Recommendations",
            "SASRec",
            "SASRecConfig",
            "SASRecTrainer",
            "SequenceBatcher",
            "SequentialRecommender",
            "SimpleBidirectionalTransformer",
            "SimpleBidirectionalTransformerConfig",
            "SimpleBidirectionalTransformerTrainer",
            "SimpleGPT",
            "SimpleGPTConfig",
            "SimpleGPTTrainer",
            "SimpleRNN",
            "SimpleRNNConfig",
            "SimpleRNNTrainer",
            "Tokenizer",
            "TransformerConfig",
            "TEASER",
            "TEASERConfig",
            "TEASERGD",
            "TEASERGDConfig",
            "TEASERGDTrainer",
            "WarmCatalogAdapter",
        },
    }

    for module, expected in expected_by_module.items():
        assert set(module.__all__) == expected
        for name in expected:
            assert hasattr(module, name)
