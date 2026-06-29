from __future__ import annotations

import compresso_recsys as cr
import compresso_recsys.checkpoint as checkpoint
import compresso_recsys.datasets as datasets


def test_top_level_public_api_is_intentional():
    expected = {
        "AmazonReviews2023",
        "build_recsys_checkpoint",
        "Goodbooks",
        "MovieLens1M",
        "MovieLens20M",
        "RecSysDataset",
        "SplitBundle",
        "load_cluster_graph_stage",
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
    }

    assert set(cr.__all__) == expected
    for name in expected:
        assert hasattr(cr, name)


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
            "SplitBundle",
            "RecSysDataset",
            "MovieLens1M",
            "MovieLens20M",
            "Goodbooks",
            "AmazonReviews2023",
        },
    }

    for module, expected in expected_by_module.items():
        assert set(module.__all__) == expected
        for name in expected:
            assert hasattr(module, name)
