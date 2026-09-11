"""Evaluation-detail counts use distinct targets and training observations."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def details(monkeypatch):
    directory = Path(__file__).parents[1] / "examples/validation"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("amazon_evaluation_details", directory / "amazon_evaluation_details.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_counts_distinct_targets_and_users_in_phase_vocabularies(details):
    payload = {
        "item_ids": np.array(["warm", "zero", "new", "cold"]),
        "train_item_ids": np.array(["warm", "zero", "new"]),
        # A catalog column with an explicit zero is still cold.
        "x_train": csr_matrix(([2., 0.], [0, 1], [0, 2, 2]), shape=(2, 3)),
        "val_item_ids": np.array(["cold", "warm", "zero"]),
        "test_item_ids": np.array(["zero", "new", "warm"]),
        "val_holdout": {"user_ids": ["u1", "u1", "u2", "u3"],
                        "source_indices": [[1], [1], [1], []],
                        "target_indices": [[0, 0, 1], [0], [2], [0]]},
        "test_holdout": {"user_ids": ["u9", "u9", "u10"],
                         "source_indices": [[2], [2], [2]],
                         "target_indices": [[1, 2], [1], []]},
    }
    result = details.evaluation_item_stats(payload)
    assert result["val_users"] == 2
    assert result["val_target_items"] == 3
    assert result["val_cold_target_items"] == 2
    assert result["val_cold_target_item_fraction"] == pytest.approx(2 / 3)
    assert result["test_users"] == 1
    assert result["test_target_items"] == 2
    assert result["test_cold_target_item_fraction"] == .5
    assert result["val_catalog_items"] == result["test_catalog_items"] == 3


def test_empty_targets_have_no_defined_cold_fraction(details):
    payload = {"item_ids": ["i"], "x_train": csr_matrix([[1.]]),
               **{f"{phase}_holdout": {"user_ids": ["u"], "source_indices": [[0]], "target_indices": [[]]}
                  for phase in ("val", "test")}}
    result = details.evaluation_item_stats(payload)
    assert result["val_users"] == result["val_target_items"] == result["val_cold_target_items"] == 0
    assert result["val_cold_target_item_fraction"] is None
    payload["val_holdout"]["target_indices"] = [[1]]
    with pytest.raises(ValueError, match="outside phase vocabulary"):
        details.evaluation_item_stats(payload)


@pytest.mark.parametrize("split", ["user_split", "item_split", "leave_last_out", "temporal"])
def test_details_match_real_builder_users_and_cold_protocols(details, split):
    rng = np.random.default_rng(13)
    frame = pd.DataFrame([
        (f"u{u}", f"i{i}", 1., 1600000000000 + rng.integers(0, 100) * 3600000)
        for u in range(45) for i in rng.choice(24, 15, replace=False)
    ], columns=["user_id", "item_id", "value", "timestamp"])
    args, _ = details.profile.builder._resolve_args(details.profile.builder._build_args(
        dataset="amazon2023", split_mode=split, min_user_support=4, item_min_support=1,
        temporal_period_hours=20, val_users=5, test_users=5, eval_draws=1))
    payload = details.profile.builder._build_split_payload(args, details.profile.RecSysDataset(), frame)
    original = details.profile.payload_stats(payload)
    result = details.evaluation_item_stats(payload)
    for phase in ("val", "test"):
        assert result[f"{phase}_users"] == original[f"{phase}_users"]
        assert 0 < result[f"{phase}_target_items"] <= result[f"{phase}_catalog_items"]
        assert 0 <= result[f"{phase}_cold_target_item_fraction"] <= 1
        if split in ("user_split", "item_split"):
            assert result[f"{phase}_cold_target_item_fraction"] == (1. if split == "item_split" else 0.)


@pytest.mark.parametrize("category", ["All_Beauty", "Books"])
@pytest.mark.parametrize("explicit_limits", [False, True])
def test_selected_profile_rebuild_preserves_graph_and_resumes(details, tmp_path, monkeypatch, category, explicit_limits):
    frame = pd.DataFrame([(f"u{u}", f"i{i}", 1., i) for u in range(12) for i in range(6)],
                         columns=["user_id", "item_id", "value", "timestamp"])
    directory = tmp_path / "inputs" / category
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "input.parquet", index=False)
    manifest = details.profile.seal_prepared(directory, category, {
        "schema": details.profile.SCHEMA, "all_ratings": True, "min_entity_text_words": 0,
        "min_user_support_upper_bound": 2, "sources": []}, len(frame))
    parameters = dict(min_user_support=2, item_min_support=1, val_users=2, test_users=2,
                      seed=42, min_entity_text_words=0, set_all_values_to=1., temporal_period_hours=8136)
    args, _ = details.profile.builder._resolve_args(details.profile.builder._build_args(
        dataset="amazon2023", amazon_category=category, split_mode="user_split", **parameters))
    payload = details.profile.builder._build_split_payload(args, details.profile.RecSysDataset(), frame)
    stats = details.profile.payload_stats(payload, preprocessed={"users": 12, "items": 6, "pairs": 72})
    record = {"category": category, "input": {"prepared_sha256": manifest["prepared"]["sha256"]},
              "splits": {"user_split": {"parameters": parameters, "stats": stats, "warnings": []}}}
    if explicit_limits:
        monkeypatch.setattr(details.profile, "limits", lambda category: (11, 5))
        record["splits"]["user_split"]["size_limits"] = {"max_users": 12, "max_items": 6}
    output = tmp_path / "output"
    details.run_category(record, tmp_path / "inputs", output, 3, details.profile.RUN_CODE)
    result = json.loads((output / category / "result.json").read_text())["splits"]["user_split"]
    heldout = 3 if category == "Books" else 2
    assert result["stats"]["val_users"] == result["stats"]["test_users"] == heldout
    assert result["stats"]["preprocessed_users"] == 12
    assert result["stats"]["preprocessed_items"] == 6
    assert result["stats"]["preprocessed_pairs"] == 72
    assert result["stats"]["train_users"] == 12 - 2 * heldout
    assert result["stats"]["val_cold_target_item_fraction"] == 0
    if explicit_limits:
        assert result["size_limits"] == {"max_users": 12, "max_items": 6}
        too_small = {**record, "splits": {"user_split": {
            **record["splits"]["user_split"], "size_limits": {"max_users": 11, "max_items": 6}}}}
        with pytest.raises(AssertionError, match="exceeds category caps"):
            details.run_category(too_small, tmp_path / "inputs", tmp_path / "too-small", 3, details.profile.RUN_CODE)
    # An archive already enriched with item counts must remain valid input.
    enriched = {**record, "splits": {"user_split": result}}
    repeat_output = tmp_path / "enriched-output"
    details.run_category(enriched, tmp_path / "inputs", repeat_output, 3, details.profile.RUN_CODE)
    repeated = json.loads((repeat_output / category / "result.json").read_text())["splits"]["user_split"]
    assert repeated["stats"] == result["stats"]
    assert repeated["changed_holdout"] is False
    monkeypatch.setattr(details.profile, "load_prepared", lambda *a, **k: pytest.fail("Rebuilt completed profile"))
    assert details.run_category(record, tmp_path / "inputs", output, 3, details.profile.RUN_CODE) == category
    with pytest.raises(ValueError, match="different code, input or parameters"):
        details.run_category(record, tmp_path / "inputs", output, 4, details.profile.RUN_CODE)


@pytest.mark.parametrize("limits", [None, {}, {"max_users": 12},
    {"max_users": 12, "max_items": 0}, {"max_users": True, "max_items": 6},
    {"max_users": 12.5, "max_items": 6}, {"max_users": 12, "max_items": 6, "extra": 1}])
def test_invalid_profile_limit_exceptions_are_rejected(details, limits):
    with pytest.raises(ValueError, match="size_limits requires"):
        details.profile_limits("Clothing_Shoes_and_Jewelry", {"size_limits": limits})


def test_profile_limits_are_scoped_to_one_entry(details):
    assert details.profile_limits("Clothing_Shoes_and_Jewelry", {}) == (100000, 20000)
    assert details.profile_limits("Clothing_Shoes_and_Jewelry", {
        "size_limits": {"max_users": 120000, "max_items": 25000}}) == (120000, 25000)
    assert details.profile_limits("Clothing_Shoes_and_Jewelry", {}) == (100000, 20000)


def test_temporal_only_rebuild_uses_stage_filtering_without_global_core(details, tmp_path, monkeypatch):
    category = "Clothing_Shoes_and_Jewelry"
    rng = np.random.default_rng(13)
    frame = pd.DataFrame([
        (f"u{u}", f"i{i}", 1., 1600000000000 + rng.integers(0, 100) * 3600000)
        for u in range(45) for i in rng.choice(24, 15, replace=False)
    ], columns=["user_id", "item_id", "value", "timestamp"])
    directory = tmp_path / "inputs" / category
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "input.parquet", index=False)
    manifest = details.profile.seal_prepared(directory, category, {
        "schema": details.profile.SCHEMA, "all_ratings": True, "min_entity_text_words": 0,
        "min_user_support_upper_bound": 2, "sources": []}, len(frame))
    parameters = dict(min_user_support=4, item_min_support=1, seed=42,
                      min_entity_text_words=0, set_all_values_to=1., temporal_period_hours=20)
    args, _ = details.profile.builder._resolve_args(details.profile.builder._build_args(
        dataset="amazon2023", amazon_category=category, split_mode="temporal", **parameters))
    payload = details.profile.builder._build_split_payload(args, details.profile.RecSysDataset(), frame)
    stats = details.profile.payload_stats(payload)
    record = {"category": category, "input": {"prepared_sha256": manifest["prepared"]["sha256"]},
              "splits": {"temporal": {"parameters": parameters, "stats": stats, "warnings": [],
                                     "size_limits": {"max_users": 45, "max_items": 24}}}}
    monkeypatch.setattr(details.profile, "CoreGraph", lambda *a: pytest.fail("Global filtering is not temporal"))
    output = tmp_path / "output"
    details.run_category(record, tmp_path / "inputs", output, 20000, details.profile.RUN_CODE)
    result = json.loads((output / category / "result.json").read_text())["splits"]["temporal"]
    assert all(result["stats"][key] == value for key, value in stats.items())
    assert result["size_limits"] == {"max_users": 45, "max_items": 24}
    assert result["stats"]["val_target_items"] > 0
    assert result["stats"]["test_target_items"] > 0
