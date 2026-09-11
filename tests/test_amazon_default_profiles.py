"""Installed defaults must agree with archived, independently measured results."""
import json
from pathlib import Path

import pytest

from compresso_recsys.builder import _build_args, _resolve_args
from compresso_recsys.datasets._amazon_defaults import AMAZON_SPLIT_DEFAULTS


ARCHIVE = json.loads((Path(__file__).parents[1] /
                     "docs/source/_static/amazon-default-profiles.json").read_text())
CASES = [(record["category"], split, entry) for record in ARCHIVE["profiles"]
         for split, entry in record["splits"].items()]


def test_every_installed_profile_has_recorded_source_and_measurement_evidence():
    assert set(AMAZON_SPLIT_DEFAULTS) == {record["category"] for record in ARCHIVE["profiles"]}
    for record in ARCHIVE["profiles"]:
        assert set(AMAZON_SPLIT_DEFAULTS[record["category"]]) == set(record["splits"])
        assert len(record["input"]["prepared_sha256"]) == 64
        assert record["input"]["sources"]
        assert all(len(source["sha256"]) == 64 and source["url"].startswith("https://")
                   for source in record["input"]["sources"])
        for entry in record["splits"].values():
            if "evaluation_verification" in entry:
                evidence = entry["evaluation_verification"]
                assert evidence["prepared_sha256"] == record["input"]["prepared_sha256"]
                assert len(evidence["profile_sha256"]) == 64


@pytest.mark.parametrize("category", ["Books", "Electronics"])
def test_large_category_holdouts_use_twenty_thousand_without_repruning(category):
    record = next(record for record in ARCHIVE["profiles"] if record["category"] == category)
    entry = record["splits"]["user_split"]
    baseline = entry["verification"]["baseline"]
    assert entry["verification"]["kind"] == "user-holdout-refinement"
    for key in ("val_users", "test_users"):
        assert entry["parameters"][key] == entry["stats"][key] == 20000
        assert baseline["parameters"][key] == 5000
    for key in ("min_user_support", "item_min_support", "seed"):
        assert entry["parameters"][key] == baseline["parameters"][key]
    for key in ("preprocessed_users", "preprocessed_items", "preprocessed_pairs"):
        assert entry["stats"][key] == baseline["stats"][key]
    assert entry["stats"]["train_users"] == entry["stats"]["preprocessed_users"] - 40000


def test_evaluation_detail_archive_has_consistent_counts_and_provenance():
    verified = 0
    for _, split, entry in CASES:
        if "evaluation_verification" not in entry:
            assert ARCHIVE["evaluation_detail_audit"]["status"] != "complete"
            continue
        verified += 1
        evidence = entry["evaluation_verification"]
        assert evidence["kind"] == "evaluation-detail-audit"
        assert len(evidence["code"]["details_sha256"]) == 64
        for key, value in ARCHIVE["source_code"].items():
            assert evidence["code"][key] == value
        stats = entry["stats"]
        for phase in ("val", "test"):
            items, cold = stats[f"{phase}_target_items"], stats[f"{phase}_cold_target_items"]
            assert 0 <= cold <= items <= stats[f"{phase}_catalog_items"]
            assert 0 < items <= stats[f"{phase}_target_pairs"]
            assert stats[f"{phase}_cold_target_item_fraction"] == pytest.approx(cold / items)
            if split in ("user_split", "item_split"):
                assert cold == (0 if split == "user_split" else items)
    assert ARCHIVE["evaluation_detail_audit"]["verified_profiles"] == verified


def test_documented_support_and_size_table_matches_measurement_archive():
    guide = (Path(__file__).parents[1] / "docs/source/datasets.rst").read_text()
    table = guide.split(".. list-table:: Installed category-specific Amazon support defaults", 1)[1]
    body = table.split("\n\n", 1)[1].split("\n\n", 1)[0]
    assert "Requested user-split val/test each" not in body
    assert body.count("   * - ") == len(ARCHIVE["profiles"]) + 1
    for record in ARCHIVE["profiles"]:
        lines = [f"   * - {record['category']}"]
        for split in ("user_split", "item_split", "leave_last_out", "temporal"):
            entry = record["splits"][split]
            parameters, stats = entry["parameters"], entry["stats"]
            users = stats["users"] if split == "temporal" else stats["preprocessed_users"]
            items = stats["items"] if split == "temporal" else stats["preprocessed_items"]
            lines.extend([
                f"     - | {parameters['min_user_support']}/{parameters['item_min_support']}",
                f"       | {users:,} users",
                f"       | {items:,} items",
            ])
            for phase in ("val", "test"):
                label = "Val" if phase == "val" else "Test"
                warning = " †" if stats[f"{phase}_users"] < 1000 else ""
                lines.append(f"       | {label}: {stats[f'{phase}_users']:,} users{warning}")
                if f"{phase}_target_items" not in stats:
                    lines.append("       | Items / cold: measuring")
                else:
                    fraction = stats[f"{phase}_cold_target_item_fraction"]
                    percent = "<0.01%" if 0 < fraction < .0001 else f"{fraction:.2%}"
                    lines.append(f"       | {stats[f'{phase}_target_items']:,} items / {percent} cold")
        assert "\n".join(lines) in body, record["category"]


def test_magazine_holdout_refinement_preserves_graph_and_meets_eval_target():
    record = next(record for record in ARCHIVE["profiles"]
                  if record["category"] == "Magazine_Subscriptions")
    entry = record["splits"]["user_split"]
    refinement = entry["verification"]
    assert refinement["kind"] == "user-holdout-refinement"
    assert all(len(value) == 64 for value in refinement["code"].values())
    baseline = refinement["baseline"]
    for field in ("min_user_support", "item_min_support", "seed"):
        assert entry["parameters"][field] == baseline["parameters"][field]
    for field in ("preprocessed_users", "preprocessed_items", "preprocessed_pairs"):
        assert entry["stats"][field] == baseline["stats"][field]
    assert entry["parameters"]["val_users"] == entry["parameters"]["test_users"] == 1200
    assert min(entry["stats"]["val_users"], entry["stats"]["test_users"]) >= 1000
    assert entry["stats"]["train_users"] > 4000
    assert any(trial["requested_each"] == 1200 and all(entry["stats"][key] == value for key, value in trial["stats"].items())
               for trial in refinement["trials"])


@pytest.mark.parametrize("category,split,entry", CASES,
                         ids=[f"{category}/{split}" for category, split, _ in CASES])
def test_profile_resolution_matches_verified_parameters_and_limits(category, split, entry):
    assert entry["status"] == "verified"
    profile = AMAZON_SPLIT_DEFAULTS[category][split]
    args, _ = _resolve_args(_build_args(dataset="amazon2023", amazon_category=category, split_mode=split))
    for key, value in profile.items():
        assert value == entry["parameters"][key]
        assert getattr(args, key) == value
    assert args.seed == entry["parameters"]["seed"] == 42
    assert args.eval_draws == 1
    assert args.min_entity_text_words == 0
    assert args.min_value_to_keep is None
    assert args.set_all_values_to == 1.
    if split == "temporal":
        assert args.temporal_period_hours == entry["parameters"]["temporal_period_hours"]
    if split == "leave_last_out":
        assert args.min_user_support >= 4
    stats = entry["stats"]
    max_users, max_items = (499999, 100000) if category in {"Books", "Electronics"} else (100000, 20000)
    if category == "Clothing_Shoes_and_Jewelry" and split == "temporal":
        max_users, max_items = 120000, 25000
    assert 0 < stats.get("preprocessed_users", stats["users"]) <= max_users
    assert 0 < stats.get("preprocessed_items", stats["items"]) <= max_items
    assert stats["pairs"] > 0 and min(stats["val_users"], stats["test_users"]) > 0
    if min(stats["val_users"], stats["test_users"]) < ARCHIVE["eval_user_target"]:
        assert any("Fewer than 1,000 eligible users" in warning for warning in entry["warnings"])
    # Defaults must never override a user's explicitly chosen protocol settings.
    overrides = dict(min_user_support=9, item_min_support=11, val_users=123, test_users=456,
                     seed=7, eval_draws=3, temporal_period_hours=72)
    custom, _ = _resolve_args(_build_args(dataset="amazon2023", amazon_category=category,
                                         split_mode=split, **overrides))
    for key, value in overrides.items():
        assert getattr(custom, key) == value


def test_all_named_categories_have_four_verified_profiles():
    assert len(ARCHIVE["profiles"]) == 33
    assert len(CASES) == 132
    for record in ARCHIVE["profiles"]:
        assert set(record["splits"]) == {"user_split", "item_split", "leave_last_out", "temporal"}


def test_clothing_exception_is_temporal_only_and_preserves_a_useful_training_set():
    record = next(r for r in ARCHIVE["profiles"] if r["category"] == "Clothing_Shoes_and_Jewelry")
    for split, entry in record["splits"].items():
        assert entry["parameters"]["min_user_support"] == 10
        assert entry["parameters"]["item_min_support"] == (17 if split == "temporal" else 35)
        if split != "temporal":
            assert "size_limits" not in entry
            assert entry["stats"]["preprocessed_users"] == 93809
            assert entry["stats"]["preprocessed_items"] == 14289
    user_split = record["splits"]["user_split"]
    assert user_split["parameters"]["val_users"] == user_split["parameters"]["test_users"] == 5000
    entry = record["splits"]["temporal"]
    assert entry["size_limits"] == {"max_users": 120000, "max_items": 25000}
    assert entry["verification"]["kind"] == "temporal-size-exception-refinement"
    assert entry["verification"]["baseline"]["stats"]["train_observed_items"] == 4
    stats = entry["stats"]
    assert (stats["users"], stats["items"]) == (113384, 24824)
    assert stats["users"] > stats["items"]
    assert stats["train_users"] >= 20000 and stats["train_observed_items"] >= 10000
    assert stats["pairs"] >= 400000
    assert min(stats["val_users"], stats["test_users"]) >= ARCHIVE["eval_user_target"]
    assert [(category, split) for category, split, e in CASES if "size_limits" in e] == [
        ("Clothing_Shoes_and_Jewelry", "temporal")]
