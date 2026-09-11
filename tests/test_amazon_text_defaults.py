"""Curated defaults enrich text without changing the verified interaction graphs."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

import compresso_recsys as cr
from compresso_recsys.builder import _build_args, _make_dataset, _resolve_args
from compresso_recsys.datasets.amazon2023 import AmazonReviews2023, DEFAULT_TEXT_FIELDS
from compresso_recsys.datasets._amazon_text_defaults import AMAZON_METADATA_TEXT_FIELDS

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "examples/validation"))
SPEC = importlib.util.spec_from_file_location("text_default_audit", ROOT /
                                            "examples/validation/amazon_metadata_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
from amazon_metadata_report import ATTRIBUTE_KEYS, recipe

RECORDS = json.loads((ROOT / "docs/source/_static/amazon-metadata-coverage.json").read_text())["categories"]
SPLITS = ("user_split", "item_split", "leave_last_out", "temporal")


def test_all_named_categories_have_curated_defaults():
    assert len(AMAZON_METADATA_TEXT_FIELDS) == 33
    assert set(AMAZON_METADATA_TEXT_FIELDS) == set(ATTRIBUTE_KEYS)


@pytest.mark.parametrize("record", RECORDS, ids=lambda r: r["category"])
def test_defaults_match_audited_recipe_and_word_counts(record):
    category = record["category"]
    fields = AmazonReviews2023.text_fields_for_category(category)
    top_level, _ = recipe(record)
    assert fields == (*top_level, *(f"details.{key}" for key in ATTRIBUTE_KEYS[category]))
    # Include every curated key, not just those in the frequent-key display.
    row = {field: np.array(["Some descriptive content", "More useful words"]) for field in top_level}
    row["details"] = {key: np.array(["Selected attribute value"]) for key in ATTRIBUTE_KEYS[category]}
    row["details"].update({"Best Sellers Rank": "FORBIDDEN", "ASIN": "FORBIDDEN", "ISBN 13": "FORBIDDEN"})
    text = AmazonReviews2023.build_entity_text(row, fields)
    assert "FORBIDDEN" not in text
    assert len(text.split()) == audit.measure_row(row, category)["curated_words"]


@pytest.mark.parametrize("category", sorted(AMAZON_METADATA_TEXT_FIELDS))
@pytest.mark.parametrize("split", SPLITS)
def test_builder_and_adapter_agree_for_every_category_and_split(category, split, tmp_path):
    args, spec = _resolve_args(_build_args(dataset="amazon2023", amazon_category=category,
                                          split_mode=split, data_dir=str(tmp_path)))
    ds = _make_dataset(args, spec)
    direct = AmazonReviews2023(data_dir=tmp_path, category=category)
    assert ds.metadata_text_fields == direct.metadata_text_fields == AMAZON_METADATA_TEXT_FIELDS[category]
    assert args.metadata_text_fields.split(",") == list(ds.metadata_text_fields)
    assert args.min_entity_text_words == ds.min_entity_text_words == 0


@pytest.mark.parametrize("category,canonical", [("toys", "Toys_and_Games"), ("electronics", "Electronics"),
                                              ("Unprofiled_Category", "Unprofiled_Category")])
def test_aliases_and_unprofiled_fallback(category, canonical, tmp_path):
    args, spec = _resolve_args(_build_args(dataset="amazon2023", amazon_category=category, data_dir=str(tmp_path)))
    ds = _make_dataset(args, spec)
    assert ds.category == canonical
    assert ds.metadata_text_fields == AMAZON_METADATA_TEXT_FIELDS.get(canonical, DEFAULT_TEXT_FIELDS)
    canonical_args, _ = _resolve_args(_build_args(dataset="amazon2023", amazon_category=canonical))
    assert (args.min_user_support, args.item_min_support) == (canonical_args.min_user_support, canonical_args.item_min_support)


@pytest.mark.parametrize("fields", [["title", "details.Brand"], "title,details.Brand", ("title", "details.Brand")])
def test_explicit_fields_replace_defaults(fields, tmp_path):
    args, spec = _resolve_args(_build_args(dataset="amazon2023", metadata_text_fields=fields, data_dir=str(tmp_path)))
    assert _make_dataset(args, spec).metadata_text_fields == ("title", "details.Brand")
    assert AmazonReviews2023(data_dir=tmp_path, metadata_text_fields=fields).metadata_text_fields == ("title", "details.Brand")


@pytest.mark.parametrize("fields", [[], "", " , "])
def test_builder_rejects_explicit_empty_fields(fields):
    with pytest.raises(ValueError, match="at least one field"):
        _resolve_args(_build_args(dataset="amazon2023", metadata_text_fields=fields))


@pytest.mark.parametrize("fields", [["details."], [".Brand"], ["details..Brand"], [None]])
def test_adapter_rejects_invalid_paths(fields, tmp_path):
    with pytest.raises(ValueError, match="nonempty field paths"):
        AmazonReviews2023(data_dir=tmp_path, metadata_text_fields=fields)


@pytest.mark.parametrize("details", [None, np.nan, "not JSON", "{}", [], '{"Brand":null}'])
def test_missing_or_malformed_nested_values_are_skipped(details):
    assert AmazonReviews2023.build_entity_text({"details": details}, ["details.Brand"]) == ""


@pytest.mark.parametrize("encoded", [False, True])
def test_nested_selectors_group_labels_and_preserve_explicit_raw_details(encoded):
    details = {"Directors": ["Ada Lovelace"], "Audio languages": ["English"], "ASIN": "SECRET"}
    row = {"details": json.dumps(details) if encoded else details}
    text = AmazonReviews2023.build_entity_text(row, ["details.Directors", "details.Audio languages", "details.Missing"])
    assert text == "Details: Directors: Ada Lovelace > Audio languages: English"
    assert AmazonReviews2023.build_entity_text(row, ["details.directors"]) == ""
    assert "ASIN: SECRET" in AmazonReviews2023.build_entity_text(row, ["details"])
    assert details == {"Directors": ["Ada Lovelace"], "Audio languages": ["English"], "ASIN": "SECRET"}


def test_parquet_lists_empty_arrays_and_nested_paths_do_not_mutate_input():
    row = {"features": np.array(["Lightweight", "Easy to use"]), "description": np.array([]),
           "details": {"Specs": {"Material": np.array(["Steel"]), "Color": "Blue"}}}
    fields = ["features", "description", "details.Specs.Material"]
    text = AmazonReviews2023.build_entity_text(row, fields)
    assert text == "Features: Lightweight\nEasy to use\n\nDetails: Specs: Material: Steel"
    assert text == AmazonReviews2023.build_entity_text(AmazonReviews2023._normalize_metadata_value(row), fields)
    assert isinstance(row["details"]["Specs"]["Material"], np.ndarray)
    assert row["details"]["Specs"]["Color"] == "Blue"
    assert "Color: Blue" in AmazonReviews2023.build_entity_text(row, ["details.Specs", "details.Specs.Material"])


@pytest.mark.parametrize("split", SPLITS)
def test_new_text_defaults_preserve_checkpoint_interactions_for_every_split(split, tmp_path, monkeypatch):
    def load(ds, config, *, split="full"):
        if config == ds.metadata_config:
            # Metadata-only Prime Video text plus a genuinely empty record.
            return pd.DataFrame([{"parent_asin": f"i{i}", "title": None,
                                  "features": np.array([]), "description": np.array([]),
                                  "details": {"Directors": ["Example Director"], "ASIN": "SECRET"} if i else {}}
                                 for i in range(12)])
        return pd.DataFrame([{"parent_asin": f"i{i}", "user_id": f"u{u}", "rating": (i % 5) + 1,
                              "timestamp": 1600000000 + (i * 24 + u) * 3600}
                             for u in range(8) for i in range(12)])

    monkeypatch.setattr(AmazonReviews2023, "_load_hf_dataframe", load)
    options = dict(dataset="amazon2023", amazon_category="Movies_and_TV", data_dir=str(tmp_path / "data"),
                   split_mode=split, val_users=2, test_users=2, val_items=2, test_items=2,
                   min_user_support=1, item_min_support=1, eval_draws=1, temporal_period_hours=48,
                   annotation_source="none", show_progress=False)
    loaded = []
    for name, fields in (("curated", None), ("legacy", DEFAULT_TEXT_FIELDS)):
        path = cr.build_recsys_checkpoint(**options, metadata_text_fields=fields,
                                          checkpoint_path=str(tmp_path / f"{name}.zip"))
        with cr.read_checkpoint(path) as root:
            loaded.append(cr.load_recsys_split(root))
            manifest = cr.load_manifest(root)
            expected = fields or AmazonReviews2023.text_fields_for_category("Movies_and_TV")
            assert manifest["stages"]["data"]["metadata_text_fields"] == list(expected)
    new, old = loaded
    for key in ("x_train", "train_source_matrix", "train_target_matrix", "val_source_matrix", "val_target_matrix",
                "test_source_matrix", "test_target_matrix"):
        assert new[key].shape == old[key].shape
        assert (new[key] != old[key]).nnz == 0
    for key in ("item_ids", "train_item_ids", "val_item_ids", "test_item_ids", "train_user_ids", "val_user_ids", "test_user_ids",
                "warm_item_indices", "val_cold_item_indices", "test_cold_item_indices"):
        np.testing.assert_array_equal(new[key], old[key])
    assert new["entity_metadata"].entity_text.fillna("").str.contains("Example Director").any()
    assert not new["entity_metadata"].entity_text.fillna("").str.contains("SECRET").any()
    assert (old["entity_metadata"].entity_text.fillna("") == "").all()
