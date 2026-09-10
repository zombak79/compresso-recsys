"""Metadata coverage is measured on retained IDs, with native format caveats."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/validation"))
SPEC = importlib.util.spec_from_file_location("metadata_audit", Path(__file__).parents[1] /
                                            "examples/validation/amazon_metadata_audit.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def test_threshold_counts_combined_text_not_only_description():
    measured = audit.measure_row({"title": "A useful title", "description": [], "categories": ["Books"]})
    assert measured["field_words"]["description"] == 0
    assert measured["native_words"] == 6  # Title/category labels also count.
    assert measured["default_words"] == 6


def test_text_arrays_are_normalized_but_image_adapter_caveat_remains():
    row = {"title": "Title", "features": np.array([]), "description": np.array([]),
           "images": np.array([{"large": "https://example.org/item.jpg"}])}
    measured = audit.measure_row(row)
    assert not measured["native_differs"]
    assert measured["native_words"] == measured["default_words"] == 2
    assert measured["source_image"]
    assert not measured["adapter_image"]
    assert measured["field_words"]["description"] == 0


def test_summary_uses_requested_catalog_and_pair_weighted_loss():
    measurements = {"a": audit.measure_row({}), "b": audit.measure_row({"title": "good item"})}
    result = audit.summarize(["a", "b"], measurements, {"a": 7, "b": 2})
    assert result["items"] == 2
    assert result["word_counts"]["native"]["below"]["1"] == 1
    assert result["word_counts"]["native"]["train_pairs_removed"]["1"] == 7
    assert result["train_pairs"] == 9
    assert audit.summarize(["b"], measurements)["items"] == 1
    with pytest.raises(ValueError, match="Missing metadata"):
        audit.summarize(["missing"], measurements)


def test_source_reader_keeps_parquet_lists_and_json_lists_distinct(tmp_path):
    row = {"parent_asin": "a", "title": "Name", "features": ["useful feature"],
           "description": [], "images": [{"large": "https://example.org/item.jpg"}]}
    parquet = tmp_path / "meta.parquet"
    pd.DataFrame([row]).to_parquet(parquet)
    native = next(audit.source_rows(parquet, "parquet"))
    assert isinstance(native["features"], np.ndarray)
    assert "images" in native
    jsonl = tmp_path / "meta.jsonl"
    jsonl.write_text(json.dumps(row) + "\n")
    assert next(audit.source_rows(jsonl, "jsonl")) == row


def test_json_encoded_details_and_images_are_measured():
    result = audit.measure_row({"details": '{"Brand":"Example", "Empty": ""}',
                               "images": '[{"hi_res":"https://example.org/a.jpg"}]'})
    assert result["details_keys"] == ["Brand"]
    assert result["source_image"] and result["adapter_image"]
    assert result["extended_words"] > result["default_words"]


def test_huggingface_struct_of_image_arrays_is_present_in_source():
    row = {"images": {"hi_res": np.array([None, "https://example.org/a.jpg"]),
                      "large": np.array(["https://example.org/b.jpg", None])}}
    result = audit.measure_row(row)
    assert result["source_image"]
    assert not result["adapter_image"]
    with pytest.raises(ValueError, match="inconsistent"):
        audit.measure_row({"images": {"hi_res": [None], "large": []}})


def test_prime_video_has_width_variant_images_and_curated_cast_text():
    row = {"title": None, "description": None, "features": None,
           "images": [{"720w": "https://example.org/movie.webp", "variant": "MAIN"}],
           "details": {"Directors": ["Director Name"], "Starring": ["Actor Name"],
                       "Best Sellers Rank": "#1", "ASIN": "B123"}}
    r = audit.measure_row(row, "Movies_and_TV")
    assert r["source_image"] and not r["adapter_image"]
    assert r["default_words"] == 0
    assert r["curated_words"] > 0
    assert r["extended_words"] > r["curated_words"]
