"""Rendered metadata tables remain tied to measured catalogs, not raw totals."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[1]
SPEC = importlib.util.spec_from_file_location("metadata_report", ROOT /
                                            "examples/validation/amazon_metadata_report.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def test_percentage_does_not_round_missing_items_to_perfect_coverage():
    assert report.percent(99999, 100000) == ">99.99%"
    assert report.percent(1, 100000) == "<0.01%"
    assert report.percent(1, 1) == "100.00%"
    assert report.percent(0, 0) == "—"


def test_recipe_uses_observed_fields_and_vetted_attributes_only():
    record = {"category": "Books", "union": {"field_nonempty": {
        "features": 90, "description": 70, "categories": 0, "store": 99}},
        "details_keys": [["Publisher", 10], ["Best Sellers Rank", 9], ["ISBN 13", 7]]}
    fields, attributes = report.recipe(record)
    assert fields == ["title", "store", "features", "description"]
    assert attributes == ["Publisher"]


def test_published_metadata_tables_match_all_measured_profiles():
    archive = json.loads((ROOT / "docs/source/_static/amazon-metadata-coverage.json").read_text())
    records = archive["categories"]
    profiles = json.loads((ROOT / "docs/source/_static/amazon-default-profiles.json").read_text())["profiles"]
    assert len(records) == len(profiles) == 33
    assert {r["category"] for r in records} == set(report.ATTRIBUTE_KEYS)
    guide = (ROOT / "docs/source/amazon-metadata.rst").read_text()
    for table in report.render(records).values():
        assert table.rstrip() in guide
    for r, p in zip(records, profiles):
        assert r["category"] == p["category"]
        assert set(r["splits"]) == set(p["splits"])
        assert r["context"]["files"]
        for split, phases in r["splits"].items():
            stats = p["splits"][split]["stats"]
            assert phases["catalog"]["items"] == stats["items"]
            assert phases["train"]["items"] == stats["train_observed_items"]
            assert phases["train"]["train_pairs"] == stats["pairs"]
            for phase in ("val", "test"):
                assert phases[phase]["items"] == stats[phase + "_target_items"]
            for values in phases.values():
                n = values["items"]
                assert 0 <= values["adapter_image"] <= values["source_image"] <= n
                for field_count in values["field_nonempty"].values():
                    assert 0 <= field_count <= n
                for counts in values["word_counts"].values():
                    losses = list(counts["below"].values())
                    assert losses == sorted(losses)
                    assert 0 <= min(losses) <= max(losses) <= n
