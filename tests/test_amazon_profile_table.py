"""The main dataset table reports the audited union, not summed split coverage."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "examples/validation/amazon_profile_table.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("profile_table", SCRIPT)
table = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(table)
PROFILE_PATH = ROOT / "docs/source/_static/amazon-default-profiles.json"
METADATA_PATH = PROFILE_PATH.with_name("amazon-metadata-coverage.json")
PROFILES = json.loads(PROFILE_PATH.read_text())
METADATA = json.loads(METADATA_PATH.read_text())


def test_complete_dataset_table_matches_both_measurement_archives():
    rendered = table.render_profiles_table(PROFILES, METADATA)
    assert rendered.rstrip() in (ROOT / "docs/source/datasets.rst").read_text()
    assert rendered.count("≥1 image:") == rendered.count("≥10 words:") == 33
    assert "Item metadata (union of splits)" in rendered
    for record in METADATA["categories"]:
        row = rendered.split(f"   * - {record['category']}\n", 1)[1].split("   * - ", 1)[0]
        union = record["union"]
        n, images = union["items"], union["source_image"]
        words = n - union["word_counts"]["curated"]["below"]["10"]
        assert 0 <= images <= n and 0 <= words <= n
        assert f"Union: {n:,} items" in row
        assert f"≥1 image: {images:,} ({table.coverage_percent(images, n)})" in row
        assert f"≥10 words: {words:,} ({table.coverage_percent(words, n)})" in row


def test_coverage_uses_curated_ten_word_boundary_and_source_images():
    profiles = {"profiles": [PROFILES["profiles"][0]]}
    metadata = {"status": "complete", "categories": [{
        "category": profiles["profiles"][0]["category"],
        "union": {"items": 10, "source_image": 8, "adapter_image": 0,
                  "word_counts": {"native": {"below": {"10": 9}},
                                  "curated": {"below": {"10": 3}}}},
    }]}
    rendered = table.render_profiles_table(profiles, metadata)
    assert "Union: 10 items" in rendered
    assert "≥1 image: 8 (80.00%)" in rendered
    assert "≥10 words: 7 (70.00%)" in rendered


@pytest.mark.parametrize("problem", ["missing", "duplicate", "incomplete"])
def test_metadata_archive_must_cover_exactly_the_same_categories(problem):
    metadata = copy.deepcopy(METADATA)
    if problem == "missing":
        metadata["categories"].pop()
    elif problem == "duplicate":
        metadata["categories"].append(metadata["categories"][0])
    else:
        metadata["status"] = "running"
    with pytest.raises(ValueError, match="Complete metadata coverage"):
        table.render_profiles_table(PROFILES, metadata)


@pytest.mark.parametrize("explicit_metadata", [False, True])
def test_table_cli_loads_the_paired_metadata_archive(explicit_metadata):
    command = [sys.executable, str(SCRIPT), str(PROFILE_PATH)]
    if explicit_metadata:
        command += ["--metadata-archive", str(METADATA_PATH)]
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert result.stdout == table.render_profiles_table(PROFILES, METADATA)
