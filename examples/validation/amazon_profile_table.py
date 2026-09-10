"""Render the Amazon docs table from measured profile and metadata archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from amazon_metadata_report import percent as coverage_percent


def render_profiles_table(archive, metadata_archive):
    coverage = {record["category"]: record["union"] for record in metadata_archive["categories"]}
    if (metadata_archive.get("status") != "complete"
            or len(coverage) != len(metadata_archive["categories"])
            or set(coverage) != {record["category"] for record in archive["profiles"]}):
        raise ValueError("Complete metadata coverage must match the profile categories")
    lines = [".. list-table:: Installed category-specific Amazon support defaults",
             "   :header-rows: 1", "", "   * - Category", "     - User split",
             "     - Item split", "     - Leave-last-out", "     - Temporal",
             "     - Item metadata (union of splits)"]
    for record in archive["profiles"]:
        lines.append(f"   * - {record['category']}")
        for split in ("user_split", "item_split", "leave_last_out", "temporal"):
            entry = record["splits"][split]
            if entry["status"] != "verified":
                raise ValueError("The defaults table accepts only verified profiles")
            parameters, stats = entry["parameters"], entry["stats"]
            users = stats["users"] if split == "temporal" else stats["preprocessed_users"]
            items = stats["items"] if split == "temporal" else stats["preprocessed_items"]
            lines.extend([f"     - | {parameters['min_user_support']}/{parameters['item_min_support']}",
                          f"       | {users:,} users", f"       | {items:,} items"])
            for phase, label in (("val", "Val"), ("test", "Test")):
                warning = " †" if stats[f"{phase}_users"] < 1000 else ""
                lines.append(f"       | {label}: {stats[f'{phase}_users']:,} users{warning}")
                if f"{phase}_target_items" not in stats:
                    lines.append("       | Items / cold: measuring")
                else:
                    fraction = stats[f"{phase}_cold_target_item_fraction"]
                    percent = "—" if fraction is None else (
                        "<0.01%" if 0 < fraction < .0001 else f"{fraction:.2%}")
                    lines.append(f"       | {stats[f'{phase}_target_items']:,} items / {percent} cold")
        union = coverage[record["category"]]
        total = union["items"]
        images = union["source_image"]
        ten_words = total - union["word_counts"]["curated"]["below"]["10"]
        lines.extend([
            f"     - | Union: {total:,} items",
            f"       | ≥1 image: {images:,} ({coverage_percent(images, total)})",
            f"       | ≥10 words: {ten_words:,} ({coverage_percent(ten_words, total)})",
        ])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--metadata-archive", type=Path,
                        help="Metadata audit JSON; defaults to amazon-metadata-coverage.json beside the profile archive")
    args = parser.parse_args()
    metadata_path = args.metadata_archive or args.archive.with_name("amazon-metadata-coverage.json")
    print(render_profiles_table(json.loads(args.archive.read_text()),
                                json.loads(metadata_path.read_text())), end="")
