"""Render measured Amazon metadata coverage tables to stdout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from compresso_recsys.datasets._amazon_text_defaults import AMAZON_METADATA_TEXT_FIELDS

def percent(number, total):
    if not total:
        return "—"
    value = number / total * 100
    if 0 < value < .01:
        return "<0.01%"
    if 99.99 < value < 100:
        return ">99.99%"
    return f"{value:.2f}%"


def rst_table(title, headers, rows):
    lines = [f".. list-table:: {title}", "   :header-rows: 1", ""]
    for row in [headers, *rows]:
        lines.append(f"   * - {row[0]}")
        lines.extend(f"     - {value}" for value in row[1:])
    return "\n".join(lines) + "\n"


# Frozen semantic-attribute allowlist used by the original coverage audit.
# recipe() displays frequent keys; measure_row() uses the entire allowlist.
# No ranks, review aggregates, IDs, crawl/release dates or shipping dimensions.
ATTRIBUTE_KEYS = {
    "All_Beauty": ("Brand", "Item Form", "Skin Type", "Hair Type", "Product Benefits"),
    "Amazon_Fashion": ("Department", "Brand", "Material", "Color", "Style"),
    "Appliances": ("Brand", "Compatible Devices", "Capacity", "Voltage", "Material"),
    "Arts_Crafts_and_Sewing": ("Brand", "Material", "Color", "Size", "Paint Type"),
    "Automotive": ("Brand", "Vehicle Service Type", "Auto Part Position", "Material"),
    "Baby_Products": ("Brand", "Age Range (Description)", "Material", "Style"),
    "Beauty_and_Personal_Care": ("Brand", "Item Form", "Skin Type", "Scent", "Product Benefits"),
    "Books": ("Publisher", "Language", "Paperback", "Hardcover"),
    "CDs_and_Vinyl": ("Label", "Language", "Media Format", "Run time"),
    "Cell_Phones_and_Accessories": ("Brand", "Compatible Devices", "Compatible Phone Models", "Material"),
    "Clothing_Shoes_and_Jewelry": ("Department", "Brand", "Material", "Color", "Style"),
    "Digital_Music": ("Label", "Genres", "Language", "Run time"),
    "Electronics": ("Brand", "Compatible Devices", "Connectivity Technology", "Screen Size", "Memory Storage Capacity"),
    "Gift_Cards": ("Brand", "Manufacturer"),
    "Grocery_and_Gourmet_Food": ("Brand", "Flavor", "Item Form", "Specialty", "Allergen Information"),
    "Handmade_Products": ("Material", "Color", "Size", "Department"),
    "Health_and_Household": ("Brand", "Item Form", "Active Ingredients", "Material", "Scent"),
    "Health_and_Personal_Care": ("Brand", "Item Form", "Active Ingredients", "Skin Type"),
    "Home_and_Kitchen": ("Brand", "Material", "Color", "Style", "Size"),
    "Industrial_and_Scientific": ("Brand", "Material", "Size", "Compatible Devices"),
    "Kindle_Store": ("Publisher", "Language", "Print length", "Author"),
    "Magazine_Subscriptions": ("Publisher", "Language", "Print length"),
    "Movies_and_TV": ("Director", "Directors", "Actors", "Starring", "Media Format", "Language", "Audio languages", "Subtitles", "Run time"),
    "Musical_Instruments": ("Brand", "Instrument", "Material", "Compatible Devices"),
    "Office_Products": ("Brand", "Material", "Color", "Sheet Size", "Ink Color"),
    "Patio_Lawn_and_Garden": ("Brand", "Material", "Power Source", "Color"),
    "Pet_Supplies": ("Brand", "Target Species", "Breed Recommendation", "Flavor", "Material"),
    "Software": ("Platform", "Operating System", "Manufacturer", "Language"),
    "Sports_and_Outdoors": ("Brand", "Sport", "Material", "Size", "Age Range (Description)"),
    "Subscription_Boxes": ("Brand", "Manufacturer", "Material"),
    "Tools_and_Home_Improvement": ("Brand", "Material", "Power Source", "Voltage"),
    "Toys_and_Games": ("Brand", "Age Range (Description)", "Material", "Theme", "Educational Objective"),
    "Video_Games": ("Platform", "Rated", "Manufacturer", "Language"),
}


def recipe(record):
    counts = record["union"]["field_nonempty"]
    # Both optional prose fields remain in the recipe when either supplies text.
    fields = ["title"]
    media = record["category"] in {"Books", "Kindle_Store", "CDs_and_Vinyl", "Digital_Music",
        "Movies_and_TV", "Magazine_Subscriptions", "Software", "Video_Games"}
    if media and counts["store"]:
        fields.append("store")
    fields += sorted((f for f in ("features", "description") if counts[f]),
                     key=lambda f: counts[f], reverse=True)
    fields += [f for f in ("categories", "store") if counts[f] and f not in fields]
    keys = {key for key, _ in record["details_keys"]}
    attributes = [key for key in ATTRIBUTE_KEYS[record["category"]] if key in keys
                  and "date" not in key.lower()]
    return fields, attributes


def render(records):
    summary, recipes, splits, native, curated = [], [], [], [], []
    for record in records:
        category = record["category"]
        u = record["union"]
        n = u["items"]
        words = u["word_counts"]["curated"]
        summary.append([category, f"{n:,}", percent(words["nonempty"], n),
            percent(u["field_nonempty"]["description"], n), percent(u["field_nonempty"]["features"], n),
            percent(u["source_image"], n), percent(words["below"]["10"], n),
            percent(words["below"]["30"], n)])
        installed = AMAZON_METADATA_TEXT_FIELDS[category]
        fields = [field for field in installed if "." not in field]
        attributes = [field for field in installed if "." in field]
        recipes.append([category, "``" + ",".join(fields) + "``",
                        "; ".join(f"``{field}``" for field in attributes)])
        native.append([category, percent(u["source_image"], n), percent(u["adapter_image"], n),
                       percent(u["native_differs"], n),
                       percent(u["word_counts"]["native"]["below"]["30"], n),
                       percent(u["word_counts"]["default"]["below"]["30"], n)])
        losses = {}
        for name in ("native", "curated"):
            values = [p["train"]["word_counts"][name]["train_pairs_removed"]["30"] /
                      p["train"]["train_pairs"] for p in record["splits"].values()]
            losses[name] = f"{min(values):.2%}–{max(values):.2%}"
        curated.append([category, percent(u["word_counts"]["default"]["nonempty"], n),
            percent(u["word_counts"]["curated"]["nonempty"], n),
            percent(u["word_counts"]["curated"]["below"]["30"], n), losses["native"], losses["curated"]])
        for split, phases in record["splits"].items():
            row = [category, split]
            for phase in ("catalog", "train", "val", "test"):
                p = phases[phase]
                row.append(f"{p['items']:,} / " + percent(p["word_counts"]["curated"]["nonempty"], p["items"]) +
                           " / " + percent(p["source_image"], p["items"]))
            train = phases["train"]
            row.append(" / ".join(percent(train["word_counts"]["curated"]["train_pairs_removed"][t],
                                          train["train_pairs"]) for t in ("1", "10", "30")))
            splits.append(row)
    return {
        "summary": rst_table("Installed-default metadata coverage and direct text-filter loss",
            ["Subset", "Union items", "Text", "Description", "Features", "Image URLs", "Below 10 words", "Below 30 words"], summary),
        "recipes": rst_table("Installed per-subset metadata_text_fields (both columns, in order)",
            ["Subset", "Top-level fields", "Included nested detail fields"], recipes),
        "native": rst_table("Historical four-field baseline: source versus pre-change adapter",
            ["Subset", "Source image URLs", "Adapter image URLs", "Text affected by old array formatting",
             "Below 30: old native", "Below 30: normalized four fields"], native),
        "curated": rst_table("Measured effect of the installed curated defaults",
            ["Subset", "Original four-field text coverage", "Installed text coverage", "Installed items below 30",
             "Original train-pair loss at 30: range over splits", "Installed train-pair loss at 30: range over splits"], curated),
        "splits": rst_table("Per-split installed-default coverage and direct training-pair loss",
            ["Subset", "Split", "Catalog: items / text / image", "Train: items / text / image",
             "Validation targets: items / text / image", "Test targets: items / text / image",
             "Train pairs lost below 1 / 10 / 30 words"], splits),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--table", choices=("summary", "recipes", "native", "splits", "curated"), default="summary")
    args = parser.parse_args()
    print(render(json.loads(args.archive.read_text())["categories"])[args.table], end="")
