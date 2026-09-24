#!/usr/bin/env python
"""Hand-run smoke test for the five large sequential datasets.

Not pytest. Run it yourself:

    python try_datasets.py                  # all five, one subprocess each
    python try_datasets.py otto yambda      # just these
    python try_datasets.py --list           # options each one will be built with
    python try_datasets.py otto --head 10
    python try_datasets.py amazon --amazon-category Gift_Cards

Every default below is chosen to HIT THE PARQUET CACHE already sitting in
data/. The caches are keyed on the constructor options that select rows
(OTTO._cache_version, Yambda._cache_version, music4all_onion.window_version),
so a different sample or window does not slice the cached frame -- it re-parses
the raw source. For OTTO that means a Python JSON loop over 11 GB. Change these
only when you mean to pay for the rebuild.

Each dataset runs in its own subprocess so its memory goes back to the OS
before the next one starts, and so one failure does not take the run with it.
"""
from __future__ import annotations

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# name -> (import path, kwargs, note). The kwargs are the cached configuration.
DATASETS = {
    "otto": (
        "OTTO",
        # 4219242697 == _cache_version() for session_sample=0.1, events=("clicks",).
        # Anything else re-parses train.jsonl (11 GB) line by line.
        dict(session_sample=0.1),
        "19.5M click events, ~1.3M sessions. No item metadata at all.",
    ),
    "yambda": (
        "Yambda",
        # 1507873275 == _cache_version() for organic_only=False, user_sample=0.2.
        dict(variant="50m", organic_only=False, user_sample=0.2),
        "9.6M listens. Timestamps are 5s ticks from the log start, not unix.",
    ),
    "music4all": (
        "Music4AllOnion",
        # 868559899 == window_version("2014-01-01", "2015-01-01"). A different
        # window re-reads a 2.2 GB bz2. The full log is 253M events.
        dict(start="2014-01-01", end="2015-01-01"),
        "33.7M listens in the 2014 window -- the heaviest frame here (~5 GB).",
    ),
    "amazon": (
        "AmazonReviews2023",
        # Overridden by --amazon-category. See SMALL_CATEGORIES: Toys_and_Games
        # is 973 MB of metadata over 890k items, which is why it hurt.
        dict(category="All_Beauty"),
        "Ratings plus real item text; entity_text is built from the metadata.",
    ),
    "movielens20m": (
        "MovieLens20M",
        dict(),
        "20M ratings, read from the extracted CSV each time (no parquet cache).",
    ),
}

#: Metadata size and raw item count per category, read off
#: docs/source/_static/amazon-metadata-coverage.json. Any of the 33 curated
#: categories works; these are the ones that fit on a laptop. A category not
#: already under data/amazon2023/ is downloaded on first use.
SMALL_CATEGORIES = {
    "Gift_Cards": "0.4 MB, 1.1k items, 29k rated pairs -- smallest download",
    "Subscription_Boxes": "1.4 MB, 641 items, 1.3k pairs -- almost too small to be interesting",
    "Magazine_Subscriptions": "4.1 MB, 3.4k items, 18k pairs",
    "Digital_Music": "13.8 MB, 71k items, 7k pairs -- sparse",
    "All_Beauty": "40 MB, 113k items, 57k pairs -- the repo's own tests use this one",
    "Health_and_Personal_Care": "118 MB, 60k items, 45k pairs",
    "Handmade_Products": "132 MB, 165k items, 51k pairs",
}

# Precisions that are genuine unix epochs, so a date is meaningful.
EPOCH_UNITS = {"seconds": 1.0, "milliseconds": 1000.0}


def describe(name: str, head: int, data_dir: str, amazon_category: str) -> None:
    """Build one dataset and print what came back. Runs in the child process."""
    import pandas as pd

    import compresso_recsys as cr

    attr, kwargs, note = DATASETS[name]
    if name == "amazon":
        kwargs = {**kwargs, "category": amazon_category}
    cls = getattr(cr, attr)

    print(f"=== {name} -> {attr}({', '.join(f'{k}={v!r}' for k, v in kwargs.items())})")
    print(f"    {note}")

    started = time.perf_counter()
    dataset = cls(data_dir=data_dir, **kwargs)
    interactions = dataset.get_interactions()
    metadata = dataset.get_item_metadata()
    elapsed = time.perf_counter() - started

    print(f"\n--- interactions  {len(interactions):,} rows in {elapsed:,.1f}s")
    print(f"    users {interactions.user_id.nunique():,}   "
          f"items {interactions.item_id.nunique():,}")
    print(f"    columns {list(interactions.columns)}")
    print(f"    dtypes  {dict(interactions.dtypes.astype(str))}")

    values = pd.to_numeric(interactions["value"], errors="coerce")
    print(f"    value   min {values.min():g}  max {values.max():g}  mean {values.mean():.3f}")

    stamps = pd.to_numeric(interactions["timestamp"], errors="coerce")
    precision = getattr(dataset, "timestamp_precision", None)
    if stamps.notna().any():
        low, high = stamps.min(), stamps.max()
        print(f"    time    min {low:,.0f}  max {high:,.0f}  (precision: {precision})")
        # Only render dates when the column really is a unix epoch. Yambda's
        # ticks are relative to the start of its log, so a date would be a lie.
        unit = EPOCH_UNITS.get(precision or "")
        if unit:
            print(f"            {pd.to_datetime(low / unit, unit='s')} .. "
                  f"{pd.to_datetime(high / unit, unit='s')}")
    else:
        print(f"    time    all null (precision: {precision})")

    per_user = interactions.groupby("user_id").size()
    print(f"    events per user  median {per_user.median():,.0f}  "
          f"p90 {per_user.quantile(0.9):,.0f}  max {per_user.max():,}")

    with pd.option_context("display.width", 200, "display.max_columns", 8):
        print(f"\n{interactions.head(head).to_string(index=False)}")

    print(f"\n--- item metadata  {len(metadata):,} rows")
    if metadata.empty:
        print("    none -- this dataset ships no item side information")
    else:
        print(f"    columns {list(metadata.columns)}")
        if "entity_text" in metadata:
            text = metadata["entity_text"].fillna("").astype(str)
            filled = text.str.strip().ne("")
            print(f"    entity_text non-empty {filled.sum():,} / {len(text):,}")
            words = text[filled].str.split().str.len()
            if len(words):
                print(f"    words   median {words.median():,.0f}  max {words.max():,}")
            sample = text[filled].head(1)
            if len(sample):
                print(f"    sample  {sample.iloc[0][:300]!r}")
        with pd.option_context("display.width", 200, "display.max_columns", 6,
                               "display.max_colwidth", 40):
            print(f"\n{metadata.head(min(head, 5)).to_string(index=False)}")

    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    print(f"\n    peak RSS {peak_gb:.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help=f"which to load, any of: {', '.join(DATASETS)} "
                             "(default: all of them)")
    parser.add_argument("--head", type=int, default=5, help="rows to print per frame")
    parser.add_argument("--data-dir", default="data", help="where the caches live")
    parser.add_argument("--amazon-category", default="All_Beauty",
                        help="any of the 33 curated categories; --list shows the small ones "
                             "(default: %(default)s)")
    parser.add_argument("--list", action="store_true", help="print the plan and exit")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child:
        describe(args.child, args.head, args.data_dir, args.amazon_category)
        return 0

    if args.list:
        for name, (attr, kwargs, note) in DATASETS.items():
            if name == "amazon":
                kwargs = {**kwargs, "category": args.amazon_category}
            options = ", ".join(f"{k}={v!r}" for k, v in kwargs.items()) or "defaults"
            print(f"{name:<13} {attr}({options})\n{'':<13} {note}")
        print("\nSmall Amazon categories for --amazon-category:")
        for category, note in SMALL_CATEGORIES.items():
            print(f"  {category:<26} {note}")
        return 0

    unknown = [name for name in args.datasets if name not in DATASETS]
    if unknown:
        parser.error(f"unknown dataset(s) {', '.join(unknown)}; "
                     f"choose from {', '.join(DATASETS)}")

    selected = args.datasets or list(DATASETS)
    failures = []
    for name in selected:
        print(f"\n{'=' * 78}")
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--child", name,
             "--head", str(args.head), "--data-dir", args.data_dir,
             "--amazon-category", args.amazon_category],
            cwd=HERE,
        )
        if result.returncode != 0:
            failures.append(name)
            print(f"!!! {name} exited {result.returncode} "
                  f"after {time.perf_counter() - started:,.1f}s")

    print(f"\n{'=' * 78}")
    print(f"ok: {len(selected) - len(failures)}/{len(selected)}"
          + (f"   failed: {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
