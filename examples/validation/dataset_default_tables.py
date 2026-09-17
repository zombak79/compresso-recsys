"""Measure installed non-Amazon defaults and render one table per dataset.

Offline by default; --download-root explicitly permits missing source downloads.
Use full source caches, not samples. No model fitting or default changes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import random
import textwrap
from unittest.mock import patch
import zipfile

import numpy as np
import pandas as pd

from compresso_recsys import builder
from compresso_recsys.datasets._public import PublicDataset
from compresso_recsys.datasets._download import download
from compresso_recsys.datasets.multimodal import SOURCE_PAGE
from compresso_recsys.multimodal import ARCHIVE_MD5, ENCODERS, _read_feature_space
from amazon_evaluation_details import evaluation_item_stats
from amazon_metadata_report import percent
from amazon_profile import payload_stats
from dataset_sweep import unsupported


NAMES = {
    "ml1m": "MovieLens 1M", "ml20m": "MovieLens 20M", "goodbooks": "Goodbooks-10k",
    "steam": "Steam", "netflix": "Netflix Prize", "taste-profile": "MSD Taste Profile",
    "gowalla": "Gowalla", "retailrocket": "Retailrocket",
    "music4all-onion": "Music4All-Onion", "otto": "OTTO", "yambda": "Yambda",
    "dbbook": "DBbook", "lfm2k": "Last.fm-2K",
}
SPLITS = ("user_split", "item_split", "leave_last_out", "temporal", "official")
HEADERS = dict(zip(SPLITS, ("User split", "Item split", "Leave-last-out", "Temporal", "Official")))
# Files actually read by prepare() when downloading/extraction is disabled.
SOURCES = {
    "ml1m": ("movielens1m/ml-1m/ratings.dat", "movielens1m/ml-1m/movies.dat",
             "movielens1m/item_text_descriptions.feather"),
    "ml20m": ("movielens20m/ml-20m/ratings.csv", "movielens20m/ml-20m/movies.csv",
              "movielens20m/item_text_descriptions.feather"),
    "goodbooks": ("goodbooks/ratings.csv", "goodbooks/books.csv", "goodbooks/item_text_descriptions.feather"),
    "steam": ("steam/steam_reviews.json.gz", "steam/steam_games.json.gz"),
    "netflix": ("netflix/nf_prize_dataset.tar.gz",),
    "taste-profile": ("taste-profile/train_triplets.txt.zip",),
    "gowalla": ("gowalla/loc-gowalla_totalCheckins.txt.gz",),
    # Kaggle exports, placed by hand; the adapter reads only these members.
    "retailrocket": ("retailrocket/events.csv",),
    "otto": ("otto/train.jsonl",),
    "music4all-onion": ("music4all-onion/userid_trackid_timestamp.tsv.bz2",),
    # The default variant. A different one is a different file and a different
    # measurement, so it would need its own entry rather than replacing this.
    "yambda": ("yambda/listens-50m.parquet",),
    "dbbook": ("dbbook/dbbook_interaction_data.zip",),
    "lfm2k": ("lfm2k/lfm2k_interaction_data.zip",),
}


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def parameters(dataset, mode):
    # Official/timestampless modes cannot be resolved, but their status is still
    # documented. annotation_source does not change interaction/text counts.
    args, spec = builder._resolve_args(builder._build_args(
        dataset=dataset, split_mode=mode, annotation_source="none", show_progress=False,
    ))
    values = {key: value for key, value in vars(args).items()
              if key not in {"data_dir", "checkpoint_path", "amazon_category"}}
    values["metadata_text_fields"] = list(spec.cls.default_text_fields)
    # Measurements record the installed defaults, which set no adapter options.
    # Omitting the empty mapping keeps archives comparable across the release
    # that introduced them, while a measured run that does set one still says so.
    if not values.get("dataset_options"):
        values.pop("dataset_options", None)
    return args, values


def split_measurement(args, ds, raw):
    """Follow the builder's preprocessing order and reuse its split implementation."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    data = raw
    if isinstance(ds, PublicDataset) and args.split_mode in ("user_split", "item_split"):
        data = data.drop_duplicates(["user_id", "item_id"], keep="first")
    if args.split_mode == "official":
        data = data[data.source_split == "train"]
    temporal = args.split_mode == "temporal"
    proc = ds.preprocess_interactions_for_recsys(
        data, min_value_to_keep=args.min_value_to_keep,
        user_min_support=1 if temporal else args.min_user_support,
        item_min_support=1 if temporal else args.item_min_support,
        set_all_values_to=args.set_all_values_to,
    )
    if proc.empty:
        raise ValueError("No interactions after preprocessing")
    payload = builder._build_split_payload(args, ds, proc)
    pre = None if temporal else {"users": int(proc.user_id.nunique()),
        "items": int(proc.item_id.nunique()), "pairs": len(proc.drop_duplicates(["user_id", "item_id"]))}
    stats = {**payload_stats(payload, preprocessed=pre), **evaluation_item_stats(payload)}
    # Include non-temporal preprocessed items, like the Amazon union denominator.
    catalog = set(map(str, payload["item_ids"]))
    if not temporal:
        catalog.update(proc.item_id.astype(str))
    return stats, catalog


def metadata_counts(metadata, item_ids):
    if metadata.item_id.astype(str).duplicated().any():
        raise ValueError("Duplicate metadata IDs")
    aligned = metadata.assign(item_id=metadata.item_id.astype(str)).set_index("item_id").reindex(sorted(item_ids))
    texts = aligned.get("entity_text", pd.Series("", index=aligned.index)).fillna("").astype(str)
    images = None
    if "image_url" in aligned:
        images = int(aligned.image_url.fillna("").astype(str).str.strip().str.startswith(("http://", "https://")).sum())
    return {"union_items": len(item_ids), "image_url_items": images,
            "text_ge10_items": int(texts.str.split().str.len().ge(10).sum())}


def feature_counts(dataset, archive_path, item_ids):
    """Coverage of valid imported vectors, not raw media or nonzero vectors."""
    md5 = hashlib.md5()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
    if md5.hexdigest() != ARCHIVE_MD5[dataset]:
        raise ValueError(f"Checksum mismatch for {archive_path}")
    spaces = {}
    with zipfile.ZipFile(archive_path) as archive:
        for name in ENCODERS:
            if dataset != "ml1m" and name.startswith("video/"):
                continue
            if dataset == "dbbook" and name.startswith("audio/"):
                continue
            print(f"Measuring {dataset} features: {name}", flush=True)
            data, dimension = _read_feature_space(archive, dataset, name)
            spaces[name] = {"items": len(item_ids.intersection(data)), "dimension": dimension,
                            "source_items": len(data),
                            "interaction_derived": dataset == "lfm2k" and name.startswith("text/")}
            del data
            gc.collect()
    return {"status": "measured", "source": fingerprint(archive_path), "verified_release": True,
            "union_items": len(item_ids), "spaces": spaces}


def measure_dataset(dataset, cache_roots, *, download_root=None, features=False, progress=None):
    root = next((root for root in cache_roots if all((root / path).is_file() for path in SOURCES[dataset])), None)
    record = {"dataset": dataset, "sources": [], "splits": {}, "metadata": {"status": "not_measured"}}
    ds, raw, union = None, None, set()
    if root is None and download_root is not None:
        args, _ = parameters(dataset, "user_split")
        args.data_dir = str(download_root)
        print(f"Downloading full {dataset} sources to {download_root}", flush=True)
        builder._make_dataset(args, builder.DATASETS[dataset]).download()
        root = download_root
    if root is not None:
        record["sources"] = [fingerprint(root / path) for path in SOURCES[dataset]]
        # Canonical caches may be reused by the adapter, so fingerprint them too.
        record["interaction_caches"] = [fingerprint(path) for source in SOURCES[dataset]
                                       for suffix in (".interactions.parquet", ".interactions.json")
                                       if (path := root / (source + suffix)).is_file()]
    for mode in SPLITS:
        reason = unsupported(dataset, mode)
        if reason:
            record["splits"][mode] = {"status": "unsupported", "reason": reason}
            continue
        args, params = parameters(dataset, mode)
        entry = {"parameters": params}
        record["splits"][mode] = entry
        if root is None:
            entry.update(status="not_measured", reason="Full source cache not available in the supplied roots")
            continue
        print(f"Measuring {dataset}/{mode}", flush=True)
        try:
            if ds is None:
                args.data_dir = str(root)
                ds = builder._make_dataset(args, builder.DATASETS[dataset])
                # Missing inputs must fail offline; never fall back to downloads.
                with patch.object(type(ds), "download", return_value=None):
                    raw = ds.get_interactions()
            if raw is None:
                raise ValueError("Dataset loading failed")
            stats, catalog = split_measurement(args, ds, raw)
            union.update(catalog)
            entry.update(status="measured", stats=stats)
        except Exception as error:
            entry.update(status="failed", reason=f"{type(error).__name__}: {error}")
        print(f"  {entry['status']}: {entry.get('stats', entry.get('reason'))}", flush=True)
        if progress is not None:
            progress(record)
        gc.collect()
    measured = [mode for mode, entry in record["splits"].items() if entry["status"] == "measured"]
    if measured:
        record["metadata"] = {"status": "measured", "splits": measured,
                              **metadata_counts(ds.get_item_metadata(), union)}
        if features and dataset in ARCHIVE_MD5:
            relative = Path("multimodal") / f"{dataset}_mm_json.zip"
            archive_path = next((p / relative for p in cache_roots if (p / relative).is_file()), None)
            if archive_path is None and download_root is not None:
                archive_path = download_root / relative
                download(f"{SOURCE_PAGE}/files/{dataset}_mm_json.zip", archive_path, show_progress=True)
            record["metadata"]["precomputed_features"] = (
                {"status": "not_measured", "reason": "Feature archive missing; use --download-root"}
                if archive_path is None else feature_counts(dataset, archive_path, union))
    return record


def render_table(record):
    dataset = record["dataset"]
    modes = SPLITS if dataset == "dbbook" else SPLITS[:-1]
    base = record["splits"]["user_split"]["parameters"]
    rating = "All feedback" if base["min_value_to_keep"] is None else f"Rating ≥{base['min_value_to_keep']:g}"
    lines = [f".. list-table:: Installed {NAMES[dataset]} preprocessing and split defaults",
             "   :header-rows: 1", "", "   * - Dataset / preprocessing"]
    lines.extend(f"     - {HEADERS[mode]}" for mode in modes)
    lines.extend(["     - Item metadata (union of measured splits)", f"   * - | {NAMES[dataset]}",
                  f"       | {rating}", f"       | Min text: {base['min_entity_text_words']} words",
                  f"       | Seed: {base['seed']}"])
    for mode in modes:
        entry = record["splits"][mode]
        if entry["status"] == "unsupported":
            cell = ["Not supported", entry["reason"]]
        else:
            p = entry["parameters"]
            cell = [f"{p['min_user_support']}/{p['item_min_support']}"]
            if entry["status"] == "measured":
                s = entry["stats"]
                users = s.get("preprocessed_users", s["users"])
                items = s.get("preprocessed_items", s["items"])
                cell += [f"{users:,} users", f"{items:,} items", f"Train: {s['train_users']:,} users"]
                for phase, label in (("val", "Val"), ("test", "Test")):
                    flag = " †" if s[f"{phase}_users"] < 1000 else ""
                    cold = percent(s[f"{phase}_cold_target_items"], s[f"{phase}_target_items"])
                    cell += [f"{label}: {s[f'{phase}_users']:,} users{flag}",
                             f"{s[f'{phase}_target_items']:,} items / {cold} cold"]
            else:
                cell += ["Not measured" if entry["status"] == "not_measured" else "Build failed",
                         *textwrap.wrap(entry["reason"], width=60, break_long_words=False,
                                        break_on_hyphens=False)]
                if mode == "user_split":
                    cell += [f"Requested val/test: {p['val_users']:,}/{p['test_users']:,} users"]
            if mode == "temporal":
                cell += [f"Window: {p['temporal_period_hours']:g} h"]
            if mode == "official":
                cell += ["Users/items above: training-only preprocessing"]
        lines += [f"     - | {cell[0]}", *(f"       | {line}" for line in cell[1:])]
    metadata = record["metadata"]
    if metadata["status"] != "measured":
        cell = ["Not measured", "No measured split catalog"]
    else:
        n, images, words = (metadata[key] for key in ("union_items", "image_url_items", "text_ge10_items"))
        cell = [f"Union: {n:,} items",
                "Image URLs: not exposed" if images is None else f"≥1 image: {images:,} ({percent(images, n)})",
                f"≥10 words: {words:,} ({percent(words, n)})"]
        features = metadata.get("precomputed_features")
        if features and features["status"] == "measured":
            cell += ["Precomputed features:"]
            for name, space in features["spaces"].items():
                cell += [f"{name}: {space['items']:,} ({percent(space['items'], n)}), {space['dimension']:,}d"]
            if dataset == "lfm2k":
                cell += ["Text: interaction-derived tags", "Media: pooled by artist ID"]
        elif features:
            cell += ["Precomputed features: not measured", features["reason"]]
    lines += [f"     - | {cell[0]}", *(f"       | {line}" for line in cell[1:])]
    return "\n".join(lines) + "\n"


def merge_archives(paths):
    """Combine disjoint completed workers without losing their provenance."""
    records, runs, seen = [], [], set()
    for path in paths:
        archive = json.loads(path.read_text())
        if "in_progress" in archive or not archive.get("datasets"):
            raise ValueError(f"Incomplete measurement archive: {path}")
        names = [record["dataset"] for record in archive["datasets"]]
        if seen.intersection(names) or len(set(names)) != len(names):
            raise ValueError(f"Duplicate datasets in measurement archives: {path}")
        seen.update(names)
        for record in archive["datasets"]:
            if set(record["splits"]) != set(SPLITS):
                raise ValueError(f"Incomplete splits: {record['dataset']}")
            for mode, entry in record["splits"].items():
                if not unsupported(record["dataset"], mode):
                    if entry["parameters"] != parameters(record["dataset"], mode)[1]:
                        raise ValueError(f"Not installed defaults: {record['dataset']}/{mode}")
        records.extend(archive["datasets"])
        runs.append({"archive": str(path), **{k: v for k, v in archive.items() if k != "datasets"},
                     "datasets": names})
    order = {name: index for index, name in enumerate(NAMES)}
    return {"schema_version": 2, "assembled_at_utc": datetime.now(timezone.utc).isoformat(),
            "runs": runs, "datasets": sorted(records, key=lambda r: order[r["dataset"]])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    measure = sub.add_parser("measure", help="Measure full caches; downloads require --download-root")
    measure.add_argument("--cache-root", type=Path, action="append", required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--datasets", nargs="+", choices=list(NAMES), default=list(NAMES))
    measure.add_argument("--download-root", type=Path, help="Allow missing source/feature downloads into this root")
    measure.add_argument("--features", action="store_true", help="Also measure official precomputed feature coverage")
    render = sub.add_parser("render")
    render.add_argument("archive", type=Path)
    render.add_argument("--dataset", choices=list(NAMES), required=True)
    merge = sub.add_parser("merge", help="Combine completed, disjoint worker archives with their provenance")
    merge.add_argument("archives", nargs="+", type=Path)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render":
        records = json.loads(args.archive.read_text())["datasets"]
        print(render_table(next(r for r in records if r["dataset"] == args.dataset)), end="")
        return
    if args.output.exists():
        parser.error("Output already exists; choose a new archive path")
    if args.command == "merge":
        archive = merge_archives(args.archives)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(archive, indent=2) + "\n")
        return
    project = Path(__file__).resolve().parents[2]
    code = [Path(__file__).resolve(), project / "examples/validation/amazon_evaluation_details.py",
            project / "examples/validation/amazon_profile.py", project / "examples/validation/dataset_sweep.py",
            project / "src/compresso_recsys/builder.py", project / "src/compresso_recsys/retrieval.py",
            project / "src/compresso_recsys/multimodal.py",
            project / "src/compresso_recsys/sequences.py", *sorted((project / "src/compresso_recsys/datasets").glob("*.py"))]
    archive = {"schema_version": 1, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
               "method": "Installed builder defaults, annotations disabled, full sources, exact builder splits",
               "downloads_permitted": args.download_root is not None,
               "code_sha256": {str(path.relative_to(project)): fingerprint(path)["sha256"] for path in code},
               "datasets": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(archive, indent=2) + "\n")
        temporary.replace(args.output)
    for dataset in args.datasets:
        def progress(record):
            archive["in_progress"] = record
            save()
        archive["datasets"].append(measure_dataset(
            dataset, args.cache_root, download_root=args.download_root,
            features=args.features, progress=progress))
        archive.pop("in_progress", None)
        save()


if __name__ == "__main__":
    main()
