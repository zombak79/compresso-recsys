"""Read-only Amazon metadata audit on verified split catalogs (no downloads).

Export catalog IDs on the audit host using its frozen profiler/builder, then
scan local raw metadata in bounded memory. Results do not change defaults.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from compresso_recsys.datasets.amazon2023 import AmazonReviews2023, DEFAULT_TEXT_FIELDS
from amazon_metadata_report import ATTRIBUTE_KEYS


FIELDS = (*DEFAULT_TEXT_FIELDS, "details", "store", "main_category")
THRESHOLDS = (1, 5, 10, 20, 30, 50, 100)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def export_category(record, input_root, output, source_code):
    import amazon_profile as profile
    assert profile.RUN_CODE == source_code, "Frozen profiler/builder mismatch"
    category = record["category"]
    destination = output / category / "catalogs.json"
    context = {"prepared_sha256": record["input"]["prepared_sha256"],
               "record_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
               "source_code": source_code}
    if destination.exists():
        assert json.loads(destination.read_text())["context"] == context
        return category
    frame, manifest = profile.load_prepared(input_root / category, category)
    assert manifest["prepared"]["sha256"] == context["prepared_sha256"]
    graph = profile.CoreGraph(frame)
    result = {"category": category, "context": context, "splits": {}}
    for split, entry in record["splits"].items():
        params = entry["parameters"]
        if split == "temporal":
            data = frame
            prepared = None
        else:
            data = frame.iloc[graph.filter(params["min_user_support"], params["item_min_support"])]
            prepared = {"users": int(data.user_id.nunique()), "items": int(data.item_id.nunique()),
                        "pairs": len(data)}
        args, _ = profile.builder._resolve_args(profile.builder._build_args(
            dataset="amazon2023", amazon_category=category, split_mode=split,
            annotation_source="none", eval_draws=1, show_progress=False, **params))
        payload = profile.builder._build_split_payload(args, profile.RecSysDataset(), data)
        stats = profile.payload_stats(payload, preprocessed=prepared)
        assert all(entry["stats"][key] == value for key, value in stats.items()), (category, split)
        item_ids = np.asarray(payload["item_ids"]).astype(str)
        train_ids = np.asarray(payload.get("train_item_ids", item_ids)).astype(str)
        train_counts = np.bincount(payload["x_train"].indices, minlength=len(train_ids))
        catalogs = {"catalog": item_ids.tolist(),
                    "train": train_ids[train_counts > 0].tolist()}
        for phase in ("val", "test"):
            ids = np.asarray(payload.get(phase + "_item_ids", item_ids)).astype(str)
            targets = payload[phase + "_holdout"]["target_indices"]
            indices = np.unique(np.concatenate(targets))
            catalogs[phase] = ids[indices].tolist()
            assert len(catalogs[phase]) == entry["stats"][phase + "_target_items"]
        if prepared is not None:
            catalogs["preprocessed"] = sorted(data.item_id.unique().astype(str).tolist())
        result["splits"][split] = {"parameters": params, "stats": stats, "ids": catalogs,
            "train_pairs_by_item": {str(i): int(n) for i, n in zip(train_ids, train_counts) if n}}
        del payload
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(destination, result)
    return category


def normalize(value):
    """Normalize Arrow/Pandas array containers without altering text content."""
    if isinstance(value, np.ndarray):
        return [normalize(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, dict):
        return {key: normalize(v) for key, v in value.items()}
    return value


def image_source_urls(value):
    """All recorded HTTP(S) image links, including Prime Video width variants."""
    if isinstance(value, dict):
        return [u for key, v in value.items() if key != "variant" for u in image_source_urls(v)]
    if isinstance(value, (list, tuple)):
        return [u for v in value for u in image_source_urls(v)]
    if isinstance(value, str) and value.strip().startswith(("https://", "http://")):
        return [value.strip()]
    return []


def measure_row(row, category=None):
    cls = AmazonReviews2023
    normalized = {key: normalize(value) for key, value in row.items()}
    text = {field: cls._metadata_value_to_text(
        cls._parse_details(normalized.get(field)) if field == "details" else normalized.get(field))
        for field in FIELDS}
    native = cls.build_entity_text(row, DEFAULT_TEXT_FIELDS)
    default = cls.build_entity_text(normalized, DEFAULT_TEXT_FIELDS)
    extended = cls.build_entity_text(normalized, (*DEFAULT_TEXT_FIELDS, "details", "store"))
    image_data = normalized.get("images")
    # Hugging Face Parquet encodes Sequence[struct] as a struct of arrays.
    if isinstance(image_data, dict) and any(isinstance(v, list) for v in image_data.values()):
        lengths = {len(v) for v in image_data.values() if isinstance(v, list)}
        if len(lengths) != 1:
            raise ValueError("Image array columns have inconsistent lengths")
        image_data = [{key: value[i] if isinstance(value, list) else value
                       for key, value in image_data.items()} for i in range(next(iter(lengths)))]
    if isinstance(image_data, str):
        image_data = cls._parse_images(image_data)
    images = image_source_urls(image_data)
    details = cls._parse_details(normalized.get("details"))
    selected = {key: details[key] for key in ATTRIBUTE_KEYS.get(category, ())
                if isinstance(details, dict) and key in details}
    curated_row = {**normalized, "details": selected}
    curated = cls.build_entity_text(curated_row, (*DEFAULT_TEXT_FIELDS, "store", "details"))
    return {"field_words": {field: len(value.split()) for field, value in text.items()},
            "native_words": len(native.split()), "default_words": len(default.split()),
            "extended_words": len(extended.split()), "curated_words": len(curated.split()),
            "source_image": any(u.startswith(("https://", "http://")) for u in images),
            "adapter_image": any(u.startswith(("https://", "http://"))
                                 for u in cls.extract_image_urls(row.get("images"))),
            "native_differs": native != default,
            "html": bool(re.search(r"<\s*(?:p|br|div|span|a|li|ul|b)\b", default, flags=re.I)),
            "details_keys": [str(key) for key, value in details.items()
                             if cls._metadata_value_to_text(value)] if isinstance(details, dict) else []}


def source_rows(path, kind):
    if kind == "parquet":
        parquet = pq.ParquetFile(path)
        columns = [c for c in ("parent_asin", *FIELDS, "images") if c in parquet.schema_arrow.names]
        for batch in parquet.iter_batches(batch_size=4096, columns=columns):
            # Keep native ndarray cells, exactly as pd.read_parquet in the adapter.
            yield from batch.to_pandas().to_dict("records")
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def summarize(ids, measurements, weights=None):
    missing = set(ids).difference(measurements)
    if missing:
        raise ValueError(f"Missing metadata for {len(missing)} retained items")
    rows = [measurements[i] for i in ids]
    count = len(rows)
    result = {"items": count,
              "field_nonempty": {f: sum(r["field_words"][f] > 0 for r in rows) for f in FIELDS},
              "description_or_features": sum(r["field_words"]["description"] > 0 or
                                              r["field_words"]["features"] > 0 for r in rows),
              "source_image": sum(r["source_image"] for r in rows),
              "adapter_image": sum(r["adapter_image"] for r in rows),
              "text_and_image": sum(r["default_words"] > 0 and r["source_image"] for r in rows),
              "native_differs": sum(r["native_differs"] for r in rows),
              "html": sum(r["html"] for r in rows),
              "word_counts": {}}
    for name in ("native", "default", "extended", "curated"):
        words = np.array([r[name + "_words"] for r in rows])
        result["word_counts"][name] = {"nonempty": int((words > 0).sum()),
            "quantiles": np.percentile(words, [0, 10, 50, 90, 99, 100]).tolist() if count else [],
            "below": {str(t): int((words < t).sum()) for t in THRESHOLDS}}
        if weights is not None:
            result["word_counts"][name]["train_pairs_removed"] = {
                str(t): sum(weights[i] for i in ids if measurements[i][name + "_words"] < t)
                for t in THRESHOLDS}
    if weights is not None:
        result["train_pairs"] = sum(weights.values())
    return result


def scan_category(category, data_dir, catalog_root, output):
    catalogs_path = catalog_root / category / "catalogs.json"
    catalogs = json.loads(catalogs_path.read_text())
    ds = AmazonReviews2023(data_dir=data_dir, category=category, show_progress=False)
    sources = ds._cached_metadata_sources()
    if not sources:
        raise FileNotFoundError(f"No complete metadata cache for {category}")
    files = [{"path": str(ds.root / s["local_path"]), "kind": s["kind"],
              "size": (ds.root / s["local_path"]).stat().st_size,
              "mtime_ns": (ds.root / s["local_path"]).stat().st_mtime_ns} for s in sources]
    context = {"catalog_sha256": hashlib.sha256(catalogs_path.read_bytes()).hexdigest(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "recipe_sha256": hashlib.sha256(json.dumps(ATTRIBUTE_KEYS, sort_keys=True).encode()).hexdigest(),
               "formatter_sha256": {
                   cls.__name__: hashlib.sha256(Path(inspect.getfile(cls)).read_bytes()).hexdigest()
                   for cls in (AmazonReviews2023, AmazonReviews2023.__bases__[0])
               },
               "files": files}
    destination = output / category / "metadata.json"
    if destination.exists():
        assert json.loads(destination.read_text())["context"] == context
        return category
    wanted = set(i for entry in catalogs["splits"].values() for ids in entry["ids"].values() for i in ids)
    measurements, total, duplicates = {}, 0, 0
    for source in files:
        for row in source_rows(Path(source["path"]), source["kind"]):
            total += 1
            item = str(row.get("parent_asin"))
            if item not in wanted:
                continue
            if item in measurements:
                duplicates += 1
                continue
            measurements[item] = measure_row(row, category)
    result = {"category": category, "context": context, "source_rows_scanned": total,
              "duplicate_retained_rows": duplicates, "splits": {},
              "union": summarize(sorted(wanted), measurements),
              "details_keys": Counter(k for r in measurements.values() for k in r["details_keys"]).most_common(60)}
    for split, entry in catalogs["splits"].items():
        result["splits"][split] = {role: summarize(ids, measurements,
            weights=entry["train_pairs_by_item"] if role == "train" else None)
            for role, ids in entry["ids"].items()}
    result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(destination, result)
    return category


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("catalogs", "scan"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--categories", nargs="+")
    args = parser.parse_args()
    archive = json.loads(args.archive.read_text())
    records = [r for r in archive["profiles"] if not args.categories or r["category"] in args.categories]
    if args.workers < 1 or (args.mode == "catalogs" and args.input_root is None) or (
            args.mode == "scan" and args.catalog_root is None):
        parser.error("Positive worker count and input-root/catalog-root are required")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_category, r, args.input_root, args.output, archive["source_code"])
                   if args.mode == "catalogs" else pool.submit(
                       scan_category, r["category"], args.data_dir, args.catalog_root, args.output): r["category"]
                   for r in records}
        for future in as_completed(futures):
            print(f"Complete: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
