"""Resumable public-source downloads for the Amazon support-profile audit.

Use with amazon_profile.py; only one downloader may own a category cache.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

from compresso_recsys.datasets import AmazonReviews2023
from amazon_profile import CATEGORIES, write_json


def download_file(ds, source, *, attempts=3):
    path = ds.root / source["local_path"]
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    error = None
    for attempt in range(attempts):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            headers = ds._headers_for_url(source["url"])
            if offset:
                headers["Range"] = f"bytes={offset}-"
            with urlopen(Request(source["url"], headers=headers), timeout=60) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if match is None or int(match[1]) != offset:
                        raise ValueError("Server returned an unexpected resume range")
                    expected = int(match[3])
                    mode = "ab"
                elif status == 200:
                    expected = int(response.headers.get("Content-Length", 0)) or source.get("size")
                    mode = "wb"  # Server ignored Range: restart, never append duplicate bytes.
                else:
                    raise ValueError(f"Unexpected download response: {status}")
                with temporary.open(mode) as output:
                    for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        output.write(chunk)
                if expected is not None and temporary.stat().st_size != expected:
                    raise ValueError(f"Incomplete download: expected {expected} bytes")
            temporary.replace(path)
            return
        except Exception as exc:
            error = exc
            print(f"{ds.category}/{path.name}: attempt {attempt + 1} failed: {exc}", flush=True)
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Could not download {source['url']}; partial data retained for resume") from error


def download_category(category, *, data_dir, output):
    ds = AmazonReviews2023(data_dir=data_dir, category=category, show_progress=False)
    completed = []
    for config in (ds.interactions_config, ds.metadata_config):
        mirror = ds._mirror_source_for_config(config)
        if config == ds.metadata_config:
            cached = ds._cached_metadata_sources()
        else:
            hf = ds._hf_source_for_config(config)
            cached = hf if all((ds.root / s["local_path"]).is_file() for s in hf) else []
            if not cached and all((ds.root / s["local_path"]).is_file() for s in mirror):
                cached = mirror
        if cached:
            completed.extend(cached)
            continue
        # HF Parquet where exported, otherwise official raw JSONL. The gzip
        # McAuley mirror remains a fallback if HF is unavailable.
        try:
            group = ds._hf_source_for_config(config)
            for source in group:
                print(f"Downloading {category}/{Path(source['local_path']).name}", flush=True)
                download_file(ds, source)
        except Exception as exc:
            print(f"{category}: trying compressed McAuley source after {exc}", flush=True)
            group = mirror
            for source in group:
                download_file(ds, source)
        completed.extend(group)
    write_json(output / category / "downloads.json", {"category": category, "status": "complete", "sources": [
        {**source, "local_path": str(source["local_path"])} for source in completed]})
    print(f"Sources ready: {category}", flush=True)
    return category


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/amazon-profiles"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(download_category, category, data_dir=args.data_dir, output=args.output): category
                for category in dict.fromkeys(args.categories)}
        for job in as_completed(jobs):
            try:
                job.result()
            except Exception as exc:
                category = jobs[job]
                failures.append(category)
                write_json(args.output / category / "downloads.json",
                           {"category": category, "status": "failed", "reason": str(exc)})
                print(f"Download FAILED: {category}: {exc}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
