"""Measure support-only Amazon defaults. See docs/source/amazon-profiling.rst.

No sampling, rating threshold, text threshold, embeddings, or model-score tuning.
The search uses the real splitter and its temporal support-filter implementation.
"""
from __future__ import annotations

import argparse
import gc
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
import time
from itertools import islice
from importlib.metadata import version

for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "2")

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys import builder
from compresso_recsys.datasets import AmazonReviews2023
from compresso_recsys.datasets.base import RecSysDataset


CATEGORIES = (
    "All_Beauty", "Amazon_Fashion", "Appliances", "Arts_Crafts_and_Sewing",
    "Automotive", "Baby_Products", "Beauty_and_Personal_Care", "Books",
    "CDs_and_Vinyl", "Cell_Phones_and_Accessories", "Clothing_Shoes_and_Jewelry",
    "Digital_Music", "Electronics", "Gift_Cards", "Grocery_and_Gourmet_Food",
    "Handmade_Products", "Health_and_Household", "Health_and_Personal_Care",
    "Home_and_Kitchen", "Industrial_and_Scientific", "Kindle_Store",
    "Magazine_Subscriptions", "Movies_and_TV", "Musical_Instruments",
    "Office_Products", "Patio_Lawn_and_Garden", "Pet_Supplies", "Software",
    "Sports_and_Outdoors", "Subscription_Boxes", "Tools_and_Home_Improvement",
    "Toys_and_Games", "Video_Games",
)
SPLITS = ("user_split", "item_split", "leave_last_out", "temporal")
USER_SUPPORTS = (2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50)
SCHEMA = 3
SELECTION_POLICY = "user-rich-short-history-v1"
EVAL_USER_TARGET = 1000


def user_supports(split):
    # LLO has a source plus separate train/validation/test targets. Lowering
    # the preprocessing threshold cannot lower this structural floor.
    floor = builder.LEAVE_LAST_OUT_MIN_HISTORY if split == "leave_last_out" else 2
    return tuple(value for value in USER_SUPPORTS if value >= floor)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_fingerprint():
    """Capture the code at process startup, not changed files midway through a run."""
    package = Path(builder.__file__).parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(package)).encode())
        digest.update(path.read_bytes())
    return {"script_sha256": sha256(Path(__file__)), "package_sha256": digest.hexdigest()}


RUN_CODE = code_fingerprint()


def seal_prepared(directory, category, manifest, rows):
    manifest = {**manifest, "prepared": {"category": category,
                "sha256": sha256(directory / "input.parquet"), "rows": rows}}
    write_json(directory / "input.json", manifest)
    return manifest


def load_prepared(directory, category, *, load_frame=True):
    """Use transferred, checksum-verified data without pretending raw files exist."""
    manifest = json.loads((directory / "input.json").read_text())
    prepared = manifest.get("prepared", {})
    if (manifest.get("schema") != SCHEMA or prepared.get("category") != category
            or manifest.get("all_ratings") is not True or manifest.get("min_entity_text_words") != 0
            or manifest.get("min_user_support_upper_bound") != min(USER_SUPPORTS)):
        raise ValueError("Prepared input has incompatible category or preprocessing policy")
    if prepared.get("sha256") != sha256(directory / "input.parquet"):
        raise ValueError("Prepared input checksum mismatch")
    if not load_frame:
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(directory / "input.parquet")
        if (parquet.metadata.num_rows != prepared.get("rows")
                or set(parquet.schema_arrow.names) != {"user_id", "item_id", "value", "timestamp"}):
            raise ValueError("Prepared input rows or schema do not match the manifest")
        return None, manifest
    frame = pd.read_parquet(directory / "input.parquet")
    if (len(frame) != prepared.get("rows")
            or set(frame.columns) != {"user_id", "item_id", "value", "timestamp"}
            or not frame.value.eq(1.).all()):
        raise ValueError("Prepared input rows, schema or binary values do not match the manifest")
    return frame, manifest


def limits(category):
    # The requested exception is strictly fewer than 500,000 users.
    return (499999, 100000) if category in {"Books", "Electronics"} else (100000, 20000)


def sources(ds, config, *, cached_only):
    """Prefer a complete cached source; never mistake partial shards for a dataset."""
    mirror = ds._mirror_source_for_config(config)
    if config == ds.metadata_config:
        cached = ds._cached_metadata_sources()
        if cached:
            return cached
    if config == ds.interactions_config:
        hf = ds._hf_source_for_config(config)
        if all((ds.root / s["local_path"]).exists() for s in hf):
            return hf
    if all((ds.root / s["local_path"]).exists() for s in mirror):
        return mirror
    if cached_only:
        raise FileNotFoundError(f"Complete cached source missing: {config}")
    errors = []
    for group in [mirror, None]:
        try:
            if group is None:
                group = ds._hf_source_for_config(config)
            for source in group:
                path = ds.root / source["local_path"]
                if not path.exists():
                    ds._download_file(source["url"], path, size=source.get("size"))
            return group
        except Exception as exc:
            errors.append(exc)
    raise RuntimeError(f"Could not obtain {config}") from errors[-1]


def rating_chunks(path):
    for chunk in pd.read_csv(path, chunksize=500000,
                             usecols=["user_id", "parent_asin", "rating", "timestamp"]):
        chunk = chunk.rename(columns={"parent_asin": "item_id", "rating": "value"})
        # Match the adapter's canonicalization, including its missing-ID handling.
        chunk["user_id"] = chunk.user_id.astype(str)
        chunk["item_id"] = chunk.item_id.astype(str)
        chunk["value"] = pd.to_numeric(chunk.value, errors="coerce")
        chunk["timestamp"] = pd.to_numeric(chunk.timestamp, errors="coerce")
        yield chunk.dropna(subset=["value"])


def metadata_ids(path, kind):
    if kind == "parquet":
        import pyarrow.parquet as pq
        for batch in pq.ParquetFile(path).iter_batches(columns=["parent_asin"]):
            yield from batch.to_pandas().parent_asin.astype(str)
    else:
        # Only keep IDs. Full metadata/text expansion can exceed laptop RAM.
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as stream:
            for line in stream:
                record = json.loads(line)
                if "parent_asin" not in record:
                    raise ValueError(f"Metadata has no parent_asin: {path}")
                yield str(record["parent_asin"])


def _lookup_ids(connection, table, ids, *, minimum=0):
    """Bound SQL parameters and memory to one input chunk, not the category."""
    found = set()
    values = iter(pd.unique(ids))
    while batch := list(islice(values, 500)):
        placeholders = ",".join("?" for _ in batch)
        condition = " AND n >= ?" if minimum else ""
        parameters = [*batch, minimum] if minimum else batch
        found.update(row[0] for row in connection.execute(
            f"SELECT id FROM {table} WHERE id IN ({placeholders}){condition}", parameters))
    return found


def prepare(ds, directory, *, cached_only=False, load_frame=True):
    """Disk-backed ID counts and streamed parquet keep laptop memory bounded.

    The server loads the resulting frame; the local hybrid coordinator passes
    load_frame=False, so it never materializes the whole prepared category.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    ratings = sources(ds, ds.interactions_config, cached_only=cached_only)
    metadata = sources(ds, ds.metadata_config, cached_only=cached_only)
    all_sources = ratings + metadata
    stamps = [{"path": str(s["local_path"]), "size": (ds.root / s["local_path"]).stat().st_size,
               "mtime_ns": (ds.root / s["local_path"]).stat().st_mtime_ns} for s in all_sources]
    manifest_path, prepared_path = directory / "input.json", directory / "input.parquet"
    if manifest_path.exists() and prepared_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("schema") == SCHEMA and manifest.get("files") == stamps
                and manifest.get("min_user_support_upper_bound") == min(USER_SUPPORTS)
                and manifest.get("all_ratings") is True and manifest.get("min_entity_text_words") == 0):
            if "prepared" not in manifest:
                manifest = seal_prepared(directory, ds.category, manifest,
                                         pq.ParquetFile(prepared_path).metadata.num_rows)
            return load_prepared(directory, ds.category, load_frame=load_frame)
    print(f"{ds.category}: counting users (all ratings)", flush=True)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = prepared_path.with_suffix(".parquet.tmp")
    schema = pa.schema([("user_id", pa.string()), ("item_id", pa.string()),
                        ("value", pa.float64()), ("timestamp", pa.float64())])
    input_events = metadata_events = prepared_rows = 0
    endpoints = None
    # Only this invocation's scratch DB is cleaned up. Raw downloads and any
    # previously completed parquet remain untouched until atomic replacement.
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=directory) as scratch:
        connection = sqlite3.connect(str(Path(scratch) / "counts.sqlite"))
        try:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA cache_size=-65536;
                PRAGMA temp_store=FILE;
                CREATE TABLE users (id TEXT PRIMARY KEY, n INTEGER NOT NULL) WITHOUT ROWID;
                CREATE TABLE metadata (id TEXT PRIMARY KEY) WITHOUT ROWID;
            """)
            for source in ratings:
                for chunk in rating_chunks(ds.root / source["local_path"]):
                    connection.executemany(
                        "INSERT INTO users VALUES (?,?) ON CONFLICT(id) DO UPDATE SET n=n+excluded.n",
                        ((str(user), int(count)) for user, count in chunk.user_id.value_counts(sort=False).items()))
                    connection.commit()
                    input_events += len(chunk)
            input_users = connection.execute("SELECT count(*) FROM users").fetchone()[0]
            eligible_users = connection.execute("SELECT count(*) FROM users WHERE n>=?",
                                                (min(USER_SUPPORTS),)).fetchone()[0]
            print(f"{ds.category}: {eligible_users:,} users can meet the minimum history", flush=True)
            print(f"{ds.category}: checking metadata IDs", flush=True)
            for source in metadata:
                values = iter(metadata_ids(ds.root / source["local_path"], source["kind"]))
                while batch := list(islice(values, 10000)):
                    connection.executemany("INSERT OR IGNORE INTO metadata VALUES (?)", ((value,) for value in batch))
                    connection.commit()
            with pq.ParquetWriter(temporary, schema) as writer:
                def append(frame):
                    nonlocal prepared_rows
                    if len(frame):
                        frame = frame.loc[:, schema.names].copy()
                        frame["value"] = 1.
                        writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
                        prepared_rows += len(frame)

                for source in ratings:
                    for chunk in rating_chunks(ds.root / source["local_path"]):
                        valid_items = _lookup_ids(connection, "metadata", chunk.item_id)
                        chunk = chunk[chunk.item_id.isin(valid_items)]
                        metadata_events += len(chunk)
                        finite = chunk[np.isfinite(chunk.timestamp)]
                        if len(finite):
                            current = finite.loc[[finite.timestamp.idxmin(), finite.timestamp.idxmax()]]
                            pool = current if endpoints is None else pd.concat([endpoints, current], ignore_index=True)
                            endpoints = pool.loc[[pool.timestamp.idxmin(), pool.timestamp.idxmax()]].drop_duplicates()
                        eligible = _lookup_ids(connection, "users", chunk.user_id, minimum=min(USER_SUPPORTS))
                        append(chunk[chunk.user_id.isin(eligible)])
                if endpoints is not None:
                    eligible = _lookup_ids(connection, "users", endpoints.user_id, minimum=min(USER_SUPPORTS))
                    # Even a one-event user must still set the original temporal
                    # boundaries. Support pruning later removes this user.
                    append(endpoints[~endpoints.user_id.isin(eligible)])
        finally:
            connection.close()
    # Removing histories with <2 TOTAL events is safe for all searched support
    # thresholds. Do not globally prune items, which would change temporal stages.
    manifest = {"schema": SCHEMA, "files": stamps, "sources": [
        {"url": s["url"], "sha256": sha256(ds.root / s["local_path"])} for s in all_sources],
        "raw_valid_rating_events": input_events, "raw_users": input_users,
        "metadata_eligible_events": metadata_events,
        "eligible_upper_bound_events": prepared_rows, "min_user_support_upper_bound": min(USER_SUPPORTS),
        "all_ratings": True, "min_entity_text_words": 0}
    temporary.replace(prepared_path)
    manifest = seal_prepared(directory, ds.category, manifest, prepared_rows)
    return load_prepared(directory, ds.category, load_frame=load_frame)


class CoreGraph:
    def __init__(self, frame):
        self.frame = frame
        self.users = pd.factorize(frame.user_id)[0].astype(np.int32)
        self.items = pd.factorize(frame.item_id)[0].astype(np.int32)

    def filter(self, user_support, item_support):
        users, items = self.users, self.items
        indices = np.arange(len(users))
        while len(indices):
            before = len(indices)
            indices = indices[np.bincount(items[indices])[items[indices]] >= item_support]
            indices = indices[np.bincount(users[indices])[users[indices]] >= user_support]
            if len(indices) == before:
                break
        return indices

    def counts(self, user_support, item_support):
        indices = self.filter(user_support, item_support)
        return {"users": len(np.unique(self.users[indices])),
                "items": len(np.unique(self.items[indices])), "pairs": len(indices)}


class TemporalGraph:
    """Reuse sparse windows, but run the builder's exact support fixed point.

    Integer IDs are sufficient for counts; the final candidate is also rebuilt
    with original IDs, event sequences and the normal split implementation.
    """
    def __init__(self, frame, period_hours):
        timestamps = builder._timestamps_in_seconds(frame.timestamp)
        valid = np.isfinite(timestamps)
        if not valid.any():
            raise ValueError("No finite timestamps")
        stamps = timestamps[valid]
        users, user_ids = pd.factorize(frame.loc[valid, "user_id"], sort=True)
        items, item_ids = pd.factorize(frame.loc[valid, "item_id"], sort=True)
        self.n_users, self.n_items = len(user_ids), len(item_ids)
        end, period = stamps.max(), period_hours * 3600
        if end - 3 * period <= stamps.min():
            raise ValueError("Three temporal windows do not fit the timestamp span")
        self.windows = []
        for offset in (3, 2, 1):
            boundary = end - offset * period
            masks = (stamps < boundary,
                     (stamps >= boundary) & ((stamps < boundary + period) if offset > 1 else True))
            matrices = []
            for mask in masks:
                matrix = csr_matrix((np.ones(mask.sum(), dtype=np.float32),
                                     (users[mask], items[mask])), shape=(self.n_users, self.n_items))
                matrix.sum_duplicates()
                matrix.data[:] = 1
                matrices.append(matrix)
            observed = np.union1d(matrices[0].indices, matrices[1].indices)
            self.windows.append((*matrices, observed))

    def counts(self, user_support, item_support):
        inherited = np.array([], dtype=np.int64)
        all_users = set()
        phases = []
        for stage, (source, target, observed) in zip(("train", "validation", "test"), self.windows):
            item_ids = np.concatenate((inherited, np.setdiff1d(observed, inherited)))
            user_ids = np.flatnonzero((source.getnnz(axis=1) >= 1) & (target.getnnz(axis=1) >= 1)
                                     & (source.maximum(target).getnnz(axis=1) >= user_support))
            source, target, user_ids, retained, _ = builder._filter_temporal_pair(
                source[user_ids][:, item_ids], target[user_ids][:, item_ids],
                user_ids=user_ids, item_ids=item_ids, inherited_items=len(inherited),
                min_user_support=user_support, item_min_support=item_support,
                min_source_items=1, min_target_items=1, stage=stage)
            inherited = retained
            all_users.update(user_ids.tolist())
            phases.append({"users": len(user_ids), "items": len(retained),
                           "pairs": source.maximum(target).nnz})
        return {"users": len(all_users), "items": len(inherited), "pairs": phases[0]["pairs"],
                "phases": phases}


def boundary_candidates(measure, max_users, max_items, user_supports=USER_SUPPORTS):
    """Smallest feasible item support per user threshold (monotone size search)."""
    measured = {}

    def check(us, it):
        key = (us, it)
        if key not in measured:
            try:
                measured[key] = measure(us, it)
            except ValueError as exc:
                if "no users after support filtering" not in str(exc):
                    raise
                measured[key] = {"users": 0, "items": 0, "pairs": 0, "error": str(exc)}
        return measured[key]

    def oversized(stats):
        return stats["users"] > max_users or stats["items"] > max_items

    for us in user_supports:
        low, high = 0, 1
        while oversized(check(us, high)):
            low, high = high, high * 2
        while high - low > 1:
            middle = (low + high) // 2
            if oversized(check(us, middle)):
                low = middle
            else:
                high = middle
        # On tiny graphs, modest extra item support may improve user-split
        # vocabulary coverage. Evaluate these alternatives rather than assume.
        if high <= 5:
            for it in (2, 3, 4, 5):
                check(us, it)
        # Probe a few stronger item thresholds too: reducing singleton/rare
        # items may improve evaluation coverage and the user/item balance.
        for it in {high + 1, math.ceil(high * 1.25), math.ceil(high * 1.5), high * 2}:
            check(us, it)
        print(f"  support {us}/{high}: {check(us, high)['users']:,} users, "
              f"{check(us, high)['items']:,} items", flush=True)
    candidates = [{"min_user_support": us, "item_min_support": it, "counts": stats}
                  for (us, it), stats in measured.items() if stats["users"] > 0 and not oversized(stats)]
    return candidates, [{"min_user_support": us, "item_min_support": it, **stats}
                        for (us, it), stats in measured.items()]


def size_score(stats):
    # Penalize item-heavy graphs, but saturate at one user per item. An extreme
    # ratio from a tiny catalog earns no extra reward; catalog coverage matters.
    coverage = min(1., stats["users"] / 50000) * min(1., stats["items"] / 10000)
    balance = min(1., stats["users"] / max(1, stats["items"]))
    return coverage * balance, stats["pairs"], stats["users"], stats["items"]


def candidate_quality(entry):
    stats = entry["stats"]
    size = {"users": stats.get("preprocessed_users", stats["users"]),
            "items": stats.get("preprocessed_items", stats["items"]), "pairs": stats["pairs"]}
    minimum_eval = min(stats["val_users"], stats["test_users"])
    sufficient_eval = minimum_eval >= EVAL_USER_TARGET
    # Prefer longer histories when they ALREADY yield the desired usable,
    # user-rich size. Otherwise let shorter histories rescue sparse categories.
    healthy_core = (entry["parameters"]["min_user_support"] >= 5
                    and size["users"] >= 50000 and size["items"] >= 10000
                    and size["users"] >= size["items"])
    score, *ties = size_score(size)
    return sufficient_eval, healthy_core, score * min(1., minimum_eval / EVAL_USER_TARGET), *ties, minimum_eval


def payload_stats(payload, *, preprocessed=None):
    train_ids = payload["train_user_ids"]
    all_users = set(train_ids)
    result = {"users": 0, "items": len(payload["item_ids"]), "pairs": int(payload["x_train"].nnz),
              "train_users": len(train_ids), "train_items": len(payload.get("train_item_ids", payload["item_ids"])),
              "train_observed_items": len(np.unique(payload["x_train"].indices))}
    for phase in ("val", "test"):
        holdout = payload[phase + "_holdout"]
        users = np.asarray(holdout["user_ids"])
        all_users.update(users.tolist())
        result[phase + "_users"] = len(np.unique(users))
        result[phase + "_target_pairs"] = sum(len(row) for row in holdout["target_indices"])
    result["users"] = len(all_users)
    if preprocessed is not None:
        # Enforce pre-split limits too: held-out users/unused items cannot hide
        # an oversized preprocessing result behind a smaller training matrix.
        result["preprocessed_users"] = preprocessed["users"]
        result["preprocessed_items"] = preprocessed["items"]
        result["preprocessed_pairs"] = preprocessed["pairs"]
    return result


def verify_candidate(frame, graph, candidate, *, category, split, period_hours, seed):
    params = {key: candidate[key] for key in ("min_user_support", "item_min_support")}
    params.update(seed=seed, min_entity_text_words=0, set_all_values_to=1.,
                  temporal_period_hours=period_hours)
    counts = candidate["counts"]
    if split == "user_split":
        # These are protocol partitions, not preprocessing subsampling.
        params.update(val_users=max(1, min(5000, counts["users"] // 10)),
                      test_users=max(1, min(5000, counts["users"] // 10)))
    args, _ = builder._resolve_args(builder._build_args(
        dataset="amazon2023", amazon_category=category, split_mode=split,
        annotation_source="none", eval_draws=1, show_progress=False, **params))
    if split == "temporal":
        proc = frame
    else:
        proc = frame.iloc[graph.filter(params["min_user_support"], params["item_min_support"])]
    payload = builder._build_split_payload(args, RecSysDataset(), proc)
    stats = payload_stats(payload, preprocessed=None if split == "temporal" else counts)
    if split == "temporal":
        actual = {key: stats[key] for key in ("users", "items", "pairs")}
        expected = {key: counts[key] for key in actual}
        if actual != expected:
            raise AssertionError(f"Temporal screening differs from real builder: {expected} != {actual}")
    if stats["val_users"] == 0 or stats["test_users"] == 0:
        raise ValueError("No eligible validation or test users")
    max_users, max_items = limits(category)
    if max(stats["users"], stats.get("preprocessed_users", 0)) > max_users or max(
            stats["items"], stats.get("preprocessed_items", 0)) > max_items:
        raise AssertionError("Verified split exceeds size limits")
    warnings = []
    if min(stats["val_users"], stats["test_users"]) < EVAL_USER_TARGET:
        warnings.append(f"Fewer than {EVAL_USER_TARGET:,} eligible users in validation or test")
    effective_users = stats.get("preprocessed_users", stats["users"])
    effective_items = stats.get("preprocessed_items", stats["items"])
    if effective_users < 50000 or effective_items < 10000:
        warnings.append("Below the preferred size range; support filtering cannot add data")
    if effective_users < effective_items:
        warnings.append("More items than users")
    if params["min_user_support"] < 5:
        warnings.append("Short-history fallback")
    return {"parameters": params, "stats": stats, "warnings": warnings}


def profile_category(category, args):
    directory = args.output / category
    result_path = directory / "result.json"
    if getattr(args, "prepared_root", None) is not None:
        frame, manifest = load_prepared(args.prepared_root / category, category)
    else:
        ds = AmazonReviews2023(data_dir=args.data_dir, category=category, min_entity_text_words=0,
                               show_progress=True)
        frame, manifest = prepare(ds, directory, cached_only=args.cached_only)
    context = {"schema": SCHEMA, **RUN_CODE, "input": manifest,
               "environment": {"python": platform.python_version(), **{
                   name: version(name) for name in ("numpy", "pandas", "scipy", "pyarrow")}},
               "seed": args.seed, "user_supports": list(USER_SUPPORTS), "limits": list(limits(category)),
               "selection_policy": SELECTION_POLICY,
               "period_hours": args.temporal_period_hours}
    result = {"category": category, "context": context, "splits": {}}
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous.get("context") == context:
            result = previous
    graph = CoreGraph(frame)
    candidates = None
    for split in args.splits:
        if result["splits"].get(split, {}).get("status") == "verified":
            print(f"{category}/{split}: resumed verified result", flush=True)
            continue
        started = time.monotonic()
        print(f"{category}/{split}: searching support thresholds", flush=True)
        try:
            if split == "temporal":
                temporal = TemporalGraph(frame, args.temporal_period_hours)
                options, measurements = boundary_candidates(temporal.counts, *limits(category),
                                                             user_supports=user_supports(split))
                del temporal
            else:
                if candidates is None:
                    candidates, core_measurements = boundary_candidates(graph.counts, *limits(category))
                options, measurements = candidates, core_measurements
                options = [option for option in options if option["min_user_support"] in user_supports(split)]
            options = sorted(options, key=lambda c: size_score(c["counts"]), reverse=True)
            # Check all feasible boundary candidates, not just the first build
            # that succeeds. User-split vocabulary filtering changes coverage.
            verified, failures = [], []
            for candidate in options:
                try:
                    verified.append(verify_candidate(frame, graph, candidate, category=category,
                                    split=split, period_hours=args.temporal_period_hours, seed=args.seed))
                except ValueError as exc:
                    failures.append({"candidate": candidate, "reason": str(exc)})
                gc.collect()
            if not verified:
                entry = {"status": "no_viable_candidate", "measurements": measurements,
                         "failures": failures, "reason": "No searched support setting gives nonempty train/val/test within limits"}
            else:
                selected = max(verified, key=candidate_quality)
                entry = {"status": "verified", **selected, "candidates": verified,
                         "measurements": measurements, "failures": failures}
                print(f"{category}/{split}: selected {selected['parameters']} {selected['stats']}", flush=True)
            entry["seconds"] = time.monotonic() - started
        except Exception as exc:
            entry = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        result["splits"][split] = entry
        write_json(result_path, result)
        if getattr(args, "write_global_report", True):
            write_report(args.output)
        gc.collect()
    return result


def write_report(output):
    records = [json.loads(path.read_text()) for path in sorted(output.glob("*/result.json"))]
    lines = ["# Amazon support-only profiling", "", "Measured all-rating, metadata-eligible interactions. "
             "User counts are preprocessed users for random/LLO splits and the unique union across temporal stages. "
             "Catalog counts include later/cold items. No sampling. No model scores were used.", "",
             "| Category | Split | Status | User/item support | Users | Items | Train pairs | Val users | Test users | Notes |",
             "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    for record in records:
        for split, entry in record["splits"].items():
            params, stats = entry.get("parameters", {}), entry.get("stats", {})
            support = f"{params['min_user_support']}/{params['item_min_support']}" if params else "—"
            values = [record["category"], split, entry["status"], support,
                      stats.get("preprocessed_users", stats.get("users", "—")),
                      stats.get("preprocessed_items", stats.get("items", "—")), stats.get("pairs", "—"),
                      stats.get("val_users", "—"), stats.get("test_users", "—"),
                      "; ".join(entry.get("warnings", [])) or entry.get("reason", "")]
            lines.append("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in values) + " |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=list(CATEGORIES), choices=CATEGORIES)
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/amazon-profiles"))
    parser.add_argument("--cached-only", action="store_true")
    parser.add_argument("--prepared-root", type=Path,
                        help="Read checksum-verified prepared inputs; no raw sources or downloads are accessed")
    parser.add_argument("--wait-for-sources", action="store_true",
                        help="With --cached-only, process ready categories first while a separate downloader runs")
    parser.add_argument("--source-wait-hours", type=float, default=24.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temporal-period-hours", type=float, default=builder.DEFAULT_TEMPORAL_PERIOD_HOURS)
    args = parser.parse_args(argv)
    if not math.isfinite(args.temporal_period_hours) or args.temporal_period_hours <= 0:
        parser.error("--temporal-period-hours must be finite and positive")
    if args.wait_for_sources and not args.cached_only:
        parser.error("--wait-for-sources requires --cached-only to avoid concurrent cache writers")
    if args.prepared_root is not None and args.wait_for_sources:
        parser.error("--prepared-root cannot be combined with --wait-for-sources")
    if not math.isfinite(args.source_wait_hours) or args.source_wait_hours <= 0:
        parser.error("--source-wait-hours must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    failed = False
    pending = list(dict.fromkeys(args.categories))
    deadline = time.monotonic() + args.source_wait_hours * 3600
    while pending:
        category = pending[0]
        if args.wait_for_sources:
            ready = []
            for candidate in pending:
                ds = AmazonReviews2023(data_dir=args.data_dir, category=candidate, show_progress=False)
                try:
                    sources(ds, ds.interactions_config, cached_only=True)
                    sources(ds, ds.metadata_config, cached_only=True)
                    ready.append(candidate)
                except FileNotFoundError:
                    continue
            if not ready:
                write_json(args.output / "run-status.json", {"status": "waiting_for_sources", "pending": pending})
                if time.monotonic() >= deadline:
                    print("Source wait expired; rerun to resume the remaining categories", flush=True)
                    failed = True
                    break
                time.sleep(30)
                continue
            category = ready[0]
        write_json(args.output / "run-status.json", {"status": "profiling", "category": category, "pending": pending})
        try:
            result = profile_category(category, args)
            failed |= any(result["splits"][split]["status"] != "verified" for split in args.splits)
        except Exception as exc:
            failed = True
            print(f"{category}: FAILED {type(exc).__name__}: {exc}", flush=True)
            write_json(args.output / category / "error.json", {"category": category,
                       "reason": f"{type(exc).__name__}: {exc}"})
        pending.remove(category)
    write_report(args.output)
    write_json(args.output / "run-status.json", {"status": "needs_attention" if failed else "complete", "pending": pending})
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
