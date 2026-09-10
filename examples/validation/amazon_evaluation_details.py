"""Rebuild selected Amazon profiles to measure evaluation items and cold coverage.

Uses checksum-verified prepared inputs, never downloads or searches thresholds.
Keep this script separate from an existing audit's frozen source snapshot.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

import amazon_profile as profile
import numpy as np


def evaluation_item_stats(payload):
    """Distinct eligible users/target items; cold means absent from x_train.

    Cold percentages are item-weighted, NOT interaction-weighted. Map column
    numbers through each phase's item IDs; temporal vocabularies can differ.
    """
    train = payload["x_train"].tocsr()
    train_ids = np.asarray(payload.get("train_item_ids", payload["item_ids"])).astype(str)
    if len(train_ids) != train.shape[1] or len(np.unique(train_ids)) != len(train_ids):
        raise ValueError("Invalid training item vocabulary")
    warm_ids = train_ids[np.unique(train.indices[train.data != 0])]
    result = {}
    for phase in ("val", "test"):
        holdout = payload[f"{phase}_holdout"]
        item_ids = np.asarray(payload.get(f"{phase}_item_ids", payload["item_ids"])).astype(str)
        users = np.asarray(holdout["user_ids"])
        sources, targets = holdout["source_indices"], holdout["target_indices"]
        if len(users) != len(sources) or len(users) != len(targets):
            raise ValueError("Holdout rows and user IDs differ")
        if len(np.unique(item_ids)) != len(item_ids):
            raise ValueError("Duplicate phase item IDs")
        seen_targets = np.zeros(len(item_ids), dtype=bool)
        eligible = np.zeros(len(users), dtype=bool)
        for row, (source, target) in enumerate(zip(sources, targets)):
            indices = np.asarray(target, dtype=np.int64)
            if indices.size and (indices.min() < 0 or indices.max() >= len(item_ids)):
                raise ValueError("Target index outside phase vocabulary")
            if len(source) and len(indices):
                eligible[row] = True
                seen_targets[indices] = True
        target_ids = item_ids[seen_targets]
        cold = int((~np.isin(target_ids, warm_ids)).sum())
        result.update({
            f"{phase}_users": int(len(np.unique(users[eligible]))),
            f"{phase}_target_items": int(len(target_ids)),
            f"{phase}_cold_target_items": cold,
            f"{phase}_cold_target_item_fraction": cold / len(target_ids) if len(target_ids) else None,
            f"{phase}_catalog_items": int(len(item_ids)),
        })
    return result


def profile_limits(category, entry):
    """Use an explicit, fingerprinted per-profile exception when supplied."""
    if "size_limits" not in entry:
        return profile.limits(category)
    limits = entry["size_limits"]
    if (not isinstance(limits, dict) or set(limits) != {"max_users", "max_items"}
            or any(type(value) is not int or value <= 0 for value in limits.values())):
        raise ValueError("size_limits requires positive integer max_users and max_items")
    return limits["max_users"], limits["max_items"]


def run_category(record, input_root, output, large_user_holdout, archive_code):
    category = record["category"]
    directory = output / category
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "profile.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        if profile.RUN_CODE != archive_code:
            raise ValueError("Builder/profiler fingerprints do not match the measurement archive")
        limits = {split: profile_limits(category, entry) for split, entry in record["splits"].items()}
        context = {"kind": "evaluation-detail-audit", "code": {
            **profile.RUN_CODE, "details_sha256": profile.sha256(Path(__file__))},
            "profile_sha256": hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest(),
            "input_sha256": record["input"]["prepared_sha256"],
            "large_user_holdout": large_user_holdout,
            "cold_definition": "distinct eligible target items absent from nonzero x_train interactions"}
        path = directory / "result.json"
        result = {"category": category, "context": context, "splits": {}}
        if path.exists():
            previous = json.loads(path.read_text())
            if previous["context"] != context:
                raise ValueError("Existing detail audit has different code, input or parameters")
            result = previous
        if set(result["splits"]) == set(record["splits"]):
            return category
        frame, manifest = profile.load_prepared(input_root / category, category)
        if manifest["prepared"]["sha256"] != context["input_sha256"]:
            raise ValueError("Prepared input does not match installed profile")
        graph = profile.CoreGraph(frame) if any(split != "temporal" for split in record["splits"]) else None
        proc, support = None, None
        for split, original in record["splits"].items():
            if split in result["splits"]:
                continue
            parameters = dict(original["parameters"])
            if split == "user_split" and category in {"Books", "Electronics"}:
                parameters.update(val_users=large_user_holdout, test_users=large_user_holdout)
            changed_holdout = parameters != original["parameters"]
            args, _ = profile.builder._resolve_args(profile.builder._build_args(
                dataset="amazon2023", amazon_category=category, split_mode=split,
                annotation_source="none", eval_draws=1, show_progress=False, **parameters))
            if split == "temporal":
                proc, support = None, None
                gc.collect()
                prepared = None
                data = frame
            else:
                requested = (parameters["min_user_support"], parameters["item_min_support"])
                if requested != support:
                    proc = frame.iloc[graph.filter(*requested)]
                    support = requested
                data = proc
                prepared = {"users": int(data.user_id.nunique()), "items": int(data.item_id.nunique()),
                            "pairs": len(data)}
            payload = profile.builder._build_split_payload(args, profile.RecSysDataset(), data)
            stats = profile.payload_stats(payload, preprocessed=prepared)
            extra = evaluation_item_stats(payload)
            for phase in ("val", "test"):
                if extra[f"{phase}_users"] != stats[f"{phase}_users"]:
                    raise AssertionError("Builder holdout contains ineligible rows")
            stats.update(extra)
            for key, value in original["stats"].items():
                if changed_holdout and not key.startswith("preprocessed_"):
                    continue
                if stats[key] != value:
                    raise AssertionError(f"{category}/{split}: {key} drifted: {stats[key]} != {value}")
            max_users, max_items = limits[split]
            if (stats.get("preprocessed_users", stats["users"]) > max_users
                    or stats.get("preprocessed_items", stats["items"]) > max_items):
                raise AssertionError("Rebuilt profile exceeds category caps")
            entry = {"status": "verified", "parameters": parameters, "stats": stats,
                     "warnings": original["warnings"],
                     "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                     "changed_holdout": changed_holdout}
            if "size_limits" in original:
                entry["size_limits"] = dict(original["size_limits"])
            result["splits"][split] = entry
            profile.write_json(path, result)
            print(f"{category}/{split}: {json.dumps(stats)}", flush=True)
            del payload, data
            gc.collect()
        return category


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--memory-budget-gib", type=float, default=80.)
    parser.add_argument("--large-user-holdout", type=int, default=20000)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.large_user_holdout < 1 or not 0 < args.memory_budget_gib <= 100:
        parser.error("Invalid worker count, holdout size or memory budget")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        child_args = [arg for arg in (sys.argv[1:] if argv is None else argv) if arg != "--detach"]
        with (args.output / "manager.log").open("a") as log:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__).resolve()), *child_args],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        profile.write_json(args.output / "manager-pid.json", {"pid": process.pid})
        print(f"Started evaluation detail audit: PID {process.pid}", flush=True)
        return 0
    with (args.output / "manager.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        archive = json.loads(args.archive.read_text())
        pending = sorted(archive["profiles"], key=lambda r: (r["category"] not in {"Books", "Electronics"}, r["category"]))
        estimates = {}
        for record in pending:
            category = record["category"]
            manifest = json.loads((args.input_root / category / "input.json").read_text())
            estimates[category] = manifest["prepared"]["rows"] * 1024 / 1024**3
            if estimates[category] > args.memory_budget_gib:
                raise ValueError(f"{category} exceeds the memory admission budget")
        running, completed, failures = {}, [], {}
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            while pending or running:
                reserved = sum(estimates[category] for category in running.values())
                for record in list(pending):
                    category = record["category"]
                    if len(running) >= args.workers:
                        break
                    if reserved + estimates[category] > args.memory_budget_gib:
                        continue
                    future = pool.submit(run_category, record, args.input_root, args.output,
                                         args.large_user_holdout, archive["source_code"])
                    running[future] = category
                    pending.remove(record)
                    reserved += estimates[category]
                profile.write_json(args.output / "state.json", {
                    "status": "running", "running": list(running.values()),
                    "completed": completed, "pending": [r["category"] for r in pending], "failures": failures,
                    "checked_at_utc": datetime.now(timezone.utc).isoformat()})
                done, _ = wait(running, timeout=1, return_when=FIRST_COMPLETED)
                for future in done:
                    category = running.pop(future)
                    try:
                        future.result()
                        completed.append(category)
                    except Exception as exc:
                        failures[category] = f"{type(exc).__name__}: {exc}"
                        print(f"{category} failed: {failures[category]}", flush=True)
        profile.write_json(args.output / "state.json", {
            "status": "complete" if not failures else "failed", "running": [], "pending": [],
            "completed": completed, "failures": failures, "checked_at_utc": datetime.now(timezone.utc).isoformat()})
        return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
