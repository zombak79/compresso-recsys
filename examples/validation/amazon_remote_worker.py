"""Run prepared Amazon categories in bounded parallel processes on a large host."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import json
import math
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import amazon_profile as profile


def run_category(category, input_root, output):
    directory = output / category
    directory.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(output=output, prepared_root=input_root, seed=42,
                              temporal_period_hours=profile.builder.DEFAULT_TEMPORAL_PERIOD_HOURS,
                              splits=profile.SPLITS, write_global_report=False)
    with (directory / "profile.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        return profile.profile_category(category, args)


def finished_result(path, manifest):
    if not path.exists():
        return False
    result = json.loads(path.read_text())
    context = result.get("context", {})
    return (all(context.get(key) == value for key, value in profile.RUN_CODE.items())
            and context.get("input", {}).get("prepared", {}).get("sha256") == manifest["prepared"]["sha256"]
            and all(result.get("splits", {}).get(split, {}).get("status") == "verified" for split in profile.SPLITS))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", choices=profile.CATEGORIES, default=list(profile.CATEGORIES))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--memory-budget-gib", type=float, default=80.,
                        help="Admission budget using a conservative 1 KiB per prepared row estimate")
    parser.add_argument("--idle-hours", type=float, default=48.)
    parser.add_argument("--detach", action="store_true", help="Keep running on the server after SSH disconnects")
    args = parser.parse_args(argv)
    if args.workers < 1 or not 0 < args.idle_hours <= 168:
        parser.error("Workers must be positive and idle-hours must be in (0,168]")
    if not math.isfinite(args.memory_budget_gib) or args.memory_budget_gib <= 0:
        parser.error("memory-budget-gib must be finite and positive")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.detach:
        child_args = [arg for arg in (sys.argv[1:] if argv is None else argv) if arg != "--detach"]
        with (args.output / "manager.log").open("a") as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *child_args],
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        profile.write_json(args.output / "manager-pid.json", {"pid": process.pid})
        print(f"Started isolated remote audit manager: PID {process.pid}", flush=True)
        return 0
    lock = (args.output / "worker.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A worker manager already owns this output directory")
    pending = list(dict.fromkeys(args.categories))
    running, completed, failures = {}, [], {}
    reservations = {}
    memory_budget = args.memory_budget_gib * 1024 ** 3
    last_activity = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        while pending or running:
            for future, category in list(running.items()):
                if not future.done():
                    continue
                del running[future]
                reservations.pop(category, None)
                last_activity = time.monotonic()
                try:
                    result = future.result()
                    bad = {split: entry for split, entry in result["splits"].items() if entry["status"] != "verified"}
                    if bad:
                        failures[category] = {split: entry.get("reason", entry["status"]) for split, entry in bad.items()}
                    else:
                        completed.append(category)
                except Exception as exc:
                    failures[category] = f"{type(exc).__name__}: {exc}"
                    profile.write_json(args.output / category / "error.json", {"reason": failures[category]})
                profile.write_report(args.output)
            for category in pending[:]:
                if len(running) >= args.workers:
                    break
                directory = args.input_root / category
                ready_path = directory / "READY.json"
                if not ready_path.exists():
                    continue
                ready = json.loads(ready_path.read_text())
                manifest = json.loads((directory / "input.json").read_text())
                if ready.get("sha256") != manifest.get("prepared", {}).get("sha256"):
                    continue
                if finished_result(args.output / category / "result.json", manifest):
                    pending.remove(category)
                    completed.append(category)
                else:
                    # Parquet size substantially understates Python strings,
                    # sparse arrays and split copies. This is an admission
                    # estimate, not a hard RSS guarantee or a data-size cap.
                    estimate = manifest["prepared"]["rows"] * 1024
                    if estimate > memory_budget:
                        pending.remove(category)
                        failures[category] = (f"Estimated working memory {estimate / 1024**3:.1f} GiB exceeds "
                                              f"{args.memory_budget_gib:g} GiB budget; review before running")
                        continue
                    if sum(reservations.values()) + estimate > memory_budget:
                        continue
                    pending.remove(category)
                    reservations[category] = estimate
                    running[pool.submit(run_category, category, args.input_root, args.output)] = category
                last_activity = time.monotonic()
            status = "running" if running else "waiting_for_inputs"
            if not pending and not running:
                status = "needs_attention" if failures else "complete"
            profile.write_json(args.output / "remote-state.json", {
                "status": status, "running": list(running.values()), "pending": pending,
                "completed": completed, "failures": failures, "code": profile.RUN_CODE,
                "reserved_memory_gib": sum(reservations.values()) / 1024 ** 3,
                "memory_budget_gib": args.memory_budget_gib,
                "categories": args.categories})
            if not running and pending and time.monotonic() - last_activity > args.idle_hours * 3600:
                profile.write_json(args.output / "remote-state.json", {
                    "status": "input_wait_expired", "pending": pending, "completed": completed, "failures": failures})
                return 1
            if pending or running:
                time.sleep(10)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
