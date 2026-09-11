"""Prepare locally, transfer compact inputs, and collect the remote Amazon audit.

Raw downloads remain in --data-dir. Start amazon_remote_worker.py separately
from an identical, isolated code snapshot on the approved host.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
from pathlib import Path
import shlex
import subprocess
import time

import amazon_profile as profile


def remote(host, command):
    return subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                                    "-o", "ClearAllForwardings=yes",
                                    host, shlex.join(command)], text=True, timeout=120)


def transfer(host, paths, destination):
    subprocess.run(["rsync", "-a", "--partial", "--timeout=120", "-e",
                    "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes", "--",
                    *map(str, paths), f"{host}:{destination}/"], check=True, timeout=3600)


def publish_input(host, remote_root, python, directory, category):
    manifest = json.loads((directory / "input.json").read_text())
    digest = manifest["prepared"]["sha256"]
    target = remote_root / "inputs" / category
    size = (directory / "input.parquet").stat().st_size
    free = int(remote(host, [python, "-c", "import shutil,sys; print(shutil.disk_usage(sys.argv[1]).free)", str(remote_root)]))
    if free < size + 20 * 1024 ** 3:
        raise RuntimeError("Transfer paused to keep at least 20 GiB free on the remote host")
    remote(host, ["mkdir", "-p", str(target)])
    transfer(host, [directory / "input.parquet", directory / "input.json"], target)
    # rsync completes before the marker is sent. The remote worker independently
    # verifies the entire parquet checksum before using it.
    profile.write_json(directory / "READY.json", {"category": category, "sha256": digest})
    transfer(host, [directory / "READY.json"], target)
    return digest


def collect(host, remote_root, output):
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-a", "--timeout=120", "-e",
                    "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ClearAllForwardings=yes", "--",
                    f"{host}:{remote_root}/results/", str(output) + "/"], check=True, timeout=300)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    parser.add_argument("--remote-python", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/amazon-profiles"))
    parser.add_argument("--categories", nargs="+", choices=profile.CATEGORIES, default=list(profile.CATEGORIES))
    parser.add_argument("--once", action="store_true", help="Process currently cached categories and return")
    args = parser.parse_args(argv)
    # rsync uses a remote shell too; keep host/path syntax intentionally narrow.
    import re
    if (not re.fullmatch(r"[A-Za-z0-9_.@-]+", args.host) or args.host.startswith("-")
            or not re.fullmatch(r"/[A-Za-z0-9_./-]+", str(args.remote_root))
            or ".." in args.remote_root.parts or len(args.remote_root.parts) < 3):
        parser.error("Use a plain SSH host alias and a dedicated absolute remote directory")
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "hybrid.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("A hybrid coordinator already owns this directory")
    state_path = args.output / "hybrid-state.json"
    identity = {"host": args.host, "remote_root": str(args.remote_root), "code": profile.RUN_CODE}
    state = {**identity, "uploaded": {}, "errors": {}}
    if state_path.exists():
        previous = json.loads(state_path.read_text())
        if all(previous.get(key) == value for key, value in identity.items()):
            state = previous
    while True:
        for category in args.categories:
            if category in state["uploaded"]:
                continue
            ds = profile.AmazonReviews2023(data_dir=args.data_dir, category=category, show_progress=False)
            try:
                profile.sources(ds, ds.interactions_config, cached_only=True)
                profile.sources(ds, ds.metadata_config, cached_only=True)
            except FileNotFoundError:
                continue
            try:
                state.update(status="preparing", category=category)
                profile.write_json(state_path, state)
                frame, manifest = profile.prepare(ds, args.output / category, cached_only=True, load_frame=False)
                del frame
                gc.collect()
                state["uploaded"][category] = publish_input(args.host, args.remote_root, args.remote_python,
                                                           args.output / category, category)
                state["errors"].pop(category, None)
                print(f"Prepared input transferred: {category}", flush=True)
            except Exception as exc:
                state["errors"][category] = f"{type(exc).__name__}: {exc}"
                print(f"Preparation/transfer failed for {category}: {exc}", flush=True)
            profile.write_json(state_path, state)
            gc.collect()
        try:
            collect(args.host, args.remote_root, args.output / "remote-results")
            remote_state_path = args.output / "remote-results/remote-state.json"
            if remote_state_path.exists():
                state["remote"] = json.loads(remote_state_path.read_text())
        except Exception as exc:
            state["collection_error"] = f"{type(exc).__name__}: {exc}"
        state["status"] = "waiting" if len(state["uploaded"]) < len(args.categories) else "all_inputs_transferred"
        profile.write_json(state_path, state)
        remote_state = state.get("remote", {})
        remote_covers_run = set(args.categories).issubset(remote_state.get("categories", []))
        if args.once or (remote_covers_run and remote_state.get("status") in {"complete", "needs_attention", "input_wait_expired"}):
            return int(bool(state["errors"]) or state.get("remote", {}).get("status") not in {None, "complete"})
        time.sleep(60)


if __name__ == "__main__":
    raise SystemExit(main())
