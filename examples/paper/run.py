"""Benchmark runner for the compresso-recsys ECIR 2027 resource paper.

One ``train`` call is one (block, dataset, model, seed, grid point)::

    python examples/paper/run.py list    --config examples/paper/ecir2027.toml --stage grid [--device gpu|cpu]
    python examples/paper/run.py train   --config ... --block B --dataset ml1m-llo --model EASE --seed 0 --grid-trial 1 --out $RESULTS
    python examples/paper/run.py select  --config ... --block B --dataset ml1m-llo --model EASE --out $RESULTS
    python examples/paper/run.py list    --config ... --stage seeds --out $RESULTS
    python examples/paper/run.py compare --config ... --out $RESULTS

``train`` is idempotent: it skips a run whose ``result.json`` holds the same
spec and refuses (exit code 2) one whose spec differs. It writes only under
``--out``. Selection uses the validation split only; the test split is scored
once per run and only read back by ``compare``. See README.md for the layout.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

SPEC_MISMATCH = 2
TABLE_METRICS = ("calibrated_recall@20", "ndcg@100")
GIT_TIER_MAX_BYTES = 50_000
# Fixed for every run so all results of one dataset pair up in compare_models.
EXCLUDE_SEEN = True


class SpecMismatch(Exception):
    """A result exists for this run with a different spec, or an input changed."""


# --------------------------------------------------------------------------- config


def _expand(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = _expand(tomllib.load(stream))
    for block_name, block in config["blocks"].items():
        for model_name, model in block["models"].items():
            if model["kind"] not in KINDS:
                raise ValueError(f"blocks.{block_name}.models.{model_name}: unknown kind {model['kind']!r}")
    return config


def _block(config: dict, name: str) -> dict:
    try:
        return config["blocks"][name]
    except KeyError:
        raise SystemExit(f"unknown block {name!r}; the config has {sorted(config['blocks'])}") from None


def model_spec(config: dict, block: str, dataset: str, model: str) -> dict:
    """The model's settings for one dataset: block defaults, then per-dataset overrides."""
    block_cfg = _block(config, block)
    if dataset not in block_cfg["datasets"]:
        raise SystemExit(f"block {block} has no dataset {dataset!r}")
    if model not in block_cfg["models"]:
        raise SystemExit(f"block {block} has no model {model!r}")
    base = dict(block_cfg["models"][model])
    override = base.pop("overrides", {}).get(dataset, {})
    spec = {
        "kind": base["kind"],
        "params": {**base.get("params", {}), **override.get("params", {})},
        "grid": {**base.get("grid", {}), **override.get("grid", {})},
        "training": {**base.get("training", {}), **override.get("training", {})},
    }
    if "datasets" in base and dataset not in base["datasets"]:
        raise SystemExit(f"model {model} does not run on {dataset} in block {block}")
    return spec


def models_for(config: dict, block: str, dataset: str) -> list[str]:
    models = _block(config, block)["models"]
    return [name for name, model in models.items() if "datasets" not in model or dataset in model["datasets"]]


def grid_points(spec: dict) -> list[dict]:
    """The grid in a fixed order: keys as written, values as listed. Ties go to the first point."""
    keys = list(spec["grid"])
    return [dict(zip(keys, values)) for values in itertools.product(*(spec["grid"][k] for k in keys))]


def seeds_for(config: dict, spec: dict) -> list[int]:
    return [0] if KINDS[spec["kind"]]["deterministic"] else list(config["run"]["seeds"])


# --------------------------------------------------------------------------- models


def _metrics():
    from compresso_recsys.metrics import MRR, NDCG, CalibratedRecall, Recall

    return [CalibratedRecall([20, 50]), NDCG([10, 20, 100]), Recall([10, 20]), MRR(20)]


def _selection_metric(name: str):
    from compresso_recsys.metrics import NDCG, CalibratedRecall, Recall

    match = re.fullmatch(r"(ndcg|recall|calibrated_recall)@(\d+)", name)
    if match is None:
        raise ValueError(f"unsupported selection metric {name!r}")
    return {"ndcg": NDCG, "recall": Recall, "calibrated_recall": CalibratedRecall}[match[1]](int(match[2]))


def _build(kind: str, params: dict, *, seed: int, device: str, epochs: int | None):
    import compresso_recsys as cr

    trainer = {"device": device, "seed": seed, "show_progress": False}
    if kind == "popularity":
        return cr.PopularityBaseline(cr.PopularityBaselineConfig(**params))
    if kind == "ease":
        return cr.EASE(cr.EASEConfig(**params))
    if kind == "itemknn":
        return cr.ItemKNNRecommender(cr.ItemKNNConfig(**params))
    if kind == "elsa":
        return cr.ELSATrainer(cr.ELSAConfig(**trainer, epochs=epochs, **params))
    if kind == "multvae":
        return cr.MultVAETrainer(cr.MultVAEConfig(**trainer, epochs=epochs, **params))
    if kind == "multdae":
        return cr.MultDAETrainer(cr.MultDAEConfig(**trainer, epochs=epochs, **params))
    if kind == "sasrec":
        return cr.SASRecTrainer(cr.SASRecConfig(**trainer, epochs=epochs, **params))
    if kind == "bert4rec":
        return cr.Bert4RecTrainer(cr.Bert4RecConfig(**trainer, epochs=epochs, **params))
    raise ValueError(kind)


# input: what the model trains and scores on. early_stopping: whether fit accepts the hook.
KINDS: dict[str, dict[str, Any]] = {
    "popularity": {"deterministic": True, "device": "cpu", "input": "matrix", "epochs": False},
    "ease": {"deterministic": True, "device": "cpu", "input": "matrix", "epochs": False},
    "itemknn": {"deterministic": True, "device": "cpu", "input": "matrix", "epochs": False},
    "elsa": {"deterministic": False, "device": "gpu", "input": "matrix", "epochs": True, "early_stopping": True},
    "multvae": {"deterministic": False, "device": "gpu", "input": "matrix", "epochs": True, "early_stopping": True},
    "multdae": {"deterministic": False, "device": "gpu", "input": "matrix", "epochs": True, "early_stopping": True},
    "sasrec": {"deterministic": False, "device": "gpu", "input": "sequence", "epochs": True, "early_stopping": True},
    "bert4rec": {"deterministic": False, "device": "gpu", "input": "sequence", "epochs": True, "early_stopping": False},
}


def _epochs(kind: str, training: dict, params: dict, split: dict, max_epochs: int | None) -> int | None:
    """The epoch budget: a maximum for early stopping, else the fixed number of epochs."""
    if not KINDS[kind]["epochs"]:
        return None
    if "target_steps" in training:
        import compresso_recsys as cr

        probe = cr.Bert4RecTrainer(cr.Bert4RecConfig(**params))
        windows = len(probe._training_windows(split["x_train_sequences"]))
        steps_per_epoch = -(-windows // probe.cfg.batch_size)
        epochs = max(1, round(training["target_steps"] / steps_per_epoch))
    else:
        epochs = int(training.get("max_epochs", training.get("epochs", 0)))
        if epochs < 1:
            raise ValueError(f"{kind}: training needs max_epochs, epochs or target_steps")
    return min(epochs, max_epochs) if max_epochs else epochs


# --------------------------------------------------------------------------- data


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_path(config: dict, block: str, dataset: str) -> Path:
    return Path(config["paths"]["checkpoints"]) / _block(config, block)["datasets"][dataset]["checkpoint"]


def verify_checkpoint(path: Path) -> tuple[str, str]:
    """The checkpoint's sha256 and how it was checked; a mismatch with SHA256SUMS is a spec mismatch."""
    if not path.exists():
        raise FileNotFoundError(f"checkpoint {path} does not exist")
    digest = sha256_file(path)
    sums = path.parent / "SHA256SUMS"
    if not sums.exists():
        return digest, "no SHA256SUMS next to the checkpoint"
    for line in sums.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and Path(parts[1].lstrip("*")).name == path.name:
            if parts[0] != digest:
                raise SpecMismatch(f"{path.name}: sha256 {digest} differs from SHA256SUMS ({parts[0]})")
            return digest, "matches SHA256SUMS"
    return digest, "not listed in SHA256SUMS"


def load_split(path: Path) -> dict:
    import compresso_recsys as cr

    with cr.read_checkpoint(path) as root:
        return cr.load_recsys_split(root)


def phase_inputs(split: dict, phase: str, kind: str) -> dict:
    source_key = f"{phase}_source_sequences" if KINDS[kind]["input"] == "sequence" else f"{phase}_source_matrix"
    source = split[source_key]
    if source is None:
        raise ValueError(f"the checkpoint has no {source_key}; {kind} needs a split mode with sequences")
    targets = split[f"{phase}_target_matrix"]
    sample_ids = split.get(f"{phase}_eval_user_ids")
    if sample_ids is None:
        sample_ids = np.arange(targets.shape[0])
    return {"source": source, "targets": targets, "sample_ids": np.asarray(sample_ids)}


# --------------------------------------------------------------------------- results layout


def run_dir(out: Path, block: str, dataset: str, model: str, seed: int, trial: int | None) -> Path:
    leaf = f"s{seed}" if trial is None else f"s{seed}-t{trial}"
    return out / "runs" / block / dataset / model / leaf


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default, allow_nan=False) + "\n")
    tmp.replace(path)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return asdict(value)
    return str(value)


def spec_digest(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True, default=_json_default).encode()).hexdigest()


def environment() -> dict:
    versions = {}
    for name in ("compresso-recsys", "compresso-pytorch", "torch", "numpy", "scipy", "pandas"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    gpu = None
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            gpu = {"name": props.name, "memory_gib": round(props.total_memory / 2**30, 1), "cuda": torch.version.cuda}
    except ImportError:
        pass
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "packages": versions,
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "source_commit": os.environ.get("CR_SOURCE_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def record_host_environment(out: Path) -> None:
    host = socket.gethostname().split(".")[0]
    freeze = out / "env" / f"{host}-pip-freeze.txt"
    if not freeze.exists():
        lines = sorted(f"{d.metadata['Name']}=={d.version}" for d in importlib_metadata.distributions())
        freeze.parent.mkdir(parents=True, exist_ok=True)
        freeze.write_text("\n".join(lines) + "\n")
    smi = out / "env" / f"{host}-nvidia-smi.txt"
    if not smi.exists():
        try:
            text = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            text = ""
        if text:
            smi.write_text(text)


def _evaluation_record(result) -> dict:
    return {
        "metrics": dict(result.metrics),
        "n_rows": result.n_rows,
        "n_scored_rows": result.n_scored_rows,
        "required_k": result.required_k,
        "target_fingerprint": result.target_fingerprint,
        "source_fingerprint": result.source_fingerprint,
        "protocol": result.protocol,
        "metadata": result.metadata,
    }


def load_evaluation(directory: Path, phase: str = "test", catalogue: str | None = None):
    """Rebuild the EvaluationResult a run wrote, for compare_models."""
    from compresso_recsys.evaluation import EvaluationResult

    result = json.loads((directory / "result.json").read_text())
    key = phase if catalogue is None else f"{phase}__{catalogue}"
    record = result["evaluation"][key]
    with np.load(directory / "per_user.npz", allow_pickle=False) as arrays:
        prefix = f"{key}/"
        per_user = {name[len(prefix):]: arrays[name] for name in arrays.files
                    if name.startswith(prefix) and name != f"{prefix}sample_ids"}
        sample_ids = arrays[f"{prefix}sample_ids"]
    return EvaluationResult(
        metrics=record["metrics"], per_user=per_user, sample_ids=sample_ids,
        n_rows=record["n_rows"], n_scored_rows=record["n_scored_rows"], required_k=record["required_k"],
        metadata=record["metadata"], target_fingerprint=record["target_fingerprint"],
        source_fingerprint=record["source_fingerprint"], protocol=record["protocol"],
    )


# --------------------------------------------------------------------------- train


def train(args, config: dict) -> int:
    spec_cfg = model_spec(config, args.block, args.dataset, args.model)
    kind = spec_cfg["kind"]
    points = grid_points(spec_cfg)
    trial = args.grid_trial
    if len(points) > 1 and trial is None:
        raise SystemExit(f"{args.model} has a grid of {len(points)} points; pass --grid-trial")
    if trial is not None and not 0 <= trial < len(points):
        raise SystemExit(f"--grid-trial must be in [0, {len(points)})")
    if len(points) == 1:
        trial = None
    if args.seed not in seeds_for(config, spec_cfg):
        raise SystemExit(f"{args.model} runs seeds {seeds_for(config, spec_cfg)}, not {args.seed}")
    params = {**spec_cfg["params"], **points[trial or 0]}
    block_cfg = _block(config, args.block)
    ckpt = checkpoint_path(config, args.block, args.dataset)
    out = run_dir(args.out, args.block, args.dataset, args.model, args.seed, trial)

    spec = {
        "block": args.block, "dataset": args.dataset, "model": args.model, "kind": kind,
        "seed": args.seed, "trial": trial, "params": params, "training": spec_cfg["training"],
        "selection_metric": block_cfg["selection"], "checkpoint": ckpt.name,
        "exclude_seen": EXCLUDE_SEEN, "max_epochs_override": args.max_epochs,
    }
    digest = spec_digest(spec)
    result_path = out / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("spec_sha256") == digest:
            print(f"skip: {result_path} exists with the same spec")
            return 0
        raise SpecMismatch(f"{result_path} exists with a different spec; move it away or fix the config")

    out.mkdir(parents=True, exist_ok=True)
    record_host_environment(args.out)
    logger = logging.getLogger(f"cr-paper.{args.block}.{args.dataset}.{args.model}.s{args.seed}")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(out / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.addHandler(console)
    logger.propagate = False
    started = time.time()
    status = {"spec": spec, "phase": "started", "started_unix": started, "host": socket.gethostname()}
    write_json(out / "status.json", status)
    phases: dict[str, float] = {}

    def mark(phase: str) -> None:
        phases[phase] = round(time.time() - started, 2)
        status.update(phase=phase, elapsed_s=phases[phase])
        write_json(out / "status.json", status)
        logger.info(f"[{phases[phase]:9.1f}s] {phase}")

    try:
        checkpoint_sha256, checkpoint_check = verify_checkpoint(ckpt)
        split = load_split(ckpt)
        mark("loaded")
        import torch

        device = args.device or ("cuda" if KINDS[kind]["device"] == "gpu" and torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        epochs = _epochs(kind, spec_cfg["training"], params, split, args.max_epochs)
        model = _build(kind, params, seed=args.seed, device=device, epochs=epochs)
        fit_kwargs: dict[str, Any] = {}
        training = spec_cfg["training"]
        if KINDS[kind].get("early_stopping"):
            from compresso_recsys.training import EarlyStopping, Validation

            val = phase_inputs(split, "val", kind)
            fit_kwargs["validation"] = Validation(
                val["source"], val["targets"], [_selection_metric(block_cfg["selection"])],
                sample_ids=val["sample_ids"], every=int(training.get("every", 1)),
                batch_size=int(config["run"].get("eval_batch_size", 1024)), exclude_seen=EXCLUDE_SEEN,
            )
            if "patience" in training:
                fit_kwargs["early_stopping"] = EarlyStopping(
                    block_cfg["selection"], patience=int(training["patience"]),
                    min_epochs=int(training.get("min_epochs", 0)),
                )
        if KINDS[kind]["epochs"]:
            fit_kwargs["logger"] = logger
        train_input = split["x_train_sequences"] if KINDS[kind]["input"] == "sequence" else split["x_train"]
        fit_started = time.time()
        model.fit(train_input, item_ids=split["train_item_ids"], **fit_kwargs)
        fit_seconds = time.time() - fit_started
        mark("fitted")

        from compresso_recsys.evaluation import evaluate_recommender

        evaluation, arrays = {}, {}
        for phase in ("val", "test"):
            inputs = phase_inputs(split, phase, kind)
            phase_started = time.time()
            result = evaluate_recommender(
                model, source=inputs["source"], targets=inputs["targets"], metrics=_metrics(),
                sample_ids=inputs["sample_ids"], exclude_seen=EXCLUDE_SEEN,
                batch_size=int(config["run"].get("eval_batch_size", 1024)),
            )
            evaluation[phase] = {**_evaluation_record(result), "seconds": round(time.time() - phase_started, 2)}
            if phase == "test":
                arrays[f"{phase}/sample_ids"] = np.asarray(result.sample_ids)
                for name, values in result.per_user.items():
                    arrays[f"{phase}/{name}"] = np.asarray(values)
            mark(f"evaluated_{phase}")
        np.savez_compressed(out / "per_user.npz", **arrays)

        history = getattr(model, "history", None)
        payload = {
            "spec": spec,
            "spec_sha256": digest,
            "selection": {"metric": block_cfg["selection"],
                          "validation": evaluation["val"]["metrics"][block_cfg["selection"]]},
            "evaluation": evaluation,
            "training": {
                "device": device,
                "epochs_budget": epochs,
                "epochs_trained": len(history) if isinstance(history, list) else None,
                "best": getattr(model, "best_", None),
                "fit_seconds": round(fit_seconds, 2),
                "history": history if isinstance(history, list) else None,
            },
            "checkpoint": {"path": str(ckpt), "sha256": checkpoint_sha256, "check": checkpoint_check},
            "env": environment(),
            "phases_s": phases,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_json(result_path, payload)
        mark("done")
        return 0
    except SpecMismatch:
        status.update(phase="spec_mismatch", error=traceback.format_exc())
        write_json(out / "status.json", status)
        raise
    except BaseException:
        status.update(phase="failed", error=traceback.format_exc())
        write_json(out / "status.json", status)
        raise
    finally:
        logger.removeHandler(handler)
        logger.removeHandler(console)
        handler.close()


# --------------------------------------------------------------------------- select


def _trial_results(out: Path, block: str, dataset: str, model: str, n_points: int) -> list[tuple[int | None, dict]]:
    trials = [None] if n_points == 1 else list(range(n_points))
    found = []
    for trial in trials:
        path = run_dir(out, block, dataset, model, 0, trial) / "result.json"
        if path.exists():
            found.append((trial, json.loads(path.read_text())))
    return found


def select(args, config: dict) -> int:
    spec_cfg = model_spec(config, args.block, args.dataset, args.model)
    points = grid_points(spec_cfg)
    metric = _block(config, args.block)["selection"]
    found = _trial_results(args.out, args.block, args.dataset, args.model, len(points))
    if len(found) < len(points):
        missing = sorted(set(range(len(points))) - {t or 0 for t, _ in found})
        raise SystemExit(f"{args.model}: grid trials {missing} have no result yet")
    best_trial, best_score = None, -math.inf
    for trial, result in found:  # grid order; a strict > keeps the first of tied points
        score = result["selection"]["validation"]
        if score > best_score:
            best_trial, best_score = trial, score
    payload = {
        "block": args.block, "dataset": args.dataset, "model": args.model,
        "metric": metric, "split": "val", "trial": best_trial,
        "params": {**spec_cfg["params"], **points[best_trial or 0]}, "validation": best_score,
        "epochs_trained": dict(found)[best_trial]["training"]["epochs_trained"],
        "trials": [{"trial": t, "params": points[t or 0], "validation": r["selection"]["validation"],
                    "epochs_trained": r["training"]["epochs_trained"]} for t, r in found],
        "tie_break": "first grid point in config order",
    }
    write_json(args.out / "selection" / args.block / args.dataset / args.model / "selected.json", payload)
    print(f"{args.model}: trial {best_trial} ({payload['params']}) {metric}={best_score:.5f}")
    return 0


def selected_trial(out: Path, config: dict, block: str, dataset: str, model: str) -> int | None:
    points = grid_points(model_spec(config, block, dataset, model))
    if len(points) == 1:
        return None
    path = out / "selection" / block / dataset / model / "selected.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run select first")
    return json.loads(path.read_text())["trial"]


# --------------------------------------------------------------------------- list


def list_commands(args, config: dict) -> int:
    script = "python examples/paper/run.py"
    out = "$RESULTS" if args.out is None else str(args.out)
    extra = f" --max-epochs {args.max_epochs}" if args.max_epochs else ""
    missing = []
    for block_name, block in config["blocks"].items():
        if args.block and block_name != args.block:
            continue
        for dataset in block["datasets"]:
            for model in models_for(config, block_name, dataset):
                spec = model_spec(config, block_name, dataset, model)
                if args.device and KINDS[spec["kind"]]["device"] != args.device:
                    continue
                points = grid_points(spec)
                if args.stage == "grid":
                    runs = [(0, None if len(points) == 1 else t) for t in range(len(points))]
                else:
                    try:
                        trial = selected_trial(Path(os.path.expandvars(out)), config, block_name, dataset, model)
                    except FileNotFoundError as error:
                        missing.append(str(error))
                        continue
                    runs = [(seed, trial) for seed in seeds_for(config, spec) if seed != 0]
                for seed, trial in runs:
                    grid = "" if trial is None else f" --grid-trial {trial}"
                    print(f"{script} train --config {args.config} --block {block_name} --dataset {dataset} "
                          f"--model {model} --seed {seed}{grid} --out {out}{extra}")
    for line in missing:
        print(line, file=sys.stderr)
    return 1 if missing else 0


# --------------------------------------------------------------------------- compare


def _tier(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    if rel.parts[0] in ("table", "casestudy", "summary", "selection", "env") or rel.name == "meta.json":
        return "git"
    if rel.name == "result.json" and path.stat().st_size < GIT_TIER_MAX_BYTES:
        return "git"
    return "archive"


def write_manifest(root: Path) -> None:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in ("MANIFEST.json",) or path.suffix == ".tmp":
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path), "tier": _tier(path, root)})
    write_json(root / "MANIFEST.json", {"files": files})


def _summary(out: Path, config: dict, block: str, dataset: str) -> dict:
    rows = {}
    for model in models_for(config, block, dataset):
        spec = model_spec(config, block, dataset, model)
        try:
            trial = selected_trial(out, config, block, dataset, model)
        except FileNotFoundError:
            continue
        seeds = {}
        for seed in seeds_for(config, spec):
            path = run_dir(out, block, dataset, model, seed, trial) / "result.json"
            if path.exists():
                result = json.loads(path.read_text())
                seeds[seed] = {"test": result["evaluation"]["test"]["metrics"],
                               "val": result["evaluation"]["val"]["metrics"],
                               "epochs_trained": result["training"]["epochs_trained"],
                               "fit_seconds": result["training"]["fit_seconds"]}
        if not seeds:
            continue
        names = sorted(next(iter(seeds.values()))["test"])
        values = {name: [s["test"][name] for s in seeds.values()] for name in names}
        rows[model] = {
            "trial": trial,
            "params": {**spec["params"], **grid_points(spec)[trial or 0]},
            "deterministic": KINDS[spec["kind"]]["deterministic"],
            "seeds": seeds,
            "test_mean": {n: float(np.mean(v)) for n, v in values.items()},
            "test_sd": {n: float(np.std(v, ddof=1)) if len(v) > 1 else None for n, v in values.items()},
            "n_seeds": len(seeds),
        }
    return rows


def compare(args, config: dict) -> int:
    from compresso_recsys.stats import compare_models

    out = args.out
    stats_cfg = config.get("statistics", {})
    written = []
    for block_name, block in config["blocks"].items():
        for dataset, dataset_cfg in block["datasets"].items():
            summary = _summary(out, config, block_name, dataset)
            if not summary:
                continue
            write_json(out / "summary" / block_name / f"{dataset}.json", summary)
            target = dataset_cfg.get("table") or dataset_cfg.get("casestudy")
            if target is None:
                continue
            # The comparison uses the seed-0 per-user vectors of each model's selected configuration.
            results = {}
            for model in summary:
                directory = run_dir(out, block_name, dataset, model, 0, summary[model]["trial"])
                results[model] = load_evaluation(directory)
            reference = block.get("reference", "EASE")
            if reference not in results or len(results) < 2:
                print(f"{block_name}/{dataset}: needs the reference {reference} and one more model; skipped",
                      file=sys.stderr)
                continue
            report = compare_models(
                results, metrics=list(TABLE_METRICS), reference=reference,
                n_resamples=int(stats_cfg.get("n_resamples", 9_999)), random_state=0,
            )
            folder = "table" if "table" in dataset_cfg else "casestudy"
            path = out / folder / f"{target}.json"
            write_json(path, report.to_frame().to_dict(orient="records"))
            written.append(str(path.relative_to(out)))
            (out / "summary" / block_name / f"{dataset}.md").write_text(report.to_markdown() + "\n")
    meta = {
        "commit": os.environ.get("CR_SOURCE_COMMIT"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hosts": sorted({json.loads(p.read_text())["env"]["host"] for p in out.glob("runs/**/result.json")}),
        "note": config["run"].get("note", ""),
        "config": str(args.config),
        "placeholder": bool(config["run"].get("placeholder", False)),
    }
    write_json(out / "meta.json", meta)
    write_manifest(out)
    print("\n".join(written) or "nothing to compare yet")
    return 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "select", "compare", "list"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name in ("train", "select"):
            sub.add_argument("--block", required=True)
            sub.add_argument("--dataset", required=True)
            sub.add_argument("--model", required=True)
        if name == "train":
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--grid-trial", type=int)
            sub.add_argument("--device", help="override the device (default: cuda when available for GPU models)")
        if name in ("train", "list"):
            sub.add_argument("--max-epochs", type=int, help="cap every epoch budget (smoke tests)")
        if name == "list":
            sub.add_argument("--stage", choices=("grid", "seeds"), required=True)
            sub.add_argument("--device", choices=("gpu", "cpu"))
            sub.add_argument("--block")
            sub.add_argument("--out", type=Path)
        else:
            sub.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    try:
        return {"train": train, "select": select, "compare": compare, "list": list_commands}[args.command](args, config)
    except SpecMismatch as error:
        print(f"spec mismatch: {error}", file=sys.stderr)
        return SPEC_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
