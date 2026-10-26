"""Benchmark runner for the compresso-recsys ECIR 2027 resource paper.

One ``train`` call is one (block, dataset, model, seed, grid point)::

    python examples/paper/run.py list    --config examples/paper/ecir2027.toml --stage grid [--device gpu|cpu]
    python examples/paper/run.py encode  --config ... --block Cp --dataset steam-temporal --out $RESULTS
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
import getpass
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
PROTOCOL_NOTE = (
    "Hyperparameters and, for trainers, the number of epochs are selected on validation: "
    "trainers run up to max_epochs and stop after `patience` validations without improvement, "
    "restoring the best epoch (max epochs + patience, selected on validation). SASRec on ML-1M "
    "and BERT4Rec keep fixed published budgets. Deterministic models run once; stochastic models "
    "rerun the selected configuration with further seeds. Statistics use the seed-0 per-user values."
)


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
        block.setdefault("mode", "standard")
        if block["mode"] not in ("standard", "catalogue"):
            raise ValueError(f"blocks.{block_name}: mode must be 'standard' or 'catalogue'")
        for model_name, model in block["models"].items():
            if model["kind"] not in KINDS:
                raise ValueError(f"blocks.{block_name}.models.{model_name}: unknown kind {model['kind']!r}")
            if KINDS[model["kind"]]["input"] == "features" and block["mode"] != "catalogue":
                raise ValueError(f"blocks.{block_name}.models.{model_name}: feature models need a catalogue block")
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
        "catalogues": catalogues_for(block_cfg, base),
    }
    if "datasets" in base and dataset not in base["datasets"]:
        raise SystemExit(f"model {model} does not run on {dataset} in block {block}")
    return spec


def catalogues_for(block_cfg: dict, model_cfg: dict) -> list[str]:
    """The catalogues a model is scored in: warm-only models never enter a cold-only catalogue."""
    if block_cfg["mode"] != "catalogue":
        return ["all"]
    names = list(model_cfg.get("catalogues", block_cfg["catalogues"]))
    if KINDS[model_cfg["kind"]]["input"] != "features":
        names = [name for name in names if name != "cold"]
    return names


def models_for(config: dict, block: str, dataset: str) -> list[str]:
    models = _block(config, block)["models"]
    return [name for name, model in models.items() if "datasets" not in model or dataset in model["datasets"]]


def grid_points(spec: dict) -> list[dict]:
    """The grid in a fixed order: keys as written, values as listed. Ties go to the first point."""
    keys = list(spec["grid"])
    return [dict(zip(keys, values)) for values in itertools.product(*(spec["grid"][k] for k in keys))]


def seeds_for(config: dict, spec: dict) -> list[int]:
    return [0] if KINDS[spec["kind"]]["deterministic"] else list(config["run"]["seeds"])


def outputs_for(block_cfg: dict, dataset: str) -> dict[str, str]:
    """catalogue -> output path without extension, e.g. {"all": "table/ml1m-llo"}."""
    dataset_cfg = block_cfg["datasets"][dataset]
    if "outputs" in dataset_cfg:
        return dict(dataset_cfg["outputs"])
    if "table" in dataset_cfg:
        return {"all": f"table/{dataset_cfg['table']}"}
    return {}


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


# input: what the model trains on ("matrix", "sequence", or item "features" plus the matrix).
# early_stopping: whether fit accepts validation= and early_stopping=.
KINDS: dict[str, dict[str, Any]] = {
    "popularity": {"deterministic": True, "device": "cpu", "input": "matrix", "epochs": False},
    "ease": {"deterministic": True, "device": "cpu", "input": "matrix", "epochs": False},
    "itemknn": {"deterministic": True, "device": "cpu", "input": "matrix", "epochs": False},
    "elsa": {"deterministic": False, "device": "gpu", "input": "matrix", "epochs": True, "early_stopping": True},
    "multvae": {"deterministic": False, "device": "gpu", "input": "matrix", "epochs": True, "early_stopping": True},
    "multdae": {"deterministic": False, "device": "gpu", "input": "matrix", "epochs": True, "early_stopping": True},
    "sasrec": {"deterministic": False, "device": "gpu", "input": "sequence", "epochs": True, "early_stopping": True},
    "bert4rec": {"deterministic": False, "device": "gpu", "input": "sequence", "epochs": True, "early_stopping": False},
    "content": {"deterministic": True, "device": "cpu", "input": "features", "epochs": False},
    "teaser": {"deterministic": True, "device": "cpu", "input": "features", "epochs": False},
    "teaser_gd": {"deterministic": False, "device": "gpu", "input": "features", "epochs": True, "early_stopping": True},
    # Learns one embedding per training user, so it needs the checkpoint's user ids.
    "clcrec": {"deterministic": False, "device": "gpu", "input": "features", "epochs": True, "early_stopping": True,
               "user_ids": True},
}

# Keys of a feature model's params that configure the wrapper rather than the inner model.
_WRAPPER_KEYS = ("features", "weights", "normalize", "missing")


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
    # Feature models go through MMConcatWrapper, also for one feature space: it imputes
    # missing rows with the training mean and L2-normalises each block the same way.
    wrapper = {key: params[key] for key in _WRAPPER_KEYS if key in params}
    inner = {key: value for key, value in params.items() if key not in _WRAPPER_KEYS}
    weights = wrapper.get("weights")
    if isinstance(weights, list):  # one weight per feature space, in the order of `features`
        weights = dict(zip(wrapper["features"], map(float, weights), strict=True))
    if "l2" in inner:  # one value for both TEASER penalties, as in the grid
        inner["l2_coefficients"] = inner["l2_encoder"] = inner.pop("l2")
    if kind == "content":
        model, model_config = cr.ContentRecommender, cr.ContentRecommenderConfig(**inner)
    elif kind == "teaser":
        model, model_config = cr.TEASER, cr.TEASERConfig(**inner)
    elif kind == "teaser_gd":
        model, model_config = cr.TEASERGDTrainer, cr.TEASERGDConfig(**trainer, epochs=epochs, **inner)
    elif kind == "clcrec":
        model, model_config = cr.CLCRecTrainer, cr.CLCRecConfig(**trainer, epochs=epochs, **inner)
    else:
        raise ValueError(kind)
    return cr.MMConcatWrapper(cr.MMConcatWrapperConfig(
        model=model, model_config=model_config, modalities=list(wrapper["features"]),
        missing=wrapper.get("missing", "mean"), normalize=wrapper.get("normalize", True),
        weights=weights, extra_modalities="ignore",
    ))


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


class WarmOnly:
    """A model fitted on the training items, scored in a larger catalogue.

    It gets the same global-width histories as every other model of the block,
    so the source fingerprints pair in ``compare_models``; it reads the training
    columns and maps its ranked items to catalogue positions. Cold candidates
    are valid targets that it can never emit.
    """

    def __init__(self, model, warm_columns: np.ndarray, catalogue_columns: np.ndarray, n_global: int) -> None:
        position = np.full(n_global, -1, dtype=np.int64)
        position[catalogue_columns] = np.arange(catalogue_columns.size)
        self.to_catalogue = position[warm_columns]
        if np.any(self.to_catalogue < 0):
            raise ValueError("every training item must be in the catalogue a warm-only model is scored in")
        self.model = model
        self.warm_columns = warm_columns
        self.n_items = int(catalogue_columns.size)

    def predict_on_batch(self, source, *, k: int, **kwargs):
        import torch
        from compresso import SRPTensor

        ranked = self.model.predict_on_batch(source[:, self.warm_columns].tocsr(), k=k, **kwargs)
        cols = torch.from_numpy(self.to_catalogue)[ranked.cols.cpu()]
        return SRPTensor(cols=cols, vals=ranked.vals.cpu(), shape=(source.shape[0], self.n_items))


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
        raise FileNotFoundError(f"checkpoint {path.name} does not exist in the checkpoint directory")
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


def load_split(path: Path, feature_names: list[str] = (), *, info: dict | None = None) -> tuple[dict, dict]:
    """The split, and the requested checkpoint feature spaces aligned to its item IDs.

    ``info``, when given, receives the checkpoint's provenance (``None`` values
    for a version 1 checkpoint) and its content digest, which is computed here
    for any version so results can be matched to a later rebuild.
    """
    import compresso_recsys as cr

    with cr.read_checkpoint(path) as root:
        split = cr.load_recsys_split(root)
        features = {name: cr.load_item_embeddings(root, name, item_ids=split["item_ids"]) for name in feature_names}
        if info is not None:
            info.update(checkpoint_provenance(root))
    return split, {name: {"embeddings": f["embeddings"], "available": np.asarray(f["available"], dtype=bool)}
                   for name, f in features.items()}


def checkpoint_provenance(root: Path) -> dict:
    from compresso_recsys import load_provenance
    from compresso_recsys._provenance import content_sha256

    record = load_provenance(root)
    return {
        "manifest_version": record["version"],
        "content_sha256": content_sha256(root),
        "recorded_content_sha256": record["content_sha256"],
        "build": record["build"],
        "sources": record["sources"],
        "provenance": record["provenance"],
    }


def encoded_features_path(out: Path, dataset: str, encoder: str) -> Path:
    return out / "features" / dataset / f"{encoder.replace('/', '__')}.npz"


def load_encoded_features(out: Path, dataset: str, encoder: str, item_ids: np.ndarray) -> dict:
    path = encoded_features_path(out, dataset, encoder)
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(out)} is missing: run encode first")
    with np.load(path, allow_pickle=False) as stored:
        position = {item: row for row, item in enumerate(stored["item_ids"].tolist())}
        rows = np.array([position.get(item, -1) for item in item_ids.tolist()])
        embeddings = np.zeros((item_ids.size, stored["embeddings"].shape[1]), dtype=np.float32)
        available = rows >= 0
        embeddings[available] = stored["embeddings"][rows[available]]
        available[available] &= stored["available"][rows[available]]
    return {"embeddings": embeddings, "available": available}


def phase_inputs(split: dict, phase: str, kind: str) -> dict:
    source_key = f"{phase}_source_sequences" if KINDS[kind]["input"] == "sequence" else f"{phase}_source_matrix"
    source = split[source_key]
    if source is None:
        raise ValueError(f"the checkpoint has no {source_key}; {kind} needs a split mode with sequences")
    targets = split[f"{phase}_target_matrix"]
    return {"source": source, "targets": targets, "sample_ids": _sample_ids(split, phase, targets.shape[0])}


def _sample_ids(split: dict, phase: str, n_rows: int) -> np.ndarray:
    sample_ids = split.get(f"{phase}_eval_user_ids")
    return np.arange(n_rows) if sample_ids is None else np.asarray(sample_ids)


def _to_global(matrix, phase_item_ids: np.ndarray, position: dict):
    """Reindex a matrix whose columns follow a phase catalogue to the checkpoint's item IDs."""
    from scipy.sparse import csr_matrix

    matrix = matrix.tocsr()
    columns = np.array([position[item] for item in phase_item_ids.tolist()], dtype=np.int64)
    return csr_matrix((matrix.data, columns[matrix.indices], matrix.indptr),
                      shape=(matrix.shape[0], len(position)))


def training_user_ids(split: dict) -> np.ndarray:
    """The ids of the training rows, for a model that learns one embedding per user.

    Evaluation then passes each row's ``*_eval_user_ids`` as its user id. Without
    real ids on both sides the runner's positional fallback would pair evaluation
    row i with training row i, a silently wrong user, so it refuses instead.
    """
    missing = [key for key in ("train_user_ids", "val_eval_user_ids", "test_eval_user_ids")
               if split.get(key) is None]
    if missing:
        raise ValueError(f"a user-embedding model needs user ids in the checkpoint; missing {missing}")
    return np.asarray(split["train_user_ids"])


def catalogue_inputs(split: dict) -> dict:
    """Global-width inputs for a catalogue block.

    Columns follow the checkpoint's ``item_ids``. Histories keep only training
    items: an item a model never trained on cannot be encoded, and every model
    must see the same histories for the comparison to pair. Catalogues are
    column index arrays into the same space.
    """
    from scipy.sparse import diags

    item_ids = np.asarray(split["item_ids"])
    position = {item: column for column, item in enumerate(item_ids.tolist())}
    warm = np.asarray(split["warm_item_indices"], dtype=np.int64)
    keep = np.zeros(item_ids.size, dtype=np.float32)
    keep[warm] = 1.0
    x_train = split["x_train"]
    if x_train.shape[1] != item_ids.size:
        x_train = _to_global(x_train, np.asarray(split["train_item_ids"]), position)
    x_train, repeats = _binary(x_train)
    binarized = {"x_train": repeats}
    phases = {}
    for phase in ("val", "test"):
        phase_ids = np.asarray(split[f"{phase}_item_ids"])
        source = _to_global(split[f"{phase}_source_matrix"], phase_ids, position) @ diags(keep)
        source.eliminate_zeros()
        source, binarized[f"{phase}_source"] = _binary(source)
        targets = _to_global(split[f"{phase}_target_matrix"], phase_ids, position)
        full = np.array(sorted(position[item] for item in phase_ids.tolist()), dtype=np.int64)
        phases[phase] = {
            "source": source.tocsr(), "targets": targets,
            "sample_ids": _sample_ids(split, phase, targets.shape[0]),
            "catalogues": {
                "warm": warm, "full": full, "mixed": full,
                "cold": np.asarray(split[f"{phase}_cold_item_indices"], dtype=np.int64),
            },
        }
    return {"item_ids": item_ids, "warm": warm, "x_train": x_train.tocsr(), "phases": phases,
            "binarized": binarized}


def _binary(matrix):
    """Implicit binary interactions, and how many entries a repeat had pushed above 1.

    Temporal splits keep repeat events (a second review of the same game), which
    sum to 2 or more. The feature models require binary input, and every model of
    a comparison must see the same data, so catalogue blocks binarise for all.
    """
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    changed = int(np.count_nonzero(matrix.data != 1))
    matrix.data = np.ones_like(matrix.data, dtype=np.float32)
    return matrix, changed


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


def host_label() -> str:
    """How results name the machine: CR_HOST_LABEL, else a digest. Hostnames never reach the results."""
    label = os.environ.get("CR_HOST_LABEL")
    if label:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", label)
    return "host-" + hashlib.sha256(socket.gethostname().encode()).hexdigest()[:8]


def environment() -> dict:
    versions = {}
    for name in ("compresso-recsys", "compresso-pytorch", "torch", "numpy", "scipy", "pandas",
                 "sentence-transformers"):
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
        "host": host_label(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
        "packages": versions,
        "gpu": gpu,
        "cpu_count": os.cpu_count(),
        "source_commit": os.environ.get("CR_SOURCE_COMMIT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def record_host_environment(out: Path) -> None:
    label = host_label()
    freeze = out / "env" / f"{label}-pip-freeze.txt"
    if not freeze.exists():
        lines = sorted({f"{d.metadata['Name']}=={d.version}" for d in importlib_metadata.distributions()})
        freeze.parent.mkdir(parents=True, exist_ok=True)
        freeze.write_text("\n".join(lines) + "\n")
    smi = out / "env" / f"{label}-nvidia-smi.txt"
    if not smi.exists():
        try:
            text = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                   "--format=csv"], capture_output=True, text=True, timeout=30).stdout
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


def evaluation_key(phase: str, catalogue: str) -> str:
    return phase if catalogue == "all" else f"{phase}__{catalogue}"


def load_evaluation(directory: Path, key: str = "test"):
    """Rebuild the EvaluationResult a run wrote, for compare_models."""
    from compresso_recsys.evaluation import EvaluationResult

    result = json.loads((directory / "result.json").read_text())
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


# --------------------------------------------------------------------------- encode


def encode(args, config: dict) -> int:
    """Encode the checkpoint's item text with the block's sentence encoder (Block C′ features)."""
    block_cfg = _block(config, args.block)
    encoder = block_cfg.get("encoder")
    if encoder is None:
        raise SystemExit(f"block {args.block} has no encoder")
    path = encoded_features_path(args.out, args.dataset, encoder)
    ckpt = checkpoint_path(config, args.block, args.dataset)
    checkpoint_sha256, _ = verify_checkpoint(ckpt)
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        existing = json.loads(sidecar.read_text())
        if existing["checkpoint_sha256"] == checkpoint_sha256 and existing["encoder"] == encoder:
            print(f"skip: {path.relative_to(args.out)} exists for this checkpoint")
            return 0
        raise SpecMismatch(f"{sidecar.relative_to(args.out)} was encoded from another checkpoint or encoder")
    import torch
    from sentence_transformers import SentenceTransformer

    split, _ = load_split(ckpt)
    metadata = split["entity_metadata"]
    if metadata is None or "entity_text" not in metadata:
        raise SystemExit(f"{ckpt.name} has no entity_text to encode")
    text = metadata.set_index("item_id")["entity_text"].reindex(split["item_ids"]).fillna("").astype(str)
    available = text.str.strip().str.len().to_numpy() > 0
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(encoder, device=device)
    started = time.time()
    embeddings = model.encode(text.tolist(), batch_size=256, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    embeddings[~available] = 0.0
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, item_ids=np.asarray(split["item_ids"]).astype(str), embeddings=embeddings,
                        available=available)
    write_json(sidecar, {
        "encoder": encoder, "text_column": "entity_text", "normalized": True,
        "items": int(available.size), "items_with_text": int(available.sum()),
        "dimensions": int(embeddings.shape[1]), "checkpoint": ckpt.name, "checkpoint_sha256": checkpoint_sha256,
        "seconds": round(time.time() - started, 1), "env": environment(), "sha256": sha256_file(path),
    })
    print(f"{path.relative_to(args.out)}: {available.sum()}/{available.size} items with text")
    return 0


# --------------------------------------------------------------------------- verify-checkpoints


def verify_checkpoints(args, config: dict) -> int:
    """Match the checkpoints results were computed on to rebuilt (manifest v2) ones.

    For each checkpoint the config names, compares the content digest of the
    checkpoint in ``--previous`` (what earlier runs used) with the one now in the
    checkpoint directory, and checks that every run's recorded ZIP SHA-256 is
    the previous or the current file. Writes ``provenance/checkpoints.json``.
    Exits 2 when any content differs: those results do not inherit the rebuild's
    provenance.
    """
    recorded: dict[str, set[str]] = {}
    for path in args.out.glob("runs/**/result.json"):
        checkpoint = json.loads(path.read_text())["checkpoint"]
        recorded.setdefault(checkpoint["name"], set()).add(checkpoint["sha256"])
    names = sorted({dataset["checkpoint"] for block in config["blocks"].values() for dataset in block["datasets"].values()})
    rows, mismatched = [], []
    for name in names:
        current = Path(config["paths"]["checkpoints"]) / name
        previous = args.previous / name if args.previous else None
        if not current.exists():
            continue
        row = {"name": name, "sha256": sha256_file(current)}
        load_split(current, info=row)
        if previous is not None and previous.exists():
            before: dict = {}
            load_split(previous, info=before)
            row["previous"] = {"sha256": sha256_file(previous), "manifest_version": before["manifest_version"],
                               "content_sha256": before["content_sha256"]}
            row["content_matches_previous"] = before["content_sha256"] == row["content_sha256"]
            if not row["content_matches_previous"]:
                mismatched.append(name)
        known = {row["sha256"]} | ({row["previous"]["sha256"]} if "previous" in row else set())
        row["runs_on_unknown_file"] = sorted(recorded.get(name, set()) - known)
        if row["runs_on_unknown_file"]:
            mismatched.append(name)
        rows.append(row)
    write_json(args.out / "provenance" / "checkpoints.json", {"checkpoints": rows})
    for row in rows:
        state = ("content matches" if row.get("content_matches_previous") else
                 "CONTENT DIFFERS" if "previous" in row else "no previous file")
        print(f"{row['name']}: v{row['manifest_version']}, {state}"
              + (f", runs on unknown files: {row['runs_on_unknown_file']}" if row["runs_on_unknown_file"] else ""))
    return SPEC_MISMATCH if mismatched else 0


# --------------------------------------------------------------------------- relabel-hosts


def relabel_dir(root: Path, old: str, new: str) -> int:
    """Replace host label ``old`` by ``new`` in every result under ``root``; returns the files changed."""
    new = re.sub(r"[^A-Za-z0-9_.-]", "-", new)
    changed = 0
    for path in list(root.glob("runs/**/result.json")) + list(root.glob("features/**/*.json")):
        record = json.loads(path.read_text())
        if record.get("env", {}).get("host") == old:
            record["env"]["host"] = new
            write_json(path, record)
            changed += 1
    for path in root.glob("runs/**/status.json"):
        record = json.loads(path.read_text())
        if record.get("host") == old:
            record["host"] = new
            write_json(path, record)
            changed += 1
    for path in (root / "env").glob(f"{old}-*"):
        path.rename(path.with_name(new + path.name[len(old):]))
        changed += 1
    meta = root / "meta.json"
    if meta.exists():
        record = json.loads(meta.read_text())
        if old in record.get("hosts", []):
            record["hosts"] = sorted({new if host == old else host for host in record["hosts"]})
            write_json(meta, record)
            changed += 1
    return changed


def relabel_hosts(args, config: dict) -> int:
    """Replace a host label in every result before publication, e.g. a cluster name by a neutral one.

    Rewrites ``env.host`` in result.json, ``host`` in status.json, the ``env/``
    file names, ``features/*.json`` and ``meta.json``. Specs are untouched, so
    nothing reruns. Run ``compare`` afterwards to refresh MANIFEST.json.
    """
    changed = relabel_dir(args.out, args.old, args.new)
    print(f"relabelled {changed} files: {args.old} -> {args.new}")
    return 0


# --------------------------------------------------------------------------- bundle


def verify_manifest(root: Path) -> list[str]:
    """Files whose bytes no longer match MANIFEST.json, and listed files that are missing."""
    manifest = json.loads((root / "MANIFEST.json").read_text())
    problems = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.exists():
            problems.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            problems.append(f"changed: {entry['path']}")
    return problems


def pseudonymise_sample_ids(root: Path) -> int:
    """Replace dataset user IDs in per_user.npz by indices, consistently within each dataset.

    Paired comparison needs the same unit to carry the same ID in every
    model's file of a dataset; it does not need the dataset's own user IDs.
    Each (block, dataset) gets one mapping: sorted distinct IDs to 0..n-1.
    """
    rewritten = 0
    for dataset_dir in sorted(root.glob("runs/*/*")):
        files = sorted(dataset_dir.glob("*/*/per_user.npz"))
        if not files:
            continue
        ids: set[str] = set()
        for path in files:
            with np.load(path, allow_pickle=False) as arrays:
                for name in arrays.files:
                    if name.endswith("/sample_ids"):
                        ids.update(np.asarray(arrays[name]).astype(str).tolist())
        index = {value: position for position, value in enumerate(sorted(ids))}
        for path in files:
            with np.load(path, allow_pickle=False) as arrays:
                data = {name: arrays[name] for name in arrays.files}
            for name in data:
                if name.endswith("/sample_ids"):
                    data[name] = np.array([index[v] for v in data[name].astype(str).tolist()], dtype=np.int64)
            np.savez_compressed(path, **data)
            rewritten += 1
    return rewritten


DATASET_TERMS = {
    "ml20m": ("MovieLens 20M", "GroupLens; research use, no redistribution (https://grouplens.org/datasets/movielens/)"),
    "netflix": ("Netflix Prize", "Netflix Prize terms; the data are no longer distributed by Netflix"),
    "ml1m-llo": ("MovieLens 1M", "GroupLens; research use, no redistribution"),
    "ml1m-item": ("MovieLens 1M with SWAP features", "GroupLens; SWAP features per their Zenodo record"),
    "amazon": ("Amazon Reviews 2023, Toys_and_Games", "McAuley Lab; research use (https://amazon-reviews-2023.github.io/)"),
    "amazon-toys": ("Amazon Reviews 2023, Toys_and_Games", "McAuley Lab; research use"),
    "steam-temporal": ("Steam (McAuley Lab)", "McAuley Lab; research use (https://cseweb.ucsd.edu/~jmcauley/datasets.html)"),
}


def deposit_readme(root: Path, config: dict, *, tag: str | None) -> str:
    meta = json.loads((root / "meta.json").read_text())
    datasets = sorted({path.name for path in (root / "runs").glob("*/*") if path.is_dir()})
    lines = [
        "# compresso-recsys ECIR 2027 benchmark results",
        "",
        f"- **Code commit:** `{meta.get('commit')}`" + (f" (tag `{tag}`)" if tag else ""),
        f"- **Created:** {meta.get('created_at')}",
        f"- **Statistics:** {meta.get('statistics')}; B = {meta.get('n_resamples')} resamples",
        f"- **Hardware:** {', '.join(meta.get('gpus', []))}",
        "",
        "## Contents",
        "",
        "| Path | What it is |",
        "|---|---|",
        "| `table/<dataset>.json`, `casestudy/<dataset>__<catalogue>.json` | The paper's tables: `ComparisonReport.to_frame()` records (Δ vs reference, 95% bootstrap CI, Holm-adjusted randomization p) |",
        "| `meta.json` | Commit, protocol per block (grids, epoch rules, seeds), cold-share figures, statistics settings |",
        "| `summary/<block>/<dataset>.json` / `.md` | Mean and sd over seeds of every metric for each model's selected configuration; Markdown versions of the comparisons |",
        "| `selection/<block>/<dataset>/<model>/selected.json` | Validation score of every grid point and the selected one |",
        "| `runs/<block>/<dataset>/<model>/s<seed>[-t<trial>]/result.json` | One fit: spec, validation and test metrics, training history, checkpoint provenance, environment |",
        "| `runs/.../per_user.npz` | Per-user test values behind every metric, keyed `<split>[__<catalogue>]/<metric>`, with `sample_ids` |",
        "| `runs/.../train.log`, `status.json` | Training logs |",
        "| `features/<dataset>/*.json` | Provenance of the text encodings used in the case study (the vectors themselves are in `.npz`) |",
        "| `env/` | Package versions and GPU of each host |",
        "| `MANIFEST.json` | Every file with size and SHA-256 |",
        "",
        "## Regenerating the paper's numbers",
        "",
        "- **Tables and figure:** the paper reads only `table/`, `casestudy/` and `meta.json`.",
        "- **Statistics:** `compare_models` can be rerun from the per-user values. `run.py compare --config examples/paper/ecir2027.toml --out <this directory>` rebuilds every table and summary.",
        "- **Everything else:** the code at the commit above. Its `examples/paper/README.md` lists the checkpoint builds and the `run.py` stages; each `result.json` records the checkpoint's `content_sha256`, so a rebuild can be matched to the one used here.",
        "",
        "## Data and licences",
        "",
        "No raw interaction data, item metadata or text is included. The files hold metrics, per-user metric values and",
        "model-free summaries derived from the datasets below, and are subject to their terms:",
        "",
    ]
    for name in datasets:
        title, terms = DATASET_TERMS.get(name, (name, "see the dataset's own terms"))
        lines.append(f"- **{title}** (`{name}`): {terms}")
    lines += [
        "",
        "**User identifiers.** `sample_ids` in `per_user.npz` are not the datasets' user IDs.",
        "",
        "- Within each dataset, the distinct IDs were replaced by their rank, so every model's file uses the same index for the same user and paired comparison still works.",
        "- The mapping to dataset IDs is not published.",
        "",
        "Encoded item text (`features/*.npz`) is derived from the datasets' item metadata.",
        "",
        "The results are released under CC BY 4.0, subject to the dataset terms above.",
        "",
    ]
    return "\n".join(lines)


def bundle(args, config: dict) -> int:
    """Build the results deposit: verify, relabel and pseudonymise a copy, then archive it.

    Works on a staging copy of the files listed in MANIFEST.json; ``--out`` is
    never modified. Exits 2 when the manifest does not match the files, or when
    the staged deposit still names this machine, user or home directory.
    """
    import shutil
    import tarfile
    import tempfile

    problems = verify_manifest(args.out)
    if problems:
        raise SpecMismatch("MANIFEST.json does not match the results (run compare again?): "
                           + "; ".join(problems[:10]) + (" …" if len(problems) > 10 else ""))
    manifest = json.loads((args.out / "MANIFEST.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / args.name
        for entry in manifest["files"]:
            target = stage / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.out / entry["path"], target)
        for relabel in args.relabel or []:
            old, _, new = relabel.partition("=")
            print(f"relabelled {relabel_dir(stage, old, new)} files: {old} -> {new}")
        print(f"pseudonymised sample IDs in {pseudonymise_sample_ids(stage)} files")
        (stage / "README.md").write_text(deposit_readme(stage, config, tag=args.tag))
        leaks = write_manifest(stage)
        if leaks:
            raise SpecMismatch("the deposit still names this machine, user or home directory: " + ", ".join(leaks[:10]))
        staged = json.loads((stage / "MANIFEST.json").read_text())["files"]
        if args.git_dir is not None:
            for entry in staged:
                if entry["tier"] == "git":
                    target = args.git_dir / entry["path"]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stage / entry["path"], target)
            for name in ("README.md", "MANIFEST.json"):
                shutil.copy2(stage / name, args.git_dir / name)
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.archive, "w:gz") as archive:
            archive.add(stage, arcname=args.name)
    git_files = sum(1 for entry in staged if entry["tier"] == "git")
    print(f"{args.archive}: {len(staged)} files ({git_files} git tier"
          + (f", copied to {args.git_dir}" if args.git_dir else "") + f"), sha256 {sha256_file(args.archive)}")
    return 0


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
    catalogue_mode = block_cfg["mode"] == "catalogue"
    selection_key = evaluation_key("val", block_cfg.get("selection_catalogue", "all") if catalogue_mode else "all")
    ckpt = checkpoint_path(config, args.block, args.dataset)
    out = run_dir(args.out, args.block, args.dataset, args.model, args.seed, trial)

    spec = {
        "block": args.block, "dataset": args.dataset, "model": args.model, "kind": kind,
        "seed": args.seed, "trial": trial, "params": params, "training": spec_cfg["training"],
        "selection_metric": block_cfg["selection"], "selection_key": selection_key,
        "catalogues": spec_cfg["catalogues"], "checkpoint": ckpt.name,
        "exclude_seen": EXCLUDE_SEEN, "max_epochs_override": args.max_epochs,
    }
    if KINDS[kind]["input"] == "features" and "encoder" in block_cfg:
        spec["encoder"] = block_cfg["encoder"]
    if catalogue_mode:
        # Since 7fa8d57 catalogue blocks binarise training data and histories. A
        # result computed before that may have seen repeat counts; making it part
        # of the spec turns such a result into a spec mismatch rather than a skip.
        spec["binary_interactions"] = True
    digest = spec_digest(spec)
    result_path = out / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("spec_sha256") == digest:
            print(f"skip: {result_path.relative_to(args.out)} exists with the same spec")
            return 0
        raise SpecMismatch(f"{result_path.relative_to(args.out)} exists with a different spec; "
                           "move it away or fix the config")

    out.mkdir(parents=True, exist_ok=True)
    record_host_environment(args.out)
    logger = logging.getLogger(f"cr-paper.{args.block}.{args.dataset}.{args.model}.s{args.seed}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(out / "train.log")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    console = logging.StreamHandler(sys.stdout)
    logger.addHandler(handler)
    logger.addHandler(console)
    started = time.time()
    status = {"spec": spec, "phase": "started", "started_unix": started, "host": host_label()}
    write_json(out / "status.json", status)
    phases: dict[str, float] = {}

    def mark(phase: str) -> None:
        phases[phase] = round(time.time() - started, 2)
        status.update(phase=phase, elapsed_s=phases[phase])
        write_json(out / "status.json", status)
        logger.info(f"[{phases[phase]:9.1f}s] {phase}")

    try:
        checkpoint_sha256, checkpoint_check = verify_checkpoint(ckpt)
        feature_names = list(params.get("features", ())) if KINDS[kind]["input"] == "features" else []
        encoded = catalogue_mode and "encoder" in block_cfg
        checkpoint_info: dict = {}
        split, features = load_split(ckpt, [] if encoded else feature_names, info=checkpoint_info)
        if encoded:
            vectors = load_encoded_features(args.out, args.dataset, block_cfg["encoder"], np.asarray(split["item_ids"]))
            features = {name: vectors for name in feature_names}
        mark("loaded")
        import torch

        device = args.device or ("cuda" if KINDS[kind]["device"] == "gpu" and torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        epochs = _epochs(kind, spec_cfg["training"], params, split, args.max_epochs)
        model = _build(kind, params, seed=args.seed, device=device, epochs=epochs)
        # Without the checkpoint's evaluation user IDs, rows are identified by
        # position. That is harmless for models that ignore user identity, but
        # a model that looks users up would silently score row i as training
        # user i, so it is refused rather than guessed.
        missing_ids = [phase for phase in ("val", "test") if split.get(f"{phase}_eval_user_ids") is None]
        if missing_ids and getattr(model, "accepts_user_ids", False):
            raise ValueError(
                f"{kind} uses user identity, but {ckpt.name} has no {'/'.join(missing_ids)} evaluation "
                "user IDs; positional IDs would pair evaluation rows with the wrong training users"
            )
        if missing_ids:
            logger.warning(f"WARNING: {ckpt.name} has no {'/'.join(missing_ids)} evaluation user IDs; "
                           "rows are identified by position")
        batch_size = int(config["run"].get("eval_batch_size", 1024))
        data = catalogue_inputs(split) if catalogue_mode else None

        # What the model trains on, and what in-training validation scores.
        fit_args: tuple = ()
        fit_kwargs: dict[str, Any] = {}
        if not catalogue_mode:
            fit_args = (split["x_train_sequences"] if KINDS[kind]["input"] == "sequence" else split["x_train"],)
            fit_kwargs["item_ids"] = split["train_item_ids"]
            val = phase_inputs(split, "val", kind)
            val_source, val_targets, val_ids = val["source"], val["targets"], val["sample_ids"]
        elif KINDS[kind]["input"] == "features":
            matrices = {name: features[name]["embeddings"] for name in feature_names}
            masks = {name: features[name]["available"] for name in feature_names}
            if kind != "content":
                fit_args = (data["x_train"],)
                fit_kwargs["train_item_indices"] = data["warm"]
            if KINDS[kind].get("user_ids"):
                fit_kwargs["user_ids"] = training_user_ids(split)
            fit_kwargs.update(item_features=matrices, item_ids=data["item_ids"], modality_masks=masks)
            # The fitted catalogue is every checkpoint item, so validation scores the full catalogue.
            val = data["phases"]["val"]
            val_source, val_targets, val_ids = val["source"], val["targets"], val["sample_ids"]
        else:
            warm = data["warm"]
            fit_args = (data["x_train"][:, warm].tocsr(),)
            fit_kwargs["item_ids"] = data["item_ids"][warm]
            # A warm-only model validates on the training items, the only ones it can rank.
            val = data["phases"]["val"]
            val_source, val_targets, val_ids = val["source"][:, warm].tocsr(), val["targets"][:, warm].tocsr(), val["sample_ids"]

        training = spec_cfg["training"]
        if KINDS[kind].get("early_stopping"):
            from compresso_recsys.training import EarlyStopping, Validation

            fit_kwargs["validation"] = Validation(
                val_source, val_targets, [_selection_metric(block_cfg["selection"])],
                sample_ids=val_ids, every=int(training.get("every", 1)),
                batch_size=batch_size, exclude_seen=EXCLUDE_SEEN,
            )
            if "patience" in training:
                fit_kwargs["early_stopping"] = EarlyStopping(
                    block_cfg["selection"], patience=int(training["patience"]),
                    min_epochs=int(training.get("min_epochs", 0)),
                )
        if KINDS[kind]["epochs"]:
            fit_kwargs["logger"] = logger
        fit_started = time.time()
        model.fit(*fit_args, **fit_kwargs)
        fit_seconds = time.time() - fit_started
        mark("fitted")

        from compresso_recsys.evaluation import evaluate_recommender

        evaluation, arrays = {}, {}
        for phase in ("val", "test"):
            for catalogue in spec_cfg["catalogues"]:
                key = evaluation_key(phase, catalogue)
                if not catalogue_mode:
                    inputs = phase_inputs(split, phase, kind)
                    scorer, source, targets = model, inputs["source"], inputs["targets"]
                    sample_ids = inputs["sample_ids"]
                else:
                    inputs = data["phases"][phase]
                    columns = inputs["catalogues"][catalogue]
                    source, sample_ids = inputs["source"], inputs["sample_ids"]
                    targets = inputs["targets"][:, columns].tocsr()
                    if KINDS[kind]["input"] == "features":
                        model.build_candidates(
                            item_ids=data["item_ids"][columns],
                            item_features={name: matrices[name][columns] for name in feature_names},
                            modality_masks={name: masks[name][columns] for name in feature_names},
                        )
                        scorer = model
                    else:
                        scorer = WarmOnly(model, data["warm"], columns, data["item_ids"].size)
                phase_started = time.time()
                result = evaluate_recommender(
                    scorer, source=source, targets=targets, metrics=_metrics(),
                    sample_ids=sample_ids, exclude_seen=EXCLUDE_SEEN, batch_size=batch_size,
                )
                evaluation[key] = {**_evaluation_record(result), "seconds": round(time.time() - phase_started, 2),
                                   "catalogue_items": int(targets.shape[1])}
                if phase == "test":
                    arrays[f"{key}/sample_ids"] = np.asarray(result.sample_ids)
                    for name, values in result.per_user.items():
                        arrays[f"{key}/{name}"] = np.asarray(values)
                mark(f"evaluated_{key}")
        np.savez_compressed(out / "per_user.npz", **arrays)

        trained = getattr(model, "inner_model", model)
        history = getattr(trained, "history", None)
        payload = {
            "spec": spec,
            "spec_sha256": digest,
            "selection": {"metric": block_cfg["selection"], "key": selection_key,
                          "validation": evaluation[selection_key]["metrics"][block_cfg["selection"]]},
            "evaluation": evaluation,
            "training": {
                "device": device,
                "epochs_budget": epochs,
                "epochs_trained": len(history) if isinstance(history, list) else None,
                "best": getattr(trained, "best_", None),
                "fit_seconds": round(fit_seconds, 2),
                "history": history if isinstance(history, list) else None,
            },
            "sample_ids": "positional" if missing_ids else "checkpoint",
            "checkpoint": {"name": ckpt.name, "sha256": checkpoint_sha256, "check": checkpoint_check,
                           **checkpoint_info},
            "env": environment(),
            "phases_s": phases,
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if catalogue_mode:
            payload["cold_share"] = cold_share(data)
            payload["binarized_entries"] = data["binarized"]
        write_json(result_path, payload)
        mark("done")
        return 0
    except SpecMismatch:
        status.update(phase="spec_mismatch", error=scrubbed(traceback.format_exc()))
        write_json(out / "status.json", status)
        raise
    except BaseException:
        status.update(phase="failed", error=scrubbed(traceback.format_exc()))
        write_json(out / "status.json", status)
        raise
    finally:
        logger.removeHandler(handler)
        logger.removeHandler(console)
        handler.close()


def cold_share_explicit(split: dict) -> dict:
    """Test-stage cold share under both definitions, with explicit keys.

    ``*_vs_training``: items absent from the training window, which is what the
    models face (they are fit once on the training window, never refit).
    ``*_new_in_stage``: items first supported in the test window. Each value is
    ``{"cold", "total", "share"}``.
    """
    data = catalogue_inputs(split)
    test = data["phases"]["test"]
    catalogue = test["catalogues"]["full"]
    per_item = np.diff(test["targets"].tocsc().indptr)
    warm = np.zeros(data["item_ids"].size, dtype=bool)
    warm[data["warm"]] = True
    new = np.zeros(data["item_ids"].size, dtype=bool)
    new[test["catalogues"]["cold"]] = True
    in_catalogue = np.zeros(data["item_ids"].size, dtype=bool)
    in_catalogue[catalogue] = True
    targeted = per_item > 0

    def entry(cold_count: int, total: int) -> dict:
        return {"cold": cold_count, "total": total, "share": round(cold_count / total, 4) if total else None}

    out = {}
    for suffix, cold in (("vs_training", ~warm), ("new_in_stage", new)):
        out[f"catalogue_items_{suffix}"] = entry(int((cold & in_catalogue).sum()), int(in_catalogue.sum()))
        out[f"target_items_{suffix}"] = entry(int((cold & targeted).sum()), int(targeted.sum()))
        out[f"target_interactions_{suffix}"] = entry(int(per_item[cold].sum()), int(per_item.sum()))
    out["definition"] = ("vs_training: not seen in the training window (models are not refit on validation "
                         "data); new_in_stage: first supported in the test window")
    return out


def cold_share(data: dict) -> dict:
    """The three cold-share definitions of benchmark-plan.md, for the test phase."""
    test = data["phases"]["test"]
    warm = np.zeros(data["item_ids"].size, dtype=bool)
    warm[data["warm"]] = True
    catalogue = test["catalogues"]["full"]
    targets = test["targets"].tocsc()
    target_items = np.flatnonzero(np.diff(targets.indptr) > 0)
    per_item = np.diff(targets.indptr)
    return {
        "catalogue_items": {"cold": int((~warm[catalogue]).sum()), "total": int(catalogue.size)},
        "target_items": {"cold": int((~warm[target_items]).sum()), "total": int(target_items.size)},
        "target_interactions": {"cold": int(per_item[~warm].sum()), "total": int(per_item.sum())},
    }


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
        "metric": metric, "key": found[0][1]["selection"]["key"], "trial": best_trial,
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
        raise FileNotFoundError(f"{path.relative_to(out)} is missing: run select first")
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
                    # A probe (report = false) only needs its grid: it is never compared.
                    reported = block["models"][model].get("report", True)
                    runs = [(seed, trial) for seed in seeds_for(config, spec) if seed != 0 and reported]
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
    if rel.parts[0] in ("features", "provenance") and rel.suffix == ".json":
        return "git"
    return "archive"


def _identifying_strings() -> list[str]:
    """What must never be published: this machine's names, the user, and the home path."""
    names = {socket.gethostname(), socket.gethostname().split(".")[0], getpass.getuser(), str(Path.home())}
    return sorted(name for name in names if len(name) >= 3)


def _host_labels(root: Path) -> set[str]:
    """The public host labels results carry: chosen to be published, so never a leak themselves."""
    labels = {host_label()}
    for path in root.glob("runs/**/result.json"):
        labels.add(json.loads(path.read_text())["env"]["host"])
    for path in (root / "env").glob("*-pip-freeze.txt"):
        labels.add(path.name[: -len("-pip-freeze.txt")])
    return {label for label in labels if label}


def scrub_check(root: Path, files: list[dict]) -> list[str]:
    """Paths of published text files that name this machine, user or home directory.

    Public host labels are removed before matching, so a label such as
    ``ostrava-h100`` on a node named ``h100`` is not reported, while the bare
    hostname still is.
    """
    patterns = [re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])") for name in _identifying_strings()]
    labels = sorted(_host_labels(root), key=len, reverse=True)
    leaks = []
    for entry in files:
        path = root / entry["path"]
        if path.suffix not in (".json", ".txt", ".md", ".log"):
            continue
        text = path.read_text(errors="replace")
        for label in labels:
            text = text.replace(label, " ")
        if any(pattern.search(text) for pattern in patterns):
            leaks.append(entry["path"])
    return leaks


def scrubbed(text: str) -> str:
    """A traceback or log line with this machine's home path, user and hostname replaced."""
    text = text.replace(str(Path.home()), "~")
    for name in _identifying_strings():
        if name != str(Path.home()):
            text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<redacted>", text)
    return text


def write_manifest(root: Path) -> list[str]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "MANIFEST.json" or path.suffix == ".tmp":
            continue
        files.append({"path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                      "sha256": sha256_file(path), "tier": _tier(path, root)})
    leaks = scrub_check(root, files)
    for entry in files:
        entry["identifying"] = entry["path"] in leaks
    write_json(root / "MANIFEST.json", {
        "files": files,
        "scrub": {"checked_for": "this host's name, the user name and the home path",
                  "flagged": leaks},
    })
    return leaks


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
                seeds[seed] = {"evaluation": {key: record["metrics"] for key, record in result["evaluation"].items()},
                               "epochs_trained": result["training"]["epochs_trained"],
                               "best_epoch": (result["training"]["best"] or {}).get("epoch"),
                               "fit_seconds": result["training"]["fit_seconds"]}
        if not seeds:
            continue
        keys = sorted(next(iter(seeds.values()))["evaluation"])
        mean, sd = {}, {}
        for key in keys:
            names = sorted(next(iter(seeds.values()))["evaluation"][key])
            values = {name: [s["evaluation"][key][name] for s in seeds.values()] for name in names}
            mean[key] = {n: float(np.mean(v)) for n, v in values.items()}
            sd[key] = {n: float(np.std(v, ddof=1)) if len(v) > 1 else None for n, v in values.items()}
        rows[model] = {
            "trial": trial,
            "params": {**spec["params"], **grid_points(spec)[trial or 0]},
            "training": spec["training"],
            "deterministic": KINDS[spec["kind"]]["deterministic"],
            "catalogues": spec["catalogues"],
            "seeds": seeds,
            "mean": mean,
            "sd": sd,
            "n_seeds": len(seeds),
        }
    return rows


def compare_pairs(out: Path, block_name: str, block: dict, dataset: str, summary: dict,
                  pairs: list, n_resamples: int) -> list[dict]:
    """Extra paired tests the paper quotes outside the tables, e.g. TEASER vs Content.

    Each pair ``[baseline, candidate]`` is its own family per catalogue: Holm over
    the table metrics of that pair only. The records say so (``family``).
    """
    from compresso_recsys.stats import compare_models

    records = []
    for baseline, candidate in pairs:
        if baseline not in summary or candidate not in summary:
            continue
        for catalogue in outputs_for(block, dataset):
            key = evaluation_key("test", catalogue)
            if not all(catalogue in summary[m]["catalogues"] for m in (baseline, candidate)):
                continue
            results = {m: load_evaluation(run_dir(out, block_name, dataset, m, 0, summary[m]["trial"]), key)
                       for m in (baseline, candidate)}
            report = compare_models(results, metrics=list(TABLE_METRICS), reference=baseline,
                                    n_resamples=n_resamples, random_state=0)
            for record in report.to_frame().to_dict(orient="records"):
                records.append({"catalogue": catalogue,
                                "family": f"{candidate} vs {baseline}, {len(TABLE_METRICS)} metrics",
                                **record})
    return records


def selection_edges(out: Path, config: dict) -> dict:
    """Per block, dataset and model: the selected point and the numeric grid axes it sits on the edge of.

    An axis is at the edge when the selected value is the smallest or largest
    value on it (by value, not position, since grids may be extended by
    appending). Non-numeric axes (feature sets, weight vectors) are skipped.
    ``at_edge`` is empty when the selection is interior. Only axes of three or
    more values count: on a two-value axis every pick is an edge. ``n_values``
    is the axis size; ``last_step_gain`` is the validation score of the selected
    point minus that of the point one step inwards on the axis, other settings
    unchanged.
    """
    edges: dict = {}
    for block_name, block in config["blocks"].items():
        for dataset in block["datasets"]:
            for model in models_for(config, block_name, dataset):
                spec = model_spec(config, block_name, dataset, model)
                path = out / "selection" / block_name / dataset / model / "selected.json"
                if not spec["grid"] or not path.exists():
                    continue
                selected = json.loads(path.read_text())
                params = selected["params"]
                # Keyed by the grid parameters each trial recorded at select time, so a
                # later change of the config cannot pair a score with the wrong point.
                scores = {json.dumps(t["params"], sort_keys=True): t["validation"] for t in selected["trials"]}
                at_edge = {}
                for axis, values in spec["grid"].items():
                    numeric = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
                    # On an axis of one or two values every pick is an edge: not a search
                    # that ran out of range, so it is not reported.
                    if len(numeric) < 3 or len(numeric) != len(values) or axis not in params:
                        continue
                    ordered = sorted(set(numeric))
                    if params[axis] == ordered[-1]:
                        side, neighbour = "max", ordered[-2]
                    elif params[axis] == ordered[0]:
                        side, neighbour = "min", ordered[1]
                    else:
                        continue
                    # Validation gain of the last grid step on this axis: the selected point
                    # minus the point one step inwards, every other setting unchanged.
                    grid_params = {key: params[key] for key in selected["trials"][0]["params"]}
                    inward = json.dumps({**grid_params, axis: neighbour}, sort_keys=True)
                    chosen = json.dumps(grid_params, sort_keys=True)
                    gain = (scores[chosen] - scores[inward]) if chosen in scores and inward in scores else None
                    at_edge[axis] = {"value": params[axis], "side": side, "n_values": len(numeric),
                                     "last_step_gain": gain}
                edges.setdefault(block_name, {}).setdefault(dataset, {})[model] = {
                    "params": params, "at_edge": at_edge}
    return edges


def protocol_description(config: dict) -> dict:
    """Per block: selection, and each model's grid and epoch rule, for the paper's captions."""
    blocks = {}
    for block_name, block in config["blocks"].items():
        models = {}
        for model_name, model in block["models"].items():
            training = model.get("training", {})
            if "patience" in training:
                rule = (f"max {training['max_epochs']} epochs, validate every {training.get('every', 1)}, "
                        f"patience {training['patience']}, best epoch restored")
            elif "target_steps" in training:
                rule = f"fixed budget of {training['target_steps']} steps"
            elif "epochs" in training:
                rule = f"fixed budget of {training['epochs']} epochs (validation logged)"
            else:
                rule = "closed form / single pass"
            models[model_name] = {"kind": model["kind"], "params": model.get("params", {}),
                                  "grid": model.get("grid", {}), "training": training, "epochs": rule,
                                  "seeds": [0] if KINDS[model["kind"]]["deterministic"] else config["run"]["seeds"],
                                  "overrides": model.get("overrides", {})}
        blocks[block_name] = {
            "datasets": {name: dataset["checkpoint"] for name, dataset in block["datasets"].items()},
            "selection": block["selection"],
            "selection_catalogue": block.get("selection_catalogue") if block["mode"] == "catalogue" else None,
            "catalogues": block.get("catalogues") if block["mode"] == "catalogue" else None,
            "reference": block.get("reference", "EASE"),
            "encoder": block.get("encoder"),
            "models": models,
        }
    return blocks


def compare(args, config: dict) -> int:
    from compresso_recsys.stats import compare_models

    out = args.out
    n_resamples = int(config.get("statistics", {}).get("n_resamples", 9_999))
    stale = [
        str(path.relative_to(out))
        for block_name, block in config["blocks"].items() if block["mode"] == "catalogue"
        for path in sorted((out / "runs" / block_name).glob("**/result.json"))
        if not json.loads(path.read_text())["spec"].get("binary_interactions")
    ]
    if stale:
        raise SpecMismatch(
            "catalogue-block results computed before interactions were binarised (7fa8d57) cannot be "
            "compared with later ones; move them aside and rerun: " + ", ".join(stale)
        )
    written = []
    for block_name, block in config["blocks"].items():
        for dataset in block["datasets"]:
            summary = _summary(out, config, block_name, dataset)
            if not summary:
                continue
            write_json(out / "summary" / block_name / f"{dataset}.json", summary)
            markdown = []
            for catalogue, target in outputs_for(block, dataset).items():
                key = evaluation_key("test", catalogue)
                # The comparison uses the seed-0 per-user values of each model's selected configuration.
                results = {
                    model: load_evaluation(run_dir(out, block_name, dataset, model, 0, row["trial"]), key)
                    for model, row in summary.items()
                    if catalogue in row["catalogues"] and block["models"][model].get("report", True)
                }
                reference = block.get("reference", "EASE")
                if reference not in results or len(results) < 2:
                    print(f"{block_name}/{dataset}/{catalogue}: needs the reference {reference} and one more "
                          "model; skipped", file=sys.stderr)
                    continue
                report = compare_models(results, metrics=list(TABLE_METRICS), reference=reference,
                                        n_resamples=n_resamples, random_state=0)
                path = out / f"{target}.json"
                write_json(path, report.to_frame().to_dict(orient="records"))
                written.append(str(path.relative_to(out)))
                markdown.append(f"## {dataset} ({catalogue})\n\n{report.to_markdown()}\n")
            if markdown:
                (out / "summary" / block_name / f"{dataset}.md").write_text("\n".join(markdown))
            pairs = block.get("pairs", [])
            if pairs:
                write_json(out / "summary" / block_name / f"{dataset}.pairs.json",
                           compare_pairs(out, block_name, block, dataset, summary, pairs, n_resamples))
    cold_shares = {}
    for block_name, block in config["blocks"].items():
        if block["mode"] != "catalogue":
            continue
        for dataset in block["datasets"]:
            path = checkpoint_path(config, block_name, dataset)
            if path.exists():
                split, _ = load_split(path)
                cold_shares[dataset] = cold_share_explicit(split)
                write_json(out / "summary" / block_name / f"{dataset}.cold_share.json", cold_shares[dataset])
            else:
                print(f"{block_name}/{dataset}: checkpoint not found, cold share not written", file=sys.stderr)
    results = list(out.glob("runs/**/result.json"))
    meta = {
        "commit": os.environ.get("CR_SOURCE_COMMIT"),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hosts": sorted({json.loads(p.read_text())["env"]["host"] for p in results}),
        "gpus": sorted({(json.loads(p.read_text())["env"]["gpu"] or {}).get("name", "cpu") for p in results}),
        "note": config["run"].get("note", ""),
        "placeholder": bool(config["run"].get("placeholder", False)),
        "n_resamples": n_resamples,
        "exclude_seen": EXCLUDE_SEEN,
        "table_metrics": list(TABLE_METRICS),
        "statistics": "compare_models on the seed-0 per-user values of each model's selected configuration; "
                      "paired bootstrap CIs, paired randomization test, Holm over each file's models and metrics",
        "protocol_note": PROTOCOL_NOTE,
        "protocol": protocol_description(config),
        "selection": selection_edges(out, config),
        "cold_share": cold_shares,
    }
    write_json(out / "meta.json", meta)
    leaks = write_manifest(out)
    print("\n".join(written) or "nothing to compare yet")
    if leaks:
        print("identifying strings found (see MANIFEST.json scrub.flagged): " + ", ".join(leaks), file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "select", "compare", "list", "encode", "verify-checkpoints", "relabel-hosts", "bundle"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
        if name in ("train", "select", "encode"):
            sub.add_argument("--block", required=True)
            sub.add_argument("--dataset", required=True)
        if name in ("train", "select"):
            sub.add_argument("--model", required=True)
        if name == "train":
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--grid-trial", type=int)
        if name in ("train", "encode"):
            sub.add_argument("--device", help="override the device (default: cuda when available for GPU models)")
        if name in ("train", "list"):
            sub.add_argument("--max-epochs", type=int, help="cap every epoch budget (smoke tests)")
        if name == "bundle":
            sub.add_argument("--archive", type=Path, required=True, help="the .tar.gz to write")
            sub.add_argument("--name", default="results-ecir2027", help="top-level directory inside the archive")
            sub.add_argument("--relabel", action="append", metavar="OLD=NEW",
                             help="replace a host label in the deposit (repeatable)")
            sub.add_argument("--git-dir", type=Path, help="copy the git-tier files here, e.g. examples/paper/results")
            sub.add_argument("--tag", help="the release tag the deposit belongs to")
        if name == "relabel-hosts":
            sub.add_argument("--old", required=True, help="the label to replace")
            sub.add_argument("--new", required=True, help="the label to publish instead")
        if name == "verify-checkpoints":
            sub.add_argument("--previous", type=Path,
                             help="directory with the checkpoints earlier runs used (e.g. the v1 builds)")
        if name == "list":
            sub.add_argument("--stage", choices=("grid", "seeds"), required=True)
            sub.add_argument("--device", choices=("gpu", "cpu"))
            sub.add_argument("--block")
            sub.add_argument("--out", type=Path)
        else:
            sub.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    commands_by_name = {"train": train, "select": select, "compare": compare, "list": list_commands,
                        "encode": encode, "verify-checkpoints": verify_checkpoints,
                        "relabel-hosts": relabel_hosts, "bundle": bundle}
    try:
        return commands_by_name[args.command](args, config)
    except SpecMismatch as error:
        print(f"spec mismatch: {error}", file=sys.stderr)
        return SPEC_MISMATCH


if __name__ == "__main__":
    sys.exit(main())
