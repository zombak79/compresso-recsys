"""Build dataset/split checkpoints, gather statistics, and evaluate CF baselines.

Run with --help. Defaults cover every registered dataset and split, but just one
Amazon category. This can download many GB and take hours; narrow --datasets and
--splits for a first run. No embeddings or media are downloaded by this script.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from importlib.metadata import version

import numpy as np
import pandas as pd
import torch
import pyarrow as pa
from scipy.sparse import csr_matrix

import compresso_recsys as cr
from compresso_recsys import builder
from compresso_recsys.datasets._public import PublicDataset
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall, HitRate, NDCG, Recall
from compresso_recsys.models import ItemKNNConfig, ItemKNNRecommender, PopularityBaseline

SPLITS = ("user_split", "item_split", "leave_last_out", "temporal", "official")
def distribution(values):
    values = np.asarray(values)
    if not values.size:
        return None
    return {"min": float(values.min()), "mean": float(values.mean()),
            "p50": float(np.quantile(values, .5)), "p90": float(np.quantile(values, .9)),
            "p99": float(np.quantile(values, .99)), "max": float(values.max())}


def frame_stats(frame):
    """Density uses unique pairs, not repeated events or rating/play-count sums."""
    pairs = frame[["user_id", "item_id"]].drop_duplicates()
    users, items = pairs.user_id.nunique(), pairs.item_id.nunique()
    timestamps = pd.to_numeric(frame.timestamp, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(timestamps)
    return {
        "n_users": int(users), "n_items": int(items), "n_events": len(frame),
        "n_unique_pairs": len(pairs),
        "sparsity": 1 - len(pairs) / (users * items) if users and items else None,
        "repeat_event_fraction": 1 - len(pairs) / len(frame) if len(frame) else None,
        "unique_items_per_user": distribution(pairs.groupby("user_id").size()),
        "unique_users_per_item": distribution(pairs.groupby("item_id").size()),
        "timestamp_coverage": float(valid.mean()) if len(frame) else None,
        "timestamp_min_native_units": float(timestamps[valid].min()) if valid.any() else None,
        "timestamp_max_native_units": float(timestamps[valid].max()) if valid.any() else None,
    }


def binary(matrix):
    matrix = matrix.tocsr(copy=True)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.data = np.ones(matrix.nnz, dtype=np.float32)
    return matrix


def matrix_sparsity(nnz, rows, items):
    """Keep all evaluation rows, including repeated users and empty histories."""
    return 1 - nnz / (rows * items) if rows and items else None


def matrix_stats(matrix, user_ids=None):
    matrix = binary(matrix)
    rows, columns = matrix.shape
    per_row = np.diff(matrix.indptr)
    per_item = np.bincount(matrix.indices, minlength=columns)
    observed_items = int((per_item > 0).sum())
    catalog_sparsity = matrix_sparsity(matrix.nnz, rows, columns)
    return {
        "n_rows": rows, "n_users": int(len(np.unique(user_ids))) if user_ids is not None else None,
        "n_items": columns, "active_rows": int((per_row > 0).sum()),
        "active_items": observed_items, "nnz": matrix.nnz,
        # Preserve the original JSON fields; n_items/sparsity describe the
        # full matrix, not just columns with observed interactions.
        "sparsity": catalog_sparsity, "catalog_sparsity": catalog_sparsity,
        "observed_item_sparsity": matrix_sparsity(matrix.nnz, rows, observed_items),
        "items_per_row": distribution(per_row), "rows_per_item": distribution(per_item),
    }


def align_columns(matrix, local_ids, global_ids):
    """Remap sparse columns by original IDs, including permuted phase vocabularies."""
    index = pd.Index(np.asarray(global_ids).astype(str))
    columns = index.get_indexer(np.asarray(local_ids).astype(str))
    if matrix.shape[1] != len(columns) or (columns < 0).any() or not index.is_unique:
        raise ValueError("Invalid checkpoint item vocabulary")
    result = csr_matrix((matrix.data.copy(), columns[matrix.indices], matrix.indptr.copy()),
                        shape=(matrix.shape[0], len(global_ids)))
    return binary(result)


def checkpoint_stats(split, root):
    global_ids = split["item_ids"]
    train = align_columns(split["x_train"], split["train_item_ids"], global_ids)
    active = np.bincount(train.indices, minlength=len(global_ids)) > 0
    stats = {"catalog_items": len(global_ids), "train_observed_items": int(active.sum()),
             "train": matrix_stats(split["x_train"], split["train_user_ids"]),
             "precomputed_features": cr.list_item_embeddings(root)}
    for phase in ("train", "val", "test"):
        ids = split[f"{phase}_item_ids"]
        users = split["train_user_ids"] if phase == "train" else split[f"{phase}_eval_user_ids"]
        source = split[f"{phase}_source_matrix"]
        target = split[f"{phase}_target_matrix"]
        global_target = align_columns(target, ids, global_ids)
        seen_target = source.multiply(target)
        candidates = pd.Index(global_ids).get_indexer(ids)
        sequence = split.get(f"{phase}_source_sequences")
        stats[phase + "_evaluation"] = {
            "source": matrix_stats(source, users), "target": matrix_stats(target, users),
            "source_target_overlap_nnz": binary(seen_target).nnz,
            "cold_candidate_items": int((~active[candidates]).sum()),
            "cold_target_fraction": (float((~active[global_target.indices]).mean())
                                     if global_target.nnz else None),
            "source_sequence_events": int(sequence.values.size) if sequence is not None else None,
        }
    metadata = split.get("entity_metadata")
    stats["metadata_coverage"] = {}
    if metadata is not None:
        for field in ("entity_text", "image_url"):
            if field in metadata:
                stats["metadata_coverage"][field] = float(
                    metadata[field].fillna("").astype(str).str.strip().ne("").mean())
    return stats


class PhaseCandidates:
    """Keep predictions in global columns while restricting to this phase's items."""

    def __init__(self, model, item_ids):
        self.model, self.item_ids = model, item_ids

    def predict_on_batch(self, source, *, k):
        return self.model.predict_on_batch(source, k=k, candidate_ids=self.item_ids)


def evaluate_phase(model, split, phase, *, cutoffs, max_users, seed, batch_size):
    ids = split[f"{phase}_item_ids"]
    source = align_columns(split[f"{phase}_source_matrix"], ids, split["item_ids"])
    targets = align_columns(split[f"{phase}_target_matrix"], ids, split["item_ids"])
    users = split[f"{phase}_eval_user_ids"]
    if users is None or len(users) != source.shape[0]:
        raise ValueError("Evaluation row user IDs are required for an auditable sweep")
    eligible = (np.diff(targets.indptr) > 0) & (len(ids) - np.diff(source.indptr) >= max(cutoffs))
    selected = eligible.copy()
    if max_users is not None:
        unique = np.unique(users[eligible])
        if len(unique) > max_users:
            chosen = np.random.default_rng(seed).choice(unique, max_users, replace=False)
            selected &= np.isin(users, chosen)
    info = {"total_rows": len(users), "eligible_rows": int(eligible.sum()),
            "evaluated_rows": int(selected.sum()), "evaluated_users": int(len(np.unique(users[selected]))),
            "excluded_rows": int((~eligible).sum()),
            "exclusion_rule": "no targets or fewer unseen candidates than largest requested cutoff"}
    if not selected.any():
        return {"status": "skipped", "reason": "No eligible rows at requested cutoffs", **info}
    started = time.perf_counter()
    result = evaluate_recommender(
        PhaseCandidates(model, ids), source=source[selected], targets=targets[selected],
        sample_ids=users[selected], metrics=[Recall(cutoffs), CalibratedRecall(cutoffs),
                                          NDCG(cutoffs), HitRate(cutoffs)],
        batch_size=batch_size, collect_per_user=False,
    )
    return {"status": "ok", **info, "seconds": time.perf_counter() - started,
            "metrics": dict(result)}


def evaluate_baselines(split, args):
    if not args.baselines:
        return {}
    train = align_columns(split["x_train"], split["train_item_ids"], split["item_ids"])
    results = {}
    for name in args.baselines:
        if name == "itemknn" and train.shape[1] > args.knn_max_items:
            results[name] = {"status": "skipped", "reason": "catalog exceeds --knn-max-items"}
            continue
        started = time.perf_counter()
        try:
            model = (PopularityBaseline() if name == "popularity" else ItemKNNRecommender(
                ItemKNNConfig(n_neighbors=args.neighbors, n_jobs=args.knn_jobs)))
            model.fit(train, item_ids=split["item_ids"])
            entry = {"status": "ok", "fit_seconds": time.perf_counter() - started,
                     "config": vars(model.cfg)}
            for phase in ("val", "test"):
                try:
                    entry[phase] = evaluate_phase(
                        model, split, phase, cutoffs=args.cutoffs, max_users=args.max_eval_users,
                        seed=args.seed, batch_size=args.batch_size,
                    )
                except Exception as error:
                    entry[phase] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
                    entry["status"] = "failed"
            results[name] = entry
            del model
        except Exception as error:
            results[name] = {"status": "failed", "error": f"{type(error).__name__}: {error}"}
    return results


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_version():
    # The script may be downloaded/copied elsewhere. Attribute Git provenance
    # to the imported package, never to the script's enclosing repository.
    source = Path(cr.__file__).resolve()
    try:
        repo = Path(subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], cwd=source.parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip())
        relative_source = source.relative_to(repo)
        # A wheel inside another project's .venv is not that project's source.
        subprocess.check_output(["git", "ls-files", "--error-unmatch", "--", str(relative_source)],
                                cwd=repo, stderr=subprocess.DEVNULL)
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                         text=True, stderr=subprocess.DEVNULL).strip()
        diff = subprocess.check_output(["git", "diff", "HEAD", "--", str(relative_source.parent)],
                                       cwd=repo, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError, ValueError):
        commit, diff = None, b""
    return {"package": version("compresso-recsys"), "commit": commit,
            "source_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "script_sha256": digest_file(Path(__file__))}


def unsupported(dataset, mode):
    if mode == "official" and dataset != "dbbook":
        return "Only DBbook supplies the supported official split"
    timed = dataset != "goodbooks" and getattr(builder.DATASETS[dataset].cls, "has_timestamps", True)
    if mode in ("temporal", "leave_last_out") and not timed:
        return "No interaction timestamps"
    return None


def dataset_label(record):
    category = record.get("amazon_category")
    return f"{record['dataset']}[{category}]" if category else record["dataset"]


def dataset_jobs(args):
    """Each category has its own cache; never concatenate Amazon categories."""
    return [(dataset, category) for dataset in args.datasets
            for category in (args.amazon_categories if dataset == "amazon2023" else [None])]


def build_parameters(args, dataset, mode, overrides, category=None):
    """Resolve sweep defaults first, then preserve explicit builder overrides."""
    period = args.temporal_period_hours
    params = dict(dataset=dataset, data_dir=str(args.data_dir.resolve()), seed=args.seed,
                  split_mode=mode, eval_draws=1, min_entity_text_words=0,
                  annotation_source="none", show_progress=False, temporal_period_hours=period)
    if dataset == "amazon2023":
        if category is None:
            raise ValueError("An Amazon category is required for each worker")
        params["amazon_category"] = category
    params.update(overrides.get(dataset, {}))
    return params


def write_summary(output, records, *, write_records=True):
    def pct(value):
        return f"{value:.6%}" if value is not None else "—"

    def safe(value):
        return str(value).replace("|", "/").replace("\n", " ")

    def sparsity(stats, item_key):
        # Derive from counts so old results can be re-rendered without
        # loading checkpoints or changing their original provenance.
        return pct(matrix_sparsity(stats.get("nnz", 0), stats.get("n_rows", 0), stats.get(item_key, 0)))

    lines = ["# Dataset sweep", "", "For loaded interactions, sparsity is 1 - unique pairs / (users × items). "
             "Loaded data is after adapter metadata filtering, before feedback/support filtering. "
             "Counts are observations, not sums of ratings or listening counts.", "",
             "For matrices, observed-item sparsity is 1 - pairs / (rows × observed items); "
             "catalog sparsity uses rows × catalog columns instead. Observed items have at least one "
             "interaction in that matrix; catalog columns may be entirely empty. Both keep all rows, "
             "including empty histories and repeated evaluation users. Undefined sparsities are shown as —.", "",
             "## Loaded datasets", "",
             "| Dataset | Users | Items | Events | Unique pairs | Sparsity | Repeat events |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    seen = set()
    for record in records:
        label = dataset_label(record)
        if label in seen or "loaded_data" not in record:
            continue
        seen.add(label)
        data = record["loaded_data"]
        lines.append(f"| {label} | {data['n_users']} | {data['n_items']} | "
                     f"{data['n_events']} | {data['n_unique_pairs']} | {pct(data['sparsity'])} | "
                     f"{pct(data['repeat_event_fraction'])} |")
    lines.extend(["", "## Training splits", "", "Observed train items exclude zero-only cold/catalog columns. "
                  "Full preprocessing settings and pre-split counts are saved in JSON.", "",
                  "| Dataset | Split | Status | Train users | Train rows | Catalog columns | Observed train items | Train pairs | Observed-item sparsity | Catalog sparsity | Reason |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"])
    for record in records:
        train = record.get("stats", {}).get("train", {})
        lines.append(f"| {dataset_label(record)} | {record['split']} | {record['status']} | "
                     f"{train.get('n_users', '—')} | {train.get('n_rows', '—')} | "
                     f"{train.get('n_items', '—')} | {train.get('active_items', '—')} | "
                     f"{train.get('nnz', '—')} | {sparsity(train, 'active_items')} | {sparsity(train, 'n_items')} | "
                     f"{safe(record.get('error', record.get('reason', '')))} |")
    lines.extend(["", "## Validation/test stages", "", "Rows can repeat users when evaluation draws "
                  "are enabled. Source/target histories overlap across stages; do not sum them as a dataset total. "
                  "Candidate catalog size is before per-user seen-item exclusion, not the number of target items. "
                  "Cold target pairs are target interactions with items absent from training; 100% is expected "
                  "for item_split, whose users can overlap across stages.", "",
                  "| Dataset | Split | Stage | Users | Rows | Candidate catalog | Observed source items | Observed target items | Source pairs | Target pairs | Target observed-item sparsity | Target catalog sparsity | Cold target pairs |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for record in records:
        for phase in ("val", "test"):
            stage = record.get("stats", {}).get(phase + "_evaluation")
            if stage is None:
                continue
            target, source = stage["target"], stage["source"]
            lines.append(f"| {dataset_label(record)} | {record['split']} | {phase} | "
                         f"{target['n_users']} | {target['n_rows']} | {target['n_items']} | "
                         f"{source['active_items']} | {target['active_items']} | "
                         f"{source['nnz']} | {target['nnz']} | {sparsity(target, 'active_items')} | "
                         f"{sparsity(target, 'n_items')} | "
                         f"{pct(stage['cold_target_fraction'])} |")
    lines.extend(["", "## Baselines", "", "Fixed training split; no tuning or refitting. "
                  "Cold items have zero collaborative signal; ties are not a cold-start capability.", "",
                  "| Dataset | Split | Model | Phase | Status | Evaluated users | Evaluated rows | Metrics / reason |",
                  "|---|---|---|---|---|---:|---:|---|"])
    for record in records:
        for name, baseline in record.get("baselines", {}).items():
            for phase in ("val", "test"):
                entry = baseline.get(phase, baseline)
                detail = entry.get("error", entry.get("reason", ""))
                if "metrics" in entry:
                    detail = ", ".join(f"{key}={value:.5f}" for key, value in entry["metrics"].items()
                                       if "@" in key)
                detail = str(detail).replace("|", "/").replace("\n", " ")
                lines.append(f"| {dataset_label(record)} | {record['split']} | {name} | {phase} | "
                             f"{entry['status']} | {entry.get('evaluated_users', '—')} | "
                             f"{entry.get('evaluated_rows', '—')} | {detail} |")
    temporary = output / "summary.md.tmp"
    temporary.write_text("\n".join(lines) + "\n")
    os.replace(temporary, output / "summary.md")
    if not write_records:
        return
    temporary = output / "results.jsonl.tmp"
    with temporary.open("w") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(temporary, output / "results.jsonl")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=sorted(builder.DATASETS), default=sorted(builder.DATASETS))
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--baselines", nargs="*", choices=["popularity", "itemknn"], default=["popularity"])
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel dataset/category processes; splits stay sequential")
    parser.add_argument("--threads-per-worker", type=int, default=4, help="Numerical-library threads per process")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/dataset-sweep"))
    parser.add_argument("--amazon-categories", "--amazon-category", nargs="+", default=["Toys_and_Games"],
                        help="Amazon subsets to run independently (default: Toys_and_Games); never combined")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--neighbors", type=int, default=100)
    parser.add_argument("--knn-max-items", type=int, default=30_000)
    parser.add_argument("--knn-jobs", type=int, default=1)
    parser.add_argument("--max-eval-users", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--temporal-period-hours", type=float, default=None,
                        help="Override target-window length (default: 720h for Gowalla, 8136h otherwise)")
    parser.add_argument("--builder-overrides", type=Path,
                        help="JSON mapping dataset names to builder keyword overrides")
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs and verified checkpoints")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate summary.md from --output/results.jsonl without running any jobs")
    args = parser.parse_args(argv)
    if any(k < 1 for k in args.cutoffs) or min(args.neighbors, args.knn_max_items, args.batch_size,
                                             args.workers, args.threads_per_worker) < 1:
        parser.error("Cutoffs, neighbors, item limit and batch size must be positive")
    if args.max_eval_users is not None and args.max_eval_users < 1:
        parser.error("--max-eval-users must be positive")
    if args.temporal_period_hours is not None and (
        not np.isfinite(args.temporal_period_hours) or args.temporal_period_hours <= 0
    ):
        parser.error("--temporal-period-hours must be finite and positive")
    args.cutoffs = sorted(set(args.cutoffs))
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.splits)) != len(args.splits):
        parser.error("Dataset and split lists must not contain duplicates")
    try:
        args.amazon_categories = [cr.AmazonReviews2023.normalize_category(category)
                                  for category in args.amazon_categories]
    except ValueError as error:
        parser.error(str(error))
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", category) or category.lower() == "all"
           for category in args.amazon_categories):
        parser.error("Select explicit Amazon category names, not 'all', paths, or a combined dataset")
    if len(set(args.amazon_categories)) != len(args.amazon_categories):
        parser.error("Amazon category list must not contain duplicates (including aliases)")
    return args


def run_dataset(args, dataset, overrides, provenance, category=None):
    """One process owns a dataset/category cache and executes its splits in order."""
    torch.set_num_threads(args.threads_per_worker)
    pa.set_cpu_count(args.threads_per_worker)
    pa.set_io_thread_count(args.threads_per_worker)
    evaluation = {key: getattr(args, key) for key in (
        "baselines", "cutoffs", "neighbors", "knn_max_items", "knn_jobs",
        "max_eval_users", "batch_size", "seed", "threads_per_worker")}
    records = []
    loaded_stats, raw, ds = None, None, None
    for mode in args.splits:
        params = build_parameters(args, dataset, mode, overrides, category)
        signature = hashlib.sha256(json.dumps([params, evaluation, provenance], sort_keys=True).encode()).hexdigest()
        slug = f"{dataset}-{category}" if category else dataset
        folder = args.output / f"{slug}-{mode}-{signature[:12]}"
        result_path, checkpoint_path = folder / "result.json", folder / "checkpoint.zip"
        record = {"dataset": dataset, "amazon_category": category, "split": mode, "signature": signature,
                  "build_parameters": params, "evaluation_parameters": evaluation,
                  "provenance": provenance, "status": "pending"}
        label = dataset_label(record)
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if not args.resume or previous.get("signature") != signature:
                raise FileExistsError(f"Existing run {folder}; use --resume or a new output directory")
            if previous["status"] in ("complete", "unsupported"):
                records.append(previous)
                print(f"Reused {label}/{mode}", flush=True)
                continue
            record = previous
            record.pop("error", None)
        reason = unsupported(dataset, mode)
        if reason:
            record.update(status="unsupported", reason=reason)
        else:
            started = time.perf_counter()
            print(f"Running {label}/{mode}", flush=True)
            try:
                if raw is None:
                    resolved, spec = builder._resolve_args(builder._build_args(**params))
                    ds = builder._make_dataset(resolved, spec)
                    raw = ds.get_interactions()
                    if loaded_stats is None:
                        loaded_stats = frame_stats(raw)
                record["loaded_data"] = loaded_stats
                resolved, _ = builder._resolve_args(builder._build_args(**params))
                record["resolved_build_parameters"] = vars(resolved)
                rating_filter = f"rating>={resolved.min_value_to_keep}" if resolved.min_value_to_keep is not None else "no rating threshold"
                print(f"  Settings {label}/{mode}: {rating_filter}, "
                      f"user/item support={resolved.min_user_support}/{resolved.item_min_support}, "
                      f"min text words={resolved.min_entity_text_words}"
                      + (f", val/test users={resolved.val_users}/{resolved.test_users}" if mode == "user_split" else "")
                      + (f", window={resolved.temporal_period_hours:g}h" if mode == "temporal" else ""), flush=True)
                pre = raw[raw.source_split == "train"] if mode == "official" else raw
                if isinstance(ds, PublicDataset) and mode in ("user_split", "item_split"):
                    pre = pre.drop_duplicates(["user_id", "item_id"])
                pre = ds.preprocess_interactions_for_recsys(
                    pre, min_value_to_keep=resolved.min_value_to_keep,
                    user_min_support=1 if mode == "temporal" else resolved.min_user_support,
                    item_min_support=1 if mode == "temporal" else resolved.item_min_support,
                    set_all_values_to=resolved.set_all_values_to,
                )
                record["pre_split_data"] = frame_stats(pre)
                del pre
                # The builder reloads cached input. Do not retain a second
                # full source dataset alongside building/fitting matrices.
                raw, ds = None, None
                if checkpoint_path.exists():
                    if record.get("checkpoint_sha256") != digest_file(checkpoint_path):
                        raise ValueError("Unverified existing checkpoint; use a new output directory")
                else:
                    atomic_json(result_path, record)
                    built = time.perf_counter()
                    cr.build_recsys_checkpoint(**params, checkpoint_path=str(checkpoint_path))
                    record["build_seconds"] = time.perf_counter() - built
                    record["checkpoint_sha256"] = digest_file(checkpoint_path)
                    atomic_json(result_path, record)
                record["checkpoint_bytes"] = checkpoint_path.stat().st_size
                with cr.read_checkpoint(checkpoint_path) as root:
                    split = cr.load_recsys_split(root)
                    record["manifest"] = cr.load_manifest(root)
                    record["stats"] = checkpoint_stats(split, root)
                    record["baselines"] = evaluate_baselines(split, args)
                del split
                record["status"] = ("failed" if any(v["status"] == "failed" for v in record["baselines"].values())
                                    else "complete")
            except Exception as error:
                record.update(status="failed", error=f"{type(error).__name__}: {error}")
                print(f"  Failed {label}/{mode}: {record['error']}", file=sys.stderr, flush=True)
            record["seconds_this_attempt"] = time.perf_counter() - started
        atomic_json(result_path, record)
        records.append(record)
    del raw, ds
    return records


def main(argv=None):
    args = parse_args(argv)
    if args.report_only:
        records = [json.loads(line) for line in (args.output / "results.jsonl").read_text().splitlines()
                   if line.strip()]
        write_summary(args.output, records, write_records=False)
        print(f"Regenerated {args.output / 'summary.md'} from saved statistics; no jobs were run.")
        return 0
    args.output.mkdir(parents=True, exist_ok=True)
    overrides = json.loads(args.builder_overrides.read_text()) if args.builder_overrides else {}
    forbidden = {"dataset", "split_mode", "checkpoint_path", "data_dir", "multimodal_features"}
    if not isinstance(overrides, dict):
        raise ValueError("Builder overrides must be an object keyed by dataset")
    for dataset, settings in overrides.items():
        if dataset not in builder.DATASETS or not isinstance(settings, dict) or forbidden & settings.keys():
            raise ValueError(f"Invalid builder overrides for {dataset}")
        if "amazon_category" in settings:
            raise ValueError("Select Amazon subsets with --amazon-categories, not builder overrides")
    provenance = code_version()
    # Spawned children import NumPy/BLAS only after inheriting these limits.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = str(args.threads_per_worker)
    jobs = dataset_jobs(args)
    job_order = {job: index for index, job in enumerate(jobs)}
    workers = min(args.workers, len(jobs))
    print(f"Starting {workers} dataset/category workers for {len(jobs)} jobs, "
          f"{args.threads_per_worker} numerical threads each", flush=True)
    records = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_dataset, args, dataset, overrides, provenance, category): (dataset, category)
                   for dataset, category in jobs}
        for future in as_completed(futures):
            dataset, category = futures[future]
            try:
                records.extend(future.result())
            except FileExistsError:
                raise
            except Exception as error:
                records.append({"dataset": dataset, "amazon_category": category,
                                "split": "worker", "status": "failed",
                                "error": f"{type(error).__name__}: {error}"})
            records.sort(key=lambda row: (job_order[(row["dataset"], row.get("amazon_category"))], row["split"]))
            write_summary(args.output, records)
    print(args.output / "summary.md")
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
