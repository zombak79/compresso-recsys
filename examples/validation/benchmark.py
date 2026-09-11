"""Build paper-inspired CF splits and validate models; see dataset-validation.rst."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from compresso_recsys import build_recsys_checkpoint, load_recsys_split, read_checkpoint
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import (
    EASE, EASEConfig, MultDAEConfig, MultDAETrainer, MultVAEConfig, MultVAETrainer,
)

# held-out users per validation/test set, minimum user/item support, rating cutoff
PROTOCOLS = {
    "ml20m": (10_000, 5, 1, 4),
    "netflix": (40_000, 5, 1, 4),
    "taste-profile": (50_000, 20, 200, None),
    "steam": (10_000, 5, 1, None),
}


def build(dataset, checkpoint, data_dir):
    heldout, users, items, threshold = PROTOCOLS[dataset]
    return build_recsys_checkpoint(
        dataset=dataset, data_dir=str(data_dir), checkpoint_path=str(checkpoint),
        seed=98765, val_users=heldout, test_users=heldout,
        min_user_support=users, item_min_support=items,
        min_value_to_keep=threshold, set_all_values_to=1.0,
        split_mode="user_split", eval_draws=1, eval_holdout_frac=0.2,
        min_entity_text_words=0, annotation_source="none",
    )


def score(model, split, phase):
    return dict(evaluate_recommender(
        model, source=split[f"{phase}_source_matrix"],
        targets=split[f"{phase}_target_matrix"],
        metrics=[CalibratedRecall([20, 50]), NDCG(100)],
        batch_size=128, collect_per_user=False,
    ))


def train(split, model_name, *, epochs, l2_values, device, seed):
    """Select on validation only; evaluate the winning model on test once."""
    for phase in ("val", "test"):
        if not np.array_equal(split["train_item_ids"], split[f"{phase}_item_ids"]):
            raise ValueError("This recipe requires a shared warm-item vocabulary")
        if split[f"{phase}_target_matrix"].nnz == 0:
            raise ValueError(f"{phase} has no targets")
    trials, best_model, best_config, best_score = [], None, None, -float("inf")
    budgets = l2_values if model_name == "ease" else epochs
    for budget in budgets:
        if model_name == "ease":
            config = EASEConfig(l2=budget, dtype="float64")
            model = EASE(config)
        else:
            kwargs = dict(epochs=budget, device=device, seed=seed,
                          preload_training_data=False, show_progress=False)
            if model_name == "multvae":
                config = MultVAEConfig(**kwargs)
                model = MultVAETrainer(config)
            elif model_name == "multdae":
                config = MultDAEConfig(**kwargs)
                model = MultDAETrainer(config)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        model.fit(split["x_train"], item_ids=split["train_item_ids"])
        validation = score(model, split, "val")
        trials.append({"config": asdict(config), "validation": validation})
        if validation["ndcg@100"] > best_score:
            best_model, best_config = model, asdict(config)
            best_score = validation["ndcg@100"]
    if best_model is None:
        raise ValueError("Supply at least one training configuration")
    return {"model": model_name, "selection_metric": "ndcg@100",
            "trials": trials, "selected_config": best_config,
            "test": score(best_model, split, "test"),
            "train_shape": list(split["x_train"].shape),
            "train_nnz": int(split["x_train"].nnz),
            "status": "adapted protocol; not a verified paper reproduction"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--dataset", choices=PROTOCOLS, required=True)
    builder.add_argument("--data-dir", type=Path, default=Path("data"))
    builder.add_argument("--checkpoint", type=Path, required=True)
    trainer = commands.add_parser("train")
    trainer.add_argument("--checkpoint", type=Path, required=True)
    trainer.add_argument("--model", choices=["ease", "multdae", "multvae"], required=True)
    trainer.add_argument("--epochs", type=int, nargs="+", default=[20, 50, 100])
    trainer.add_argument("--l2", type=float, nargs="+", default=[200, 500, 1000])
    trainer.add_argument("--device", default="cpu")
    trainer.add_argument("--seed", type=int, default=98765)
    trainer.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        if args.checkpoint.exists():
            parser.error("Checkpoint already exists; choose a new path")
        build(args.dataset, args.checkpoint, args.data_dir)
        return
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    with read_checkpoint(args.checkpoint) as root:
        result = train(load_recsys_split(root), args.model, epochs=args.epochs,
                       l2_values=args.l2, device=args.device, seed=args.seed)
    with args.checkpoint.open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        result["checkpoint_sha256"] = digest.hexdigest()
    result["checkpoint"] = str(args.checkpoint.resolve())
    result["package_version"] = version("compresso-recsys")
    result["seed"] = args.seed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(args.output)


if __name__ == "__main__":
    main()
