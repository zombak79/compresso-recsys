from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso_recsys.checkpoint import ELSA_DIR, load_recsys_split, save_json, update_checkpoint, update_stage_manifest
from compresso_recsys.models.elsa import fit_elsa
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout


def eval_three_metrics(item_embs, source_indices, target_indices, eval_batch_size):
    out = {}
    for k in (20, 50, 100):
        m = evaluate_item_embeddings_with_holdout(
            item_embeddings=item_embs,
            source_indices=source_indices,
            target_indices=target_indices,
            k=k,
            score_batch_size=eval_batch_size,
        )
        out.update({kk: vv for kk, vv in m.items() if kk != "n_eval_users"})
    return {"recall@20": out.get("recall@20", 0.0), "recall@50": out.get("recall@50", 0.0), "ndcg@100": out.get("ndcg@100", 0.0)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--elsa_dim", type=int, default=2048)
    p.add_argument("--elsa_epochs", type=int, default=10)
    p.add_argument("--elsa_batch_size", type=int, default=1024)
    p.add_argument("--elsa_lr", type=float, default=0.1)
    p.add_argument("--elsa_weight_decay", type=float, default=0.0)
    p.add_argument("--elsa_beta1", type=float, default=0.9)
    p.add_argument("--elsa_beta2", type=float, default=0.999)
    p.add_argument("--elsa_eps", type=float, default=1e-8)
    p.add_argument("--elsa_momentum_decay", type=float, default=0.004)
    p.add_argument("--elsa_decoupled_weight_decay", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--elsa_grad_clip_norm", type=float, default=None)
    p.add_argument("--elsa_grad_accum_steps", type=int, default=1)
    p.add_argument("--elsa_use_ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--elsa_ema_momentum", type=float, default=0.99)
    p.add_argument("--elsa_ema_overwrite_frequency", type=int, default=150)
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--device", type=str, default="mps")
    return p.parse_args()


def resolve_device(requested: str) -> str:
    req = requested.lower()
    if req == "cpu":
        return "cpu"
    if req == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return req


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    with update_checkpoint(args.checkpoint_path) as root:
        split = load_recsys_split(root)
        x_train = split["x_train"]
        val_source_indices = split["val_source_indices"]
        val_target_indices = split["val_target_indices"]
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]

        def _elsa_val_callback(model):
            return eval_three_metrics(model.export_item_embeddings(), val_source_indices, val_target_indices, args.eval_batch_size)

        elsa = fit_elsa(
            x_train,
            n_factors=args.elsa_dim,
            epochs=args.elsa_epochs,
            batch_size=args.elsa_batch_size,
            lr=args.elsa_lr,
            weight_decay=args.elsa_weight_decay,
            beta1=args.elsa_beta1,
            beta2=args.elsa_beta2,
            eps=args.elsa_eps,
            momentum_decay=args.elsa_momentum_decay,
            decoupled_weight_decay=args.elsa_decoupled_weight_decay,
            grad_clip_norm=args.elsa_grad_clip_norm,
            grad_accum_steps=args.elsa_grad_accum_steps,
            use_ema=args.elsa_use_ema,
            ema_momentum=args.elsa_ema_momentum,
            ema_overwrite_frequency=args.elsa_ema_overwrite_frequency,
            device=device,
            val_callback=_elsa_val_callback,
        )
        item_embs = elsa.export_item_embeddings()
        metrics = eval_three_metrics(item_embs, test_source_indices, test_target_indices, args.eval_batch_size)
        print("ELSA checkpoint metrics:", metrics)

        stage_dir = root / ELSA_DIR
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / "item_embeddings.npy", item_embs.astype(np.float32))
        torch.save(
            {
                "model_state_dict": elsa.state_dict(),
                "config": vars(args),
                "n_items": int(x_train.shape[1]),
                "n_factors": int(args.elsa_dim),
            },
            stage_dir / "model.pt",
        )
        save_json(root, f"{ELSA_DIR}/metrics.json", {"test_metrics": metrics})
        update_stage_manifest(root, "elsa", {"n_factors": args.elsa_dim, "metrics": metrics})
    print(f"Saved ELSA stage to checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
