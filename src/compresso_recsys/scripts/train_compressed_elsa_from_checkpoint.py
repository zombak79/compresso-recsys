from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso_recsys.checkpoint import COMPRESSED_ELSA_DIR, load_recsys_split, save_json, update_checkpoint, update_stage_manifest
from compresso_recsys.models.elsa import fit_compressed_elsa
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout
from compresso.io import save_srp_tensor


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
    p.add_argument("--sparse_k_target", type=int, default=128)
    p.add_argument("--sparse_num_stages", type=int, default=10)
    p.add_argument("--sparse_stability_window", type=int, default=5)
    p.add_argument("--sparse_change_threshold", type=float, default=0.01)
    p.add_argument("--sparse_mask_update_interval", type=int, default=10)
    p.add_argument("--sparse_score_mode", type=str, default="abs", choices=["abs", "raw", "relu"])
    p.add_argument("--sparse_ste_alpha", type=float, default=1.0)
    p.add_argument("--sparse_post_norm_l1", action=argparse.BooleanOptionalAction, default=False)
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

        def _val_callback(model):
            return eval_three_metrics(model.export_item_embeddings(), val_source_indices, val_target_indices, args.eval_batch_size)

        model = fit_compressed_elsa(
            x_train,
            n_factors=args.elsa_dim,
            k_target=args.sparse_k_target,
            num_stages=args.sparse_num_stages,
            stability_window=args.sparse_stability_window,
            change_threshold=args.sparse_change_threshold,
            score_mode=args.sparse_score_mode,
            ste_alpha=args.sparse_ste_alpha,
            post_norm_l1=args.sparse_post_norm_l1,
            mask_update_interval=args.sparse_mask_update_interval,
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
            device=device,
            val_callback=_val_callback,
        )
        item_embs = model.export_item_embeddings()
        metrics = eval_three_metrics(item_embs, test_source_indices, test_target_indices, args.eval_batch_size)
        print("Compressed ELSA checkpoint metrics:", metrics)

        stage_dir = root / COMPRESSED_ELSA_DIR
        stage_dir.mkdir(parents=True, exist_ok=True)
        np.save(stage_dir / "item_embeddings.npy", item_embs.astype(np.float32))
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": vars(args),
                "n_items": int(x_train.shape[1]),
                "n_factors": int(args.elsa_dim),
                "k_target": int(args.sparse_k_target),
            },
            stage_dir / "model.pt",
        )
        save_srp_tensor(stage_dir / "sparse_embeddings.srp.pt", model.A.srp())
        save_json(root, f"{COMPRESSED_ELSA_DIR}/metrics.json", {"test_metrics": metrics})
        update_stage_manifest(
            root,
            "compressed_elsa",
            {"n_factors": args.elsa_dim, "k_target": args.sparse_k_target, "metrics": metrics},
        )
    print(f"Saved CompressedELSA stage to checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
