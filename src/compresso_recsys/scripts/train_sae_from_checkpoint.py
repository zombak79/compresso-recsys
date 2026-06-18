from __future__ import annotations

import argparse
import random

import numpy as np
import torch

from compresso_recsys.checkpoint import ELSA_DIR, SAE_DIR, SBERT_DIR, SBERT_SAE_DIR, load_json, load_recsys_split, save_json, update_checkpoint, update_stage_manifest
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout
from compresso_recsys.models.sae import fit_sae_on_embeddings
from compresso.io import save_srp_tensor
from compresso.params.srp import SRPTensor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding_stage", type=str, default="elsa", choices=["elsa", "sbert"])

    p.add_argument("--sae_hidden_dim", type=int, default=4096)
    p.add_argument("--sae_k", type=int, default=128)
    p.add_argument("--sae_ste_alpha", type=float, default=0.0)
    p.add_argument("--sae_score_mode", type=str, default="abs", choices=["abs", "raw", "relu"])
    p.add_argument("--sae_post_norm_l1", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sae_loss", type=str, default="mse", choices=["mse", "cosine"])
    p.add_argument("--sae_epochs", type=int, default=10)
    p.add_argument("--sae_batch_size", type=int, default=1024)
    p.add_argument("--sae_lr", type=float, default=1e-3)
    p.add_argument("--debug_l1", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--eval_batch_size", type=int, default=1024)
    p.add_argument("--eval_debug", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--eval_debug_users", type=int, default=10)
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


def select_metrics(metrics: dict) -> dict:
    return {
        "recall@20": metrics.get("recall@20", 0.0),
        "recall@50": metrics.get("recall@50", 0.0),
        "ndcg@100": metrics.get("ndcg@100", 0.0),
    }


def percent_drop(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return ((base - new) / base) * 100.0


def stage_dirs(embedding_stage: str) -> tuple[str, str, str]:
    if embedding_stage == "elsa":
        return ELSA_DIR, SAE_DIR, "Original embedding"
    if embedding_stage == "sbert":
        return SBERT_DIR, SBERT_SAE_DIR, "Original SBERT embedding"
    raise ValueError(f"Unknown embedding stage: {embedding_stage}")


def eval_multi_k(embs, source_indices, target_indices, *, eval_batch_size, eval_debug, eval_debug_users):
    out = {}
    debug_rows = None
    for k in (10, 20, 50, 100):
        m = evaluate_item_embeddings_with_holdout(
            item_embeddings=embs,
            source_indices=source_indices,
            target_indices=target_indices,
            k=k,
            score_batch_size=eval_batch_size,
            debug=eval_debug and k == 100,
            debug_users=eval_debug_users,
        )
        out.update({kk: vv for kk, vv in m.items() if kk != "n_eval_users"})
        if "debug" in m:
            debug_rows = m["debug"]
    if debug_rows is not None:
        out["debug_ndcg@100"] = debug_rows
    return out


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    with update_checkpoint(args.checkpoint_path) as root:
        split = load_recsys_split(root)
        base_dir, output_dir, base_label = stage_dirs(args.embedding_stage)
        embeddings_path = root / base_dir / "item_embeddings.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found in checkpoint: {base_dir}/item_embeddings.npy")
        item_embs = np.load(embeddings_path).astype(np.float32)
        val_source_indices = split["val_source_indices"]
        val_target_indices = split["val_target_indices"]
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]

        base_metrics_path = root / base_dir / "metrics.json"
        if base_metrics_path.exists():
            base_all = select_metrics(load_json(root, f"{base_dir}/metrics.json").get("test_metrics", {}))
            print(f"{base_label} metrics:", base_all, "(from checkpoint)")
        else:
            base_all = eval_multi_k(
                item_embs,
                test_source_indices,
                test_target_indices,
                eval_batch_size=args.eval_batch_size,
                eval_debug=args.eval_debug,
                eval_debug_users=args.eval_debug_users,
            )
            print(f"{base_label} metrics:", select_metrics(base_all))
            if args.eval_debug and "debug_ndcg@100" in base_all:
                print(f"{base_label} ndcg@100 debug rows:", base_all["debug_ndcg@100"])

        def _sae_val_callback(model):
            with torch.no_grad():
                x_val = torch.from_numpy(item_embs.astype(np.float32)).to(device)
                _, codes_val, _ = model(x_val)
                embs_val = codes_val.detach().cpu().numpy().astype(np.float32)
            m20 = evaluate_item_embeddings_with_holdout(
                item_embeddings=embs_val,
                source_indices=val_source_indices,
                target_indices=val_target_indices,
                k=20,
                score_batch_size=args.eval_batch_size,
            )
            m50 = evaluate_item_embeddings_with_holdout(
                item_embeddings=embs_val,
                source_indices=val_source_indices,
                target_indices=val_target_indices,
                k=50,
                score_batch_size=args.eval_batch_size,
            )
            m100 = evaluate_item_embeddings_with_holdout(
                item_embeddings=embs_val,
                source_indices=val_source_indices,
                target_indices=val_target_indices,
                k=100,
                score_batch_size=args.eval_batch_size,
            )
            return {
                "recall@20": m20.get("recall@20", 0.0),
                "recall@50": m50.get("recall@50", 0.0),
                "ndcg@100": m100.get("ndcg@100", 0.0),
            }

        sae = fit_sae_on_embeddings(
            item_embs,
            hidden_dim=args.sae_hidden_dim,
            k=args.sae_k,
            sparsify_score_mode=args.sae_score_mode,
            sparsify_ste_alpha=args.sae_ste_alpha,
            post_norm_p=1.0 if args.sae_post_norm_l1 else None,
            epochs=args.sae_epochs,
            batch_size=args.sae_batch_size,
            lr=args.sae_lr,
            device=device,
            loss_type=args.sae_loss,
            debug_l1=args.debug_l1,
            val_callback=_sae_val_callback,
        )

        with torch.no_grad():
            x = torch.from_numpy(item_embs.astype(np.float32)).to(device)
            _, codes, stats = sae(x)
            sae_embs = codes.detach().cpu().numpy().astype(np.float32)
            dead = int(stats["dead_features"].item()) if hasattr(stats["dead_features"], "item") else int(stats["dead_features"])
            dead_frac = dead / float(args.sae_hidden_dim)
            print(f"SAE dead neurons: {dead}/{args.sae_hidden_dim} ({dead_frac:.2%})")

        sae_all = eval_multi_k(
            sae_embs,
            test_source_indices,
            test_target_indices,
            eval_batch_size=args.eval_batch_size,
            eval_debug=args.eval_debug,
            eval_debug_users=args.eval_debug_users,
        )
        base_metrics = select_metrics(base_all)
        sae_metrics = select_metrics(sae_all)
        print("SAE embedding metrics:", sae_metrics)
        if args.eval_debug and "debug_ndcg@100" in sae_all:
            print("SAE ndcg@100 debug rows:", sae_all["debug_ndcg@100"])
        print(
            f"Perf drop vs {args.embedding_stage}: "
            f"recall@20={percent_drop(base_metrics['recall@20'], sae_metrics['recall@20']):.2f}% "
            f"recall@50={percent_drop(base_metrics['recall@50'], sae_metrics['recall@50']):.2f}% "
            f"ndcg@100={percent_drop(base_metrics['ndcg@100'], sae_metrics['ndcg@100']):.2f}%"
        )

        stage_dir = root / output_dir
        stage_dir.mkdir(parents=True, exist_ok=True)
        model_path = stage_dir / "model.pt"
        sparse_path = stage_dir / "sparse_embeddings.srp.pt"
        torch.save(
            {
                "model_state_dict": sae.state_dict(),
                "config": vars(args),
                "embedding_stage": args.embedding_stage,
                "base_stage": base_dir,
                "output_stage": output_dir,
                "hidden_dim": args.sae_hidden_dim,
                "k": args.sae_k,
                "score_mode": args.sae_score_mode,
                "ste_alpha": args.sae_ste_alpha,
                "post_norm_l1": bool(args.sae_post_norm_l1),
            },
            model_path,
        )
        srp_codes = SRPTensor.from_dense(
            torch.from_numpy(sae_embs),
            k=args.sae_k,
            score_mode=args.sae_score_mode,
        )
        save_srp_tensor(sparse_path, srp_codes)
        save_json(
            root,
            f"{output_dir}/metrics.json",
            {
                "original_metrics": base_metrics,
                "sae_metrics": sae_metrics,
                "perf_drop_percent": {
                    "recall@20": percent_drop(base_metrics["recall@20"], sae_metrics["recall@20"]),
                    "recall@50": percent_drop(base_metrics["recall@50"], sae_metrics["recall@50"]),
                    "ndcg@100": percent_drop(base_metrics["ndcg@100"], sae_metrics["ndcg@100"]),
                },
                "dead_neurons": {"count": dead, "total": args.sae_hidden_dim, "fraction": dead_frac},
            },
        )
        update_stage_manifest(
            root,
            output_dir,
            {
                "embedding_stage": args.embedding_stage,
                "base_stage": base_dir,
                "hidden_dim": args.sae_hidden_dim,
                "k": args.sae_k,
                "score_mode": args.sae_score_mode,
                "ste_alpha": args.sae_ste_alpha,
                "post_norm_l1": bool(args.sae_post_norm_l1),
                "metrics": sae_metrics,
                "dead_neurons": {"count": dead, "total": args.sae_hidden_dim, "fraction": dead_frac},
            },
        )
    print(f"Saved {output_dir} stage to checkpoint: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
