from __future__ import annotations

import argparse

import numpy as np
import torch

from compresso_recsys.checkpoint import (
    COMPRESSED_ELSA_DIR,
    ELSA_DIR,
    SAE_DIR,
    SBERT_DIR,
    SBERT_SAE_DIR,
    load_json,
    load_recsys_split,
    save_json,
    update_checkpoint,
)
from compresso_recsys.retrieval import evaluate_item_embeddings_with_holdout
from compresso.io import load_srp_tensor
from compresso.nn import TopKSAE


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--eval_batch_size", type=int, default=1024)
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


def saved_metrics(root, relpath: str, key: str) -> dict | None:
    path = root / relpath
    if not path.exists():
        return None
    metrics = load_json(root, relpath).get(key)
    return select_metrics(metrics) if metrics is not None else None


def merge_metrics(root, relpath: str, **values) -> None:
    path = root / relpath
    data = load_json(root, relpath) if path.exists() else {}
    data.update(values)
    save_json(root, relpath, data)


def percent_drop(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return ((base - new) / base) * 100.0


def bytes_to_mb(n_bytes: int) -> float:
    return float(n_bytes) / (1024.0 * 1024.0)


def srp_size_mb(srp) -> float:
    return bytes_to_mb(
        int(srp.cols.numel() * srp.cols.element_size())
        + int(srp.vals.numel() * srp.vals.element_size())
    )


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
    return select_metrics(out)


def _calibrated_recall(target_sets: list[set[int]], pred_ranked: list[np.ndarray], k: int) -> float:
    vals = []
    for tset, pred in zip(target_sets, pred_ranked):
        if not tset:
            continue
        hits = sum(1 for i in pred[:k] if int(i) in tset)
        denom = min(k, len(tset))
        vals.append(hits / denom if denom > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _ndcg(target_sets: list[set[int]], pred_ranked: list[np.ndarray], k: int) -> float:
    vals = []
    for tset, pred in zip(target_sets, pred_ranked):
        if not tset:
            continue
        dcg = 0.0
        for rank, item_idx in enumerate(pred[:k], start=1):
            if int(item_idx) in tset:
                dcg += 1.0 / np.log2(rank + 1)
        ideal_len = min(k, len(tset))
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, ideal_len + 1))
        vals.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(vals)) if vals else 0.0


def _compute_topk_kernel_trick(
    z: torch.Tensor,
    decoder_map: torch.Tensor,
    source_indices: list[np.ndarray],
    k: int,
    *,
    batch_size: int,
) -> list[np.ndarray]:
    device = z.device
    n_items = z.shape[0]
    k_eff = min(k, n_items)
    k_mat = decoder_map @ decoder_map.t()
    norms_sq = (z @ k_mat * z).sum(dim=1).clamp_min(1e-12)
    z_scaled = z / torch.sqrt(norms_sq).unsqueeze(1)
    preds: list[np.ndarray] = []
    for start in range(0, len(source_indices), batch_size):
        batch = source_indices[start : start + batch_size]
        lengths = [len(x) for x in batch]
        flat_src = np.concatenate(batch, axis=0)
        flat_src_t = torch.from_numpy(flat_src).long().to(device)
        owners = torch.repeat_interleave(
            torch.arange(len(batch), device=device, dtype=torch.long),
            torch.tensor(lengths, device=device, dtype=torch.long),
        )
        x = torch.zeros((len(batch), n_items), device=device, dtype=z.dtype)
        x[owners, flat_src_t] = 1.0
        scores = torch.relu(((x @ z_scaled) @ k_mat) @ z_scaled.t())
        scores[owners, flat_src_t] = -torch.inf
        topk_idx = torch.topk(scores, k_eff, dim=1, largest=True, sorted=True).indices
        preds.extend([row.detach().cpu().numpy() for row in topk_idx])
    return preds


def eval_kernel_trick(
    *,
    z_codes: np.ndarray,
    decoder_map: np.ndarray,
    source_indices: list[np.ndarray],
    target_indices: list[np.ndarray],
    k: int,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    z = torch.from_numpy(z_codes.astype(np.float32)).to(device)
    w = torch.from_numpy(decoder_map.astype(np.float32)).to(device)
    pred_ranked = _compute_topk_kernel_trick(z, w, source_indices, k=k, batch_size=batch_size)
    target_sets = [set(x.tolist()) for x in target_indices]
    return {f"recall@{k}": _calibrated_recall(target_sets, pred_ranked, k), f"ndcg@{k}": _ndcg(target_sets, pred_ranked, k)}


def print_drop(base_label: str, variant_label: str, base: dict, metrics: dict) -> None:
    print(
        f"Perf drop vs {base_label} ({variant_label}): "
        f"recall@20={percent_drop(base['recall@20'], metrics['recall@20']):.2f}% "
        f"recall@50={percent_drop(base['recall@50'], metrics['recall@50']):.2f}% "
        f"ndcg@100={percent_drop(base['ndcg@100'], metrics['ndcg@100']):.2f}%"
    )


def load_or_eval_dense_stage(root, *, stage_dir: str, label: str, source_indices, target_indices, eval_batch_size):
    emb_path = root / stage_dir / "item_embeddings.npy"
    if not emb_path.exists():
        return None, None
    embs = np.load(emb_path).astype(np.float32)
    metrics = saved_metrics(root, f"{stage_dir}/metrics.json", "test_metrics")
    if metrics is None:
        metrics = eval_three_metrics(embs, source_indices, target_indices, eval_batch_size)
        merge_metrics(root, f"{stage_dir}/metrics.json", test_metrics=metrics)
        print(f"{label} metrics:", metrics)
    else:
        print(f"{label} metrics:", metrics, "(from checkpoint)")
    print(f"Inference size (MB) {label} dense: {bytes_to_mb(int(embs.nbytes)):.2f}")
    return embs, metrics


def eval_sae_stage(
    root,
    *,
    base_embs: np.ndarray,
    base_metrics: dict,
    base_label: str,
    sae_dir: str,
    variant_label: str,
    source_indices,
    target_indices,
    eval_batch_size: int,
    device: str,
) -> None:
    sae_model_path = root / sae_dir / "model.pt"
    sae_sparse_path = root / sae_dir / "sparse_embeddings.srp.pt"
    if not (sae_model_path.exists() and sae_sparse_path.exists()):
        return

    srp_codes = load_srp_tensor(sae_sparse_path)
    z_codes = None
    srp_metrics = saved_metrics(root, f"{sae_dir}/metrics.json", "sae_metrics")
    if srp_metrics is None:
        z_codes = srp_codes.to_dense().detach().cpu().numpy().astype(np.float32)
        srp_metrics = eval_three_metrics(z_codes, source_indices, target_indices, eval_batch_size)
        merge_metrics(root, f"{sae_dir}/metrics.json", sae_metrics=srp_metrics)
        print(f"{variant_label} sparse code metrics:", srp_metrics)
    else:
        print(f"{variant_label} sparse code metrics:", srp_metrics, "(from checkpoint)")

    kernel_metrics = saved_metrics(root, f"{sae_dir}/metrics.json", "kernel_metrics")
    blob = torch.load(sae_model_path, map_location="cpu", weights_only=False)
    cfg = blob.get("config", {})
    hidden_dim = int(blob.get("hidden_dim", cfg.get("sae_hidden_dim", srp_codes.shape[1])))
    k = int(blob.get("k", cfg.get("sae_k", 128)))
    score_mode = str(blob.get("score_mode", cfg.get("sae_score_mode", "abs")))
    ste_alpha = float(blob.get("ste_alpha", cfg.get("sae_ste_alpha", 0.0)))
    decoder_bias = "decoder.bias" in blob["model_state_dict"]
    sae = TopKSAE(
        input_dim=base_embs.shape[1],
        hidden_dim=hidden_dim,
        k=k,
        decoder_bias=decoder_bias,
        sparsify_score_mode=score_mode,
        sparsify_ste_alpha=ste_alpha,
    )
    sae.load_state_dict(blob["model_state_dict"], strict=True)
    decoder_map = (
        sae.encoder.weight.detach().cpu().numpy().astype(np.float32)
        if sae.tied
        else sae.decoder.weight.detach().cpu().numpy().astype(np.float32).T
    )
    kernel = decoder_map @ decoder_map.T
    if kernel_metrics is None:
        if z_codes is None:
            z_codes = srp_codes.to_dense().detach().cpu().numpy().astype(np.float32)
        k20 = eval_kernel_trick(
            z_codes=z_codes,
            decoder_map=decoder_map,
            source_indices=source_indices,
            target_indices=target_indices,
            k=20,
            batch_size=eval_batch_size,
            device=device,
        )
        k50 = eval_kernel_trick(
            z_codes=z_codes,
            decoder_map=decoder_map,
            source_indices=source_indices,
            target_indices=target_indices,
            k=50,
            batch_size=eval_batch_size,
            device=device,
        )
        k100 = eval_kernel_trick(
            z_codes=z_codes,
            decoder_map=decoder_map,
            source_indices=source_indices,
            target_indices=target_indices,
            k=100,
            batch_size=eval_batch_size,
            device=device,
        )
        kernel_metrics = {"recall@20": k20["recall@20"], "recall@50": k50["recall@50"], "ndcg@100": k100["ndcg@100"]}
        merge_metrics(root, f"{sae_dir}/metrics.json", kernel_metrics=kernel_metrics)
        print(f"{variant_label} + decoder kernel-trick metrics:", kernel_metrics)
    else:
        print(f"{variant_label} + decoder kernel-trick metrics:", kernel_metrics, "(from checkpoint)")
    print_drop(base_label, f"{variant_label} sparse", base_metrics, srp_metrics)
    print_drop(base_label, f"{variant_label} kernel trick", base_metrics, kernel_metrics)
    print(f"Inference size (MB) {variant_label} SRP sparse: {srp_size_mb(srp_codes):.2f}")
    print(f"Inference size (MB) {variant_label} kernel_K: {bytes_to_mb(int(kernel.nbytes)):.2f}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    with update_checkpoint(args.checkpoint_path) as root:
        split = load_recsys_split(root)
        test_source_indices = split["test_source_indices"]
        test_target_indices = split["test_target_indices"]

        elsa_embs, elsa_metrics = load_or_eval_dense_stage(
            root,
            stage_dir=ELSA_DIR,
            label="ELSA",
            source_indices=test_source_indices,
            target_indices=test_target_indices,
            eval_batch_size=args.eval_batch_size,
        )
        if elsa_embs is not None and elsa_metrics is not None:
            eval_sae_stage(
                root,
                base_embs=elsa_embs,
                base_metrics=elsa_metrics,
                base_label="ELSA",
                sae_dir=SAE_DIR,
                variant_label="SAE",
                source_indices=test_source_indices,
                target_indices=test_target_indices,
                eval_batch_size=args.eval_batch_size,
                device=device,
            )

            compressed_path = root / COMPRESSED_ELSA_DIR / "sparse_embeddings.srp.pt"
            if compressed_path.exists():
                compressed_srp = load_srp_tensor(compressed_path)
                compressed_metrics = saved_metrics(root, f"{COMPRESSED_ELSA_DIR}/metrics.json", "test_metrics")
                if compressed_metrics is None:
                    compressed_dense = compressed_srp.to_dense().detach().cpu().numpy().astype(np.float32)
                    compressed_metrics = eval_three_metrics(compressed_dense, test_source_indices, test_target_indices, args.eval_batch_size)
                    merge_metrics(root, f"{COMPRESSED_ELSA_DIR}/metrics.json", test_metrics=compressed_metrics)
                    print("CompressedELSA SRP metrics:", compressed_metrics)
                else:
                    print("CompressedELSA SRP metrics:", compressed_metrics, "(from checkpoint)")
                print_drop("ELSA", "CompressedELSA SRP", elsa_metrics, compressed_metrics)
                print(f"Inference size (MB) CompressedELSA SRP: {srp_size_mb(compressed_srp):.2f}")

        sbert_embs, sbert_metrics = load_or_eval_dense_stage(
            root,
            stage_dir=SBERT_DIR,
            label="SBERT",
            source_indices=test_source_indices,
            target_indices=test_target_indices,
            eval_batch_size=args.eval_batch_size,
        )
        if sbert_embs is not None and sbert_metrics is not None:
            eval_sae_stage(
                root,
                base_embs=sbert_embs,
                base_metrics=sbert_metrics,
                base_label="SBERT",
                sae_dir=SBERT_SAE_DIR,
                variant_label="SBERT-SAE",
                source_indices=test_source_indices,
                target_indices=test_target_indices,
                eval_batch_size=args.eval_batch_size,
                device=device,
            )


if __name__ == "__main__":
    main()
