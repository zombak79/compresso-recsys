from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from compresso.nn import TopKSAE


class _LpNormalize(nn.Module):
    def __init__(self, p: float, dim: int = -1) -> None:
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(x, p=self.p, dim=self.dim)


def fit_sae_on_embeddings(
    embeddings: np.ndarray,
    *,
    hidden_dim: int,
    k: int,
    sparsify_score_mode: str = "abs",
    sparsify_ste_alpha: float = 0.0,
    post_norm_p: float | None = None,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    loss_type: str = "mse",
    log_every_epoch: bool = True,
    debug_l1: bool = False,
    val_callback=None,
) -> TopKSAE:
    if loss_type not in {"mse", "cosine"}:
        raise ValueError("loss_type must be 'mse' or 'cosine'")
    if post_norm_p is not None and post_norm_p <= 0.0:
        raise ValueError("post_norm_p must be > 0 when provided")

    post_sparsify = _LpNormalize(post_norm_p, dim=-1) if post_norm_p is not None else None
    model = TopKSAE(
        input_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        k=k,
        sparsify_score_mode=sparsify_score_mode,
        sparsify_ste_alpha=sparsify_ste_alpha,
        post_sparsify=post_sparsify,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    x = torch.from_numpy(embeddings.astype(np.float32))
    loader = torch.utils.data.DataLoader(x, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        n_batches = 0
        running_code_abs_mean = 0.0
        running_code_nonzero_abs_mean = 0.0
        for xb in loader:
            xb = xb.to(device)
            recon, codes, stats = model(xb)
            if loss_type == "mse":
                recon_loss = stats["reconstruction_mse"]
            else:
                xb_flat = xb.reshape(xb.shape[0], -1)
                recon_flat = recon.reshape(recon.shape[0], -1)
                recon_loss = (1.0 - F.cosine_similarity(xb_flat, recon_flat, dim=-1)).mean()
            loss = recon_loss
            if debug_l1:
                code_abs = codes.abs()
                running_code_abs_mean += float(code_abs.mean().item())
                nz_mask = code_abs > 0
                running_code_nonzero_abs_mean += float(code_abs[nz_mask].mean().item()) if nz_mask.any() else 0.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += float(loss.item())
            n_batches += 1
        if log_every_epoch:
            avg_loss = running_loss / max(1, n_batches)
            msg = f"[SAE] epoch={epoch}/{epochs} {loss_type}={avg_loss:.6f}"
            if debug_l1:
                msg += (
                    f" code_abs_mean={running_code_abs_mean / max(1, n_batches):.6f}"
                    f" code_nonzero_abs_mean={running_code_nonzero_abs_mean / max(1, n_batches):.6f}"
                )
            if val_callback is not None:
                val_metrics = val_callback(model)
                msg += (
                    f" val_recall@20={val_metrics.get('recall@20', 0.0):.6f}"
                    f" val_recall@50={val_metrics.get('recall@50', 0.0):.6f}"
                    f" val_ndcg@100={val_metrics.get('ndcg@100', 0.0):.6f}"
                )
            print(msg)

    return model
