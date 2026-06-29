from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from contextlib import contextmanager
from compresso.params.masked import MaskedParam
from compresso.utils.controllers import SparsityController


class TorchELSA(nn.Module):
    """Thin ELSA-style linear autoencoder for item embedding learning.

    Uses normalized item embedding matrix ``A`` and predicts
    ``y = relu((x @ A) @ A.T - x)``, where ``x`` is a user interaction vector.
    """

    def __init__(self, n_items: int, n_factors: int) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = n_factors
        self.A = nn.Parameter(torch.empty(n_items, n_factors))
        nn.init.xavier_uniform_(self.A)

    def normalized_A(self) -> torch.Tensor:
        return F.normalize(self.A, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.normalized_A()
        x_a = x @ a
        x_aat = x_a @ a.T
        return F.relu(x_aat - x)

    @torch.no_grad()
    def export_item_embeddings(self) -> np.ndarray:
        return self.normalized_A().detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self(x)

    def loss_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(F.normalize(y_pred, dim=-1), F.normalize(y_true, dim=-1))

    def train_step(
        self,
        xb: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        grad_clip_norm: float | None = None,
    ) -> dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        y = self(xb)
        loss = self.loss_fn(y, xb)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        optimizer.step()
        return {"loss": float(loss.detach().item())}

    def fit(
        self,
        x_train: csr_matrix,
        *,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
        grad_clip_norm: float | None = None,
        grad_accum_steps: int = 1,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        device: str = "cpu",
        log_every_epoch: bool = True,
        val_callback=None,
    ) -> dict[str, list[dict[str, float]]]:
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        self.to(device)
        opt = torch.optim.NAdam(
            self.parameters(),
            lr=lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
        )

        ema_params = [p.detach().clone() for p in self.parameters()] if use_ema else None

        @torch.no_grad()
        def _ema_update():
            if ema_params is None:
                return
            one_minus = 1.0 - ema_momentum
            for ema_p, p in zip(ema_params, self.parameters()):
                ema_p.mul_(ema_momentum).add_(p.detach(), alpha=one_minus)

        @torch.no_grad()
        def _ema_overwrite_model():
            if ema_params is None:
                return
            for p, ema_p in zip(self.parameters(), ema_params):
                p.copy_(ema_p)

        @contextmanager
        def _ema_eval_scope():
            if ema_params is None:
                yield
                return
            backup = [p.detach().clone() for p in self.parameters()]
            _ema_overwrite_model()
            try:
                yield
            finally:
                with torch.no_grad():
                    for p, b in zip(self.parameters(), backup):
                        p.copy_(b)

        history: dict[str, list[dict[str, float]]] = {"train": [], "eval": []}
        opt_steps = 0
        self.train()
        for epoch in range(1, epochs + 1):
            running_loss = 0.0
            n_batches = 0
            opt.zero_grad()
            for xb in _iter_csr_batches(x_train, batch_size=batch_size):
                xb = xb.to(device)
                y = self(xb)
                loss = self.loss_fn(y, xb) / grad_accum_steps
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                if (n_batches + 1) % grad_accum_steps == 0:
                    opt.step()
                    opt_steps += 1
                    _ema_update()
                    if (
                        use_ema
                        and ema_overwrite_frequency is not None
                        and ema_overwrite_frequency > 0
                        and (opt_steps % ema_overwrite_frequency == 0)
                    ):
                        _ema_overwrite_model()
                    opt.zero_grad()
                running_loss += float(loss.item())
                n_batches += 1
            if n_batches % grad_accum_steps != 0:
                opt.step()
                opt_steps += 1
                _ema_update()
                if (
                    use_ema
                    and ema_overwrite_frequency is not None
                    and ema_overwrite_frequency > 0
                    and (opt_steps % ema_overwrite_frequency == 0)
                ):
                    _ema_overwrite_model()
                opt.zero_grad()

            avg_loss = (running_loss * grad_accum_steps) / max(1, n_batches)
            history["train"].append({"epoch": epoch, "loss": avg_loss})
            if log_every_epoch:
                msg = f"[ELSA] epoch={epoch}/{epochs} loss={avg_loss:.6f}"
                if val_callback is not None:
                    with _ema_eval_scope():
                        val_metrics = val_callback(self)
                    history["eval"].append({"epoch": epoch, **val_metrics})
                    msg += (
                        f" val_recall@20={val_metrics.get('recall@20', 0.0):.6f}"
                        f" val_recall@50={val_metrics.get('recall@50', 0.0):.6f}"
                        f" val_ndcg@100={val_metrics.get('ndcg@100', 0.0):.6f}"
                    )
                print(msg)

        if use_ema:
            _ema_overwrite_model()
        return history


class CompressedELSA(nn.Module):
    """ELSA-style model with directly learned sparse item factors via MaskedParam."""

    def __init__(
        self,
        n_items: int,
        n_factors: int,
        *,
        k_target: int,
        k_schedule: list[int] | None = None,
        num_stages: int = 10,
        stability_window: int = 5,
        change_threshold: float = 0.01,
        score_mode: str = "abs",
        ste_alpha: float = 1.0,
        post_norm_l1: bool = False,
    ) -> None:
        super().__init__()
        self.n_items = n_items
        self.n_factors = n_factors
        w = torch.empty(n_items, n_factors)
        nn.init.xavier_uniform_(w)
        self.A = MaskedParam(
            weight=w,
            k_target=k_target,
            k_schedule=k_schedule,
            num_stages=num_stages,
            stability_window=stability_window,
            change_threshold=change_threshold,
            sparsity="row",
            allow_regrowth=True,
            score_mode=score_mode,
            ste_alpha=ste_alpha,
            post_norm_l1=post_norm_l1,
        )

    def normalized_A(self) -> torch.Tensor:
        return F.normalize(self.A(), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.normalized_A()
        x_a = x @ a
        x_aat = x_a @ a.T
        return F.relu(x_aat - x)

    @torch.no_grad()
    def export_item_embeddings(self) -> np.ndarray:
        return self.normalized_A().detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def export_srp(self):
        return self.A.srp()

    def loss_fn(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(F.normalize(y_pred, dim=-1), F.normalize(y_true, dim=-1))

    def train_step(
        self,
        xb: torch.Tensor,
        *,
        optimizer: torch.optim.Optimizer,
        controller: SparsityController | None = None,
        grad_clip_norm: float | None = None,
    ) -> dict[str, float | bool]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        y = self(xb)
        loss = self.loss_fn(y, xb)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
        optimizer.step()
        cinfo = controller.step() if controller is not None else {}
        return {
            "loss": float(loss.detach().item()),
            "rewind_triggered": bool(cinfo.get("rewind_triggered", False)),
        }

    def fit(
        self,
        x_train: csr_matrix,
        *,
        mask_update_interval: int = 10,
        epochs: int = 10,
        batch_size: int = 512,
        lr: float = 1e-2,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        momentum_decay: float = 0.004,
        decoupled_weight_decay: bool = False,
        grad_clip_norm: float | None = None,
        grad_accum_steps: int = 1,
        device: str = "cpu",
        log_every_epoch: bool = True,
        val_callback=None,
    ) -> dict[str, list[dict[str, float]]]:
        if grad_accum_steps < 1:
            raise ValueError("grad_accum_steps must be >= 1")
        self.to(device)

        def _make_optimizer():
            return torch.optim.NAdam(
                self.parameters(),
                lr=lr,
                betas=(beta1, beta2),
                eps=eps,
                weight_decay=weight_decay,
                momentum_decay=momentum_decay,
                decoupled_weight_decay=decoupled_weight_decay,
            )

        opt = _make_optimizer()
        controller = SparsityController(
            self,
            mask_update_interval=mask_update_interval,
            freeze_at_schedule_end=True,
            method="all",
        )

        history: dict[str, list[dict[str, float]]] = {"train": [], "eval": []}
        self.train()
        epoch_in_stage = 0
        global_epoch = 0
        while True:
            epoch_in_stage += 1
            global_epoch += 1
            running_loss = 0.0
            n_batches = 0
            opt.zero_grad()
            rewinds_in_epoch = 0

            for xb in _iter_csr_batches(x_train, batch_size=batch_size):
                xb = xb.to(device)
                y = self(xb)
                loss = self.loss_fn(y, xb) / grad_accum_steps
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_clip_norm)
                if (n_batches + 1) % grad_accum_steps == 0:
                    opt.step()
                    cinfo = controller.step()
                    if cinfo.get("rewind_triggered", False):
                        opt = _make_optimizer()
                        rewinds_in_epoch += 1
                    opt.zero_grad()
                running_loss += float(loss.item())
                n_batches += 1

            if n_batches % grad_accum_steps != 0:
                opt.step()
                cinfo = controller.step()
                if cinfo.get("rewind_triggered", False):
                    opt = _make_optimizer()
                    rewinds_in_epoch += 1
                opt.zero_grad()

            avg_loss = (running_loss * grad_accum_steps) / max(1, n_batches)
            history["train"].append(
                {
                    "epoch": float(epoch_in_stage),
                    "global_epoch": float(global_epoch),
                    "loss": avg_loss,
                    "k": float(self.A.k_current),
                    "stage": float(self.A.stage_idx),
                    "rewinds": float(rewinds_in_epoch),
                }
            )
            if log_every_epoch:
                cur_mask = self.A.mask if self.A.mask_frozen else self.A.topk_mask(k=self.A.k_current)
                dead_cols = int((cur_mask.sum(dim=0) == 0).sum().item())
                msg = (
                    f"[CompressedELSA] epoch={epoch_in_stage}/{epochs} global_epoch={global_epoch} loss={avg_loss:.6f}"
                    f" k={self.A.k_current} stage={self.A.stage_idx}/{self.A.num_stages - 1}"
                    f" rewinds={rewinds_in_epoch}"
                    f" change={float(self.A.last_change):.6f}"
                    f" threshold={float(self.A.change_threshold):.6f}"
                    f" dead_features={dead_cols}/{self.n_factors} ({dead_cols / float(self.n_factors):.2%})"
                )
                if val_callback is not None:
                    val_metrics = val_callback(self)
                    history["eval"].append({"epoch": global_epoch, **val_metrics})
                    msg += (
                        f" val_recall@20={val_metrics.get('recall@20', 0.0):.6f}"
                        f" val_recall@50={val_metrics.get('recall@50', 0.0):.6f}"
                        f" val_ndcg@100={val_metrics.get('ndcg@100', 0.0):.6f}"
                    )
                print(msg)

            if rewinds_in_epoch > 0:
                epoch_in_stage = 0
                continue

            if epoch_in_stage >= epochs and not self.A.schedule_done:
                self.A.stage_completed = True
                stats = self.A.rewind()
                print(f"[CompressedELSA] Forced rewind at stage budget end: {stats}")
                opt = _make_optimizer()
                epoch_in_stage = 0
                continue

            if self.A.schedule_done and epoch_in_stage >= epochs:
                break
        return history


def _iter_csr_batches(x: csr_matrix, batch_size: int):
    n = x.shape[0]
    for i in range(0, n, batch_size):
        m = x[i : i + batch_size]
        yield torch.from_numpy(m.toarray().astype(np.float32))


def fit_elsa(
    x_train: csr_matrix,
    *,
    n_factors: int,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    momentum_decay: float = 0.004,
    decoupled_weight_decay: bool = False,
    grad_clip_norm: float | None = None,
    grad_accum_steps: int = 1,
    use_ema: bool = False,
    ema_momentum: float = 0.99,
    ema_overwrite_frequency: int | None = None,
    device: str = "cpu",
    log_every_epoch: bool = True,
    val_callback=None,
) -> TorchELSA:
    model = TorchELSA(n_items=x_train.shape[1], n_factors=n_factors).to(device)
    model.fit(
        x_train,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        momentum_decay=momentum_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        grad_clip_norm=grad_clip_norm,
        grad_accum_steps=grad_accum_steps,
        use_ema=use_ema,
        ema_momentum=ema_momentum,
        ema_overwrite_frequency=ema_overwrite_frequency,
        device=device,
        log_every_epoch=log_every_epoch,
        val_callback=val_callback,
    )
    return model


def fit_compressed_elsa(
    x_train: csr_matrix,
    *,
    n_factors: int,
    k_target: int,
    k_schedule: list[int] | None = None,
    num_stages: int = 10,
    stability_window: int = 5,
    change_threshold: float = 0.01,
    score_mode: str = "abs",
    ste_alpha: float = 1.0,
    post_norm_l1: bool = False,
    mask_update_interval: int = 10,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-2,
    weight_decay: float = 0.0,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    momentum_decay: float = 0.004,
    decoupled_weight_decay: bool = False,
    grad_clip_norm: float | None = None,
    grad_accum_steps: int = 1,
    device: str = "cpu",
    log_every_epoch: bool = True,
    val_callback=None,
) -> CompressedELSA:
    model = CompressedELSA(
        n_items=x_train.shape[1],
        n_factors=n_factors,
        k_target=k_target,
        k_schedule=k_schedule,
        num_stages=num_stages,
        stability_window=stability_window,
        change_threshold=change_threshold,
        score_mode=score_mode,
        ste_alpha=ste_alpha,
        post_norm_l1=post_norm_l1,
    ).to(device)
    model.fit(
        x_train,
        mask_update_interval=mask_update_interval,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        momentum_decay=momentum_decay,
        decoupled_weight_decay=decoupled_weight_decay,
        grad_clip_norm=grad_clip_norm,
        grad_accum_steps=grad_accum_steps,
        device=device,
        log_every_epoch=log_every_epoch,
        val_callback=val_callback,
    )
    return model
