from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, isspmatrix_csr
from torch import nn

from compresso import MaskedParam, SRPParam, SRPTensor, SparsityController

__all__ = [
    "CompressedELSA",
    "ELSA",
    "ELSACompressionConfig",
    "ELSAConfig",
    "ELSATrainer",
]

OptimizerName = Literal["NAdam", "AdamW"]
FactorNorm = Literal["l2", "l1", "none"]
CompressionScoreMode = Literal["abs", "raw", "relu"]


def _canonical_csr(matrix: csr_matrix, *, name: str) -> csr_matrix:
    if not isspmatrix_csr(matrix):
        raise TypeError(f"{name} must be a scipy.sparse.csr_matrix")
    needs_copy = not matrix.has_canonical_format or bool(np.any(matrix.data == 0))
    out = matrix.copy() if needs_copy else matrix
    if needs_copy:
        out.sum_duplicates()
        out.eliminate_zeros()
        out.sort_indices()
    if not np.all(np.isfinite(out.data)):
        raise ValueError(f"{name} values must be finite")
    return out


def _normalized_mse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = F.normalize(predictions, dim=-1)
    targets = F.normalize(targets, dim=-1)
    return (predictions - targets).square().sum(dim=-1).mean()


def _dense_training_target(
    x: torch.Tensor,
    *,
    sources: torch.Tensor,
    candidates: torch.Tensor | None,
    input_dim: int,
) -> torch.Tensor:
    if candidates is None:
        target = x.new_zeros((x.shape[0], input_dim))
        target[:, sources] = x
        return target
    if candidates.numel() < x.shape[1]:
        raise ValueError("the candidate prefix must contain every source item")
    return F.pad(x, (0, candidates.numel() - x.shape[1]))


def _normalize_dense_factors(
    factors: torch.Tensor,
    *,
    mode: FactorNorm,
) -> torch.Tensor:
    if mode == "none":
        return factors
    p = 1.0 if mode == "l1" else 2.0
    return F.normalize(factors, p=p, dim=-1)


def _normalize_srp(
    factors: SRPTensor,
    *,
    mode: FactorNorm,
) -> SRPTensor:
    if mode == "none":
        values = factors.vals
    else:
        p = 1.0 if mode == "l1" else 2.0
        values = F.normalize(
            factors.vals,
            p=p,
            dim=-1,
        )
    return SRPTensor(
        cols=factors.cols,
        vals=values,
        shape=factors.shape,
        validate=False,
    )


def _score_candidates(
    x: torch.Tensor,
    *,
    embeddings: torch.Tensor,
    sources: torch.Tensor | None,
    candidates: torch.Tensor | None,
    x_out: torch.Tensor | None,
    use_relu: bool,
) -> torch.Tensor:
    if candidates is None:
        source_embeddings = (
            embeddings if sources is None else embeddings[sources]
        )
        candidate_embeddings = embeddings
    else:
        candidate_embeddings = embeddings[candidates]
        if x.shape[1] > candidate_embeddings.shape[0]:
            raise ValueError(
                "the candidate prefix must contain every source item"
            )
        source_embeddings = candidate_embeddings[: x.shape[1]]
    scores = (x @ source_embeddings) @ candidate_embeddings.T
    if x_out is not None:
        scores = scores - x_out
    return F.relu(scores) if use_relu else scores


class _ELSAInteractionDataset:
    """Batched sparse interactions with optional output candidate sampling."""

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        device: torch.device,
        batch_size: int,
        shuffle: bool,
        max_output: int | None,
        seed: int,
    ) -> None:
        self.interactions = interactions
        self.device = device
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.max_output = max_output
        self.user_indices = np.arange(interactions.shape[0], dtype=np.int64)
        self.item_indices = np.arange(interactions.shape[1], dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        if self.shuffle:
            self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(self.interactions.shape[0] / self.batch_size)

    def __getitem__(
        self,
        batch_index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
    ]:
        start = int(batch_index) * self.batch_size
        end = min(start + self.batch_size, self.interactions.shape[0])
        if start < 0 or start >= self.interactions.shape[0]:
            raise IndexError(batch_index)

        matrix = self.interactions[self.user_indices[start:end]]
        source_columns = np.flatnonzero(
            np.asarray(matrix.getnnz(axis=0)).ravel()
        ).astype(np.int64, copy=False)
        if self.max_output is None:
            candidate_columns = None
        else:
            negative_mask = np.ones(matrix.shape[1], dtype=bool)
            negative_mask[source_columns] = False
            negative_pool = self.item_indices[negative_mask]
            n_negatives = min(
                len(negative_pool),
                max(0, int(self.max_output) - len(source_columns)),
            )
            if n_negatives == len(negative_pool):
                negative_columns = negative_pool
            elif n_negatives > 0:
                negative_columns = self.rng.choice(
                    negative_pool,
                    size=n_negatives,
                    replace=False,
                    shuffle=False,
                )
            else:
                negative_columns = np.empty(0, dtype=np.int64)
            candidate_columns = np.concatenate(
                (source_columns, negative_columns)
            )

        row_indices = np.repeat(
            np.arange(matrix.shape[0], dtype=np.int64),
            np.diff(matrix.indptr),
        )
        source_lookup = np.empty(matrix.shape[1], dtype=np.int64)
        source_lookup[source_columns] = np.arange(
            len(source_columns),
            dtype=np.int64,
        )
        source_local_columns = source_lookup[matrix.indices]
        values = matrix.data.astype(np.float32, copy=False)
        x = self._sparse_tensor(
            row_indices,
            source_local_columns,
            values,
            shape=(matrix.shape[0], len(source_columns)),
        )
        if candidate_columns is None:
            sources = torch.from_numpy(source_columns).to(self.device)
            candidates = None
        else:
            candidates = torch.from_numpy(candidate_columns).to(self.device)
            sources = candidates[: len(source_columns)]
        return (
            x,
            sources,
            candidates,
        )

    def _sparse_tensor(
        self,
        rows: np.ndarray,
        columns: np.ndarray,
        values: np.ndarray,
        *,
        shape: tuple[int, int],
    ) -> torch.Tensor:
        indices = torch.from_numpy(np.vstack((rows, columns)))
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse invariant checks are implicitly disabled.*",
                category=UserWarning,
            )
            tensor = torch.sparse_coo_tensor(
                indices,
                torch.from_numpy(values),
                shape,
                is_coalesced=True,
                check_invariants=False,
            )
            return tensor.to(self.device)

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.user_indices)


@dataclass(frozen=True)
class ELSACompressionConfig:
    """Lottery-ticket compression settings for :class:`ELSATrainer`.

    Mask-search stages advance only when the proposed mask remains below
    ``change_threshold`` for ``stability_window`` mask updates. Once the final
    ticket is found, it is converted to an :class:`compresso.SRPParam` and its
    values are trained for ``ELSAConfig.epochs``.
    """

    k_target: int
    k_schedule: tuple[int, ...] | None = None
    num_stages: int = 10
    stability_window: int = 5
    change_threshold: float = 0.01
    mask_update_interval: int = 10
    score_mode: CompressionScoreMode = "abs"
    ste_alpha: float = 1.0
    factor_norm: FactorNorm = "l2"

    def __post_init__(self) -> None:
        if self.k_target < 1:
            raise ValueError("k_target must be >= 1")
        if self.k_schedule is not None:
            if not self.k_schedule:
                raise ValueError("k_schedule must not be empty")
            if any(k < 1 for k in self.k_schedule):
                raise ValueError("every k_schedule value must be >= 1")
            if any(
                current < following
                for current, following in zip(
                    self.k_schedule,
                    self.k_schedule[1:],
                )
            ):
                raise ValueError("k_schedule must be non-increasing")
            if self.k_schedule[-1] != self.k_target:
                raise ValueError("the last k_schedule value must equal k_target")
        if self.num_stages < 1:
            raise ValueError("num_stages must be >= 1")
        if self.stability_window < 1:
            raise ValueError("stability_window must be >= 1")
        if not np.isfinite(self.change_threshold) or self.change_threshold < 0:
            raise ValueError("change_threshold must be finite and >= 0")
        if self.mask_update_interval < 1:
            raise ValueError("mask_update_interval must be >= 1")
        if self.score_mode not in {"abs", "raw", "relu"}:
            raise ValueError("score_mode must be 'abs', 'raw', or 'relu'")
        if not np.isfinite(self.ste_alpha) or not 0 <= self.ste_alpha <= 1:
            raise ValueError("ste_alpha must be finite and in [0, 1]")
        if self.factor_norm not in {"l2", "l1", "none"}:
            raise ValueError("factor_norm must be 'l2', 'l1', or 'none'")


@dataclass(frozen=True)
class ELSAConfig:
    """Configuration for :class:`ELSATrainer`.

    ``max_output`` limits the number of output candidates used by a training
    batch. Every item with a positive interaction in the batch is retained,
    and the remaining budget is sampled without replacement from items absent
    from the entire batch. Consequently, a batch with more positive columns
    than ``max_output`` exceeds the requested limit rather than dropping
    positive targets. ``None`` evaluates the full item output during training.
    """

    latent_dim: int = 1024
    batch_size: int = 1024
    max_output: int | None = None
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    use_relu: bool = True
    optimizer: OptimizerName = "NAdam"
    compression: ELSACompressionConfig | None = None

    def __post_init__(self) -> None:
        if self.latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.max_output is not None and self.max_output < 1:
            raise ValueError("max_output must be >= 1 or None")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not np.isfinite(self.lr) or self.lr <= 0:
            raise ValueError("lr must be finite and > 0")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and >= 0")
        if self.optimizer not in {"NAdam", "AdamW"}:
            raise ValueError("optimizer must be 'NAdam' or 'AdamW'")
        if self.compression is not None:
            if self.compile:
                raise ValueError(
                    "torch.compile is not supported during compressed ELSA "
                    "mask search"
                )
            if self.compression.k_target > self.latent_dim:
                raise ValueError("compression.k_target must be <= latent_dim")
            if self.compression.k_schedule is not None:
                if self.compression.k_schedule[0] != self.latent_dim:
                    raise ValueError(
                        "compression.k_schedule must start with latent_dim"
                    )
                if any(k > self.latent_dim for k in self.compression.k_schedule):
                    raise ValueError(
                        "compression.k_schedule values must be <= latent_dim"
                    )


class ELSA(nn.Module):
    """Scalable linear shallow autoencoder with normalized item embeddings."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        *,
        use_relu: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.use_relu = bool(use_relu)
        self.A = nn.Parameter(torch.empty(self.input_dim, self.latent_dim))
        nn.init.xavier_uniform_(self.A)

    def normalized_item_embeddings(self) -> torch.Tensor:
        """Return row-normalized item embeddings."""
        return F.normalize(self.A, dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score items, using candidate rows as the source prefix when given."""
        return _score_candidates(
            x,
            embeddings=self.normalized_item_embeddings(),
            sources=sources,
            candidates=candidates,
            x_out=x_out,
            use_relu=self.use_relu,
        )


class CompressedELSA(nn.Module):
    """ELSA item factors compressed to fixed row-wise sparsity.

    The model starts with a dense :class:`compresso.MaskedParam`. After its
    mask schedule is complete, :meth:`convert_to_srp` replaces that parameter
    with an :class:`compresso.SRPParam` whose structure is fixed and whose
    values remain trainable.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        compression: ELSACompressionConfig,
        *,
        use_relu: bool = True,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if compression.k_target > latent_dim:
            raise ValueError("compression.k_target must be <= latent_dim")

        self.input_dim = int(input_dim)
        self.latent_dim = int(latent_dim)
        self.compression = compression
        self.use_relu = bool(use_relu)
        weight = torch.empty(self.input_dim, self.latent_dim)
        nn.init.xavier_uniform_(weight)
        self.masked_A: MaskedParam | None = MaskedParam(
            weight=weight,
            k_target=compression.k_target,
            k_schedule=compression.k_schedule,
            num_stages=compression.num_stages,
            stability_window=compression.stability_window,
            change_threshold=compression.change_threshold,
            sparsity="row",
            score_mode=compression.score_mode,
            ste_alpha=compression.ste_alpha,
            post_norm_l1=False,
        )
        self.sparse_A: SRPParam | None = None
        self.phase = "mask_search"
        self._inference_srp: SRPTensor | None = None
        self._inference_csr: torch.Tensor | None = None

    def _invalidate_inference_cache(self) -> None:
        self._inference_srp = None
        self._inference_csr = None

    def _apply(self, fn):
        result = super()._apply(fn)
        self._invalidate_inference_cache()
        return result

    def train(self, mode: bool = True) -> CompressedELSA:
        result = super().train(mode)
        if mode:
            self._invalidate_inference_cache()
            if self.is_sparse:
                self.phase = "sparse_finetune"
        return result

    @property
    def is_sparse(self) -> bool:
        """Whether the final fixed SRP structure has been installed."""
        return self.sparse_A is not None

    def normalized_item_embeddings(self) -> torch.Tensor:
        """Return normalized dense factors for the current training phase."""
        if self.masked_A is not None:
            factors = self.masked_A()
        elif self.sparse_A is not None:
            factors = self.sparse_A().to_dense()
        else:  # pragma: no cover - defensive invariant
            raise RuntimeError("compressed ELSA has no item parameter")
        return _normalize_dense_factors(
            factors,
            mode=self.compression.factor_norm,
        )

    def normalized_item_srp(self) -> SRPTensor:
        """Return normalized sparse factors after mask search completes."""
        if self.sparse_A is None:
            raise RuntimeError(
                "SRP factors are unavailable until mask search completes"
            )
        return _normalize_srp(
            self.sparse_A(),
            mode=self.compression.factor_norm,
        )

    @torch.no_grad()
    def convert_to_srp(self) -> None:
        """Install Compresso's final fixed SRP parameter."""
        if self.sparse_A is not None:
            return
        if self.masked_A is None or not self.masked_A.schedule_done:
            raise RuntimeError(
                "mask search must complete before conversion to SRP"
            )

        sparse_A = self.masked_A.maskedparam_to_srp()
        self.sparse_A = sparse_A
        self.masked_A = None
        self.phase = "sparse_finetune"
        self._invalidate_inference_cache()

    @torch.no_grad()
    def prepare_inference(self) -> None:
        """Cache normalized SRP and CSR factors for sparse inference."""
        factors = self.normalized_item_srp().detach()
        self._inference_srp = factors
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Sparse CSR tensor support is in beta state.*",
                category=UserWarning,
            )
            self._inference_csr = factors.to_csr()
        self.phase = "inference"

    @torch.no_grad()
    def export_item_embeddings(self) -> SRPTensor:
        """Return a detached copy of the normalized final item factors."""
        return self.normalized_item_srp().detach().clone()

    def forward(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        x_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates during mask search or sparse-value training."""
        return _score_candidates(
            x,
            embeddings=self.normalized_item_embeddings(),
            sources=sources,
            candidates=candidates,
            x_out=x_out,
            use_relu=self.use_relu,
        )

    @torch.no_grad()
    def score_all_items(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor,
    ) -> torch.Tensor:
        """Score the full catalog through the cached SRP-to-CSR path."""
        if self._inference_srp is None or self._inference_csr is None:
            self.prepare_inference()
        assert self._inference_srp is not None
        assert self._inference_csr is not None

        source_factors = SRPTensor(
            cols=self._inference_srp.cols.index_select(0, sources),
            vals=self._inference_srp.vals.index_select(0, sources),
            shape=(sources.numel(), self.latent_dim),
            validate=False,
        ).to_dense()
        user_factors = x @ source_factors
        scores = torch.sparse.mm(
            self._inference_csr,
            user_factors.T,
        ).T
        return F.relu(scores) if self.use_relu else scores


class ELSATrainer:
    """Fit and run ELSA with sparse interaction matrices."""

    def __init__(self, config: ELSAConfig | None = None) -> None:
        self.cfg = config if config is not None else ELSAConfig()
        self.input_dim: int | None = None
        self.elsa: ELSA | CompressedELSA | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.sparsity_controller: SparsityController | None = None
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float | str]] = []
        self._last_controller_info: dict[str, object] = {}
        self._is_fitted = False

    @property
    def is_built(self) -> bool:
        """Whether the underlying ELSA model has been initialized."""
        return self.elsa is not None

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has completed at least once."""
        return self._is_fitted

    def build(self, input_dim: int) -> ELSATrainer:
        """Initialize the ELSA model and optimizer."""
        if self.is_built:
            if int(input_dim) != self.input_dim:
                raise ValueError(
                    f"trainer is already built for input_dim={self.input_dim}, "
                    f"got {input_dim}"
                )
            return self
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")

        torch.manual_seed(int(self.cfg.seed))
        if self.cfg.compression is None:
            model: ELSA | CompressedELSA | nn.Module = ELSA(
                input_dim=int(input_dim),
                latent_dim=int(self.cfg.latent_dim),
                use_relu=bool(self.cfg.use_relu),
            ).to(self.device)
            if self.cfg.compile:
                model = torch.compile(model)
        else:
            model = CompressedELSA(
                input_dim=int(input_dim),
                latent_dim=int(self.cfg.latent_dim),
                compression=self.cfg.compression,
                use_relu=bool(self.cfg.use_relu),
            ).to(self.device)
        self.input_dim = int(input_dim)
        self.elsa = model
        self._reset_optimizer()
        if self.cfg.compression is not None:
            self.sparsity_controller = SparsityController(
                self.elsa,
                mask_update_interval=self.cfg.compression.mask_update_interval,
                freeze_at_schedule_end=False,
                method="all",
            )
        return self

    def _reset_optimizer(self) -> None:
        if self.elsa is None:
            raise RuntimeError("trainer must be built before creating optimizer")
        optimizer_class = (
            torch.optim.AdamW
            if self.cfg.optimizer == "AdamW"
            else torch.optim.NAdam
        )
        self.optimizer = optimizer_class(
            self.elsa.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _progress(self, iterable, *, total: int, desc: str):
        if not self.cfg.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency fallback
            return iterable
        return tqdm(iterable, total=total, desc=desc)

    def _set_lr(self, learning_rate: float) -> None:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before setting learning rate")
        for group in self.optimizer.param_groups:
            group["lr"] = float(learning_rate)

    def _current_lr(self) -> float:
        if self.optimizer is None:
            raise RuntimeError("trainer must be built before reading learning rate")
        return float(self.optimizer.param_groups[0]["lr"])

    def train_step(
        self,
        x: torch.Tensor,
        sources: torch.Tensor,
        candidates: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        """Run one optimization step."""
        if self.elsa is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")
        x = x.to_dense()
        assert self.input_dim is not None
        y = _dense_training_target(
            x,
            sources=sources,
            candidates=candidates,
            input_dim=self.input_dim,
        )
        self.elsa.train()
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.elsa(
            x,
            sources=sources,
            candidates=candidates,
            x_out=y,
        )
        loss = _normalized_mse(predictions, y)
        cosine_loss = 1.0 - F.cosine_similarity(
            predictions,
            y,
            dim=-1,
        ).mean()
        loss.backward()
        self.optimizer.step()
        self._last_controller_info = (
            self.sparsity_controller.step()
            if self.sparsity_controller is not None
            else {}
        )
        return {
            "loss": loss.detach(),
            "cosine_loss": cosine_loss.detach(),
        }

    def _run_epoch(
        self,
        dataset: _ELSAInteractionDataset,
        *,
        desc: str,
    ) -> tuple[dict[str, float], bool]:
        sums: dict[str, float] = {}
        n_batches = 0
        rewind_triggered = False
        batch_iter = self._progress(
            range(len(dataset)),
            total=len(dataset),
            desc=desc,
        )
        for batch_index in batch_iter:
            stats = self.train_step(*dataset[batch_index])
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + float(value.cpu().item())
            n_batches += 1
            if bool(self._last_controller_info.get("rewind_triggered", False)):
                rewind_triggered = True
                break
        dataset.on_epoch_end()
        return (
            {
                key: value / max(1, n_batches)
                for key, value in sums.items()
            },
            rewind_triggered,
        )

    def _fit_fixed_epochs(
        self,
        dataset: _ELSAInteractionDataset,
        *,
        phase: str | None,
    ) -> None:
        self._set_lr(float(self.cfg.lr))
        assert self.optimizer is not None
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.epochs,
                eta_min=0.0,
            )
            if self.cfg.decay
            else None
        )
        epoch_iter = self._progress(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="ELSA fit" if phase is None else "ELSA sparse fine-tune",
        )
        for epoch in epoch_iter:
            record: dict[str, float | str] = self._run_epoch(
                dataset,
                desc=f"ELSA epoch {epoch}",
            )[0]
            record["epoch"] = float(epoch)
            record["lr"] = self._current_lr()
            if phase is not None:
                record["phase"] = phase
            self.history.append(record)
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    {
                        "loss": f"{record['loss']:.4f}",
                        "cosine": f"{record['cosine_loss']:.4f}",
                        "lr": f"{record['lr']:.2E}",
                    }
                )
            if scheduler is not None:
                scheduler.step()

    def _fit_compressed_mask_search(
        self,
        dataset: _ELSAInteractionDataset,
    ) -> CompressedELSA:
        if not isinstance(self.elsa, CompressedELSA):
            raise RuntimeError("compressed ELSA model was not built")
        search_epoch = 0
        stage_epoch = 0
        while True:
            masked_A = self.elsa.masked_A
            if masked_A is None:  # pragma: no cover - defensive invariant
                raise RuntimeError("mask-search parameter is unavailable")
            search_epoch += 1
            stage_epoch += 1
            stage_idx = int(masked_A.stage_idx)
            k_current = int(masked_A.k_current)
            record: dict[str, float | str] = self._run_epoch(
                dataset,
                desc=(
                    f"ELSA mask stage {stage_idx + 1} "
                    f"epoch {stage_epoch}"
                ),
            )[0]
            rewind_triggered = bool(
                self._last_controller_info.get("rewind_triggered", False)
            )
            record.update(
                {
                    "epoch": float(search_epoch),
                    "stage_epoch": float(stage_epoch),
                    "stage": float(stage_idx),
                    "k": float(k_current),
                    "mask_change": float(masked_A.last_change),
                    "lr": self._current_lr(),
                    "phase": "mask_search",
                }
            )
            self.history.append(record)
            if not rewind_triggered:
                continue

            schedule_done = bool(masked_A.schedule_done)
            self._reset_optimizer()
            self._set_lr(float(self.cfg.lr))
            stage_epoch = 0
            if schedule_done:
                break

        self.elsa.convert_to_srp()
        self.sparsity_controller = None
        self._last_controller_info = {}
        self._reset_optimizer()
        return self.elsa

    def fit(self, interactions: csr_matrix) -> ELSATrainer:
        """Fit dense ELSA or search and fine-tune a compressed ELSA ticket."""
        interactions = _canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        self.build(interactions.shape[1])
        dataset = _ELSAInteractionDataset(
            interactions,
            device=self.device,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            max_output=self.cfg.max_output,
            seed=self.cfg.seed,
        )
        self._set_lr(float(self.cfg.lr))
        if self.cfg.compression is None:
            self._fit_fixed_epochs(dataset, phase=None)
        else:
            if not isinstance(self.elsa, CompressedELSA):
                raise RuntimeError("compressed ELSA model was not built")
            model = self.elsa
            if model.is_sparse:
                model.train()
                self._reset_optimizer()
            else:
                model = self._fit_compressed_mask_search(dataset)
            self._fit_fixed_epochs(dataset, phase="sparse_finetune")
            model.prepare_inference()
        self._is_fitted = True
        return self

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if not self.is_fitted or self.elsa is None or self.input_dim is None:
            raise RuntimeError("ELSATrainer must be fitted before prediction")
        source = _canonical_csr(source, name="source")
        if source.shape[1] != self.input_dim:
            raise ValueError(
                f"source has {source.shape[1]} items; expected {self.input_dim}"
            )
        return source

    @torch.no_grad()
    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        """Predict ranked items for one source batch.

        Seen source items are excluded unless ``exclude_seen`` is false.
        """
        source = self._prepare_source(source)
        assert self.elsa is not None and self.input_dim is not None
        if not 1 <= int(k) <= self.input_dim:
            raise ValueError(f"k must be in [1, {self.input_dim}], got {k}")
        if exclude_seen:
            unseen_counts = self.input_dim - np.diff(source.indptr)
            if unseen_counts.size and np.any(unseen_counts < k):
                row = int(np.flatnonzero(unseen_counts < k)[0])
                raise ValueError(
                    f"source row {row} has only {unseen_counts[row]} unseen "
                    f"items, fewer than k={k}"
                )

        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=source.shape,
            )

        self.elsa.eval()
        source_columns = np.unique(source.indices).astype(np.int64, copy=False)
        x = torch.from_numpy(
            source[:, source_columns].toarray().astype(np.float32, copy=False)
        ).to(self.device)
        source_columns_tensor = torch.from_numpy(source_columns).long().to(self.device)
        if isinstance(self.elsa, CompressedELSA):
            scores = self.elsa.score_all_items(
                x,
                sources=source_columns_tensor,
            )
        else:
            scores = self.elsa(
                x,
                sources=source_columns_tensor,
                candidates=None,
                x_out=None,
            )
        if exclude_seen:
            seen = source.tocoo()
            seen_rows = torch.from_numpy(seen.row.astype(np.int64)).to(self.device)
            seen_columns = torch.from_numpy(seen.col.astype(np.int64)).to(self.device)
            scores[seen_rows, seen_columns] = -torch.inf
        return SRPTensor.from_dense(scores, k=int(k), score_mode="raw")

    @torch.no_grad()
    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int | None = None,
        exclude_seen: bool = True,
        show_progress: bool | None = None,
    ) -> SRPTensor:
        """Predict ranked items for all source rows in batches.

        Each batch delegates to :meth:`predict_on_batch`. Seen source items
        are excluded unless ``exclude_seen`` is false.
        """
        source = self._prepare_source(source)
        resolved_batch_size = (
            self.cfg.batch_size if batch_size is None else int(batch_size)
        )
        if resolved_batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 1 <= int(k) <= source.shape[1]:
            raise ValueError(f"k must be in [1, {source.shape[1]}], got {k}")
        progress_enabled = (
            self.cfg.show_progress
            if show_progress is None
            else bool(show_progress)
        )

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], resolved_batch_size)
        if progress_enabled:
            try:
                from tqdm.auto import tqdm
            except Exception:  # pragma: no cover - optional dependency fallback
                pass
            else:
                starts = tqdm(starts, desc=f"ELSA predict@{k}")
        for start in starts:
            end = min(start + resolved_batch_size, source.shape[0])
            predictions = self.predict_on_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
            )
            columns.append(predictions.cols)
            values.append(predictions.vals)

        if not columns:
            return self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
            )
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=source.shape,
        )
