from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, isspmatrix_csr
from torch import nn

from compresso import SRPTensor

__all__ = ["ELSA", "ELSAConfig", "ELSATrainer"]

OptimizerName = Literal["NAdam", "AdamW"]


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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        start = int(batch_index) * self.batch_size
        end = min(start + self.batch_size, self.interactions.shape[0])
        if start < 0 or start >= self.interactions.shape[0]:
            raise IndexError(batch_index)

        matrix = self.interactions[self.user_indices[start:end]]
        source_columns = np.unique(matrix.indices).astype(np.int64, copy=False)
        negative_pool = np.setdiff1d(
            self.item_indices,
            source_columns,
            assume_unique=True,
        )
        if self.max_output is None:
            n_negatives = len(negative_pool)
        else:
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
            )
        else:
            negative_columns = np.empty(0, dtype=np.int64)

        candidate_columns = np.concatenate((source_columns, negative_columns))
        x = matrix[:, source_columns].toarray().astype(np.float32, copy=False)
        y = matrix[:, candidate_columns].toarray().astype(np.float32, copy=False)
        return (
            torch.from_numpy(x).to(self.device),
            torch.from_numpy(y).to(self.device),
            torch.from_numpy(source_columns).long().to(self.device),
            torch.from_numpy(candidate_columns).long().to(self.device),
        )

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self.rng.shuffle(self.user_indices)


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
        """Score candidate items from optionally column-restricted inputs."""
        embeddings = self.normalized_item_embeddings()
        source_embeddings = embeddings if sources is None else embeddings[sources]
        candidate_embeddings = (
            embeddings if candidates is None else embeddings[candidates]
        )
        scores = (x @ source_embeddings) @ candidate_embeddings.T
        if x_out is not None:
            scores = scores - x_out
        return F.relu(scores) if self.use_relu else scores


class ELSATrainer:
    """Fit and run ELSA with sparse interaction matrices."""

    def __init__(self, config: ELSAConfig | None = None) -> None:
        self.cfg = config if config is not None else ELSAConfig()
        self.input_dim: int | None = None
        self.elsa: ELSA | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
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
        model: ELSA | nn.Module = ELSA(
            input_dim=int(input_dim),
            latent_dim=int(self.cfg.latent_dim),
            use_relu=bool(self.cfg.use_relu),
        ).to(self.device)
        if self.cfg.compile:
            model = torch.compile(model)
        self.input_dim = int(input_dim)
        self.elsa = model
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
        return self

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
        y: torch.Tensor,
        sources: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run one optimization step."""
        if self.elsa is None or self.optimizer is None:
            raise RuntimeError("trainer must be built before train_step")
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
        return {
            "loss": loss.detach(),
            "cosine_loss": cosine_loss.detach(),
        }

    def fit(self, interactions: csr_matrix) -> ELSATrainer:
        """Fit ELSA from a CSR user-item interaction matrix."""
        interactions = _canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
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
            desc="ELSA fit",
        )
        for epoch in epoch_iter:
            sums: dict[str, float] = {}
            n_batches = 0
            batch_iter = self._progress(
                range(len(dataset)),
                total=len(dataset),
                desc=f"ELSA epoch {epoch}",
            )
            for batch_index in batch_iter:
                stats = self.train_step(*dataset[batch_index])
                for key, value in stats.items():
                    sums[key] = sums.get(key, 0.0) + float(value.cpu().item())
                n_batches += 1
            dataset.on_epoch_end()
            record = {
                key: value / max(1, n_batches)
                for key, value in sums.items()
            }
            record["epoch"] = float(epoch)
            record["lr"] = self._current_lr()
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
