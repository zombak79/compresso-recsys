"""Variational autoencoder for multinomial implicit collaborative filtering."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Hashable, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix
from torch import nn
from torch.nn import functional as F

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
    _validate_log_every_n_steps,
)
from compresso_recsys.models._autoencoder_batching import (
    dense_training_batch,
    prepare_dense_training_data,
)
from compresso_recsys.models._ranking import validate_candidate_topk
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.base import BaseCollaborativeRecommender
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = ["MultVAE", "MultVAEConfig", "MultVAETrainer"]


@dataclass
class MultVAEConfig:
    """Configuration for :class:`MultVAETrainer`.

    ``kl_cap`` is the maximum coefficient on KL divergence.
    ``kl_anneal_steps`` is the denominator in ``updates / kl_anneal_steps``;
    the coefficient is clipped at ``kl_cap``. It therefore reaches the cap
    after ``kl_cap * kl_anneal_steps`` updates. Set the step count to zero to
    use ``kl_cap`` from the first update.
    ``preload_training_data=True`` caches the dense interaction matrix on the
    training device by default. Set it to ``False`` to stream CSR minibatches
    when the complete dense matrix does not fit.
    """

    latent_dim: int = 200
    hidden_dim: int = 600
    dropout: float = 0.5
    epochs: int = 20
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    kl_cap: float = 0.2
    kl_anneal_steps: int = 200_000
    preload_training_data: bool = True
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "MultVAE"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in ("latent_dim", "hidden_dim", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if isinstance(self.kl_anneal_steps, (bool, np.bool_)) or not isinstance(
            self.kl_anneal_steps, (int, np.integer)
        ):
            raise TypeError("kl_anneal_steps must be an integer")
        if self.kl_anneal_steps < 0:
            raise ValueError("kl_anneal_steps must be >= 0")
        if not np.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not np.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(
                "weight_decay must be finite and >= 0, got "
                f"{self.weight_decay}"
            )
        if not np.isfinite(self.kl_cap) or self.kl_cap < 0.0:
            raise ValueError(f"kl_cap must be finite and >= 0, got {self.kl_cap}")
        if not isinstance(self.preload_training_data, (bool, np.bool_)):
            raise TypeError("preload_training_data must be a bool")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        torch.device(self.device)


class MultVAE(nn.Module):
    """Symmetric multinomial VAE with a Gaussian latent representation."""

    def __init__(
        self,
        n_items: int,
        latent_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError("n_items must be >= 1")
        if latent_dim < 1:
            raise ValueError("latent_dim must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.n_items = int(n_items)
        self.input_dropout = nn.Dropout(float(dropout))
        self.encoder = nn.Linear(self.n_items, int(hidden_dim))
        self.mean = nn.Linear(int(hidden_dim), int(latent_dim))
        self.log_variance = nn.Linear(int(hidden_dim), int(latent_dim))
        self.decoder_hidden = nn.Linear(int(latent_dim), int(hidden_dim))
        self.decoder = nn.Linear(int(hidden_dim), self.n_items)

    def encode(self, interactions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return posterior mean and log variance for interaction rows."""
        if interactions.ndim != 2 or interactions.shape[1] != self.n_items:
            raise ValueError(
                "interactions must have shape (rows, "
                f"{self.n_items}), got {tuple(interactions.shape)}"
            )
        normalized = F.normalize(interactions, p=2, dim=1)
        hidden = torch.tanh(self.encoder(self.input_dropout(normalized)))
        return self.mean(hidden), self.log_variance(hidden)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent rows to unnormalized multinomial item scores."""
        return self.decoder(torch.tanh(self.decoder_hidden(latent)))

    def forward(
        self,
        interactions: torch.Tensor,
        *,
        sample: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return item logits, posterior mean, and posterior log variance.

        Sampling defaults to the module's training mode. Evaluation therefore
        uses the posterior mean and produces deterministic rankings.
        """
        mean, log_variance = self.encode(interactions)
        should_sample = self.training if sample is None else bool(sample)
        if should_sample:
            standard_deviation = torch.exp(0.5 * log_variance)
            latent = mean + standard_deviation * torch.randn_like(mean)
        else:
            latent = mean
        return self.decode(latent), mean, log_variance


class MultVAETrainer(BaseCollaborativeRecommender):
    """Train and serve Mult-VAE on implicit-feedback user rows."""

    checkpoint_type = "mult_vae_trainer"

    def __init__(
        self,
        config: MultVAEConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config if config is not None else MultVAEConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.model: MultVAE | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, float]] = []
        self._n_items: int | None = None
        self._updates = 0
        self.training_data_preloaded_: bool | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_items(self) -> int | None:
        return self._n_items

    def _reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        return _resolve_reporter(
            default_logger=self.logger,
            logger=logger,
            default_show_progress=self.cfg.show_progress,
            show_progress=show_progress,
            prefix=self.cfg.log_prefix,
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

    def _train_step(
        self,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Optimize one dense user batch and return detached objectives."""
        assert self.model is not None and self.optimizer is not None
        logits, mean, log_variance = self.model(target, sample=True)
        reconstruction = -(
            target * F.log_softmax(logits, dim=1)
        ).sum(dim=1).mean()
        kl = -0.5 * (
            1.0 + log_variance - mean.square() - log_variance.exp()
        ).sum(dim=1).mean()
        kl_weight = self._kl_weight()
        loss = reconstruction + kl_weight * kl
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self._updates += 1
        return loss.detach(), reconstruction.detach(), kl.detach(), kl_weight

    def _kl_weight(self) -> float:
        if self.cfg.kl_anneal_steps == 0:
            return float(self.cfg.kl_cap)
        return min(
            float(self.cfg.kl_cap),
            self._updates / float(self.cfg.kl_anneal_steps),
        )

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> MultVAETrainer:
        """Fit Mult-VAE with multinomial likelihood and annealed KL loss."""
        reporter = self._reporter(logger, show_progress)
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError(
                "interactions must contain at least one user and one item"
            )
        if np.any(interactions.data < 0):
            raise ValueError("interactions must contain nonnegative values")
        active_rows = np.flatnonzero(np.diff(interactions.indptr) > 0)
        if active_rows.size == 0:
            raise ValueError("interactions must contain at least one nonempty user")

        # A full fit installs a fresh model, optimizer, history, and update
        # counter before any update. The shallow snapshot therefore remains
        # untouched and can be restored if setup or training fails.
        previous_state = (
            self.model,
            self.optimizer,
            self.history,
            self._n_items,
            self._updates,
            self.training_data_preloaded_,
            self._is_fitted,
        )
        had_vocabulary = "_item_vocabulary" in self.__dict__
        previous_vocabulary = self.__dict__.get("_item_vocabulary")
        self._is_fitted = False
        try:
            return self._fit_validated(
                interactions,
                active_rows=active_rows,
                item_ids=item_ids,
                reporter=reporter,
            )
        except BaseException:
            (
                self.model,
                self.optimizer,
                self.history,
                self._n_items,
                self._updates,
                self.training_data_preloaded_,
                self._is_fitted,
            ) = previous_state
            if had_vocabulary:
                self.__dict__["_item_vocabulary"] = previous_vocabulary
            else:
                self.__dict__.pop("_item_vocabulary", None)
            raise

    def _fit_validated(
        self,
        interactions: csr_matrix,
        *,
        active_rows: np.ndarray,
        item_ids: Sequence[Hashable] | np.ndarray | None,
        reporter: _Reporter,
    ) -> MultVAETrainer:
        """Train replacement state after public input validation."""

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))
        self._n_items = int(interactions.shape[1])
        self._set_item_ids(item_ids, n_items=self._n_items)
        self.model = self._build_model()
        self._build_checkpoint_optimizer()
        assert self.optimizer is not None
        self.history = []
        self._updates = 0
        steps_per_epoch = (active_rows.size + int(self.cfg.batch_size) - 1) // int(
            self.cfg.batch_size
        )
        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{active_rows.size} active users | {interactions.shape[1]} items | "
            f"{interactions.nnz} interactions | {steps_per_epoch} batches of "
            f"{self.cfg.batch_size} | {self.cfg.epochs} epochs | device {self.device}"
        )
        training_data = prepare_dense_training_data(
            interactions,
            device=self.device,
            preload=self.cfg.preload_training_data,
        )
        self.training_data_preloaded_ = training_data is not None

        epochs = reporter.wrap(
            range(1, int(self.cfg.epochs) + 1),
            total=int(self.cfg.epochs),
            desc="MultVAE fit",
        )
        batch_bar = reporter.bar(total=steps_per_epoch, desc="MultVAE epoch 1")
        try:
            for epoch in epochs:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(active_rows)
                loss_sum = torch.zeros((), device=self.device)
                reconstruction_sum = torch.zeros((), device=self.device)
                kl_sum = torch.zeros((), device=self.device)
                users = 0
                last_kl_weight = self._kl_weight()
                if batch_bar is not None:
                    batch_bar.reset(total=steps_per_epoch)
                    batch_bar.set_description(f"MultVAE epoch {epoch}")
                for step, start in enumerate(
                    range(0, order.size, int(self.cfg.batch_size)),
                    start=1,
                ):
                    selected = order[start : start + int(self.cfg.batch_size)]
                    target = dense_training_batch(
                        interactions,
                        selected,
                        device=self.device,
                        preloaded=training_data,
                    )
                    loss, reconstruction, kl, last_kl_weight = self._train_step(
                        target
                    )

                    batch_users = int(selected.size)
                    loss_sum += loss * batch_users
                    reconstruction_sum += reconstruction * batch_users
                    kl_sum += kl * batch_users
                    users += batch_users
                    if batch_bar is not None:
                        batch_bar.update(1)
                    log_steps = reporter.log_every_n_steps
                    if log_steps and step % log_steps == 0:
                        running_loss, running_reconstruction, running_kl = torch.stack(
                            (
                                loss_sum / users,
                                reconstruction_sum / users,
                                kl_sum / users,
                            )
                        ).tolist()
                        reporter.step(
                            f"epoch {epoch}/{self.cfg.epochs} step {step}/{steps_per_epoch}",
                            step,
                            steps_per_epoch,
                            epoch_started,
                            {
                                "loss": running_loss,
                                "reconstruction_loss": running_reconstruction,
                                "kl_loss": running_kl,
                                "kl_weight": last_kl_weight,
                            },
                        )
                mean_loss, mean_reconstruction, mean_kl = torch.stack(
                    (
                        loss_sum / users,
                        reconstruction_sum / users,
                        kl_sum / users,
                    )
                ).tolist()
                record = {
                    "epoch": float(epoch),
                    "loss": mean_loss,
                    "reconstruction_loss": mean_reconstruction,
                    "kl_loss": mean_kl,
                    "kl_weight": last_kl_weight,
                }
                self.history.append(record)
                reporter.epoch(
                    f"epoch {epoch}/{self.cfg.epochs}",
                    record,
                    epoch_started,
                )
                if hasattr(epochs, "set_postfix"):
                    epochs.set_postfix(
                        {
                            "loss": f"{mean_loss:.4f}",
                            "kl_weight": f"{last_kl_weight:.4f}",
                        }
                    )
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epochs, "close"):
                epochs.close()
        self._is_fitted = True
        reporter.log(
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} total | "
            f"{len(self.history)} epochs recorded"
        )
        return self

    def _build_model(self) -> MultVAE:
        if self._n_items is None:
            raise RuntimeError("MultVAE catalog size is unavailable")
        return MultVAE(
            self._n_items,
            int(self.cfg.latent_dim),
            int(self.cfg.hidden_dim),
            float(self.cfg.dropout),
        ).to(self.device)

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        source = self._prepare_source(source)
        candidate_rows = self._candidate_rows(candidate_ids)
        validate_candidate_topk(
            source,
            candidate_rows,
            k=k,
            exclude_seen=exclude_seen,
        )
        assert self.model is not None and self._n_items is not None
        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=source.shape,
            )

        # Cast while still sparse so a non-float32 source does not create a
        # second catalog-wide dense allocation.
        dense = source.astype(np.float32, copy=False).toarray()
        inputs = torch.from_numpy(dense).to(self.device)
        candidates = torch.from_numpy(candidate_rows).long().to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(inputs, sample=False)
            # _candidate_rows returns sorted unique rows, so matching the
            # catalog width means this is the identity selection. Reuse the
            # model output instead of advanced-indexing a full-size copy.
            selected_logits = (
                logits
                if candidate_rows.size == self._n_items
                else logits[:, candidates]
            )
            if exclude_seen and source.indices.size:
                candidate_to_local = np.full(source.shape[1], -1, dtype=np.int64)
                candidate_to_local[candidate_rows] = np.arange(candidate_rows.size)
                seen_counts = np.diff(source.indptr)
                seen_rows = np.repeat(
                    np.arange(source.shape[0], dtype=np.int64),
                    seen_counts,
                )
                seen_local = candidate_to_local[source.indices]
                included = seen_local >= 0
                selected_logits[
                    torch.as_tensor(
                        seen_rows[included], dtype=torch.long, device=self.device
                    ),
                    torch.as_tensor(
                        seen_local[included], dtype=torch.long, device=self.device
                    ),
                ] = -torch.inf
            values, local_columns = torch.topk(selected_logits, int(k), dim=1)
            columns = candidates[local_columns]
        return SRPTensor(cols=columns, vals=values, shape=source.shape)

    def _checkpoint_config(self) -> dict[str, Any]:
        config = asdict(self.cfg)
        config["device"] = str(self.device)
        return config

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> MultVAETrainer:
        state = reader.read_json("state/trainer.json")
        n_items = state.get("n_items")
        if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items < 1:
            raise ValueError("MultVAE n_items must be a positive integer")
        config = dict(config)
        config["device"] = str(device)
        trainer = cls(MultVAEConfig(**config))
        trainer._n_items = n_items
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self._n_items is not None
        writer.write_json(
            "state/trainer.json",
            {
                "n_items": self._n_items,
                "history": self.history,
                "updates": self._updates,
            },
        )

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        state = reader.read_json("state/trainer.json")
        history = state.get("history")
        updates = state.get("updates")
        if not isinstance(history, list):
            raise ValueError("MultVAE training history must be a list")
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("MultVAE update count must be a nonnegative integer")
        self.history = list(history)
        self._updates = updates

    def _finish_checkpoint_load(self) -> None:
        self._is_fitted = True

    def _build_checkpoint_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("MultVAE model must be built before its optimizer")
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )
