"""The training procedure, and the fitted model's prediction path."""

from __future__ import annotations

from dataclasses import asdict
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
    TrainingProgress,
)
from compresso_recsys.models.core.autoencoder_batching import (
    dense_training_batch,
    prepare_dense_training_data,
)
from compresso_recsys.models.core.ranking import validate_candidate_topk
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.base import BaseCollaborativeRecommender
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.models.mult_vae.config import (
    MultVAEConfig,
)
from compresso_recsys.models.mult_vae.model import (
    MultVAE,
)


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
        with TrainingProgress(
            reporter,
            label="MultVAE",
            epochs=int(self.cfg.epochs),
            batches=steps_per_epoch,
        ) as progress:
            progress.start(
                f"{active_rows.size} active users | {interactions.shape[1]} items | "
                f"{interactions.nnz} interactions | {steps_per_epoch} batches of "
                f"{self.cfg.batch_size} | {self.cfg.epochs} epochs | "
                f"device {self.device}"
            )
            # Inside the block, so preloading counts toward the reported total
            # exactly as it did when fit timed itself.
            training_data = prepare_dense_training_data(
                interactions,
                device=self.device,
                preload=self.cfg.preload_training_data,
            )
            self.training_data_preloaded_ = training_data is not None
            for epoch in progress.epochs():
                self.model.train()
                order = rng.permutation(active_rows)
                loss_sum = torch.zeros((), device=self.device)
                reconstruction_sum = torch.zeros((), device=self.device)
                kl_sum = torch.zeros((), device=self.device)
                users = 0
                last_kl_weight = self._kl_weight()

                def running() -> dict[str, float]:
                    """Stack the three sums into one sync, only when logging."""
                    mean_loss, mean_reconstruction, mean_kl = torch.stack(
                        (
                            loss_sum / users,
                            reconstruction_sum / users,
                            kl_sum / users,
                        )
                    ).tolist()
                    return {
                        "loss": mean_loss,
                        "reconstruction_loss": mean_reconstruction,
                        "kl_loss": mean_kl,
                        "kl_weight": last_kl_weight,
                    }

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
                    progress.batch(step, metrics=running)
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
                progress.epoch_done(record, postfix=("loss", "kl_weight"))
            self._is_fitted = True
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
