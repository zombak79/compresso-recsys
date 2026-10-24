"""The training procedure, and the fitted model's prediction path."""

from __future__ import annotations

import time
from typing import Any, Hashable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch import nn

from compresso import SRPParam, SRPTensor, SparsityController
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.models.core.batching import (
    InteractionBatchSampler,
)
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.base import BaseCollaborativeRecommender
from compresso_recsys.models.elsa.config import (
    ELSACompressionConfig,
    ELSAConfig,
    SparseInferenceBackend,
    _dense_training_target,
    _normalized_mse,
)
from compresso_recsys.models.elsa.model import (
    CompressedELSA,
    ELSA,
)


class _ELSAInteractionDataset(InteractionBatchSampler):
    """Backward-compatible alias for the shared interaction batch sampler."""

    def __getitem__(
        self,
        batch_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        batch = super().__getitem__(batch_index)
        return batch.x, batch.sources, batch.candidates


class ELSATrainer(BaseCollaborativeRecommender):
    """Fit and run ELSA with sparse interaction matrices."""

    checkpoint_type = "elsa_trainer"

    def __init__(
        self,
        config: ELSAConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config if config is not None else ELSAConfig()
        self.logger = logger
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

    @property
    def n_items(self) -> int | None:
        """Number of fitted item columns, or ``None`` before building."""
        return self.input_dim

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
            torch.optim.AdamW if self.cfg.optimizer == "AdamW" else torch.optim.NAdam
        )
        self.optimizer = optimizer_class(
            self.elsa.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> ELSATrainer:
        config = dict(config)
        compression_state = config.get("compression")
        if compression_state is not None:
            compression_state = dict(compression_state)
            schedule = compression_state.get("k_schedule")
            if schedule is not None:
                compression_state["k_schedule"] = tuple(schedule)
            config["compression"] = ELSACompressionConfig(**compression_state)
        config["device"] = str(device)
        # Compilation is runtime state; checkpoints always rebuild an eager model.
        config["compile"] = False
        trainer = cls(ELSAConfig(**config))
        state = reader.read_json("state/trainer.json")
        input_dim = int(state["input_dim"])
        trainer.input_dim = input_dim
        if trainer.cfg.compression is None:
            trainer.elsa = ELSA(
                input_dim=input_dim,
                latent_dim=trainer.cfg.latent_dim,
                use_relu=trainer.cfg.use_relu,
            ).to(device)
        else:
            trainer.elsa = CompressedELSA(
                input_dim=input_dim,
                latent_dim=trainer.cfg.latent_dim,
                compression=trainer.cfg.compression,
                use_relu=trainer.cfg.use_relu,
            ).to(device)
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.elsa

    def _prepare_checkpoint_module_state(self, state: dict[str, object]) -> None:
        if not isinstance(self.elsa, CompressedELSA):
            return
        columns = state.get("sparse_A.cols")
        values = state.get("sparse_A.values")
        if not isinstance(columns, torch.Tensor) or not isinstance(
            values, torch.Tensor
        ):
            raise ValueError(
                "fitted compressed ELSA checkpoint is missing sparse factors"
            )
        self.elsa.masked_A = None
        self.elsa.sparse_A = SRPParam(
            cols=columns,
            values=torch.zeros_like(values),
            shape=(self.elsa.input_dim, self.elsa.latent_dim),
        ).to(self.device)
        self.elsa.phase = "sparse_finetune"

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.input_dim is not None
        writer.write_json(
            "state/trainer.json",
            {
                "input_dim": self.input_dim,
                "history": self.history,
            },
        )

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        state = reader.read_json("state/trainer.json")
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("ELSA training history must be a list")
        self.history = list(history)

    def _build_checkpoint_optimizer(self) -> None:
        self._reset_optimizer()

    def _finish_checkpoint_load(self) -> None:
        self.sparsity_controller = None
        self._last_controller_info = {}
        self._is_fitted = True
        if isinstance(self.elsa, CompressedELSA):
            self.elsa.prepare_inference()

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
        cosine_loss = (
            1.0
            - F.cosine_similarity(
                predictions,
                y,
                dim=-1,
            ).mean()
        )
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
        reporter: _Reporter,
        started: float,
        bar=None,
    ) -> tuple[dict[str, float], bool]:
        """Run one epoch, reporting into a caller-owned bar when given.

        The bar belongs to the caller so that one bar is rewound and relabelled
        per epoch, rather than a finished bar being left behind for each.
        """
        sums: dict[str, float] = {}
        n_batches = 0
        rewind_triggered = False
        if bar is not None:
            bar.reset(total=len(dataset))
            bar.set_description(desc)
        for batch_index in range(len(dataset)):
            stats = self.train_step(*dataset[batch_index])
            for key, value in stats.items():
                sums[key] = sums.get(key, 0.0) + float(value.cpu().item())
            n_batches += 1
            if bar is not None:
                bar.update(1)
            log_steps = reporter.log_every_n_steps
            if log_steps and n_batches % log_steps == 0:
                reporter.step(
                    f"{desc} step {n_batches}/{len(dataset)}",
                    n_batches,
                    len(dataset),
                    started,
                    {
                        key: value / n_batches
                        for key, value in sums.items()
                    },
                )
            if bool(self._last_controller_info.get("rewind_triggered", False)):
                rewind_triggered = True
                break
        dataset.on_epoch_end()
        return (
            {key: value / max(1, n_batches) for key, value in sums.items()},
            rewind_triggered,
        )

    def _fit_fixed_epochs(
        self,
        dataset: _ELSAInteractionDataset,
        *,
        phase: str | None,
        reporter: _Reporter,
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
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="ELSA fit" if phase is None else "ELSA sparse fine-tune",
        )
        batch_bar = reporter.bar(total=len(dataset), desc="ELSA epoch 1")
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                record: dict[str, float | str] = self._run_epoch(
                    dataset,
                    desc=f"ELSA epoch {epoch}",
                    reporter=reporter,
                    started=epoch_started,
                    bar=batch_bar,
                )[0]
                record["epoch"] = float(epoch)
                record["lr"] = self._current_lr()
                if phase is not None:
                    record["phase"] = phase
                self.history.append(record)
                reporter.epoch(
                    f"epoch {epoch}/{self.cfg.epochs}",
                    record,
                    epoch_started,
                )
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
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epoch_iter, "close"):
                epoch_iter.close()

    def _fit_compressed_mask_search(
        self,
        dataset: _ELSAInteractionDataset,
        *,
        reporter: _Reporter,
    ) -> CompressedELSA:
        if not isinstance(self.elsa, CompressedELSA):
            raise RuntimeError("compressed ELSA model was not built")
        search_epoch = 0
        stage_epoch = 0
        # Mask search runs an unbounded number of epochs, so one reused bar
        # matters even more here than during fixed-epoch training.
        batch_bar = reporter.bar(
            total=len(dataset),
            desc="ELSA mask stage 1 epoch 1",
        )
        try:
            while True:
                masked_A = self.elsa.masked_A
                if masked_A is None:  # pragma: no cover - defensive invariant
                    raise RuntimeError("mask-search parameter is unavailable")
                search_epoch += 1
                stage_epoch += 1
                stage_idx = int(masked_A.stage_idx)
                k_current = int(masked_A.k_current)
                epoch_started = time.monotonic()
                epoch_desc = f"ELSA mask stage {stage_idx + 1} epoch {stage_epoch}"
                record: dict[str, float | str] = self._run_epoch(
                    dataset,
                    desc=epoch_desc,
                    reporter=reporter,
                    started=epoch_started,
                    bar=batch_bar,
                )[0]
                rewind_triggered = bool(
                    self._last_controller_info.get("rewind_triggered", False)
                )
                transition = "stable" if rewind_triggered else "none"
                max_stage_epochs = (
                    self.cfg.compression.max_epochs_per_stage
                    if self.cfg.compression is not None
                    else None
                )
                if (
                    not rewind_triggered
                    and max_stage_epochs is not None
                    and stage_epoch >= max_stage_epochs
                ):
                    # Temporary compatibility path until Compresso exposes a
                    # public forced-stage transition on SparsityController.
                    masked_A.stage_completed = True
                    rewind_stats = masked_A.rewind()
                    message = (
                        "Forced rewind "
                        f"(max_epochs_per_stage={max_stage_epochs}): "
                        f"{rewind_stats}"
                    )
                    if reporter.logger is not None:
                        reporter.log(message)
                    elif reporter.allow_stdout_fallback:
                        print(f"[ELSATrainer] {message}")
                    if self.sparsity_controller is not None:
                        self.sparsity_controller.num_restarts += 1
                    rewind_triggered = True
                    transition = "forced"
                record.update(
                    {
                        "epoch": float(search_epoch),
                        "stage_epoch": float(stage_epoch),
                        "stage": float(stage_idx),
                        "k": float(k_current),
                        "mask_change": float(masked_A.last_change),
                        "lr": self._current_lr(),
                        "phase": "mask_search",
                        "transition": transition,
                    }
                )
                self.history.append(record)
                reporter.epoch(epoch_desc, record, epoch_started)
                if not rewind_triggered:
                    continue

                schedule_done = bool(masked_A.schedule_done)
                self._reset_optimizer()
                self._set_lr(float(self.cfg.lr))
                stage_epoch = 0
                if schedule_done:
                    break
        finally:
            if batch_bar is not None:
                batch_bar.close()

        self.elsa.convert_to_srp()
        self.sparsity_controller = None
        self._last_controller_info = {}
        self._reset_optimizer()
        return self.elsa

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> ELSATrainer:
        """Fit dense ELSA or search and fine-tune a compressed ELSA ticket."""
        reporter = self._reporter(logger, show_progress)
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
        vocabulary = self._prepare_item_vocabulary(
            item_ids,
            n_items=int(interactions.shape[1]),
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
        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{interactions.shape[0]} users | {interactions.shape[1]} items | "
            f"{interactions.nnz} interactions | {len(dataset)} batches of "
            f"{self.cfg.batch_size} | {self.cfg.epochs} epochs | device {self.device}"
        )
        if self.cfg.compression is None:
            self._fit_fixed_epochs(dataset, phase=None, reporter=reporter)
        else:
            if not isinstance(self.elsa, CompressedELSA):
                raise RuntimeError("compressed ELSA model was not built")
            model = self.elsa
            if model.is_sparse:
                model.train()
                self._reset_optimizer()
            else:
                model = self._fit_compressed_mask_search(
                    dataset,
                    reporter=reporter,
                )
            self._fit_fixed_epochs(
                dataset,
                phase="sparse_finetune",
                reporter=reporter,
            )
            model.prepare_inference()
        assert self.input_dim is not None
        self._publish_item_vocabulary(vocabulary)
        self._is_fitted = True
        reporter.log(
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} total | "
            f"{len(self.history)} epochs recorded"
        )
        return self

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if not self.is_fitted or self.elsa is None or self.input_dim is None:
            raise RuntimeError("ELSATrainer must be fitted before prediction")
        source = canonical_csr(source, name="source")
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
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        sparse_inference_backend: SparseInferenceBackend | None = None,
    ) -> SRPTensor:
        """Predict ranked items for one source batch.

        Seen source items are excluded unless ``exclude_seen`` is false. For
        compressed ELSA, ``sparse_inference_backend`` overrides the configured
        inference backend for this call.
        """
        source = self._prepare_source(source)
        assert self.elsa is not None and self.input_dim is not None
        if sparse_inference_backend is not None and not isinstance(
            self.elsa, CompressedELSA
        ):
            raise ValueError(
                "sparse_inference_backend is only available for compressed ELSA"
            )
        candidate_rows = self._candidate_rows(candidate_ids)
        candidate_count = int(candidate_rows.size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")
        if exclude_seen:
            selected = np.zeros(self.input_dim, dtype=bool)
            selected[candidate_rows] = True
            seen_counts = np.diff(source.indptr)
            seen_rows = np.repeat(
                np.arange(source.shape[0], dtype=np.int64),
                seen_counts,
            )
            selected_seen = selected[source.indices]
            selected_seen_counts = np.bincount(
                seen_rows[selected_seen],
                minlength=source.shape[0],
            )
            unseen_counts = candidate_count - selected_seen_counts
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
                backend=sparse_inference_backend,
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
        candidate_tensor = torch.from_numpy(candidate_rows).long().to(self.device)
        local = SRPTensor.from_dense(
            scores[:, candidate_tensor],
            k=int(k),
            score_mode="raw",
        )
        return SRPTensor(
            cols=candidate_tensor[local.cols],
            vals=local.vals,
            shape=source.shape,
        )

    @torch.no_grad()
    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int | None = None,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
        sparse_inference_backend: SparseInferenceBackend | None = None,
    ) -> SRPTensor:
        """Predict ranked items for all source rows in batches.

        Each batch delegates to :meth:`predict_on_batch`. Seen source items
        are excluded unless ``exclude_seen`` is false. For compressed ELSA,
        ``sparse_inference_backend`` overrides the configured inference backend
        for every batch.
        """
        source = self._prepare_source(source)
        resolved_batch_size = (
            self.cfg.batch_size if batch_size is None else int(batch_size)
        )
        if resolved_batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        candidate_count = int(self._candidate_rows(candidate_ids).size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")
        reporter = self._reporter(logger, show_progress)

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], resolved_batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.shape[0]} rows | "
            f"{steps} batches of {resolved_batch_size} | device {self.device}"
        )
        for step, start in enumerate(
            reporter.wrap(starts, total=steps, desc=f"ELSA predict@{k}"),
            start=1,
        ):
            end = min(start + resolved_batch_size, source.shape[0])
            predictions = self.predict_on_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
                candidate_ids=candidate_ids,
                sparse_inference_backend=sparse_inference_backend,
            )
            columns.append(predictions.cols)
            values.append(predictions.vals)
            log_steps = reporter.log_every_n_steps
            if log_steps and step % log_steps == 0:
                reporter.step(
                    f"predict@{k} step {step}/{steps}",
                    step,
                    steps,
                    started,
                )

        if not columns:
            result = self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                candidate_ids=candidate_ids,
                sparse_inference_backend=sparse_inference_backend,
            )
        else:
            result = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=source.shape,
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.shape[0]} rows"
        )
        return result
