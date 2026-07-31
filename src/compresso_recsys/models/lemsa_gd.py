from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models._batching import (
    LeaveOneOutInteractionBatch,
    LeaveOneOutInteractionBatchSampler,
    SymmetricInteractionBatch,
    SymmetricInteractionBatchSampler,
    dense_training_target,
    normalized_mse,
)
from compresso_recsys.models.teaser_gd import (
    TEASERGD,
    TEASERGDTrainer,
    _feature_tensor,
    _score_feature_rows,
)

__all__ = ["LEMSAGD", "LEMSAGDConfig", "LEMSAGDTrainer"]

OptimizerName = Literal["NAdam", "AdamW"]
EncoderInit = Literal["xavier", "features"]
TrainingMode = Literal["leave_one_out", "symmetric"]


def _cross_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    source: torch.Tensor,
    source_positions: torch.Tensor,
) -> torch.Tensor:
    """Normalized reconstruction with observed source entries unscored."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if source.shape[0] != predictions.shape[0]:
        raise ValueError("source rows must match prediction rows")
    if source_positions.shape != (source.shape[1],):
        raise ValueError("source_positions must map every source column")
    if source_positions.numel() and (
        int(source_positions.min().item()) < 0
        or int(source_positions.max().item()) >= predictions.shape[1]
    ):
        raise ValueError("source_positions are out of prediction bounds")
    source_mask = torch.zeros_like(predictions, dtype=torch.bool)
    source_mask[:, source_positions] = source.bool()
    return normalized_mse(
        predictions.masked_fill(source_mask, 0),
        targets,
    )


class LEMSAGD(TEASERGD):
    """Unconstrained feature-decoder autoencoder for split-history training."""

    def forward(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor,
        candidate_features: torch.Tensor,
        source_features: torch.Tensor | None = None,
        source_candidate_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score candidates with ``x @ E @ S.T`` without diagonal correction."""
        if x.shape[1] != sources.numel():
            raise ValueError("x columns must match sources")
        if candidate_features.shape[1] != self.feature_dim:
            raise ValueError("candidate_features has an incompatible feature dimension")
        source_encoder = self.encoder_weights(sources)
        scores = _score_feature_rows(x @ source_encoder, candidate_features)
        return torch.nn.functional.relu(scores) if self.use_relu else scores

    def exact_coefficient_squared_norm(
        self,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the unconstrained ``||E S.T||_F^2`` for diagnostics."""
        if item_features.layout != torch.strided:
            item_features = item_features.to_dense()
        if item_features.shape != (self.input_dim, self.feature_dim):
            raise ValueError("item_features shape must match input_dim and feature_dim")
        coefficients = self.encoder_weights() @ item_features.T
        return coefficients.square().sum()


@dataclass(frozen=True)
class LEMSAGDConfig:
    """Configuration for symmetric split-history LEMSA training.

    ``training_mode="leave_one_out"`` visits every eligible interaction once
    per epoch, removes it from its user history, and predicts it as a one-hot
    target. ``"symmetric"`` randomly divides histories into two non-empty
    views and optimizes both directions. Active source entries are excluded
    from each loss, so the encoder-decoder diagonal is neither subtracted nor
    penalized.
    """

    batch_size: int = 1024
    max_output: int | None = None
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    l2_encoder: float = 0.0
    training_mode: TrainingMode = "leave_one_out"
    split_probability: float = 0.5
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    use_relu: bool = True
    include_popularity: bool = False
    optimizer: OptimizerName = "NAdam"
    encoder_init: EncoderInit = "xavier"
    normalize_encoder: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.max_output is not None and (
            isinstance(self.max_output, bool)
            or not isinstance(self.max_output, int)
            or self.max_output < 1
        ):
            raise ValueError("max_output must be >= 1 or None")
        for name in ("lr", "weight_decay", "l2_encoder"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                relation = "> 0" if name == "lr" else ">= 0"
                raise ValueError(f"{name} must be finite and {relation}")
        if not np.isfinite(self.split_probability) or not (
            0 < self.split_probability < 1
        ):
            raise ValueError("split_probability must be finite and in (0, 1)")
        if self.training_mode not in {"leave_one_out", "symmetric"}:
            raise ValueError(
                "training_mode must be 'leave_one_out' or 'symmetric'"
            )
        if self.optimizer not in {"NAdam", "AdamW"}:
            raise ValueError("optimizer must be 'NAdam' or 'AdamW'")
        if self.encoder_init not in {"xavier", "features"}:
            raise ValueError("encoder_init must be 'xavier' or 'features'")
        for name in (
            "decay",
            "compile",
            "show_progress",
            "use_relu",
            "include_popularity",
            "normalize_encoder",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")


class LEMSAGDTrainer(TEASERGDTrainer):
    """Fit an unconstrained encoder with symmetric held-out reconstruction."""

    _fit_name = "LEMSAGD"

    def __init__(self, config: LEMSAGDConfig | None = None) -> None:
        self.cfg = config if config is not None else LEMSAGDConfig()
        super().__init__(self.cfg)  # type: ignore[arg-type]

    def _build_base_model(self, *, input_dim: int, feature_dim: int) -> LEMSAGD:
        return LEMSAGD(
            input_dim=input_dim,
            feature_dim=feature_dim,
            use_relu=self.cfg.use_relu,
            normalize_encoder=self.cfg.normalize_encoder,
        )

    def _build_batch_sampler(
        self,
        interactions: csr_matrix,
    ) -> SymmetricInteractionBatchSampler | LeaveOneOutInteractionBatchSampler:
        if self.cfg.training_mode == "leave_one_out":
            return LeaveOneOutInteractionBatchSampler(
                interactions,
                device=self.device,
                batch_size=self.cfg.batch_size,
                shuffle=True,
                max_output=self.cfg.max_output,
                seed=self.cfg.seed,
            )
        return SymmetricInteractionBatchSampler(
            interactions,
            device=self.device,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            max_output=self.cfg.max_output,
            seed=self.cfg.seed,
            split_probability=self.cfg.split_probability,
        )

    def _empty_epoch_sums(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "reconstruction": 0.0,
            "encoder_l2": 0.0,
        }

    def _train_step(
        self,
        batch: SymmetricInteractionBatch | LeaveOneOutInteractionBatch,
        training_features: csr_matrix | np.ndarray,
    ) -> dict[str, float]:
        if isinstance(batch, LeaveOneOutInteractionBatch):
            return self._leave_one_out_train_step(batch, training_features)
        return self._symmetric_train_step(batch, training_features)

    def _candidate_features(
        self,
        candidates: torch.Tensor | None,
        training_features: csr_matrix | np.ndarray,
    ) -> torch.Tensor:
        full_features = self._training_tensor()
        if candidates is None:
            return full_features
        if full_features.layout == torch.strided:
            return full_features[candidates]
        candidate_rows = candidates.detach().cpu().numpy()
        return _feature_tensor(
            training_features[candidate_rows],
            device=self.device,
        )

    @staticmethod
    def _source_positions(
        sources: torch.Tensor,
        candidates: torch.Tensor | None,
    ) -> torch.Tensor:
        if candidates is None:
            return sources
        return torch.arange(sources.numel(), device=sources.device)

    def _finish_train_step(self, reconstruction: torch.Tensor) -> dict[str, float]:
        assert self.optimizer is not None
        encoder_l2 = self._base_model.encoder.square().mean()
        loss = reconstruction + float(self.cfg.l2_encoder) * encoder_l2
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "encoder_l2": float(encoder_l2.detach().cpu()),
        }

    def _symmetric_train_step(
        self,
        batch: SymmetricInteractionBatch,
        training_features: csr_matrix | np.ndarray,
    ) -> dict[str, float]:
        assert self.teaser is not None and self.optimizer is not None
        x = batch.x.to_dense()
        y = batch.y.to_dense()
        x_targets = dense_training_target(
            x,
            sources=batch.sources,
            candidates=batch.candidates,
            input_dim=training_features.shape[0],
        )
        y_targets = dense_training_target(
            y,
            sources=batch.sources,
            candidates=batch.candidates,
            input_dim=training_features.shape[0],
        )
        candidate_features = self._candidate_features(
            batch.candidates,
            training_features,
        )
        source_positions = self._source_positions(
            batch.sources,
            batch.candidates,
        )

        self.teaser.train()
        self.optimizer.zero_grad(set_to_none=True)
        x_predictions = self.teaser(
            x,
            sources=batch.sources,
            candidate_features=candidate_features,
        )
        y_predictions = self.teaser(
            y,
            sources=batch.sources,
            candidate_features=candidate_features,
        )
        reconstruction = 0.5 * (
            _cross_reconstruction_loss(
                x_predictions,
                y_targets,
                source=x,
                source_positions=source_positions,
            )
            + _cross_reconstruction_loss(
                y_predictions,
                x_targets,
                source=y,
                source_positions=source_positions,
            )
        )
        return self._finish_train_step(reconstruction)

    def _leave_one_out_train_step(
        self,
        batch: LeaveOneOutInteractionBatch,
        training_features: csr_matrix | np.ndarray,
    ) -> dict[str, float]:
        assert self.teaser is not None and self.optimizer is not None
        x = batch.x.to_dense()
        candidate_features = self._candidate_features(
            batch.candidates,
            training_features,
        )
        targets = x.new_zeros((x.shape[0], candidate_features.shape[0]))
        target_rows = torch.arange(x.shape[0], device=self.device)
        targets[target_rows, batch.target_positions.long()] = 1
        source_positions = self._source_positions(
            batch.sources,
            batch.candidates,
        )

        self.teaser.train()
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.teaser(
            x,
            sources=batch.sources,
            candidate_features=candidate_features,
        )
        reconstruction = _cross_reconstruction_loss(
            predictions,
            targets,
            source=x,
            source_positions=source_positions,
        )
        return self._finish_train_step(reconstruction)
