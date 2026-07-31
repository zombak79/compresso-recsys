from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models._batching import (
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


def _cross_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    source: torch.Tensor,
) -> torch.Tensor:
    """Normalized reconstruction with observed source entries unscored."""
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if source.shape[0] != predictions.shape[0] or source.shape[1] > predictions.shape[1]:
        raise ValueError("source must be a prefix-shaped view of predictions")
    source_mask = torch.nn.functional.pad(
        source.bool(),
        (0, predictions.shape[1] - source.shape[1]),
    )
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

    Each eligible user history is randomly divided into two non-empty views
    every epoch. Training minimizes both ``x -> y`` and ``y -> x`` normalized
    reconstruction losses. Active source entries are excluded from each loss,
    so the encoder-decoder diagonal is neither subtracted nor penalized.
    """

    batch_size: int = 1024
    max_output: int | None = None
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    l2_encoder: float = 0.0
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
    ) -> SymmetricInteractionBatchSampler:
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
        full_features = self._training_tensor()
        if batch.candidates is None:
            candidate_features = full_features
        elif full_features.layout == torch.strided:
            candidate_features = full_features[batch.candidates]
        else:
            candidate_rows = batch.candidates.detach().cpu().numpy()
            candidate_features = _feature_tensor(
                training_features[candidate_rows],
                device=self.device,
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
            )
            + _cross_reconstruction_loss(
                y_predictions,
                x_targets,
                source=y,
            )
        )
        encoder_l2 = self._base_model.encoder.square().mean()
        loss = reconstruction + float(self.cfg.l2_encoder) * encoder_l2
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "encoder_l2": float(encoder_l2.detach().cpu()),
        }

