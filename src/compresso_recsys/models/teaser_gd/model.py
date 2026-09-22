"""The architecture: modules holding the learned parameters."""

from __future__ import annotations

from __future__ import annotations


import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



def _score_feature_rows(
    profiles: torch.Tensor,
    candidate_features: torch.Tensor,
) -> torch.Tensor:
    if candidate_features.layout == torch.strided:
        return profiles @ candidate_features.T
    return torch.sparse.mm(candidate_features, profiles.T).T


class TEASERGD(nn.Module):
    """Trainable TEASER encoder with a fixed feature decoder.

    The model represents the coefficient matrix as ``E @ S.T`` without
    constructing it. ``source_candidate_positions`` identifies each source
    item's position in the candidate output so ``diagonal_scale`` can remove
    all or part of the diagonal contribution ``(E * S).sum(-1)`` exactly.
    """

    def __init__(
        self,
        input_dim: int,
        feature_dim: int,
        *,
        use_relu: bool = True,
        normalize_encoder: bool = False,
        diagonal_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if feature_dim < 1:
            raise ValueError("feature_dim must be >= 1")
        if (
            isinstance(diagonal_scale, bool)
            or not np.isfinite(diagonal_scale)
            or not 0 <= diagonal_scale <= 1
        ):
            raise ValueError("diagonal_scale must be finite and in [0, 1]")
        self.input_dim = int(input_dim)
        self.feature_dim = int(feature_dim)
        self.use_relu = bool(use_relu)
        self.normalize_encoder = bool(normalize_encoder)
        self.diagonal_scale = float(diagonal_scale)
        self.encoder = nn.Parameter(torch.empty(self.input_dim, self.feature_dim))
        nn.init.xavier_uniform_(self.encoder)

    def encoder_weights(
        self,
        rows: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return effective encoder rows, optionally normalized as in ELSA."""
        weights = self.encoder if rows is None else self.encoder[rows]
        if self.normalize_encoder:
            return F.normalize(weights, p=2.0, dim=-1)
        return weights

    def forward(
        self,
        x: torch.Tensor,
        *,
        sources: torch.Tensor,
        source_features: torch.Tensor,
        candidate_features: torch.Tensor,
        source_candidate_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Score candidates and scale represented self-coefficient removal."""
        if x.shape[1] != sources.numel():
            raise ValueError("x columns must match sources")
        if source_features.shape != (sources.numel(), self.feature_dim):
            raise ValueError("source_features shape must match sources and feature_dim")
        if candidate_features.shape[1] != self.feature_dim:
            raise ValueError("candidate_features has an incompatible feature dimension")
        if source_candidate_positions.shape != sources.shape:
            raise ValueError("source_candidate_positions must match sources")

        source_encoder = self.encoder_weights(sources)
        profiles = x @ source_encoder
        scores = _score_feature_rows(profiles, candidate_features)
        diagonal = (source_encoder * source_features).sum(dim=-1)
        valid = source_candidate_positions >= 0
        positions = source_candidate_positions[valid]
        correction = -self.diagonal_scale * (x[:, valid] * diagonal[valid])
        scores = scores.scatter_add(
            1,
            positions.expand(x.shape[0], -1),
            correction,
        )
        return F.relu(scores) if self.use_relu else scores

    def exact_coefficient_squared_norm(
        self,
        item_features: torch.Tensor,
    ) -> torch.Tensor:
        """Return the exact effective coefficient squared norm."""
        if item_features.layout != torch.strided:
            item_features = item_features.to_dense()
        if item_features.shape != (self.input_dim, self.feature_dim):
            raise ValueError("item_features shape must match input_dim and feature_dim")
        encoder = self.encoder_weights()
        coefficients = encoder @ item_features.T
        diagonal = (encoder * item_features).sum(dim=-1)
        removed_fraction = self.diagonal_scale * (2.0 - self.diagonal_scale)
        return coefficients.square().sum() - removed_fraction * diagonal.square().sum()
