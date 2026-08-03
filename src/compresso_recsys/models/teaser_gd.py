from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix, isspmatrix_csr
from torch import nn

from compresso import SRPTensor
from compresso_recsys.models._batching import (
    InteractionBatchSampler,
    dense_training_target,
    normalized_mse,
)
from compresso_recsys.models._validation import (
    canonical_csr,
    canonical_train_item_indices,
)
from compresso_recsys.models.cold_start import (
    BaseColdStartRecommender,
    CandidateCatalog,
    CandidateSelection,
    ItemFeatures,
    append_column,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_item_ids,
    canonical_metadata,
    take_features,
)

__all__ = ["TEASERGD", "TEASERGDConfig", "TEASERGDTrainer"]

OptimizerName = Literal["NAdam", "AdamW"]
TEASERGDLoss = Literal["normalized_mse", "teaser"]
EncoderInit = Literal["xavier", "features"]


def _teaser_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    n_source_items: int,
    n_items: int,
) -> torch.Tensor:
    """Estimate the original TEASER Frobenius reconstruction per user."""
    squared_error = (predictions - targets).square()
    n_sampled_negatives = predictions.shape[1] - n_source_items
    n_available_negatives = n_items - n_source_items
    if 0 < n_sampled_negatives < n_available_negatives:
        negative_weight = n_available_negatives / n_sampled_negatives
        squared_error = torch.cat(
            (
                squared_error[:, :n_source_items],
                squared_error[:, n_source_items:] * negative_weight,
            ),
            dim=1,
        )
    return squared_error.sum(dim=-1).mean()


def _dense_feature_rows(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    selected = features[rows]
    array = selected.toarray() if isspmatrix_csr(selected) else np.asarray(selected)
    return torch.from_numpy(np.array(array, dtype=np.float32, order="C", copy=True)).to(
        device
    )


def _feature_tensor(
    features: csr_matrix | np.ndarray,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not isspmatrix_csr(features):
        return torch.from_numpy(
            np.array(features, dtype=np.float32, order="C", copy=True)
        ).to(device)

    coo = features.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row.astype(np.int64), coo.col.astype(np.int64)))
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=True))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Sparse invariant checks are implicitly disabled.*",
            category=UserWarning,
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            coo.shape,
            is_coalesced=True,
            check_invariants=False,
        ).to(device)


def _score_feature_rows(
    profiles: torch.Tensor,
    candidate_features: torch.Tensor,
) -> torch.Tensor:
    if candidate_features.layout == torch.strided:
        return profiles @ candidate_features.T
    return torch.sparse.mm(candidate_features, profiles.T).T


@torch.no_grad()
def _initialize_encoder_from_features(
    model: TEASERGD,
    features: csr_matrix | np.ndarray,
) -> None:
    """Copy dense or sparse fixed features into the trainable encoder."""
    if features.shape != (model.input_dim, model.feature_dim):
        raise ValueError("features shape must match input_dim and feature_dim")
    if not isspmatrix_csr(features):
        model.encoder.copy_(
            torch.from_numpy(
                np.array(features, dtype=np.float32, order="C", copy=True)
            ).to(model.encoder.device)
        )
        return

    coo = features.tocoo()
    model.encoder.zero_()
    rows = torch.from_numpy(coo.row.astype(np.int64, copy=False)).to(
        model.encoder.device
    )
    columns = torch.from_numpy(coo.col.astype(np.int64, copy=False)).to(
        model.encoder.device
    )
    values = torch.from_numpy(coo.data.astype(np.float32, copy=True)).to(
        model.encoder.device
    )
    model.encoder[rows, columns] = values


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


@dataclass(frozen=True)
class TEASERGDConfig:
    """Configuration for gradient-trained TEASER.

    ``loss="normalized_mse"`` preserves the ELSA-style objective, while
    ``loss="teaser"`` uses the original TEASER Frobenius reconstruction and
    regularization scale. ``max_output`` uses the same source-prefix candidate
    sampling as ELSA. In TEASER mode, sampled negatives are importance-weighted
    to estimate full-output reconstruction.
    ``coefficient_regularization_samples`` controls a Monte Carlo estimate of
    the effective coefficient norm; zero disables that term.
    """

    batch_size: int = 1024
    max_output: int | None = None
    epochs: int = 1
    lr: float = 1e-3
    weight_decay: float = 0.0
    l2_coefficients: float = 0.05
    l2_encoder: float = 0.05
    coefficient_regularization_samples: int = 4096
    decay: bool = False
    compile: bool = False
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    use_relu: bool = True
    include_popularity: bool = True
    optimizer: OptimizerName = "NAdam"
    loss: TEASERGDLoss = "normalized_mse"
    encoder_init: EncoderInit = "xavier"
    normalize_encoder: bool = False
    diagonal_scale: float = 1.0

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
        if (
            isinstance(self.coefficient_regularization_samples, bool)
            or not isinstance(self.coefficient_regularization_samples, int)
            or self.coefficient_regularization_samples < 0
        ):
            raise ValueError("coefficient_regularization_samples must be >= 0")
        for name in (
            "lr",
            "weight_decay",
            "l2_coefficients",
            "l2_encoder",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0 or (name == "lr" and value == 0):
                relation = "> 0" if name == "lr" else ">= 0"
                raise ValueError(f"{name} must be finite and {relation}")
        if (
            isinstance(self.diagonal_scale, bool)
            or not np.isfinite(self.diagonal_scale)
            or not 0 <= self.diagonal_scale <= 1
        ):
            raise ValueError("diagonal_scale must be finite and in [0, 1]")
        if self.optimizer not in {"NAdam", "AdamW"}:
            raise ValueError("optimizer must be 'NAdam' or 'AdamW'")
        if self.loss not in {"normalized_mse", "teaser"}:
            raise ValueError("loss must be 'normalized_mse' or 'teaser'")
        if self.encoder_init not in {"xavier", "features"}:
            raise ValueError("encoder_init must be 'xavier' or 'features'")
        for name in (
            "decay",
            "compile",
            "show_progress",
            "use_relu",
            "normalize_encoder",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")
        if not isinstance(self.include_popularity, bool):
            raise ValueError("include_popularity must be a bool")


class TEASERGDTrainer(BaseColdStartRecommender):
    """Fit TEASER with PyTorch, sampled outputs, and cold candidate catalogs."""

    _fit_name = "TEASERGD"

    def __init__(self, config: TEASERGDConfig | None = None) -> None:
        super().__init__()
        self.cfg = config if config is not None else TEASERGDConfig()
        self.device = torch.device(self.cfg.device)
        self.teaser: TEASERGD | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, float]] = []
        self.decoder_features_: csr_matrix | np.ndarray | None = None
        self.train_item_indices_: np.ndarray | None = None
        self.train_item_mask_: np.ndarray | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.n_items_: int | None = None
        self.n_features_: int | None = None
        self._training_features: csr_matrix | np.ndarray | None = None
        self._candidate_tensor_cache: tuple[int, torch.Tensor] | None = None
        self._training_tensor_cache: torch.Tensor | None = None
        self._regularizer_rng: np.random.Generator | None = None
        self._n_training_users: int | None = None
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _on_catalog_published(self, catalog: CandidateCatalog) -> None:
        self.decoder_features_ = catalog.item_features
        self._candidate_tensor_cache = None

    def _reset_optimizer(self) -> None:
        if self.teaser is None:
            raise RuntimeError("trainer must be built before creating optimizer")
        optimizer_class = (
            torch.optim.AdamW if self.cfg.optimizer == "AdamW" else torch.optim.NAdam
        )
        self.optimizer = optimizer_class(
            self.teaser.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _build_base_model(self, *, input_dim: int, feature_dim: int) -> TEASERGD:
        return TEASERGD(
            input_dim=input_dim,
            feature_dim=feature_dim,
            use_relu=self.cfg.use_relu,
            normalize_encoder=self.cfg.normalize_encoder,
            diagonal_scale=self.cfg.diagonal_scale,
        )

    def _build_batch_sampler(
        self,
        interactions: csr_matrix,
    ) -> InteractionBatchSampler:
        return InteractionBatchSampler(
            interactions,
            device=self.device,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            max_output=self.cfg.max_output,
            seed=self.cfg.seed,
        )

    def _empty_epoch_sums(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "reconstruction": 0.0,
            "coefficient_l2": 0.0,
            "encoder_l2": 0.0,
        }

    def _coefficient_penalty(self) -> torch.Tensor:
        assert self._training_features is not None
        assert self._regularizer_rng is not None
        model = self._base_model
        n_items = self._training_features.shape[0]
        n_samples = self.cfg.coefficient_regularization_samples
        if n_samples == 0 or n_items < 2 or self.cfg.l2_coefficients == 0:
            return model.encoder.new_zeros(())
        left = self._regularizer_rng.integers(0, n_items, size=n_samples)
        right = self._regularizer_rng.integers(0, n_items - 1, size=n_samples)
        right += right >= left
        left_tensor = torch.from_numpy(left).long().to(self.device)
        right_features = self._training_feature_rows(right)
        coefficients = (model.encoder_weights(left_tensor) * right_features).sum(-1)
        penalty = coefficients.square().mean()
        residual_diagonal_scale = 1.0 - model.diagonal_scale
        diagonal_penalty = model.encoder.new_zeros(())
        if residual_diagonal_scale:
            left_features = self._training_feature_rows(left)
            diagonal = (
                model.encoder_weights(left_tensor) * left_features
            ).sum(-1)
            diagonal_penalty = (
                residual_diagonal_scale**2 * diagonal.square().mean()
            )
        if self.cfg.loss == "teaser":
            assert self._n_training_users is not None
            penalty = penalty * (n_items * (n_items - 1) / self._n_training_users)
            penalty = penalty + (
                diagonal_penalty * n_items / self._n_training_users
            )
        else:
            penalty = penalty + diagonal_penalty / (n_items - 1)
        return penalty

    def _training_tensor(self) -> torch.Tensor:
        assert self._training_features is not None
        if self._training_tensor_cache is None:
            self._training_tensor_cache = _feature_tensor(
                self._training_features,
                device=self.device,
            )
        return self._training_tensor_cache

    def _training_feature_rows(self, rows: np.ndarray) -> torch.Tensor:
        features = self._training_tensor()
        if features.layout == torch.strided:
            indices = torch.from_numpy(rows).long().to(self.device)
            return features[indices]
        assert self._training_features is not None
        return _dense_feature_rows(
            self._training_features,
            rows,
            device=self.device,
        )

    def fit(
        self,
        interactions: csr_matrix,
        item_features: ItemFeatures,
        *,
        train_item_indices: np.ndarray | Sequence[int] | None = None,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
        feature_names: Sequence[str] | None = None,
        show_progress: bool | None = None,
    ) -> TEASERGDTrainer:
        """Fit the encoder while keeping item features fixed."""
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
        if interactions.data.size and not np.all(interactions.data == 1):
            raise ValueError(
                "interactions must contain binary implicit values equal to 1"
            )

        dtype = np.dtype("float32")
        features = canonical_item_features(item_features, dtype=dtype)
        n_items = int(interactions.shape[1])
        if features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but interactions "
                f"has {n_items} items"
            )
        resolved_ids = canonical_item_ids(
            np.arange(n_items, dtype=np.int64) if item_ids is None else item_ids,
            expected_rows=n_items,
        )
        resolved_metadata = canonical_metadata(metadata, item_ids=resolved_ids)
        resolved_space = canonical_feature_space_id(feature_space_id)
        train_indices = canonical_train_item_indices(
            train_item_indices,
            n_items=n_items,
        )
        names = None if feature_names is None else tuple(map(str, feature_names))
        if names is not None and len(names) != features.shape[1]:
            raise ValueError(
                "feature_names length must match the number of item_features columns"
            )

        training_interactions = interactions[:, train_indices].astype(
            np.float32,
            copy=False,
        )
        if training_interactions.nnz < 1:
            raise ValueError(
                "training item columns must contain at least one interaction"
            )
        training_features = take_features(features, train_indices)
        decoder_features = features
        source_popularity = np.zeros(n_items, dtype=np.float32)
        if self.cfg.include_popularity:
            popularity = np.asarray(training_interactions.sum(axis=0)).ravel()
            max_popularity = float(popularity.max(initial=0))
            if max_popularity <= 0:
                raise ValueError(
                    "cannot compute popularity without training interactions"
                )
            popularity = np.asarray(popularity / max_popularity, dtype=np.float32)
            source_popularity[train_indices] = popularity
            training_features = append_column(training_features, popularity)
            decoder_features = append_column(decoder_features, source_popularity)
            if names is not None:
                names = (*names, "popularity")

        if show_progress is not None and not isinstance(show_progress, bool):
            raise ValueError("show_progress must be a bool or None")
        torch.manual_seed(int(self.cfg.seed))
        base_model = self._build_base_model(
            input_dim=train_indices.size,
            feature_dim=training_features.shape[1],
        ).to(self.device)
        if self.cfg.encoder_init == "features":
            _initialize_encoder_from_features(base_model, training_features)
        model: TEASERGD | nn.Module = base_model
        if self.cfg.compile:
            model = torch.compile(model)
        self.teaser = model
        self._reset_optimizer()
        self._training_features = training_features
        self._training_tensor_cache = None
        self._regularizer_rng = np.random.default_rng(self.cfg.seed + 1)
        self._n_training_users = int(training_interactions.shape[0])
        self.train_item_indices_ = train_indices.copy()
        self.train_item_mask_ = np.zeros(n_items, dtype=bool)
        self.train_item_mask_[train_indices] = True
        self.feature_names_ = names
        self.n_items_ = n_items
        self.n_features_ = int(training_features.shape[1])
        self.history = []
        self._install_feature_catalog(
            source_item_ids=resolved_ids,
            source_popularity=source_popularity,
            n_input_features=int(features.shape[1]),
            candidate_features=decoder_features,
            metadata=resolved_metadata,
            feature_space_id=resolved_space,
            dtype=dtype,
            include_popularity=self.cfg.include_popularity,
        )

        dataset = self._build_batch_sampler(training_interactions)
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
        progress_enabled = (
            self.cfg.show_progress if show_progress is None else show_progress
        )
        epoch_iter = self._progress_with_override(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc=f"{self._fit_name} fit",
            enabled=progress_enabled,
        )
        for epoch in epoch_iter:
            sums = self._empty_epoch_sums()
            batches = 0
            batch_iter = self._progress_with_override(
                range(len(dataset)),
                total=len(dataset),
                desc=f"{self._fit_name} epoch {epoch}",
                enabled=progress_enabled,
            )
            for batch_index in batch_iter:
                batch = dataset[batch_index]
                stats = self._train_step(batch, training_features)
                for key, value in stats.items():
                    sums[key] += float(value)
                batches += 1
            dataset.on_epoch_end()
            record = {key: value / max(1, batches) for key, value in sums.items()}
            record["epoch"] = float(epoch)
            record["lr"] = float(self.optimizer.param_groups[0]["lr"])
            self.history.append(record)
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{record['loss']:.4f}")
            if scheduler is not None:
                scheduler.step()
        self._is_fitted = True
        return self

    @staticmethod
    def _progress_with_override(iterable, *, total: int, desc: str, enabled: bool):
        if not enabled:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover
            return iterable
        return tqdm(iterable, total=total, desc=desc)

    def _train_step(
        self,
        batch,
        training_features: csr_matrix | np.ndarray,
    ) -> dict[str, float]:
        assert self.teaser is not None and self.optimizer is not None
        x = batch.x.to_dense()
        targets = dense_training_target(
            x,
            sources=batch.sources,
            candidates=batch.candidates,
            input_dim=training_features.shape[0],
        )
        candidate_rows = (
            np.arange(training_features.shape[0], dtype=np.int64)
            if batch.candidates is None
            else batch.candidates.detach().cpu().numpy()
        )
        source_rows = batch.sources.detach().cpu().numpy()
        source_features = self._training_feature_rows(source_rows)
        full_training_features = self._training_tensor()
        if batch.candidates is None:
            candidate_features = full_training_features
        elif full_training_features.layout == torch.strided:
            candidate_features = full_training_features[batch.candidates]
        else:
            candidate_features = _feature_tensor(
                take_features(training_features, candidate_rows),
                device=self.device,
            )
        positions = (
            batch.sources
            if batch.candidates is None
            else torch.arange(batch.sources.numel(), device=self.device)
        )
        self.teaser.train()
        self.optimizer.zero_grad(set_to_none=True)
        predictions = self.teaser(
            x,
            sources=batch.sources,
            source_features=source_features,
            candidate_features=candidate_features,
            source_candidate_positions=positions,
        )
        if self.cfg.loss == "normalized_mse":
            reconstruction = normalized_mse(predictions, targets)
        else:
            reconstruction = _teaser_reconstruction_loss(
                predictions,
                targets,
                n_source_items=x.shape[1],
                n_items=training_features.shape[0],
            )
        base_model = self._base_model
        coefficient_l2 = self._coefficient_penalty()
        if self.cfg.loss == "normalized_mse":
            encoder_l2 = base_model.encoder.square().mean()
        else:
            assert self._n_training_users is not None
            encoder_l2 = base_model.encoder.square().sum() / self._n_training_users
        loss = (
            reconstruction
            + float(self.cfg.l2_coefficients) * coefficient_l2
            + float(self.cfg.l2_encoder) * encoder_l2
        )
        loss.backward()
        self.optimizer.step()
        return {
            "loss": float(loss.detach().cpu()),
            "reconstruction": float(reconstruction.detach().cpu()),
            "coefficient_l2": float(coefficient_l2.detach().cpu()),
            "encoder_l2": float(encoder_l2.detach().cpu()),
        }

    @property
    def _base_model(self) -> TEASERGD:
        if self.teaser is None:
            raise RuntimeError("TEASERGDTrainer must be fitted before use")
        model = getattr(self.teaser, "_orig_mod", self.teaser)
        if not isinstance(model, TEASERGD):
            raise RuntimeError("underlying TEASERGD model is unavailable")
        return model

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if not self.is_fitted or self.n_items_ is None or self.train_item_mask_ is None:
            raise RuntimeError("TEASERGDTrainer must be fitted before prediction")
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items_:
            raise ValueError(
                f"source has {source.shape[1]} items, but TEASERGD was fitted "
                f"with {self.n_items_} items"
            )
        if source.data.size and not np.all(source.data == 1):
            raise ValueError("source must contain binary implicit values equal to 1")
        cold_positions = np.flatnonzero(~self.train_item_mask_[source.indices])
        if cold_positions.size:
            cold_item = int(source.indices[cold_positions[0]])
            raise ValueError(
                f"source contains item {cold_item}, which has no fitted encoder row"
            )
        return source

    def _selection_tensor(self, selection: CandidateSelection) -> torch.Tensor:
        if selection.rows.size == selection.catalog.n_items:
            cached = self._candidate_tensor_cache
            if cached is not None and cached[0] == selection.catalog.version:
                return cached[1]
            tensor = _feature_tensor(selection.features, device=self.device)
            self._candidate_tensor_cache = (selection.catalog.version, tensor)
            return tensor
        return _feature_tensor(selection.features, device=self.device)

    @torch.no_grad()
    def _predict_prepared_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        selection: CandidateSelection,
        candidate_features: torch.Tensor,
    ) -> SRPTensor:
        if not 1 <= int(k) <= selection.rows.size:
            raise ValueError(f"k must be in [1, {selection.rows.size}], got {k}")
        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=(0, selection.catalog.n_items),
            )
        assert self.train_item_indices_ is not None
        compact = source[:, self.train_item_indices_]
        active = np.unique(compact.indices).astype(np.int64, copy=False)
        x = torch.from_numpy(
            compact[:, active].toarray().astype(np.float32, copy=False)
        ).to(self.device)
        source_rows = torch.from_numpy(active).long().to(self.device)
        global_source_rows = self.train_item_indices_[active]
        candidate_catalog_rows = selection.source_to_candidate[global_source_rows]
        candidate_local_rows = np.full(candidate_catalog_rows.shape, -1, dtype=np.int64)
        registered = candidate_catalog_rows >= 0
        candidate_local_rows[registered] = selection.candidate_to_local[
            candidate_catalog_rows[registered]
        ]
        source_features = torch.zeros(
            (active.size, self.n_features_),
            dtype=torch.float32,
            device=self.device,
        )
        represented = candidate_local_rows >= 0
        if bool(represented.any()):
            represented_rows = torch.from_numpy(np.flatnonzero(represented)).to(
                self.device
            )
            if candidate_features.layout == torch.strided:
                local_rows = torch.from_numpy(candidate_local_rows[represented]).to(
                    self.device
                )
                source_features[represented_rows] = candidate_features[local_rows]
            else:
                source_features[represented_rows] = _dense_feature_rows(
                    selection.features,
                    candidate_local_rows[represented],
                    device=self.device,
                )
        positions = torch.from_numpy(candidate_local_rows).long().to(self.device)
        self.teaser.eval()
        scores = self.teaser(
            x,
            sources=source_rows,
            source_features=source_features,
            candidate_features=candidate_features,
            source_candidate_positions=positions,
        )

        seen_counts = np.diff(source.indptr)
        seen_users = np.repeat(np.arange(source.shape[0], dtype=np.int64), seen_counts)
        seen_catalog = selection.source_to_candidate[source.indices]
        registered = seen_catalog >= 0
        seen_local = np.full(seen_catalog.shape, -1, dtype=np.int64)
        seen_local[registered] = selection.candidate_to_local[seen_catalog[registered]]
        selected = seen_local >= 0
        if exclude_seen:
            available = selection.rows.size - np.bincount(
                seen_users[selected],
                minlength=source.shape[0],
            )
            if available.size and np.any(available < k):
                row = int(np.flatnonzero(available < k)[0])
                raise ValueError(
                    f"source row {row} has only {available[row]} unseen items "
                    f"among the selected candidates, fewer than k={k}"
                )
            if bool(selected.any()):
                rows = torch.from_numpy(seen_users[selected]).long().to(self.device)
                columns = torch.from_numpy(seen_local[selected]).long().to(self.device)
                scores[rows, columns] = -torch.inf
        local = SRPTensor.from_dense(scores, k=int(k), score_mode="raw")
        global_rows = torch.from_numpy(selection.rows).long().to(self.device)
        return SRPTensor(
            cols=global_rows[local.cols],
            vals=local.vals,
            shape=(source.shape[0], selection.catalog.n_items),
        )

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict one batch against the current or restricted catalog."""
        source = self._prepare_source(source)
        selection = self._resolve_candidate_selection(candidate_ids)
        return self._predict_prepared_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            selection=selection,
            candidate_features=self._selection_tensor(selection),
        )

    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        show_progress: bool = False,
    ) -> SRPTensor:
        """Predict all source rows while reusing one candidate tensor."""
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        selection = self._resolve_candidate_selection(candidate_ids)
        candidate_features = self._selection_tensor(selection)
        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        for start in self._progress_with_override(
            starts,
            total=len(starts),
            desc=f"TEASERGD predict@{k}",
            enabled=show_progress,
        ):
            result = self._predict_prepared_batch(
                source[start : start + batch_size],
                k=k,
                exclude_seen=exclude_seen,
                selection=selection,
                candidate_features=candidate_features,
            )
            columns.append(result.cols)
            values.append(result.vals)
        if not columns:
            return self._predict_prepared_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                selection=selection,
                candidate_features=candidate_features,
            )
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=(source.shape[0], selection.catalog.n_items),
        )
