"""SEMCo: sparse contrastive learning for content-based cold-item recommendation.

A content encoder trained directly from interactions with a sampled
entmax/sparsemax objective, with no pretrained collaborative embeddings. Item
features are a mapping of modality name to matrix; one encoder stack runs per
modality and an attention-weighted sum fuses them.

Reference: Sparse Contrastive Learning for Content-Based Cold Item
Recommendation, SIGIR 2026. https://github.com/gmeehan96/SEMCo
"""

from __future__ import annotations

import math
import time
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, issparse
from torch import nn

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
    _validate_log_every_n_steps,
)
from compresso_recsys.models._validation import (
    canonical_csr,
    canonical_train_item_indices,
)
from compresso_recsys.models.cold_start import (
    canonical_feature_space_id,
    canonical_item_features,
    canonical_item_ids,
    canonical_metadata,
    take_features,
)
from compresso_recsys.models.multimodal import (
    BaseMultiModalRecommender,
    MultiModalCandidateCatalog,
    MultiModalItemFeatures,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = ["SEMCo", "SEMCoConfig", "SEMCoTrainer"]

SEMCoObjective = Literal["softmax", "entmax15", "sparsemax"]

_ALPHA: dict[str, float] = {"entmax15": 1.5, "sparsemax": 2.0}

_Matrix = csr_matrix | np.ndarray


def _entmax_function(objective: str):
    """Resolve the entmax projection lazily, as ItemKNN does for scikit-learn.

    ``entmax`` rather than ``adasplash.triton_entmax``: AdaSplash is faster but
    hard-depends on ``triton``, which rules out CPU and macOS.
    """
    try:
        from entmax import entmax15, sparsemax
    except ImportError as error:  # pragma: no cover - environment dependent
        raise ImportError(
            "SEMCoTrainer requires entmax for the entmax15 and sparsemax "
            "objectives; install compresso-recsys[semco]"
        ) from error
    return entmax15 if objective == "entmax15" else sparsemax


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise L2 normalization, matching ``F.normalize(x, dim=-1)``."""
    return x / x.norm(dim=1, keepdim=True).clamp_min(eps)


def _omega_alpha(p: torch.Tensor, alpha: float) -> torch.Tensor:
    """Tsallis entropy of a probability row, the Fenchel-Young penalty."""
    return (1.0 - p.pow(alpha).sum(dim=-1)) / (alpha * (alpha - 1.0))


def _canonical_modalities(
    item_features: MultiModalItemFeatures,
    *,
    n_items: int,
    dtype: np.dtype,
) -> dict[str, _Matrix]:
    """Validate a modality mapping and fix its order.

    Runs before the catalog because the encoder needs the per-modality widths
    first. Caller order becomes the installed and ``nn.ModuleList`` order.
    """
    if not isinstance(item_features, Mapping):
        raise TypeError("item_features must be a mapping of modality names to matrices")
    if not item_features:
        raise ValueError("item_features must contain at least one modality")
    resolved: dict[str, _Matrix] = {}
    for name, matrix in item_features.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("modality names must be non-empty strings")
        canonical = canonical_item_features(matrix, dtype=dtype)
        if canonical.shape[0] != n_items:
            raise ValueError(
                f"item_features[{name!r}] has {canonical.shape[0]} rows, but "
                f"interactions has {n_items} items"
            )
        resolved[name] = canonical
    return resolved


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SEMCoConfig:
    """Configuration for :class:`SEMCoTrainer`.

    ``objective`` selects the projection used by the contrastive loss.
    ``softmax`` is plain InfoNCE; ``entmax15`` and ``sparsemax`` are the sparse
    Fenchel-Young variants the paper is named for and require the optional
    ``entmax`` dependency.

    ``temperature`` is the paper's tau. Its useful range is objective-specific:
    around 6-20 for sparsemax, smaller for entmax15, smaller again for softmax.

    Modality names and widths come from the feature mapping passed to
    :meth:`SEMCoTrainer.fit`, not from here.
    """

    # Architecture.
    hidden_size: int = 256
    emb_size: int = 128

    # Objective.
    objective: SEMCoObjective = "sparsemax"
    temperature: float = 10.0

    # Optimization.
    epochs: int = 20
    batch_size: int = 1024
    lr: float = 1e-3
    weight_decay: float = 0.0
    lr_schedule: Literal["constant", "cosine"] = "cosine"

    # Standard trainer fields.
    device: str | torch.device = "cpu"
    seed: int = 0
    show_progress: bool = True
    log_prefix: str = "SEMCo"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        for name in ("hidden_size", "emb_size", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.objective not in ("softmax", "entmax15", "sparsemax"):
            raise ValueError(
                "objective must be one of 'softmax', 'entmax15', 'sparsemax', "
                f"got {self.objective!r}"
            )
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if not np.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be finite and > 0, got {self.temperature}"
            )
        if not np.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError(f"lr must be finite and > 0, got {self.lr}")
        if not np.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be finite and >= 0, got {self.weight_decay}"
            )
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(
            self.seed, (int, np.integer)
        ):
            raise TypeError("seed must be an integer")
        torch.device(self.device)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class LearnedWeightedSum(nn.Module):
    """Attention-weighted sum over per-modality encodings.

    Weight and bias are zero-initialized so the softmax starts uniform.
    """

    def __init__(self, input_dim: int, num_inputs: int) -> None:
        super().__init__()
        self.num_inputs = int(num_inputs)
        self.attention = nn.Linear(input_dim * self.num_inputs, self.num_inputs)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def forward(self, inputs: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(inputs) != self.num_inputs:
            raise ValueError(
                f"expected {self.num_inputs} modality encodings, got {len(inputs)}"
            )
        stacked = torch.stack(tuple(inputs), dim=1)
        weights = torch.softmax(
            self.attention(stacked.reshape(stacked.shape[0], -1)), dim=1
        )
        return torch.einsum("bi,bid->bd", weights, stacked)


class SEMCo(nn.Module):
    """Content encoder mapping per-modality item features into the retrieval space.

    One ``Linear -> BatchNorm -> ReLU`` stack per modality, fused by attention,
    then projected and L2-normalized. Stacks live in an ``nn.ModuleList`` ordered
    by ``modalities``, so that order is part of the checkpoint.
    """

    def __init__(
        self,
        *,
        feature_dims: Mapping[str, int],
        hidden_size: int,
        emb_size: int,
    ) -> None:
        super().__init__()
        if not feature_dims:
            raise ValueError("feature_dims must contain at least one modality")
        self.modalities = tuple(feature_dims)
        # track_running_stats=False follows the paper: BatchNorm normalizes by
        # batch statistics even in eval mode, so an item's embedding depends on
        # what it is encoded alongside. Every encode here is therefore over a
        # large item set -- never a candidate subset, where a handful of rows
        # would shift embeddings and a single row would zero them. See
        # _catalog_embeddings and the two tests named there.
        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(int(feature_dims[name]), hidden_size),
                    nn.BatchNorm1d(hidden_size, track_running_stats=False),
                    nn.ReLU(),
                )
                for name in self.modalities
            ]
        )
        self.fuser = LearnedWeightedSum(hidden_size, len(self.modalities))
        self.final_layer = nn.Linear(hidden_size, emb_size)

    def forward(self, features: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Encode one dense tensor per modality into L2-normalized embeddings."""
        missing = [name for name in self.modalities if name not in features]
        if missing:
            raise KeyError(f"missing modality features: {sorted(missing)}")
        # The reference L2-normalizes raw features before the first layer.
        encoded = [
            stack(_l2_normalize(features[name]))
            for name, stack in zip(self.modalities, self.encoders)
        ]
        return _l2_normalize(self.final_layer(self.fuser(encoded)))


# ---------------------------------------------------------------------------
# Training batches
# ---------------------------------------------------------------------------


class _InteractionPairSampler:
    """Shuffled (user, item) interaction pairs, the batch's own items as candidates.

    None of the package's samplers fit: ``InteractionBatchSampler`` and
    ``SymmetricInteractionBatchSampler`` are user-row shaped, and
    ``LeaveOneOutInteractionBatchSampler`` removes the positive from the user's
    history and drops users with a single interaction.

    Uniform shuffle-and-chunk, so a user or an item may appear several times in
    one batch. Both match the reference; see ``_batch_item_rows``.
    """

    def __init__(
        self,
        interactions: csr_matrix,
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        coo = interactions.tocoo()
        self.users = coo.row.astype(np.int64, copy=True)
        self.items = coo.col.astype(np.int64, copy=True)
        self.batch_size = int(batch_size)
        self._rng = np.random.default_rng(seed)
        self._order = np.arange(self.users.size, dtype=np.int64)
        self._shuffle()

    def _shuffle(self) -> None:
        self._rng.shuffle(self._order)

    def __len__(self) -> int:
        return math.ceil(self.users.size / self.batch_size)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.batch_size
        selected = self._order[start : start + self.batch_size]
        return self.users[selected], self.items[selected]

    def on_epoch_end(self) -> None:
        self._shuffle()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class SEMCoTrainer(BaseMultiModalRecommender):
    """Train a multimodal content encoder with a sparse contrastive loss.

    ``_prepare_source`` is deliberately not overridden. TEASER rejects a history
    containing an item with no fitted encoder row; SEMCo has no per-item
    parameters, so any item with features can contribute to a profile, including
    one held out of training.

    ``predict`` is inherited, so ``resolve_selection`` runs once per batch rather
    than once per call. See ``TEASERGDTrainer.predict`` for the resolve-once
    shape, which is worth adopting above roughly 10^5 items.

    Missing modalities are not handled: ``MultiModalCandidateCatalog`` has no
    availability mask and ``MMConcatWrapper``'s ``missing`` policy is unreachable
    from a model that stores its own modalities. Every supplied row is treated as
    present, so impute before fitting if the features have coverage gaps.
    """

    checkpoint_type = "semco_trainer"
    _fit_name = "SEMCo"

    def __init__(
        self,
        config: SEMCoConfig | None = None,
        logger: Any | None = None,
    ) -> None:
        super().__init__()
        self.cfg = config if config is not None else SEMCoConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.encoder: SEMCo | nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.history: list[dict[str, float]] = []
        self.source_features_: dict[str, _Matrix] | None = None
        # Warm rows of source_features_, aligned with the training matrix
        # columns. Transient: rebuilt by fit, never checkpointed.
        self._training_features: dict[str, _Matrix] | None = None
        self.train_item_indices_: np.ndarray | None = None
        self.feature_dims_: dict[str, int] | None = None
        self.n_items_: int | None = None
        self._source_embedding_cache: torch.Tensor | None = None
        self._candidate_cache: tuple[int, torch.Tensor] | None = None
        self._is_fitted = False

    # -- reporting ---------------------------------------------------------

    def _reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        return _resolve_reporter(
            default_logger=self.logger,
            logger=logger,
            default_show_progress=self.cfg.show_progress,
            show_progress=show_progress,
            prefix=self.cfg.log_prefix,
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

    # -- state -------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def _on_catalog_published(self, catalog: MultiModalCandidateCatalog) -> None:
        del catalog
        self._candidate_cache = None

    def _build_encoder(self) -> SEMCo:
        if not self.feature_dims_:
            raise RuntimeError("feature widths must be known before building SEMCo")
        return SEMCo(
            feature_dims=self.feature_dims_,
            hidden_size=self.cfg.hidden_size,
            emb_size=self.cfg.emb_size,
        )

    def _reset_optimizer(self) -> None:
        if self.encoder is None:
            raise RuntimeError("encoder must be built before creating optimizer")
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=float(self.cfg.lr),
            weight_decay=float(self.cfg.weight_decay),
        )

    def _dense_rows(
        self,
        features: Mapping[str, _Matrix],
        rows: np.ndarray | None = None,
    ) -> dict[str, torch.Tensor]:
        """One dense device tensor per modality, optionally restricted to rows."""
        tensors: dict[str, torch.Tensor] = {}
        for name, matrix in features.items():
            selected = matrix if rows is None else matrix[rows]
            array = selected.toarray() if issparse(selected) else np.asarray(selected)
            tensors[name] = torch.from_numpy(
                np.array(array, dtype=np.float32, order="C", copy=True)
            ).to(self.device)
        return tensors

    def _sparse_tensor(self, matrix: csr_matrix) -> torch.Tensor:
        """A scipy CSR as a coalesced torch sparse tensor on the device."""
        coo = matrix.tocoo()
        indices = torch.as_tensor(
            np.vstack([coo.row, coo.col]), dtype=torch.long, device=self.device
        )
        values = torch.as_tensor(
            coo.data.astype(np.float32), dtype=torch.float32, device=self.device
        )
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=matrix.shape,
            device=self.device,
            check_invariants=False,
        ).coalesce()

    # -- loss --------------------------------------------------------------

    def _loss(
        self,
        user_vectors: torch.Tensor,
        item_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive loss over one batch of (user, positive item) pairs.

        Both arguments are L2-normalized and aligned row-wise: row ``i`` of
        ``item_vectors`` is the positive for row ``i`` of ``user_vectors``.
        """
        tau = float(self.cfg.temperature)
        scores = (user_vectors @ item_vectors.T) / tau

        if self.cfg.objective == "softmax":
            return -torch.diag(torch.nn.functional.log_softmax(scores, dim=1)).mean()

        projection = _entmax_function(self.cfg.objective)
        alpha = _ALPHA[self.cfg.objective]
        p_star = projection(scores / tau, dim=1)
        identity = torch.eye(
            scores.shape[0], dtype=scores.dtype, device=scores.device
        )
        alignment = torch.einsum("ij,ij->i", p_star - identity, scores).mean()
        return alignment + _omega_alpha(p_star, alpha).mean()

    def _batch_item_rows(
        self,
        users: np.ndarray,
        items: np.ndarray,
        interactions: csr_matrix,
    ) -> np.ndarray:
        """Items to encode for one batch: positives first, then the rest.

        Positives come first in pair order, so encoded rows ``[:items.size]``
        are the positives aligned with ``users``. The remaining distinct items
        the batch's users interacted with follow, which covers every nonzero of
        their interaction rows, so profiles are exact rather than approximate.

        Positives keep their duplicates, matching the reference: an item that is
        the positive for several pairs is encoded and counted once per pair.
        ``np.unique`` plus a lookup table would drop that.
        """
        histories = np.unique(interactions[np.unique(users)].indices)
        positives = np.asarray(items, dtype=np.int64)
        remaining = np.setdiff1d(histories, np.unique(positives), assume_unique=True)
        return np.concatenate((positives, remaining))

    def _train_step(
        self,
        users: np.ndarray,
        items: np.ndarray,
        interactions: csr_matrix,
    ) -> torch.Tensor:
        """Optimize one batch of interaction pairs and return the detached loss."""
        assert self.encoder is not None and self.optimizer is not None
        assert self._training_features is not None

        rows = self._batch_item_rows(users, items, interactions)
        embeddings = self.encoder(self._dense_rows(self._training_features, rows))

        # Row-scaling the interaction submatrix is omitted: it is a per-row
        # positive scalar and the L2 normalization below removes it.
        profile_source = interactions[users][:, rows]
        user_vectors = _l2_normalize(
            torch.sparse.mm(self._sparse_tensor(profile_source), embeddings)
        )
        item_vectors = embeddings[: items.size]

        loss = self._loss(user_vectors, item_vectors)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    # -- fitting -----------------------------------------------------------

    def fit(
        self,
        interactions: csr_matrix,
        item_features: MultiModalItemFeatures,
        *,
        train_item_indices: np.ndarray | Sequence[int] | None = None,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SEMCoTrainer:
        """Learn the content encoder and publish the initial candidate catalog.

        ``item_features`` maps modality names to aligned item-by-feature
        matrices; iteration order fixes the encoder and catalog order.
        """
        reporter = self._reporter(logger, show_progress)

        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
        if np.any(interactions.data < 0):
            raise ValueError("interactions must contain nonnegative values")

        dtype = np.dtype("float32")
        n_items = int(interactions.shape[1])
        features = _canonical_modalities(item_features, n_items=n_items, dtype=dtype)

        resolved_ids = canonical_item_ids(
            np.arange(n_items, dtype=np.int64) if item_ids is None else item_ids,
            expected_rows=n_items,
        )
        resolved_metadata = canonical_metadata(metadata, item_ids=resolved_ids)
        resolved_space = canonical_feature_space_id(feature_space_id)
        train_indices = canonical_train_item_indices(train_item_indices, n_items=n_items)

        training_interactions = interactions[:, train_indices].astype(
            np.float32, copy=False
        )
        if training_interactions.nnz < 1:
            raise ValueError(
                "training item columns must contain at least one interaction"
            )

        # A full fit installs fresh state before any update, so the snapshot
        # below can be restored if setup or training fails.
        previous_state = (
            self.encoder,
            self.optimizer,
            self.history,
            self.source_features_,
            self._training_features,
            self.train_item_indices_,
            self.feature_dims_,
            self.n_items_,
            self._is_fitted,
        )
        self._is_fitted = False
        try:
            return self._fit_validated(
                training_interactions,
                features=features,
                train_indices=train_indices,
                resolved_ids=resolved_ids,
                resolved_metadata=resolved_metadata,
                resolved_space=resolved_space,
                dtype=dtype,
                reporter=reporter,
            )
        except BaseException:
            (
                self.encoder,
                self.optimizer,
                self.history,
                self.source_features_,
                self._training_features,
                self.train_item_indices_,
                self.feature_dims_,
                self.n_items_,
                self._is_fitted,
            ) = previous_state
            raise

    def _fit_validated(
        self,
        training_interactions: csr_matrix,
        *,
        features: dict[str, _Matrix],
        train_indices: np.ndarray,
        resolved_ids: np.ndarray,
        resolved_metadata: pd.DataFrame | None,
        resolved_space: str | None,
        dtype: np.dtype,
        reporter: _Reporter,
    ) -> SEMCoTrainer:
        """Train replacement state after public input validation."""
        fit_started = time.monotonic()
        torch.manual_seed(self.cfg.seed)

        self.n_items_ = int(resolved_ids.size)
        self.feature_dims_ = {
            name: int(matrix.shape[1]) for name, matrix in features.items()
        }
        self.train_item_indices_ = train_indices
        self.source_features_ = features
        self._training_features = {
            name: take_features(matrix, train_indices)
            for name, matrix in features.items()
        }
        self.history = []
        self._source_embedding_cache = None
        self._candidate_cache = None

        self.encoder = self._build_encoder().to(self.device)
        self._reset_optimizer()

        sampler = _InteractionPairSampler(
            training_interactions,
            batch_size=self.cfg.batch_size,
            seed=self.cfg.seed,
        )
        steps_per_epoch = len(sampler)

        scheduler = None
        if self.cfg.lr_schedule == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=steps_per_epoch * self.cfg.epochs,
                eta_min=0.0,
            )

        reporter.log(
            "fit started: "
            f"{training_interactions.shape[0]} users | "
            f"{training_interactions.shape[1]} items | "
            f"{training_interactions.nnz} interactions | "
            f"{steps_per_epoch} batches of {self.cfg.batch_size} | "
            f"{self.cfg.epochs} epochs | objective {self.cfg.objective} | "
            f"modalities {list(self.feature_dims_)} | device {self.device}"
        )
        epochs = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc=f"{self._fit_name} fit",
        )
        # A single batch bar is rewound and relabelled each epoch, rather than a
        # finished bar being left behind per epoch.
        batch_bar = reporter.bar(
            total=steps_per_epoch,
            desc=f"{self._fit_name} epoch 1",
        )
        try:
            self.encoder.train()
            for epoch in epochs:
                epoch_started = time.monotonic()
                loss_sum = 0.0
                pairs = 0
                if batch_bar is not None:
                    batch_bar.reset(total=steps_per_epoch)
                    batch_bar.set_description(f"{self._fit_name} epoch {epoch}")
                for step in range(1, steps_per_epoch + 1):
                    users, items = sampler[step - 1]
                    loss = self._train_step(users, items, training_interactions)

                    loss_sum += float(loss) * users.size
                    pairs += int(users.size)
                    if scheduler is not None:
                        scheduler.step()
                    if batch_bar is not None:
                        batch_bar.update(1)

                    log_steps = reporter.log_every_n_steps
                    if log_steps and step % log_steps == 0:
                        reporter.step(
                            f"epoch {epoch}/{self.cfg.epochs} "
                            f"step {step}/{steps_per_epoch}",
                            step,
                            steps_per_epoch,
                            epoch_started,
                            {"loss": loss_sum / max(1, pairs)},
                        )

                record = {
                    "epoch": float(epoch),
                    "loss": loss_sum / max(1, pairs),
                    "pairs": float(pairs),
                }
                self.history.append(record)
                reporter.epoch(f"epoch {epoch}/{self.cfg.epochs}", record, epoch_started)
                if hasattr(epochs, "set_postfix"):
                    epochs.set_postfix(loss=f"{record['loss']:.4f}")
                sampler.on_epoch_end()
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epochs, "close"):
                epochs.close()

        self.encoder.eval()
        self.candidates.install(
            source_item_ids=resolved_ids,
            candidate_features=features,
            metadata=resolved_metadata,
            feature_space_id=resolved_space,
            dtype=dtype,
        )
        self._is_fitted = True
        reporter.log(
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} total | "
            f"{len(self.history)} epochs recorded"
        )
        return self

    # -- prediction --------------------------------------------------------

    def _encode(self, features: Mapping[str, _Matrix]) -> torch.Tensor:
        """Encode a full modality mapping under no_grad, in eval mode."""
        assert self.encoder is not None
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(self._dense_rows(features)).contiguous()

    def _source_embeddings(self) -> torch.Tensor:
        """Encoded embeddings of the fitted source items, cached."""
        if self._source_embedding_cache is None:
            assert self.source_features_ is not None
            self._source_embedding_cache = self._encode(self.source_features_)
        return self._source_embedding_cache

    def _catalog_embeddings(self, catalog: MultiModalCandidateCatalog) -> torch.Tensor:
        """Embeddings for the whole catalog, cached by snapshot version.

        Never encodes a candidate subset: BatchNorm uses batch statistics, so a
        subset would give an item a different vector than a full-catalog
        prediction. Guarded by
        ``test_candidate_subset_scores_match_full_catalog`` and
        ``test_registering_one_item_at_a_time_gives_distinct_embeddings``.
        """
        cached = self._candidate_cache
        if cached is not None and cached[0] == catalog.version:
            return cached[1]
        matrix = self._encode(catalog.item_features)
        self._candidate_cache = (catalog.version, matrix)
        return matrix

    def _profiles(self, source: csr_matrix) -> torch.Tensor:
        """User profiles as the normalized sum of interacted item embeddings."""
        return _l2_normalize(
            torch.sparse.mm(self._sparse_tensor(source), self._source_embeddings())
        )

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for one source batch."""
        source = self._prepare_source(source)
        selection = self.candidates.resolve_selection(candidate_ids)
        n_candidates = int(selection.rows.size)
        if not 1 <= int(k) <= n_candidates:
            raise ValueError(f"k must be in [1, {n_candidates}], got {k}")

        # Two hops: source-vocabulary column -> catalog row -> selection-local
        # column. Lifted from ContentRecommender; indexing scores with
        # source.indices directly is only correct for a whole-catalog selection.
        seen_counts = np.diff(source.indptr)
        seen_users = np.repeat(np.arange(source.shape[0], dtype=np.int64), seen_counts)
        seen_catalog = selection.source_to_candidate[source.indices]
        registered = seen_catalog >= 0
        seen_local = np.full(seen_catalog.shape, -1, dtype=np.int64)
        seen_local[registered] = selection.candidate_to_local[seen_catalog[registered]]
        selected_seen = seen_local >= 0

        if exclude_seen:
            available = n_candidates - np.bincount(
                seen_users[selected_seen], minlength=source.shape[0]
            )
            if available.size and np.any(available < k):
                row = int(np.flatnonzero(available < k)[0])
                raise ValueError(
                    f"source row {row} has only {available[row]} unseen items "
                    f"among the selected candidates, fewer than k={k}"
                )

        if source.shape[0] == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long),
                vals=torch.empty((0, k), dtype=torch.float32),
                shape=(0, selection.catalog.n_items),
            )

        embeddings = self._catalog_embeddings(selection.catalog)
        if n_candidates != selection.catalog.n_items:
            embeddings = embeddings[
                torch.from_numpy(np.ascontiguousarray(selection.rows)).to(self.device)
            ]
        scores = self._profiles(source) @ embeddings.T

        if exclude_seen and bool(selected_seen.any()):
            rows = torch.as_tensor(
                seen_users[selected_seen], dtype=torch.long, device=self.device
            )
            cols = torch.as_tensor(
                seen_local[selected_seen], dtype=torch.long, device=self.device
            )
            scores[rows, cols] = -torch.inf

        local = SRPTensor.from_dense(scores.cpu(), k=int(k), score_mode="raw")
        catalog_rows = torch.from_numpy(np.ascontiguousarray(selection.rows))
        return SRPTensor(
            cols=catalog_rows[local.cols],
            vals=local.vals,
            shape=(source.shape[0], selection.catalog.n_items),
        )

    # -- persistence -------------------------------------------------------

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
    ) -> SEMCoTrainer:
        state = reader.read_json("state/semco.json")
        n_items = state.get("n_items")
        if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items < 1:
            raise ValueError("SEMCo n_items must be a positive integer")
        modalities = state.get("modalities")
        if not isinstance(modalities, list) or not modalities:
            raise ValueError("SEMCo checkpoint must record at least one modality")
        dims: dict[str, int] = {}
        for entry in modalities:
            if not isinstance(entry, dict):
                raise ValueError("SEMCo modality entries must be objects")
            name = entry.get("name")
            width = entry.get("n_features")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("SEMCo modality names must be non-empty strings")
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                raise ValueError(f"SEMCo modality {name!r} width must be positive")
            if name in dims:
                raise ValueError(f"duplicate SEMCo modality {name!r}")
            dims[name] = width

        config = dict(config)
        config["device"] = str(device)
        trainer = cls(SEMCoConfig(**config))
        trainer.n_items_ = int(n_items)
        # Insertion order here is the checkpoint's recorded order, which must
        # match the ModuleList order the state_dict was written against.
        trainer.feature_dims_ = dims
        trainer.encoder = trainer._build_encoder()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.encoder

    def _move_checkpoint_state(self, device: torch.device) -> None:
        self._source_embedding_cache = None
        self._candidate_cache = None

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        # The multimodal base persists the catalog for us.
        super()._save_checkpoint_state(writer)
        assert self.source_features_ is not None
        assert self.train_item_indices_ is not None
        assert self.feature_dims_ is not None and self.n_items_ is not None

        writer.write_numpy("state/train_item_indices.npy", self.train_item_indices_)
        # Index-based paths, as the catalog does: modality names are not
        # necessarily path-safe.
        modalities = []
        for index, (name, matrix) in enumerate(self.source_features_.items()):
            storage = writer.write_features(f"state/source_features/{index}", matrix)
            modalities.append(
                {
                    "name": name,
                    "n_features": int(matrix.shape[1]),
                    "storage": storage,
                }
            )
        writer.write_json(
            "state/semco.json",
            {
                "n_items": self.n_items_,
                "modalities": modalities,
                "history": self.history,
            },
        )

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        super()._load_checkpoint_state(reader)
        state = reader.read_json("state/semco.json")
        self.train_item_indices_ = np.asarray(
            reader.read_numpy("state/train_item_indices.npy"), dtype=np.int64
        )
        self.n_items_ = int(state["n_items"])

        features: dict[str, _Matrix] = {}
        dims: dict[str, int] = {}
        for index, entry in enumerate(state["modalities"]):
            name = str(entry["name"])
            matrix = reader.read_features(
                f"state/source_features/{index}",
                storage=str(entry["storage"]),
            )
            if matrix.shape != (self.n_items_, int(entry["n_features"])):
                raise ValueError(
                    f"SEMCo source features for {name!r} have the wrong shape"
                )
            features[name] = matrix
            dims[name] = int(entry["n_features"])
        self.source_features_ = features
        self.feature_dims_ = dims
        self._training_features = {
            name: take_features(matrix, self.train_item_indices_)
            for name, matrix in features.items()
        }

        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("SEMCo training history must be a list")
        self.history = list(history)

        catalog = self.candidates.snapshot()
        if dict(catalog.feature_dims) != dims:
            raise ValueError(
                "SEMCo modality schema does not match the restored catalog"
            )
        source_ids = self.candidates.source_item_ids
        assert source_ids is not None
        if source_ids.size != self.n_items_:
            raise ValueError("SEMCo source catalog does not match n_items")

        self._source_embedding_cache = None
        self._candidate_cache = None

    def _build_checkpoint_optimizer(self) -> None:
        self._reset_optimizer()

    def _finish_checkpoint_load(self) -> None:
        self._is_fitted = True
        if self.encoder is not None:
            self.encoder.eval()