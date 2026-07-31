from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse

from compresso_recsys.models._validation import (
    canonical_csr,
    canonical_train_item_indices,
)
from compresso_recsys.models.cold_start import (
    CandidateCatalog,
    ItemFeatures,
    _LinearFeatureRecommenderMixin,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_item_ids,
    canonical_metadata,
)

__all__ = ["LEMSA", "LEMSAConfig"]

LEMSADataType = Literal["float32", "float64"]
LEMSASolver = Literal["eigen", "direct"]
LEMSAEncoderInit = Literal["zeros", "features"]


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


def _as_dense(
    matrix: csr_matrix | np.ndarray,
    *,
    dtype: np.dtype,
) -> np.ndarray:
    dense = matrix.toarray() if issparse(matrix) else np.asarray(matrix)
    return np.asarray(dense, dtype=dtype, order="C")


def _feature_gram(features: csr_matrix | np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    gram = features.T @ features
    dense = gram.toarray() if issparse(gram) else np.asarray(gram)
    return np.asarray(dense, dtype=dtype)


def _item_semantic_sums(
    interactions: csr_matrix,
    features: csr_matrix | np.ndarray,
    *,
    batch_size: int,
    dtype: np.dtype,
) -> np.ndarray:
    """Compute ``X.T @ X @ features`` without a full user-feature buffer."""
    result = np.zeros((interactions.shape[1], features.shape[1]), dtype=dtype)
    for start in range(0, interactions.shape[0], batch_size):
        batch = interactions[start : start + batch_size]
        user_semantics = batch @ features
        if issparse(user_semantics):
            user_semantics = user_semantics.toarray()
        result += np.asarray(batch.T @ user_semantics, dtype=dtype)
    return result


def _solve_eigen_row(
    *,
    support: int,
    feature: np.ndarray,
    eigenvalues: np.ndarray,
    right_hand_side: np.ndarray,
    l2_encoder: float,
) -> tuple[np.ndarray, bool]:
    """Solve a diagonal-minus-rank-one LEMSA row system."""
    if support == 0:
        return np.zeros_like(right_hand_side), False

    diagonal_inverse = 1.0 / (support * eigenvalues + l2_encoder)
    base = diagonal_inverse * right_hand_side
    scaled_feature = diagonal_inverse * feature
    denominator = 1.0 - support * float(feature @ scaled_feature)
    threshold = 32.0 * np.finfo(right_hand_side.dtype).eps
    if np.isfinite(denominator) and denominator > threshold:
        correction = support * scaled_feature * float(feature @ base) / denominator
        return base + correction, False

    matrix = support * (np.diag(eigenvalues) - np.outer(feature, feature))
    matrix.flat[:: matrix.shape[0] + 1] += l2_encoder
    return np.linalg.solve(matrix, right_hand_side), True


def _solve_eigen_rows(
    *,
    supports: np.ndarray,
    features: np.ndarray,
    eigenvalues: np.ndarray,
    right_hand_sides: np.ndarray,
    l2_encoder: float,
) -> tuple[np.ndarray, int]:
    """Solve a block of diagonal-minus-rank-one LEMSA row systems."""
    result = np.zeros_like(right_hand_sides)
    active = supports > 0
    if not np.any(active):
        return result, 0

    active_supports = supports[active, None]
    active_features = features[active]
    active_rhs = right_hand_sides[active]
    diagonal_inverse = 1.0 / (active_supports * eigenvalues + l2_encoder)
    base = diagonal_inverse * active_rhs
    scaled_features = diagonal_inverse * active_features
    denominators = 1.0 - supports[active] * np.sum(
        active_features * scaled_features,
        axis=1,
    )
    threshold = 32.0 * np.finfo(right_hand_sides.dtype).eps
    stable = np.isfinite(denominators) & (denominators > threshold)

    solved = base.copy()
    solved[stable] += (
        active_supports[stable]
        * scaled_features[stable]
        * (
            np.sum(active_features[stable] * base[stable], axis=1)
            / denominators[stable]
        )[:, None]
    )

    fallback_indices = np.flatnonzero(active)[~stable]
    for item in fallback_indices:
        solved_row, _ = _solve_eigen_row(
            support=int(supports[item]),
            feature=features[item],
            eigenvalues=eigenvalues,
            right_hand_side=right_hand_sides[item],
            l2_encoder=l2_encoder,
        )
        result[item] = solved_row
    result[np.flatnonzero(active)[stable]] = solved[stable]
    return result, int(fallback_indices.size)


def _solve_direct_row(
    *,
    support: int,
    feature: np.ndarray,
    feature_gram: np.ndarray,
    right_hand_side: np.ndarray,
    l2_encoder: float,
) -> np.ndarray:
    """Solve one literal dense system from the LEMSA closed form."""
    if support == 0:
        return np.zeros_like(right_hand_side)
    gram_without_item = feature_gram - np.outer(feature, feature)
    matrix = support * gram_without_item
    matrix = matrix.copy()
    matrix.flat[:: matrix.shape[0] + 1] += l2_encoder
    return np.linalg.solve(matrix, right_hand_side)


@dataclass(frozen=True)
class LEMSAConfig:
    """Configuration for Language Embeddings Meet Shallow Autoencoders.

    Parameters
    ----------
    l2_encoder:
        Positive ridge regularization applied to each encoder-row update.
    epochs:
        Maximum number of block coordinate-descent sweeps.
    solver:
        ``"eigen"`` rotates the feature Gram matrix to diagonal form and uses
        an exact rank-one solve. ``"direct"`` solves the literal dense system
        and is intended as a small-problem reference implementation.
    encoder_init:
        Initialize encoder rows with zeros or their fixed feature rows.
    tolerance:
        Optional relative encoder-change threshold for early stopping.
    update_batch_size:
        Number of encoder rows computed from the same frozen model snapshot
        before their updates are applied together. ``1`` selects sequential
        Gauss-Seidel updates, while ``None`` uses one full Jacobi update per
        epoch.
    precompute_batch_size:
        User batch size used while computing fixed semantic target sums.
    dtype:
        Numerical precision used by fitting and prediction.
    """

    l2_encoder: float = 0.05
    epochs: int = 10
    solver: LEMSASolver = "eigen"
    encoder_init: LEMSAEncoderInit = "zeros"
    tolerance: float | None = None
    update_batch_size: int | None = 1
    precompute_batch_size: int = 8192
    dtype: LEMSADataType = "float32"

    def __post_init__(self) -> None:
        if not np.isfinite(self.l2_encoder) or self.l2_encoder <= 0:
            raise ValueError("l2_encoder must be finite and > 0")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, (int, np.integer))
            or self.epochs < 1
        ):
            raise ValueError("epochs must be >= 1")
        if self.solver not in {"eigen", "direct"}:
            raise ValueError("solver must be 'eigen' or 'direct'")
        if self.encoder_init not in {"zeros", "features"}:
            raise ValueError("encoder_init must be 'zeros' or 'features'")
        if self.tolerance is not None and (
            not np.isfinite(self.tolerance) or self.tolerance <= 0
        ):
            raise ValueError("tolerance must be finite and > 0, or None")
        if self.update_batch_size is not None and (
            isinstance(self.update_batch_size, bool)
            or not isinstance(self.update_batch_size, (int, np.integer))
            or self.update_batch_size < 1
        ):
            raise ValueError("update_batch_size must be >= 1, or None")
        if (
            isinstance(self.precompute_batch_size, bool)
            or not isinstance(self.precompute_batch_size, (int, np.integer))
            or self.precompute_batch_size < 1
        ):
            raise ValueError("precompute_batch_size must be >= 1")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")


class LEMSA(_LinearFeatureRecommenderMixin):
    """Gated shallow autoencoder with a fixed item-feature decoder.

    LEMSA learns one semantic direction per warm item. While updating item
    ``i``, it reconstructs histories of users who interacted with ``i`` after
    removing ``i`` from both the target and decoder. This blocks only the
    current item's trivial self-copy path and leaves all other context terms
    unconstrained.
    """

    _model_name = "LEMSA"

    def __init__(self, config: LEMSAConfig | None = None) -> None:
        self.cfg = config if config is not None else LEMSAConfig()
        self._init_feature_catalog_state()
        self.encoder_: np.ndarray | None = None
        self.decoder_features_: csr_matrix | np.ndarray | None = None
        self.train_item_indices_: np.ndarray | None = None
        self.train_item_mask_: np.ndarray | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.feature_rotation_: np.ndarray | None = None
        self.feature_eigenvalues_: np.ndarray | None = None
        self.n_items_: int | None = None
        self.n_features_: int | None = None
        self.n_epochs_: int = 0
        self.fit_history_: list[dict[str, float]] = []

    @property
    def is_fitted(self) -> bool:
        """Whether the encoder and fixed decoder have been fitted."""
        return self.encoder_ is not None and self.decoder_features_ is not None

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype used by the model."""
        return np.dtype(self.cfg.dtype)

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
        show_progress: bool = False,
    ) -> LEMSA:
        """Fit the gated alternating closed-form objective.

        ``item_features`` must have one row per source item. When
        ``train_item_indices`` is supplied, only those columns receive encoder
        rows, while every supplied feature row remains an immediately scoreable
        candidate.
        """
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
        if interactions.data.size and not np.all(interactions.data == 1):
            raise ValueError(
                "interactions must contain binary implicit values equal to 1"
            )

        features = canonical_item_features(item_features, dtype=self.dtype)
        n_items = int(interactions.shape[1])
        if features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but interactions "
                f"has {n_items} items"
            )

        resolved_item_ids = canonical_item_ids(
            np.arange(n_items, dtype=np.int64) if item_ids is None else item_ids,
            expected_rows=n_items,
        )
        resolved_metadata = canonical_metadata(metadata, item_ids=resolved_item_ids)
        resolved_feature_space_id = canonical_feature_space_id(feature_space_id)
        train_indices = canonical_train_item_indices(
            train_item_indices,
            n_items=n_items,
        )

        if feature_names is None:
            names = None
        else:
            names = tuple(str(name) for name in feature_names)
            if len(names) != features.shape[1]:
                raise ValueError(
                    "feature_names length must match the number of "
                    "item_features columns"
                )

        x = interactions[:, train_indices].astype(self.dtype, copy=False)
        if x.nnz < 1:
            raise ValueError(
                "training item columns must contain at least one interaction"
            )
        training_features = features[train_indices]
        feature_gram = _feature_gram(training_features, dtype=self.dtype)

        rotation: np.ndarray | None
        eigenvalues: np.ndarray | None
        if self.cfg.solver == "eigen":
            eigenvalues, rotation = np.linalg.eigh(feature_gram)
            numerical_scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            negative_tolerance = (
                64.0
                * np.finfo(self.dtype).eps
                * feature_gram.shape[0]
                * numerical_scale
            )
            if float(eigenvalues.min(initial=0.0)) < -negative_tolerance:
                raise np.linalg.LinAlgError(
                    "item-feature Gram matrix has unexpectedly negative eigenvalues"
                )
            np.maximum(eigenvalues, 0, out=eigenvalues)
            solver_features = np.asarray(
                training_features @ rotation,
                dtype=self.dtype,
                order="C",
            )
            eigenvalues = np.asarray(eigenvalues, dtype=self.dtype)
            rotation = np.asarray(rotation, dtype=self.dtype)
        else:
            rotation = None
            eigenvalues = None
            solver_features = _as_dense(training_features, dtype=self.dtype)

        if self.cfg.encoder_init == "zeros":
            encoder = np.zeros_like(solver_features)
            user_profiles = np.zeros(
                (x.shape[0], solver_features.shape[1]),
                dtype=self.dtype,
            )
        else:
            encoder = solver_features.copy()
            user_profiles = np.asarray(x @ encoder, dtype=self.dtype)

        target_sums = _item_semantic_sums(
            x,
            solver_features,
            batch_size=int(self.cfg.precompute_batch_size),
            dtype=self.dtype,
        )
        x_csc = x.tocsc()
        supports = np.diff(x_csc.indptr)
        history: list[dict[str, float]] = []
        update_batch_size = (
            x.shape[1]
            if self.cfg.update_batch_size is None
            else min(int(self.cfg.update_batch_size), x.shape[1])
        )

        for epoch in range(int(self.cfg.epochs)):
            squared_change = 0.0
            max_row_change = 0.0
            fallbacks = 0
            blocks = _progress(
                range(0, x.shape[1], update_batch_size),
                enabled=show_progress,
                desc=f"LEMSA epoch {epoch + 1}",
            )
            for block_start in blocks:
                block_end = min(block_start + update_batch_size, x.shape[1])
                block = slice(block_start, block_end)
                block_supports = supports[block]
                block_features = solver_features[block]
                old_rows = encoder[block]

                # Context and every solve read the same frozen block snapshot.
                context_sums = np.asarray(
                    x_csc[:, block].T @ user_profiles,
                    dtype=self.dtype,
                )
                context_sums -= block_supports[:, None] * old_rows
                target = target_sums[block] - block_supports[:, None] * block_features

                if self.cfg.solver == "eigen":
                    assert eigenvalues is not None
                    gram_context = eigenvalues * context_sums
                    gram_context -= block_features * np.sum(
                        block_features * context_sums,
                        axis=1,
                        keepdims=True,
                    )
                    next_rows, block_fallbacks = _solve_eigen_rows(
                        supports=block_supports,
                        features=block_features,
                        eigenvalues=eigenvalues,
                        right_hand_sides=target - gram_context,
                        l2_encoder=float(self.cfg.l2_encoder),
                    )
                    fallbacks += block_fallbacks
                else:
                    next_rows = np.empty_like(old_rows)
                    for offset in range(block_end - block_start):
                        next_rows[offset] = _solve_direct_row(
                            support=int(block_supports[offset]),
                            feature=block_features[offset],
                            feature_gram=feature_gram,
                            right_hand_side=(
                                target[offset]
                                - (
                                    feature_gram
                                    - np.outer(
                                        block_features[offset],
                                        block_features[offset],
                                    )
                                )
                                @ context_sums[offset]
                            ),
                            l2_encoder=float(self.cfg.l2_encoder),
                        )

                deltas = np.asarray(
                    next_rows - old_rows,
                    dtype=self.dtype,
                )
                encoder[block] = next_rows

                # Commit profile changes only after every row has been solved.
                for offset, item in enumerate(range(block_start, block_end)):
                    start, end = x_csc.indptr[item : item + 2]
                    users = x_csc.indices[start:end]
                    if users.size:
                        user_profiles[users] += deltas[offset]

                row_changes = np.linalg.norm(deltas, axis=1)
                squared_change += float(row_changes @ row_changes)
                max_row_change = max(
                    max_row_change,
                    float(row_changes.max(initial=0.0)),
                )

            encoder_norm = float(np.linalg.norm(encoder))
            relative_change = float(np.sqrt(squared_change)) / max(
                encoder_norm,
                float(np.finfo(self.dtype).tiny),
            )
            history.append(
                {
                    "relative_change": relative_change,
                    "max_row_change": max_row_change,
                    "solver_fallbacks": float(fallbacks),
                }
            )
            if self.cfg.tolerance is not None and relative_change <= self.cfg.tolerance:
                break

        if rotation is not None:
            encoder = np.asarray(encoder @ rotation.T, dtype=self.dtype, order="C")
        if not np.all(np.isfinite(encoder)):
            raise np.linalg.LinAlgError(
                "LEMSA fitting produced non-finite encoder values"
            )

        self.encoder_ = encoder
        self.train_item_indices_ = train_indices.copy()
        self.train_item_mask_ = np.zeros(n_items, dtype=bool)
        self.train_item_mask_[train_indices] = True
        self.feature_names_ = names
        self.feature_rotation_ = rotation
        self.feature_eigenvalues_ = eigenvalues
        self.n_items_ = n_items
        self.n_features_ = int(features.shape[1])
        self.n_epochs_ = len(history)
        self.fit_history_ = history
        self._install_feature_catalog(
            source_item_ids=resolved_item_ids,
            source_popularity=np.zeros(n_items, dtype=self.dtype),
            n_input_features=int(features.shape[1]),
            candidate_features=features,
            metadata=resolved_metadata,
            feature_space_id=resolved_feature_space_id,
            dtype=self.dtype,
            include_popularity=False,
        )
        return self
