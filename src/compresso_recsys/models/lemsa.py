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
        Maximum number of sequential coordinate-descent sweeps.
    solver:
        ``"eigen"`` rotates the feature Gram matrix to diagonal form and uses
        an exact rank-one solve. ``"direct"`` solves the literal dense system
        and is intended as a small-problem reference implementation.
    encoder_init:
        Initialize encoder rows with zeros or their fixed feature rows.
    tolerance:
        Optional relative encoder-change threshold for early stopping.
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

        for epoch in range(int(self.cfg.epochs)):
            squared_change = 0.0
            max_row_change = 0.0
            fallbacks = 0
            rows = _progress(
                range(x.shape[1]),
                enabled=show_progress,
                desc=f"LEMSA epoch {epoch + 1}",
            )
            for item in rows:
                start, end = x_csc.indptr[item : item + 2]
                users = x_csc.indices[start:end]
                support = int(supports[item])
                feature = solver_features[item]
                old_encoder = encoder[item].copy()

                if support == 0:
                    next_encoder = np.zeros_like(old_encoder)
                else:
                    target_sum = target_sums[item] - support * feature
                    context_sum = user_profiles[users].sum(axis=0)
                    context_sum -= support * old_encoder

                    if self.cfg.solver == "eigen":
                        assert eigenvalues is not None
                        gram_context = eigenvalues * context_sum
                        gram_context -= feature * float(feature @ context_sum)
                        right_hand_side = target_sum - gram_context
                        next_encoder, used_fallback = _solve_eigen_row(
                            support=support,
                            feature=feature,
                            eigenvalues=eigenvalues,
                            right_hand_side=right_hand_side,
                            l2_encoder=float(self.cfg.l2_encoder),
                        )
                        fallbacks += int(used_fallback)
                    else:
                        gram_without_item = feature_gram - np.outer(feature, feature)
                        right_hand_side = target_sum - gram_without_item @ context_sum
                        next_encoder = _solve_direct_row(
                            support=support,
                            feature=feature,
                            feature_gram=feature_gram,
                            right_hand_side=right_hand_side,
                            l2_encoder=float(self.cfg.l2_encoder),
                        )

                delta = np.asarray(next_encoder - old_encoder, dtype=self.dtype)
                encoder[item] = next_encoder
                if users.size:
                    user_profiles[users] += delta
                row_change = float(np.linalg.norm(delta))
                squared_change += row_change * row_change
                max_row_change = max(max_row_change, row_change)

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
