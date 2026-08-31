from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, issparse, isspmatrix_csr

from compresso_recsys.models._validation import (
    canonical_csr,
    canonical_train_item_indices,
)
from compresso_recsys.models.cold_start import (
    CandidateCatalog,
    ItemFeatures,
    _LinearFeatureRecommenderMixin,
    append_column,
    canonical_feature_space_id,
    canonical_item_features,
    canonical_item_ids,
    canonical_metadata,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = ["CandidateCatalog", "TEASER", "TEASERConfig"]

TEASERDataType = Literal["float32", "float64"]


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


def _feature_gram(features: csr_matrix | np.ndarray) -> np.ndarray:
    gram = features.T @ features
    return gram.toarray() if issparse(gram) else np.asarray(gram)


def _diagonal_product(
    left: np.ndarray,
    right_transposed: csr_matrix | np.ndarray,
) -> np.ndarray:
    if issparse(right_transposed):
        return np.asarray(right_transposed.T.multiply(left).sum(axis=1)).ravel()
    return np.sum(left * right_transposed.T, axis=1)


def _right_multiply_features(
    left: np.ndarray,
    features: csr_matrix | np.ndarray,
) -> np.ndarray:
    if isspmatrix_csr(features):
        return np.asarray((features.T @ left.T).T)
    return left @ features


@dataclass(frozen=True)
class TEASERConfig:
    """Configuration for the reference ADMM implementation of TEASER.

    Parameters
    ----------
    l2_coefficients:
        L2 regularization on the diagonal-free item coefficient matrix.
    l2_encoder:
        L2 regularization on the learned item-to-feature encoder.
    rho:
        Positive ADMM penalty parameter.
    max_iterations:
        Number of fixed ADMM iterations. The reference implementation uses 10.
    include_popularity:
        Append normalized training-item popularity as an additional feature.
    dtype:
        Numerical precision used by fitting and prediction. ``float64`` matches
        the reference implementation.
    """

    l2_coefficients: float = 0.05
    l2_encoder: float = 0.05
    rho: float = 0.05
    max_iterations: int = 10
    include_popularity: bool = False
    dtype: TEASERDataType = "float64"

    def __post_init__(self) -> None:
        for name in ("l2_coefficients", "l2_encoder", "rho"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and > 0")
        if (
            isinstance(self.max_iterations, bool)
            or not isinstance(self.max_iterations, (int, np.integer))
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be >= 1")
        if not isinstance(self.include_popularity, bool):
            raise ValueError("include_popularity must be a bool")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")


class TEASER(_LinearFeatureRecommenderMixin):
    """Transparent and explainable aspect-space recommender.

    TEASER learns an item-to-feature encoder from binary implicit interactions
    while keeping the supplied item-feature matrix fixed as its decoder. The
    ADMM implementation follows the original algorithm and supports warm-item
    training with metadata-only cold candidate items.
    """

    _model_name = "TEASER"
    checkpoint_type = "teaser"

    def __init__(self, config: TEASERConfig | None = None) -> None:
        super().__init__()
        self.cfg = config if config is not None else TEASERConfig()
        self.encoder_: np.ndarray | None = None
        self.decoder_features_: csr_matrix | np.ndarray | None = None
        self.diagonal_: np.ndarray | None = None
        self.dual_: np.ndarray | None = None
        self.train_item_indices_: np.ndarray | None = None
        self.train_item_mask_: np.ndarray | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.n_items_: int | None = None
        self.n_features_: int | None = None
        self.admm_history_: list[dict[str, float]] = []

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
    ) -> TEASER:
        """Fit the original TEASER objective with ADMM.

        ``item_features`` must have one row per source item. When
        ``train_item_indices`` is supplied, only those item columns and feature
        rows participate in fitting. All supplied feature rows initialize the
        candidate catalog, allowing the remaining items to be scored cold.
        ``item_ids`` defaults to positional integer IDs for compatibility.
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
        resolved_metadata = canonical_metadata(
            metadata,
            item_ids=resolved_item_ids,
        )
        resolved_feature_space_id = canonical_feature_space_id(feature_space_id)

        train_indices = canonical_train_item_indices(
            train_item_indices,
            n_items=n_items,
        )
        if feature_names is not None:
            names = tuple(str(name) for name in feature_names)
            if len(names) != features.shape[1]:
                raise ValueError(
                    "feature_names length must match the number of item_features columns"
                )
        else:
            names = None

        x = interactions[:, train_indices].astype(self.dtype, copy=False)
        if x.nnz < 1:
            raise ValueError(
                "training item columns must contain at least one interaction"
            )
        training_features = features[train_indices]
        decoder_features = features
        source_popularity = np.zeros(n_items, dtype=self.dtype)

        if self.cfg.include_popularity:
            popularity = np.asarray(x.sum(axis=0), dtype=self.dtype).ravel()
            max_popularity = float(popularity.max(initial=0))
            if max_popularity <= 0:
                raise ValueError(
                    "cannot compute popularity without training interactions"
                )
            popularity /= max_popularity
            source_popularity[train_indices] = popularity
            training_features = append_column(training_features, popularity)
            decoder_features = append_column(decoder_features, source_popularity)
            if names is not None:
                names = (*names, "popularity")

        n_train_items = int(train_indices.size)
        feature_count = int(training_features.shape[1])
        gram = (x.T @ x).toarray()
        gram_diagonal = np.diag(gram).copy()
        feature_gram = _feature_gram(training_features)

        item_eigenvalues, item_eigenvectors = np.linalg.eigh(gram)
        feature_eigenvalues, feature_eigenvectors = np.linalg.eigh(feature_gram)
        item_eigenvalues += float(self.cfg.l2_coefficients)
        inverse_spectrum = 1.0 / (
            np.outer(item_eigenvalues, feature_eigenvalues) + float(self.cfg.l2_encoder)
        )

        diagonal = np.zeros(n_train_items, dtype=self.dtype)
        dual = np.zeros(n_train_items, dtype=self.dtype)
        diagonal_indices = np.diag_indices(n_train_items)
        encoder: np.ndarray | None = None
        history: list[dict[str, float]] = []
        iterations = _progress(
            range(self.cfg.max_iterations),
            enabled=show_progress,
            desc="TEASER ADMM",
        )
        for _ in iterations:
            right_hand_side = gram * (1.0 + diagonal)
            right_hand_side[diagonal_indices] += (
                float(self.cfg.rho) * (diagonal + dual)
                + float(self.cfg.l2_coefficients) * diagonal
            )
            right_hand_side = _right_multiply_features(
                right_hand_side,
                training_features,
            )

            transformed = item_eigenvectors.T @ right_hand_side @ feature_eigenvectors
            next_encoder = (
                item_eigenvectors
                @ (transformed * inverse_spectrum)
                @ feature_eigenvectors.T
            )

            encoder_diagonal = _diagonal_product(
                next_encoder,
                training_features.T,
            )
            next_diagonal = (
                _diagonal_product(gram.T @ next_encoder, training_features.T)
                - gram_diagonal
                + (float(self.cfg.rho) + float(self.cfg.l2_coefficients))
                * encoder_diagonal
                - float(self.cfg.rho) * dual
            )
            next_diagonal /= (
                gram_diagonal
                + float(self.cfg.l2_coefficients)
                + 2.0 * float(self.cfg.rho)
            )
            np.maximum(next_diagonal, 0, out=next_diagonal)
            dual += next_diagonal - encoder_diagonal

            history.append(
                {
                    "primal_residual": float(
                        np.linalg.norm(next_diagonal - encoder_diagonal)
                    ),
                    "dual_residual": (
                        0.0
                        if encoder is None
                        else float(
                            self.cfg.rho * np.linalg.norm(next_encoder - encoder)
                        )
                    ),
                }
            )
            encoder = next_encoder
            diagonal = next_diagonal

        assert encoder is not None
        if not np.all(np.isfinite(encoder)):
            raise np.linalg.LinAlgError(
                "TEASER fitting produced non-finite encoder values"
            )

        self.encoder_ = np.asarray(encoder, dtype=self.dtype)
        self.diagonal_ = np.asarray(diagonal, dtype=self.dtype)
        self.dual_ = np.asarray(dual, dtype=self.dtype)
        self.train_item_indices_ = train_indices.copy()
        self.train_item_mask_ = np.zeros(n_items, dtype=bool)
        self.train_item_mask_[train_indices] = True
        self.feature_names_ = names
        self.n_items_ = n_items
        self.n_features_ = feature_count
        self.admm_history_ = history
        self.candidates.install(
            source_item_ids=resolved_item_ids,
            source_popularity=source_popularity,
            n_input_features=int(features.shape[1]),
            candidate_features=decoder_features,
            metadata=resolved_metadata,
            feature_space_id=resolved_feature_space_id,
            dtype=self.dtype,
            include_popularity=self.cfg.include_popularity,
        )
        return self

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> TEASER:
        del reader, device
        return cls(TEASERConfig(**config))

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.encoder_ is not None
        assert self.diagonal_ is not None
        assert self.dual_ is not None
        assert self.train_item_indices_ is not None
        assert self.train_item_mask_ is not None
        assert self.n_items_ is not None
        assert self.n_features_ is not None
        writer.write_numpy("state/encoder.npy", self.encoder_)
        writer.write_numpy("state/diagonal.npy", self.diagonal_)
        writer.write_numpy("state/dual.npy", self.dual_)
        writer.write_numpy("state/train_item_indices.npy", self.train_item_indices_)
        writer.write_numpy("state/train_item_mask.npy", self.train_item_mask_)
        writer.write_json(
            "state/trainer.json",
            {
                "n_items": self.n_items_,
                "n_features": self.n_features_,
                "feature_names": self.feature_names_,
                "admm_history": self.admm_history_,
            },
        )
        self.candidates._save_checkpoint(writer)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        state = reader.read_json("state/trainer.json")
        self.encoder_ = np.asarray(
            reader.read_numpy("state/encoder.npy"),
            dtype=self.dtype,
        )
        self.diagonal_ = np.asarray(
            reader.read_numpy("state/diagonal.npy"),
            dtype=self.dtype,
        )
        self.dual_ = np.asarray(
            reader.read_numpy("state/dual.npy"),
            dtype=self.dtype,
        )
        self.train_item_indices_ = np.asarray(
            reader.read_numpy("state/train_item_indices.npy"),
            dtype=np.int64,
        )
        self.train_item_mask_ = np.asarray(
            reader.read_numpy("state/train_item_mask.npy"),
            dtype=bool,
        )
        self.n_items_ = int(state["n_items"])
        self.n_features_ = int(state["n_features"])
        names = state.get("feature_names")
        self.feature_names_ = None if names is None else tuple(map(str, names))
        history = state.get("admm_history")
        if not isinstance(history, list):
            raise ValueError("TEASER ADMM history must be a list")
        self.admm_history_ = list(history)
        n_train = int(self.train_item_indices_.size)
        if self.encoder_.shape != (n_train, self.n_features_):
            raise ValueError("TEASER encoder shape does not match its metadata")
        if self.diagonal_.shape != (n_train,) or self.dual_.shape != (n_train,):
            raise ValueError("TEASER diagnostic vectors do not match training items")
        if self.train_item_mask_.shape != (self.n_items_,):
            raise ValueError("TEASER training-item mask does not match n_items")
        self.candidates._load_checkpoint(reader)
        source_ids = self.candidates.source_item_ids
        assert source_ids is not None
        if source_ids.size != self.n_items_:
            raise ValueError("TEASER source catalog does not match n_items")
