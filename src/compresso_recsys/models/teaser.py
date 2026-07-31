from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix, hstack, issparse, isspmatrix_csr

from compresso import SRPTensor

__all__ = ["TEASER", "TEASERConfig"]

TEASERDataType = Literal["float32", "float64"]
ItemFeatures = csr_matrix | SRPTensor | np.ndarray | torch.Tensor


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


def _canonical_csr(matrix: csr_matrix, *, name: str) -> csr_matrix:
    if not isspmatrix_csr(matrix):
        raise TypeError(f"{name} must be a scipy.sparse.csr_matrix")
    needs_copy = not matrix.has_canonical_format or bool(np.any(matrix.data == 0))
    out = matrix.copy() if needs_copy else matrix
    if needs_copy:
        out.sum_duplicates()
        out.eliminate_zeros()
        out.sort_indices()
    if not np.all(np.isfinite(out.data)):
        raise ValueError(f"{name} values must be finite")
    return out


def _torch_sparse_to_csr(features: torch.Tensor) -> csr_matrix:
    coo = features.detach().cpu().to_sparse_coo().coalesce()
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    return csr_matrix(
        (values, (indices[0], indices[1])),
        shape=tuple(features.shape),
    )


def _canonical_item_features(
    features: ItemFeatures,
    *,
    dtype: np.dtype,
) -> csr_matrix | np.ndarray:
    if isinstance(features, SRPTensor):
        if features.dim() != 2:
            raise ValueError("item_features must be two-dimensional")
        if torch.is_complex(features.vals):
            raise TypeError("item_features must contain real numeric values")
        torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
        features = features.to(device="cpu", dtype=torch_dtype).to_scipy_csr()
    elif isinstance(features, torch.Tensor):
        if features.ndim != 2:
            raise ValueError("item_features must be two-dimensional")
        if torch.is_complex(features):
            raise TypeError("item_features must contain real numeric values")
        torch_dtype = torch.float32 if dtype == np.dtype("float32") else torch.float64
        features = features.detach().to(device="cpu", dtype=torch_dtype)
        if features.layout == torch.strided:
            features = features.numpy()
        else:
            features = _torch_sparse_to_csr(features)

    if isspmatrix_csr(features):
        out = _canonical_csr(features, name="item_features")
        if out.ndim != 2:
            raise ValueError("item_features must be two-dimensional")
        if out.shape[0] < 1 or out.shape[1] < 1:
            raise ValueError("item_features must contain at least one item and one feature")
        if np.iscomplexobj(out.data):
            raise TypeError("item_features must contain real numeric values")
        return out.astype(dtype, copy=False)

    if not isinstance(features, np.ndarray):
        raise TypeError(
            "item_features must be a scipy.sparse.csr_matrix, "
            "compresso.SRPTensor, numpy.ndarray, or torch.Tensor"
        )
    if features.ndim != 2:
        raise ValueError("item_features must be two-dimensional")
    if features.shape[0] < 1 or features.shape[1] < 1:
        raise ValueError("item_features must contain at least one item and one feature")
    if not np.issubdtype(features.dtype, np.number):
        raise TypeError("item_features must contain numeric values")
    if np.iscomplexobj(features):
        raise TypeError("item_features must contain real numeric values")
    if not np.all(np.isfinite(features)):
        raise ValueError("item_features values must be finite")
    return np.asarray(features, dtype=dtype, order="C")


def _canonical_train_item_indices(
    train_item_indices: np.ndarray | Sequence[int] | None,
    *,
    n_items: int,
) -> np.ndarray:
    if train_item_indices is None:
        return np.arange(n_items, dtype=np.int64)

    indices = np.asarray(train_item_indices)
    if indices.ndim != 1:
        raise ValueError("train_item_indices must be one-dimensional")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("train_item_indices must contain integers")
    indices = indices.astype(np.int64, copy=False)
    if indices.size < 1:
        raise ValueError("train_item_indices must contain at least one item")
    if np.any(indices < 0) or np.any(indices >= n_items):
        raise ValueError(f"train_item_indices must be in [0, {n_items - 1}]")
    if np.unique(indices).size != indices.size:
        raise ValueError("train_item_indices must not contain duplicates")
    return indices


def _append_column(
    matrix: csr_matrix | np.ndarray,
    column: np.ndarray,
) -> csr_matrix | np.ndarray:
    if isspmatrix_csr(matrix):
        return hstack((matrix, csr_matrix(column[:, None])), format="csr")
    return np.concatenate((matrix, column[:, None]), axis=1)


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
    include_popularity: bool = True
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


class TEASER:
    """Transparent and explainable aspect-space recommender.

    TEASER learns an item-to-feature encoder from binary implicit interactions
    while keeping the supplied item-feature matrix fixed as its decoder. The
    ADMM implementation follows the original algorithm and supports warm-item
    training with metadata-only cold candidate items.
    """

    def __init__(self, config: TEASERConfig | None = None) -> None:
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
        feature_names: Sequence[str] | None = None,
        show_progress: bool = False,
    ) -> TEASER:
        """Fit the original TEASER objective with ADMM.

        ``item_features`` must have one row per global item. When
        ``train_item_indices`` is supplied, only those item columns and feature
        rows participate in fitting. All feature rows remain available to the
        fixed decoder, allowing the remaining items to be scored cold.
        """
        interactions = _canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")
        if interactions.data.size and not np.all(interactions.data == 1):
            raise ValueError("interactions must contain binary implicit values equal to 1")

        features = _canonical_item_features(item_features, dtype=self.dtype)
        n_items = int(interactions.shape[1])
        if features.shape[0] != n_items:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but interactions "
                f"has {n_items} items"
            )

        train_indices = _canonical_train_item_indices(
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
            raise ValueError("training item columns must contain at least one interaction")
        training_features = features[train_indices]
        decoder_features = features

        if self.cfg.include_popularity:
            popularity = np.asarray(x.sum(axis=0), dtype=self.dtype).ravel()
            max_popularity = float(popularity.max(initial=0))
            if max_popularity <= 0:
                raise ValueError("cannot compute popularity without training interactions")
            popularity /= max_popularity
            global_popularity = np.zeros(n_items, dtype=self.dtype)
            global_popularity[train_indices] = popularity
            training_features = _append_column(training_features, popularity)
            decoder_features = _append_column(decoder_features, global_popularity)
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
            np.outer(item_eigenvalues, feature_eigenvalues)
            + float(self.cfg.l2_encoder)
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

            transformed = (
                item_eigenvectors.T
                @ right_hand_side
                @ feature_eigenvectors
            )
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
                            self.cfg.rho
                            * np.linalg.norm(next_encoder - encoder)
                        )
                    ),
                }
            )
            encoder = next_encoder
            diagonal = next_diagonal

        assert encoder is not None
        if not np.all(np.isfinite(encoder)):
            raise np.linalg.LinAlgError("TEASER fitting produced non-finite encoder values")

        self.encoder_ = np.asarray(encoder, dtype=self.dtype)
        self.decoder_features_ = decoder_features
        self.diagonal_ = np.asarray(diagonal, dtype=self.dtype)
        self.dual_ = np.asarray(dual, dtype=self.dtype)
        self.train_item_indices_ = train_indices.copy()
        self.train_item_mask_ = np.zeros(n_items, dtype=bool)
        self.train_item_mask_[train_indices] = True
        self.feature_names_ = names
        self.n_items_ = n_items
        self.n_features_ = feature_count
        self.admm_history_ = history
        return self

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if (
            not self.is_fitted
            or self.n_items_ is None
            or self.train_item_indices_ is None
            or self.train_item_mask_ is None
        ):
            raise RuntimeError("TEASER must be fitted before prediction")
        source = _canonical_csr(source, name="source")
        if source.shape[1] != self.n_items_:
            raise ValueError(
                f"source has {source.shape[1]} items, but TEASER was fitted "
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

    def user_profiles(self, source: csr_matrix) -> np.ndarray:
        """Transform binary source histories into item-feature profiles."""
        source = self._prepare_source(source)
        return self._profiles_from_prepared_source(source)

    def _profiles_from_prepared_source(self, source: csr_matrix) -> np.ndarray:
        assert self.encoder_ is not None
        assert self.train_item_indices_ is not None
        return np.asarray(
            source[:, self.train_item_indices_] @ self.encoder_,
            dtype=self.dtype,
        )

    def _score_profiles(self, profiles: np.ndarray) -> np.ndarray:
        assert self.decoder_features_ is not None
        if isspmatrix_csr(self.decoder_features_):
            scores = (self.decoder_features_ @ profiles.T).T
        else:
            scores = profiles @ self.decoder_features_.T
        return np.asarray(scores, dtype=self.dtype)

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for one source batch."""
        source = self._prepare_source(source)
        if not 1 <= int(k) <= source.shape[1]:
            raise ValueError(f"k must be in [1, {source.shape[1]}], got {k}")

        seen_counts = np.diff(source.indptr)
        if exclude_seen:
            available_counts = source.shape[1] - seen_counts
            if available_counts.size and np.any(available_counts < k):
                row = int(np.flatnonzero(available_counts < k)[0])
                raise ValueError(
                    f"source row {row} has only {available_counts[row]} unseen "
                    f"items, fewer than k={k}"
                )

        if source.shape[0] == 0:
            value_dtype = torch.from_numpy(np.empty(0, dtype=self.dtype)).dtype
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long),
                vals=torch.empty((0, k), dtype=value_dtype),
                shape=source.shape,
            )

        scores = self._score_profiles(
            self._profiles_from_prepared_source(source)
        )
        if exclude_seen:
            seen_rows = np.repeat(
                np.arange(source.shape[0], dtype=np.int64),
                seen_counts,
            )
            scores[seen_rows, source.indices] = -np.inf
        return SRPTensor.from_dense(
            torch.from_numpy(scores),
            k=int(k),
            score_mode="raw",
        )

    def predict(
        self,
        source: csr_matrix,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        show_progress: bool = False,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for all source rows in batches."""
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if not 1 <= int(k) <= source.shape[1]:
            raise ValueError(f"k must be in [1, {source.shape[1]}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        for start in _progress(
            starts,
            enabled=show_progress,
            desc=f"TEASER predict@{k}",
        ):
            end = min(start + batch_size, source.shape[0])
            predictions = self.predict_on_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
            )
            columns.append(predictions.cols)
            values.append(predictions.vals)

        if not columns:
            return self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
            )
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=source.shape,
        )
