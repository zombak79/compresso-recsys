from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from types import MappingProxyType
from typing import Hashable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack, issparse, isspmatrix_csr, vstack

from compresso import SRPTensor

__all__ = ["CandidateCatalog", "TEASER", "TEASERConfig"]

TEASERDataType = Literal["float32", "float64"]
ItemFeatures = csr_matrix | SRPTensor | np.ndarray | torch.Tensor
CandidateConflict = Literal["error", "replace", "ignore"]


@dataclass(frozen=True, init=False)
class CandidateCatalog:
    """Immutable snapshot of the candidates available to TEASER.

    Candidate rows define the column space of prediction tensors. ``version``
    increases whenever the owning model replaces, updates, or removes
    candidates, allowing callers to identify the snapshot used for serving.
    """

    item_ids: np.ndarray
    item_features: csr_matrix | np.ndarray
    _metadata: pd.DataFrame | None = field(repr=False)
    feature_space_id: str | None
    version: int
    id_to_row: Mapping[Hashable, int]

    def __init__(
        self,
        *,
        item_ids: np.ndarray,
        item_features: csr_matrix | np.ndarray,
        metadata: pd.DataFrame | None,
        feature_space_id: str | None,
        version: int,
        id_to_row: Mapping[Hashable, int],
    ) -> None:
        object.__setattr__(self, "item_ids", item_ids)
        object.__setattr__(self, "item_features", item_features)
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "feature_space_id", feature_space_id)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "id_to_row", id_to_row)

    @property
    def n_items(self) -> int:
        """Number of candidate items in this snapshot."""
        return int(self.item_ids.size)

    @property
    def metadata(self) -> pd.DataFrame | None:
        """A defensive copy of metadata aligned with candidate rows."""
        return None if self._metadata is None else self._metadata.copy(deep=True)

    def rows_for(self, item_ids: Sequence[Hashable]) -> np.ndarray:
        """Resolve registered item IDs to candidate rows in request order."""
        ids = _canonical_item_ids(item_ids, name="candidate_ids")
        rows = np.empty(ids.size, dtype=np.int64)
        for position, item_id in enumerate(ids.tolist()):
            try:
                rows[position] = self.id_to_row[item_id]
            except KeyError as error:
                raise KeyError(f"unknown candidate item ID: {item_id!r}") from error
        return rows

    def ids_for(self, rows: np.ndarray | torch.Tensor) -> np.ndarray:
        """Resolve candidate row indices to stable item IDs."""
        if isinstance(rows, torch.Tensor):
            rows = rows.detach().cpu().numpy()
        row_array = np.asarray(rows)
        if not np.issubdtype(row_array.dtype, np.integer):
            raise TypeError("candidate rows must contain integers")
        if row_array.size and (
            int(row_array.min()) < 0 or int(row_array.max()) >= self.n_items
        ):
            raise IndexError("candidate row is out of bounds")
        return self.item_ids[row_array]


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


def _canonical_item_ids(
    item_ids: Sequence[Hashable] | np.ndarray,
    *,
    expected_rows: int | None = None,
    name: str = "item_ids",
) -> np.ndarray:
    if isinstance(item_ids, (str, bytes)):
        raise TypeError(f"{name} must be a one-dimensional sequence of item IDs")
    try:
        values = list(item_ids)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a one-dimensional sequence of item IDs"
        ) from error
    ids = np.empty(len(values), dtype=object)
    ids[:] = values
    if expected_rows is not None and ids.size != expected_rows:
        raise ValueError(
            f"{name} has {ids.size} entries, but item_features has "
            f"{expected_rows} rows"
        )
    if ids.size < 1:
        raise ValueError(f"{name} must contain at least one item")

    seen: dict[Hashable, int] = {}
    for position, item_id in enumerate(ids.tolist()):
        try:
            hash(item_id)
        except TypeError as error:
            raise TypeError(
                f"{name} entry at position {position} is not hashable"
            ) from error
        try:
            missing = bool(pd.isna(item_id))
        except (TypeError, ValueError):
            missing = False
        if item_id is None or missing:
            raise ValueError(f"{name} must not contain missing IDs")
        if item_id in seen:
            raise ValueError(f"{name} must not contain duplicate IDs: {item_id!r}")
        seen[item_id] = position

    ids.setflags(write=False)
    return ids


def _canonical_metadata(
    metadata: pd.DataFrame | None,
    *,
    item_ids: np.ndarray,
) -> pd.DataFrame | None:
    if metadata is None:
        return None
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError("metadata must be a pandas.DataFrame or None")
    if len(metadata) != item_ids.size:
        raise ValueError(
            f"metadata has {len(metadata)} rows, but item_ids has {item_ids.size} entries"
        )
    out = metadata.reset_index(drop=True).copy(deep=True)
    if "item_id" in out.columns:
        metadata_ids = _canonical_item_ids(
            out["item_id"].tolist(),
            expected_rows=item_ids.size,
            name="metadata['item_id']",
        )
        if not np.array_equal(metadata_ids, item_ids):
            raise ValueError("metadata['item_id'] must match item_ids in row order")
    return out


def _canonical_feature_space_id(feature_space_id: str | None) -> str | None:
    if feature_space_id is None:
        return None
    if not isinstance(feature_space_id, str) or not feature_space_id.strip():
        raise ValueError("feature_space_id must be a non-empty string or None")
    return feature_space_id


def _freeze_features(features: csr_matrix | np.ndarray) -> csr_matrix | np.ndarray:
    out = features.copy()
    if isspmatrix_csr(out):
        out.data.setflags(write=False)
        out.indices.setflags(write=False)
        out.indptr.setflags(write=False)
    else:
        out.setflags(write=False)
    return out


def _make_catalog(
    *,
    item_ids: np.ndarray,
    item_features: csr_matrix | np.ndarray,
    metadata: pd.DataFrame | None,
    feature_space_id: str | None,
    version: int,
) -> CandidateCatalog:
    frozen_ids = item_ids.copy()
    frozen_ids.setflags(write=False)
    return CandidateCatalog(
        item_ids=frozen_ids,
        item_features=_freeze_features(item_features),
        metadata=None if metadata is None else metadata.copy(deep=True),
        feature_space_id=feature_space_id,
        version=int(version),
        id_to_row=MappingProxyType(
            {item_id: row for row, item_id in enumerate(frozen_ids.tolist())}
        ),
    )


def _take_features(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
) -> csr_matrix | np.ndarray:
    selected = features[rows]
    return selected.tocsr() if issparse(selected) else np.asarray(selected)


def _stack_features(
    top: csr_matrix | np.ndarray,
    bottom: csr_matrix | np.ndarray,
) -> csr_matrix | np.ndarray:
    if isspmatrix_csr(top) or isspmatrix_csr(bottom):
        return vstack((csr_matrix(top), csr_matrix(bottom)), format="csr")
    return np.concatenate((top, bottom), axis=0)


def _replace_feature_rows(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
    replacements: csr_matrix | np.ndarray,
) -> csr_matrix | np.ndarray:
    if rows.size == 0:
        return features
    if isinstance(features, np.ndarray) and isinstance(replacements, np.ndarray):
        out = features.copy()
        out[rows] = replacements
        return out
    out = csr_matrix(features).tolil(copy=True)
    out[rows] = csr_matrix(replacements)
    return out.tocsr()


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
        self._catalog_lock = RLock()
        self._candidates: CandidateCatalog | None = None
        self.encoder_: np.ndarray | None = None
        self.decoder_features_: csr_matrix | np.ndarray | None = None
        self.diagonal_: np.ndarray | None = None
        self.dual_: np.ndarray | None = None
        self.train_item_indices_: np.ndarray | None = None
        self.train_item_mask_: np.ndarray | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.source_item_ids_: np.ndarray | None = None
        self.source_id_to_row_: Mapping[Hashable, int] | None = None
        self.source_popularity_: np.ndarray | None = None
        self.feature_space_id_: str | None = None
        self.n_input_features_: int | None = None
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

    @property
    def candidates(self) -> CandidateCatalog:
        """Current immutable candidate-catalog snapshot."""
        catalog = self._candidates
        if catalog is None:
            raise RuntimeError("TEASER must be fitted before accessing candidates")
        return catalog

    @property
    def n_candidates_(self) -> int | None:
        """Number of candidates currently available for prediction."""
        return None if self._candidates is None else self._candidates.n_items

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

        resolved_item_ids = _canonical_item_ids(
            np.arange(n_items, dtype=np.int64) if item_ids is None else item_ids,
            expected_rows=n_items,
        )
        resolved_metadata = _canonical_metadata(
            metadata,
            item_ids=resolved_item_ids,
        )
        resolved_feature_space_id = _canonical_feature_space_id(feature_space_id)

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
        source_popularity = np.zeros(n_items, dtype=self.dtype)

        if self.cfg.include_popularity:
            popularity = np.asarray(x.sum(axis=0), dtype=self.dtype).ravel()
            max_popularity = float(popularity.max(initial=0))
            if max_popularity <= 0:
                raise ValueError("cannot compute popularity without training interactions")
            popularity /= max_popularity
            source_popularity[train_indices] = popularity
            training_features = _append_column(training_features, popularity)
            decoder_features = _append_column(decoder_features, source_popularity)
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

        catalog = _make_catalog(
            item_ids=resolved_item_ids,
            item_features=decoder_features,
            metadata=resolved_metadata,
            feature_space_id=resolved_feature_space_id,
            version=1,
        )
        source_item_ids = resolved_item_ids.copy()
        source_item_ids.setflags(write=False)

        self.encoder_ = np.asarray(encoder, dtype=self.dtype)
        self.decoder_features_ = catalog.item_features
        self.diagonal_ = np.asarray(diagonal, dtype=self.dtype)
        self.dual_ = np.asarray(dual, dtype=self.dtype)
        self.train_item_indices_ = train_indices.copy()
        self.train_item_mask_ = np.zeros(n_items, dtype=bool)
        self.train_item_mask_[train_indices] = True
        self.feature_names_ = names
        self.source_item_ids_ = source_item_ids
        self.source_id_to_row_ = MappingProxyType(
            {item_id: row for row, item_id in enumerate(source_item_ids.tolist())}
        )
        source_popularity.setflags(write=False)
        self.source_popularity_ = source_popularity
        self.feature_space_id_ = resolved_feature_space_id
        self.n_input_features_ = int(features.shape[1])
        self.n_items_ = n_items
        self.n_features_ = feature_count
        self.admm_history_ = history
        with self._catalog_lock:
            self._candidates = catalog
        return self

    def _prepare_catalog_features(
        self,
        item_ids: np.ndarray,
        item_features: ItemFeatures,
    ) -> csr_matrix | np.ndarray:
        if (
            not self.is_fitted
            or self.n_input_features_ is None
            or self.source_id_to_row_ is None
            or self.source_popularity_ is None
        ):
            raise RuntimeError("TEASER must be fitted before changing candidates")
        features = _canonical_item_features(item_features, dtype=self.dtype)
        if features.shape[0] != item_ids.size:
            raise ValueError(
                f"item_features has {features.shape[0]} rows, but item_ids "
                f"has {item_ids.size} entries"
            )
        if features.shape[1] != self.n_input_features_:
            raise ValueError(
                f"item_features has {features.shape[1]} columns, but TEASER was "
                f"fitted with {self.n_input_features_} input features"
            )
        if self.cfg.include_popularity:
            popularity = np.zeros(item_ids.size, dtype=self.dtype)
            for row, item_id in enumerate(item_ids.tolist()):
                source_row = self.source_id_to_row_.get(item_id)
                if source_row is not None:
                    popularity[row] = self.source_popularity_[source_row]
            features = _append_column(features, popularity)
        return features

    def _resolve_catalog_feature_space_id(
        self,
        feature_space_id: str | None,
    ) -> str | None:
        resolved = _canonical_feature_space_id(feature_space_id)
        if resolved is None:
            return self.feature_space_id_
        if resolved != self.feature_space_id_:
            raise ValueError(
                "feature_space_id must match the feature space used to fit TEASER; "
                "set feature_space_id during fit to enable this check"
            )
        return resolved

    def build_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Atomically replace the complete candidate catalog.

        The fitted source vocabulary and encoder are unchanged. New IDs are
        decoder-only candidates and therefore cannot appear in source history.
        """
        ids = _canonical_item_ids(item_ids)
        candidate_metadata = _canonical_metadata(metadata, item_ids=ids)
        features = self._prepare_catalog_features(ids, item_features)
        resolved_space = self._resolve_catalog_feature_space_id(feature_space_id)

        with self._catalog_lock:
            current = self.candidates
            catalog = _make_catalog(
                item_ids=ids,
                item_features=features,
                metadata=candidate_metadata,
                feature_space_id=resolved_space,
                version=current.version + 1,
            )
            self._candidates = catalog
            self.decoder_features_ = catalog.item_features
        return catalog

    def update_candidates(
        self,
        *,
        item_ids: Sequence[Hashable] | np.ndarray,
        item_features: ItemFeatures,
        metadata: pd.DataFrame | None = None,
        on_conflict: CandidateConflict = "error",
        feature_space_id: str | None = None,
    ) -> CandidateCatalog:
        """Add or update candidates and atomically publish a new snapshot.

        Existing rows retain their positions. New rows are appended in request
        order. ``replace`` updates conflicting rows, while ``ignore`` leaves
        them unchanged.
        """
        if on_conflict not in {"error", "replace", "ignore"}:
            raise ValueError("on_conflict must be 'error', 'replace', or 'ignore'")
        ids = _canonical_item_ids(item_ids)
        incoming_metadata = _canonical_metadata(metadata, item_ids=ids)
        incoming_features = self._prepare_catalog_features(ids, item_features)
        resolved_space = self._resolve_catalog_feature_space_id(feature_space_id)

        with self._catalog_lock:
            current = self.candidates
            conflicts = np.array(
                [item_id in current.id_to_row for item_id in ids.tolist()],
                dtype=bool,
            )
            if on_conflict == "error" and bool(conflicts.any()):
                first = ids[int(np.flatnonzero(conflicts)[0])]
                raise ValueError(f"candidate item ID already exists: {first!r}")

            replace_input_rows = (
                np.flatnonzero(conflicts)
                if on_conflict == "replace"
                else np.empty(0, dtype=np.int64)
            )
            replace_catalog_rows = np.asarray(
                [current.id_to_row[ids[row]] for row in replace_input_rows],
                dtype=np.int64,
            )
            additions = np.flatnonzero(~conflicts)
            if replace_input_rows.size == 0 and additions.size == 0:
                return current

            features = _replace_feature_rows(
                current.item_features,
                replace_catalog_rows,
                _take_features(incoming_features, replace_input_rows),
            )
            if additions.size:
                features = _stack_features(
                    features,
                    _take_features(incoming_features, additions),
                )
                combined_ids = np.concatenate((current.item_ids, ids[additions]))
            else:
                combined_ids = current.item_ids.copy()

            combined_metadata = self._updated_metadata(
                current=current,
                incoming=incoming_metadata,
                replace_input_rows=replace_input_rows,
                replace_catalog_rows=replace_catalog_rows,
                addition_input_rows=additions,
            )
            catalog = _make_catalog(
                item_ids=combined_ids,
                item_features=features,
                metadata=combined_metadata,
                feature_space_id=resolved_space,
                version=current.version + 1,
            )
            self._candidates = catalog
            self.decoder_features_ = catalog.item_features
        return catalog

    @staticmethod
    def _updated_metadata(
        *,
        current: CandidateCatalog,
        incoming: pd.DataFrame | None,
        replace_input_rows: np.ndarray,
        replace_catalog_rows: np.ndarray,
        addition_input_rows: np.ndarray,
    ) -> pd.DataFrame | None:
        if current.metadata is None and incoming is None:
            return None
        old = (
            current.metadata.copy(deep=True)
            if current.metadata is not None
            else pd.DataFrame(index=range(current.n_items))
        )
        new = incoming if incoming is not None else pd.DataFrame()
        columns = old.columns.union(new.columns, sort=False)
        result = old.reindex(columns=columns)

        if addition_input_rows.size:
            additions = pd.DataFrame(
                pd.NA,
                index=range(addition_input_rows.size),
                columns=columns,
            )
            if incoming is not None:
                additions.loc[:, incoming.columns] = (
                    incoming.iloc[addition_input_rows].reset_index(drop=True).to_numpy()
                )
            result = pd.concat((result, additions), ignore_index=True)

        if incoming is not None and replace_input_rows.size:
            result.loc[replace_catalog_rows, incoming.columns] = (
                incoming.iloc[replace_input_rows].to_numpy()
            )
        return result.reset_index(drop=True)

    def remove_candidates(
        self,
        item_ids: Sequence[Hashable] | np.ndarray,
        *,
        missing: Literal["error", "ignore"] = "error",
    ) -> CandidateCatalog:
        """Remove registered candidates and atomically publish a new snapshot."""
        if missing not in {"error", "ignore"}:
            raise ValueError("missing must be 'error' or 'ignore'")
        ids = _canonical_item_ids(item_ids)
        with self._catalog_lock:
            current = self.candidates
            unknown = [item_id for item_id in ids.tolist() if item_id not in current.id_to_row]
            if unknown and missing == "error":
                raise KeyError(f"unknown candidate item ID: {unknown[0]!r}")
            removed = {item_id for item_id in ids.tolist() if item_id in current.id_to_row}
            if not removed:
                return current
            keep = np.asarray(
                [item_id not in removed for item_id in current.item_ids.tolist()],
                dtype=bool,
            )
            if not bool(keep.any()):
                raise ValueError("candidate catalog must contain at least one item")
            rows = np.flatnonzero(keep)
            metadata = (
                None
                if current.metadata is None
                else current.metadata.iloc[rows].reset_index(drop=True)
            )
            catalog = _make_catalog(
                item_ids=current.item_ids[rows],
                item_features=_take_features(current.item_features, rows),
                metadata=metadata,
                feature_space_id=current.feature_space_id,
                version=current.version + 1,
            )
            self._candidates = catalog
            self.decoder_features_ = catalog.item_features
        return catalog

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

    def _score_profiles(
        self,
        profiles: np.ndarray,
        *,
        candidate_features: csr_matrix | np.ndarray,
    ) -> np.ndarray:
        if isspmatrix_csr(candidate_features):
            scores = (candidate_features @ profiles.T).T
        else:
            scores = profiles @ candidate_features.T
        return np.asarray(scores, dtype=self.dtype)

    @staticmethod
    def _resolve_candidate_rows(
        catalog: CandidateCatalog,
        candidate_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> np.ndarray:
        if candidate_ids is None:
            return np.arange(catalog.n_items, dtype=np.int64)
        # Catalog order makes tie behavior deterministic regardless of request order.
        return np.sort(catalog.rows_for(candidate_ids))

    @staticmethod
    def _candidate_features_for_rows(
        catalog: CandidateCatalog,
        candidate_rows: np.ndarray,
    ) -> csr_matrix | np.ndarray:
        if candidate_rows.size == catalog.n_items:
            return catalog.item_features
        return _take_features(catalog.item_features, candidate_rows)

    def _source_to_candidate_rows(self, catalog: CandidateCatalog) -> np.ndarray:
        assert self.source_item_ids_ is not None
        return np.fromiter(
            (
                catalog.id_to_row.get(item_id, -1)
                for item_id in self.source_item_ids_.tolist()
            ),
            dtype=np.int64,
            count=self.source_item_ids_.size,
        )

    def _predict_prepared_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool,
        catalog: CandidateCatalog,
        candidate_rows: np.ndarray,
        candidate_features: csr_matrix | np.ndarray,
        source_to_candidate_rows: np.ndarray,
        candidate_to_local: np.ndarray,
    ) -> SRPTensor:
        if not 1 <= int(k) <= candidate_rows.size:
            raise ValueError(
                f"k must be in [1, {candidate_rows.size}], got {k}"
            )

        seen_counts = np.diff(source.indptr)
        source_rows = np.repeat(
            np.arange(source.shape[0], dtype=np.int64),
            seen_counts,
        )
        seen_candidate_rows = source_to_candidate_rows[source.indices]
        registered = seen_candidate_rows >= 0
        seen_local_rows = np.full(seen_candidate_rows.shape, -1, dtype=np.int64)
        seen_local_rows[registered] = candidate_to_local[seen_candidate_rows[registered]]
        selected_seen = seen_local_rows >= 0

        if exclude_seen:
            selected_seen_counts = np.bincount(
                source_rows[selected_seen],
                minlength=source.shape[0],
            )
            available_counts = candidate_rows.size - selected_seen_counts
            if available_counts.size and np.any(available_counts < k):
                row = int(np.flatnonzero(available_counts < k)[0])
                raise ValueError(
                    f"source row {row} has only {available_counts[row]} unseen "
                    f"items among the selected candidates, fewer than k={k}"
                )

        if source.shape[0] == 0:
            value_dtype = torch.from_numpy(np.empty(0, dtype=self.dtype)).dtype
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long),
                vals=torch.empty((0, k), dtype=value_dtype),
                shape=(0, catalog.n_items),
            )

        scores = self._score_profiles(
            self._profiles_from_prepared_source(source),
            candidate_features=candidate_features,
        )
        if exclude_seen and bool(selected_seen.any()):
            scores[source_rows[selected_seen], seen_local_rows[selected_seen]] = -np.inf
        local_predictions = SRPTensor.from_dense(
            torch.from_numpy(scores),
            k=int(k),
            score_mode="raw",
        )
        global_columns = torch.from_numpy(candidate_rows).to(local_predictions.cols.device)[
            local_predictions.cols
        ]
        return SRPTensor(
            cols=global_columns,
            vals=local_predictions.vals,
            shape=(source.shape[0], catalog.n_items),
        )

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Predict ranked top-``k`` items for one source batch.

        ``candidate_ids`` optionally restricts scoring to registered IDs. The
        returned columns still refer to rows of the complete catalog snapshot.
        """
        source = self._prepare_source(source)
        catalog = self.candidates
        candidate_rows = self._resolve_candidate_rows(catalog, candidate_ids)
        candidate_features = self._candidate_features_for_rows(catalog, candidate_rows)
        source_to_candidate_rows = self._source_to_candidate_rows(catalog)
        candidate_to_local = np.full(catalog.n_items, -1, dtype=np.int64)
        candidate_to_local[candidate_rows] = np.arange(candidate_rows.size, dtype=np.int64)
        return self._predict_prepared_batch(
            source,
            k=k,
            exclude_seen=exclude_seen,
            catalog=catalog,
            candidate_rows=candidate_rows,
            candidate_features=candidate_features,
            source_to_candidate_rows=source_to_candidate_rows,
            candidate_to_local=candidate_to_local,
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
        """Predict ranked top-``k`` items for all source rows in batches."""
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        catalog = self.candidates
        candidate_rows = self._resolve_candidate_rows(catalog, candidate_ids)
        candidate_features = self._candidate_features_for_rows(catalog, candidate_rows)
        source_to_candidate_rows = self._source_to_candidate_rows(catalog)
        candidate_to_local = np.full(catalog.n_items, -1, dtype=np.int64)
        candidate_to_local[candidate_rows] = np.arange(candidate_rows.size, dtype=np.int64)
        if not 1 <= int(k) <= candidate_rows.size:
            raise ValueError(f"k must be in [1, {candidate_rows.size}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        for start in _progress(
            starts,
            enabled=show_progress,
            desc=f"TEASER predict@{k}",
        ):
            end = min(start + batch_size, source.shape[0])
            predictions = self._predict_prepared_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
                catalog=catalog,
                candidate_rows=candidate_rows,
                candidate_features=candidate_features,
                source_to_candidate_rows=source_to_candidate_rows,
                candidate_to_local=candidate_to_local,
            )
            columns.append(predictions.cols)
            values.append(predictions.vals)

        if not columns:
            return self._predict_prepared_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                catalog=catalog,
                candidate_rows=candidate_rows,
                candidate_features=candidate_features,
                source_to_candidate_rows=source_to_candidate_rows,
                candidate_to_local=candidate_to_local,
            )
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=(source.shape[0], catalog.n_items),
        )
