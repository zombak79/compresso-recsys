from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Literal, Sequence

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.models._validation import canonical_csr
from compresso_recsys.models.base import BaseCollaborativeRecommender
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

__all__ = ["EASE", "EASEConfig"]

EASEDataType = Literal["float32", "float64"]


def _progress(iterable, *, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except Exception:  # pragma: no cover - optional display helper
        return iterable
    return tqdm(iterable, desc=desc)


@dataclass(frozen=True)
class EASEConfig:
    """Configuration for :class:`EASE`.

    Parameters
    ----------
    l2:
        Positive L2 regularization added to the item Gram matrix diagonal.
    dtype:
        Floating-point precision used to fit and score the model. ``float32``
        is the memory-efficient default; ``float64`` is available for
        experiments that need additional numerical precision.
    """

    l2: float = 500.0
    dtype: EASEDataType = "float32"

    def __post_init__(self) -> None:
        if not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("l2 must be finite and > 0")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")


class EASE(BaseCollaborativeRecommender):
    """Embarrassingly Shallow Autoencoder recommender.

    EASE learns a closed-form item-to-item coefficient matrix from a sparse
    user-item interaction matrix. Predictions are returned as ranked
    :class:`compresso.SRPTensor` objects, with seen source items excluded by
    default.
    """

    checkpoint_type = "ease"

    def __init__(self, config: EASEConfig | None = None) -> None:
        self.cfg = config if config is not None else EASEConfig()
        self.coefficients_: np.ndarray | None = None
        self.n_items_: int | None = None

    @property
    def is_fitted(self) -> bool:
        """Whether the item coefficient matrix has been fitted."""
        return self.coefficients_ is not None

    @property
    def n_items(self) -> int | None:
        """Number of fitted item columns, or ``None`` before fitting."""
        return self.n_items_

    @property
    def dtype(self) -> np.dtype:
        """NumPy dtype used by the model."""
        return np.dtype(self.cfg.dtype)

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> EASE:
        """Fit EASE from a CSR user-item interaction matrix."""
        interactions = canonical_csr(interactions, name="interactions")
        if interactions.shape[0] < 1 or interactions.shape[1] < 1:
            raise ValueError("interactions must contain at least one user and one item")

        x = interactions.astype(self.dtype, copy=False)
        gram = (x.T @ x).toarray()
        diagonal_indices = np.diag_indices(gram.shape[0])
        gram[diagonal_indices] += float(self.cfg.l2)

        coefficients = np.linalg.inv(gram)
        del gram
        precision_diagonal = np.diag(coefficients).copy()
        if np.any(precision_diagonal == 0):
            raise np.linalg.LinAlgError("EASE precision matrix has a zero diagonal")

        coefficients /= -precision_diagonal
        coefficients[diagonal_indices] = 0
        self.coefficients_ = coefficients
        self.n_items_ = int(interactions.shape[1])
        self._set_item_ids(item_ids, n_items=self.n_items_)
        return self

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> EASE:
        del reader, device
        return cls(EASEConfig(**config))

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        assert self.coefficients_ is not None
        writer.write_numpy("state/coefficients.npy", self.coefficients_)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        coefficients = reader.read_numpy("state/coefficients.npy")
        if coefficients.ndim != 2 or coefficients.shape[0] != coefficients.shape[1]:
            raise ValueError("EASE coefficients must be a square matrix")
        if coefficients.shape[0] < 1:
            raise ValueError("EASE coefficients must contain at least one item")
        if coefficients.dtype != self.dtype:
            raise ValueError(
                f"EASE coefficients use {coefficients.dtype}, expected {self.dtype}"
            )
        self.coefficients_ = coefficients
        self.n_items_ = int(coefficients.shape[0])

    def _prepare_source(self, source: csr_matrix) -> csr_matrix:
        if not self.is_fitted or self.n_items_ is None:
            raise RuntimeError("EASE must be fitted before prediction")
        source = canonical_csr(source, name="source")
        if source.shape[1] != self.n_items_:
            raise ValueError(
                f"source has {source.shape[1]} items, but EASE was fitted with "
                f"{self.n_items_} items"
            )
        return source

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
        candidate_rows = self._candidate_rows(candidate_ids)
        candidate_count = int(candidate_rows.size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

        seen_counts = np.diff(source.indptr)
        if exclude_seen:
            selected = np.zeros(source.shape[1], dtype=bool)
            selected[candidate_rows] = True
            selected_seen = selected[source.indices]
            seen_rows = np.repeat(
                np.arange(source.shape[0], dtype=np.int64),
                seen_counts,
            )
            selected_seen_counts = np.bincount(
                seen_rows[selected_seen],
                minlength=source.shape[0],
            )
            available_counts = candidate_count - selected_seen_counts
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

        assert self.coefficients_ is not None
        scores = np.asarray(
            source @ self.coefficients_[:, candidate_rows],
            dtype=self.dtype,
        )
        seen_rows = np.repeat(
            np.arange(source.shape[0], dtype=np.int64),
            seen_counts,
        )
        if exclude_seen:
            candidate_to_local = np.full(source.shape[1], -1, dtype=np.int64)
            candidate_to_local[candidate_rows] = np.arange(candidate_count)
            seen_local = candidate_to_local[source.indices]
            in_selection = seen_local >= 0
            scores[seen_rows[in_selection], seen_local[in_selection]] = -np.inf
        local = SRPTensor.from_dense(
            torch.from_numpy(scores),
            k=int(k),
            score_mode="raw",
        )
        global_rows = torch.from_numpy(candidate_rows).to(local.cols.device)
        return SRPTensor(
            cols=global_rows[local.cols],
            vals=local.vals,
            shape=source.shape,
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
        candidate_count = int(self._candidate_rows(candidate_ids).size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.shape[0], batch_size)
        for start in _progress(starts, enabled=show_progress, desc=f"EASE predict@{k}"):
            end = min(start + batch_size, source.shape[0])
            predictions = self.predict_on_batch(
                source[start:end],
                k=k,
                exclude_seen=exclude_seen,
                candidate_ids=candidate_ids,
            )
            columns.append(predictions.cols)
            values.append(predictions.vals)

        if not columns:
            return self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                candidate_ids=candidate_ids,
            )
        return SRPTensor(
            cols=torch.vstack(columns),
            vals=torch.vstack(values),
            shape=source.shape,
        )
