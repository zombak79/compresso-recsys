"""Canonical item-feature matrices, and the checks that keep them comparable.

A feature space is identified by its canonical form, so two catalogs built from
the same features agree on what column means what.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Hashable,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    runtime_checkable,
)

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix, hstack, issparse, isspmatrix_csr, vstack

from compresso import SRPTensor
from compresso_recsys.models.core.validation import canonical_csr
from compresso_recsys.models.core.identifiers import (
    ItemVocabulary,
    canonical_item_ids,
)

__all__ = [
    "BaseColdStartRecommender",
    "CandidateCatalog",
    "ColdStartRecommender",
    "ItemVocabulary",
    "MutableCandidateCatalog",
    "WarmCatalogAdapter",
]

ItemFeatures = csr_matrix | SRPTensor | np.ndarray | torch.Tensor
CandidateConflict = Literal["error", "replace", "ignore"]

_NOT_INSTALLED = (
    "no candidate catalog is installed: the model has not been fitted, or "
    "install() was never called on the catalog"
)




def canonical_metadata(
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
            f"metadata has {len(metadata)} rows, but item_ids has "
            f"{item_ids.size} entries"
        )
    out = metadata.reset_index(drop=True).copy(deep=True)
    if "item_id" in out.columns:
        metadata_ids = canonical_item_ids(
            out["item_id"].tolist(),
            expected_rows=item_ids.size,
            name="metadata['item_id']",
        )
        if not np.array_equal(metadata_ids, item_ids):
            raise ValueError("metadata['item_id'] must match item_ids in row order")
    return out


def canonical_feature_space_id(feature_space_id: str | None) -> str | None:
    if feature_space_id is None:
        return None
    if not isinstance(feature_space_id, str) or not feature_space_id.strip():
        raise ValueError("feature_space_id must be a non-empty string or None")
    return feature_space_id


def _torch_sparse_to_csr(features: torch.Tensor) -> csr_matrix:
    coo = features.detach().cpu().to_sparse_coo().coalesce()
    indices = coo.indices().numpy()
    values = coo.values().numpy()
    return csr_matrix(
        (values, (indices[0], indices[1])),
        shape=tuple(features.shape),
    )


def canonical_item_features(
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
        out = canonical_csr(features, name="item_features")
        if out.ndim != 2:
            raise ValueError("item_features must be two-dimensional")
        if out.shape[0] < 1 or out.shape[1] < 1:
            raise ValueError(
                "item_features must contain at least one item and one feature"
            )
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


def append_column(
    matrix: csr_matrix | np.ndarray,
    column: np.ndarray,
) -> csr_matrix | np.ndarray:
    if isspmatrix_csr(matrix):
        return hstack((matrix, csr_matrix(column[:, None])), format="csr")
    return np.concatenate((matrix, column[:, None]), axis=1)


def take_features(
    features: csr_matrix | np.ndarray,
    rows: np.ndarray,
) -> csr_matrix | np.ndarray:
    selected = features[rows]
    return selected.tocsr() if issparse(selected) else np.asarray(selected)


def _freeze_features(features: csr_matrix | np.ndarray) -> csr_matrix | np.ndarray:
    out = features.copy()
    if isspmatrix_csr(out):
        out.data.setflags(write=False)
        out.indices.setflags(write=False)
        out.indptr.setflags(write=False)
    else:
        out.setflags(write=False)
    return out


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
