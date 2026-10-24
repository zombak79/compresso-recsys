"""Named feature matrices and their metadata, canonicalised.

A multi-modal catalog holds one matrix per modality rather than one matrix
overall, so every check the single-matrix family does per array has to be
done per name, and the names themselves have to survive a checkpoint.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.models.core.features import (
    ItemFeatures,
    canonical_item_features,
)
from compresso_recsys.persistence import (
    _decoded_item_id,
    _encoded_item_id,
)


MultiModalItemFeatures = Mapping[str, ItemFeatures]


_StoredFeatures = Mapping[str, csr_matrix | np.ndarray]


def _canonical_features(
    features: MultiModalItemFeatures,
    *,
    n_items: int,
    dtype: np.dtype,
    dimensions: Mapping[str, int] | None = None,
) -> dict[str, csr_matrix | np.ndarray]:
    if not isinstance(features, Mapping):
        raise TypeError("item_features must be a mapping of modality names to matrices")
    if not features:
        raise ValueError("item_features must contain at least one modality")
    if any(not isinstance(name, str) or not name.strip() for name in features):
        raise ValueError("modality names must be non-empty strings")
    if dimensions is not None and set(features) != set(dimensions):
        raise ValueError(
            "modality names must match the installed schema; "
            f"missing={sorted(set(dimensions) - set(features))}, "
            f"extra={sorted(set(features) - set(dimensions))}"
        )
    result = {}
    # Use the installed order on updates, never the caller's dictionary order.
    for name in features if dimensions is None else dimensions:
        matrix = canonical_item_features(features[name], dtype=dtype)
        values = matrix.data if isinstance(matrix, csr_matrix) else matrix
        if not np.all(np.isfinite(values)):
            raise ValueError(f"item_features[{name!r}] must remain finite in {dtype}")
        if matrix.shape[0] != n_items:
            raise ValueError(
                f"item_features[{name!r}] has {matrix.shape[0]} rows, "
                f"but item_ids has {n_items} entries"
            )
        if dimensions is not None and matrix.shape[1] != dimensions[name]:
            raise ValueError(
                f"item_features[{name!r}] has {matrix.shape[1]} columns, "
                f"expected {dimensions[name]}"
            )
        result[name] = matrix
    return result


def _canonical_dtype(dtype: str | np.dtype) -> np.dtype:
    # NumPy interprets None as float64; an unset config must not choose precision.
    if dtype is None:
        raise ValueError("dtype must be float32 or float64")
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype("float32"), np.dtype("float64")):
        raise ValueError("dtype must be float32 or float64")
    return resolved


def _encode_metadata_label(label: Hashable) -> dict:
    """Reuse the safe scalar codec, adding null, NaN and tuple metadata labels."""
    if label is None:
        return {"type": "none"}
    # Pandas can coerce missing column labels to NaN; keep it distinct from None.
    if isinstance(label, (float, np.floating)) and np.isnan(label):
        return {"type": "nan"}
    if isinstance(label, tuple):
        return {"type": "tuple", "value": [_encode_metadata_label(v) for v in label]}
    try:
        return _encoded_item_id(label)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"unsupported metadata column/index label: {label!r} "
            f"({type(label).__name__})"
        ) from error


def _decode_metadata_label(state: object) -> Hashable:
    if isinstance(state, dict):
        if state.get("type") == "none":
            return None
        if state.get("type") == "nan":
            return np.nan
        if state.get("type") == "tuple" and isinstance(state.get("value"), list):
            return tuple(_decode_metadata_label(v) for v in state["value"])
    return _decoded_item_id(state)


def _metadata_columns_state(columns: pd.Index) -> dict:
    state = {
        "labels": [_encode_metadata_label(label) for label in columns],
        "names": [_encode_metadata_label(name) for name in columns.names],
    }
    if isinstance(columns, pd.MultiIndex):
        state["kind"] = "multi"
        state["level_dtypes"] = [str(level.dtype) for level in columns.levels]
    elif isinstance(columns, pd.RangeIndex):
        state.update(
            kind="range", start=columns.start, stop=columns.stop, step=columns.step
        )
    else:
        state.update(kind="index", dtype=str(columns.dtype))
    return state


def _restore_metadata_columns(state: object, *, n_columns: int) -> pd.Index:
    try:
        if not isinstance(state, dict):
            raise TypeError("column description must be an object")
        labels, names = state["labels"], state["names"]
        if not isinstance(labels, list) or not isinstance(names, list):
            raise TypeError("column labels and names must be lists")
        if len(labels) != n_columns:
            raise ValueError("column count does not match metadata")
        labels = [_decode_metadata_label(label) for label in labels]
        names = [_decode_metadata_label(name) for name in names]
        if state["kind"] == "multi":
            columns = pd.MultiIndex.from_tuples(labels, names=names)
            if "level_dtypes" in state:
                dtypes = state["level_dtypes"]
                if (
                    not isinstance(dtypes, list)
                    or len(dtypes) != columns.nlevels
                    or any(not isinstance(dtype, str) for dtype in dtypes)
                ):
                    raise ValueError("invalid multi-level column dtypes")
                # Iterating a level with NaN can turn its integer labels into
                # floats. Restore the level dtype independently of missing codes.
                columns = columns.set_levels(
                    [
                        level.astype(dtype)
                        for level, dtype in zip(columns.levels, dtypes)
                    ]
                )
            return columns
        if len(names) != 1:
            raise ValueError("flat columns must have one index name")
        if state["kind"] == "range":
            columns = pd.RangeIndex(
                state["start"], state["stop"], state["step"], name=names[0]
            )
            if len(columns) != n_columns or columns.tolist() != labels:
                raise ValueError("range does not match column labels")
            return columns
        if state["kind"] == "index" and isinstance(state.get("dtype"), str):
            return pd.Index(
                labels, dtype=state["dtype"], name=names[0], tupleize_cols=False
            )
        raise ValueError("unsupported column index description")
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid checkpoint metadata column description") from error
