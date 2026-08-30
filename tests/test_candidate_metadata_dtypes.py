from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from compresso_recsys.models import ContentRecommender

N_ITEMS = 6
N_FEAT = 4


def _model():
    features = np.random.default_rng(0).normal(size=(N_ITEMS, N_FEAT)).astype("float32")
    item_ids = np.array([f"item-{i}" for i in range(N_ITEMS)], dtype=object)
    return ContentRecommender().fit(features, item_ids=item_ids), features, item_ids


def _with_metadata(metadata: pd.DataFrame):
    model, features, item_ids = _model()
    model.build_candidates(
        item_ids=item_ids, item_features=features, metadata=metadata
    )
    return model, features


@pytest.mark.parametrize(
    ("column", "values", "expected_dtype_kind"),
    [
        ("score", np.arange(N_ITEMS, dtype="float64"), "f"),
        ("released", pd.to_datetime(["2020-01-01"] * N_ITEMS), "M"),
        ("count", np.arange(N_ITEMS, dtype="int64"), "f"),
    ],
)
def test_appending_items_preserves_metadata_dtype(column, values, expected_dtype_kind):
    """Appending rows without metadata must not widen existing columns to object.

    pandas resolves concat dtypes while excluding all-NA operands, warns that it
    will stop, and then silently returns object once it does. Extending the index
    instead keeps each column's own promotion rules, so the dtype is the same on
    pandas 2 and 3.
    """
    model, features = _with_metadata(pd.DataFrame({column: values}))

    model.update_candidates(item_ids=["new-a", "new-b"], item_features=features[:2])

    metadata = model.candidates.snapshot().metadata
    assert metadata is not None
    assert len(metadata) == N_ITEMS + 2
    assert metadata[column].dtype.kind == expected_dtype_kind
    # The appended rows carry no metadata.
    assert metadata[column].iloc[N_ITEMS:].isna().all()
    # The originals are untouched.
    assert not metadata[column].iloc[:N_ITEMS].isna().any()


def test_appending_items_emits_no_pandas_warning():
    model, features = _with_metadata(
        pd.DataFrame({"score": np.arange(N_ITEMS, dtype="float64")})
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.update_candidates(item_ids=["new-a"], item_features=features[:1])

    concat_warnings = [
        w for w in caught if "concatenation" in str(w.message).lower()
    ]
    assert not concat_warnings, [str(w.message) for w in concat_warnings]


def test_metadata_introduced_by_an_update_accepts_non_numeric_values():
    """A column absent from the catalog must accept whatever the update carries."""
    model, features, item_ids = _model()

    model.update_candidates(
        item_ids=["new-a", "new-b"],
        item_features=features[:2],
        metadata=pd.DataFrame({"tag": ["x", "y"]}),
    )

    metadata = model.candidates.snapshot().metadata
    assert metadata is not None
    assert list(metadata["tag"].iloc[-2:]) == ["x", "y"]
    assert metadata["tag"].iloc[:N_ITEMS].isna().all()


def test_appended_metadata_values_land_on_the_appended_rows():
    model, features = _with_metadata(
        pd.DataFrame({"score": np.arange(N_ITEMS, dtype="float64")})
    )

    model.update_candidates(
        item_ids=["new-a", "new-b"],
        item_features=features[:2],
        metadata=pd.DataFrame({"score": [90.0, 91.0]}),
    )

    metadata = model.candidates.snapshot().metadata
    assert metadata is not None
    assert metadata["score"].dtype.kind == "f"
    assert list(metadata["score"].iloc[-2:]) == [90.0, 91.0]
    assert list(metadata["score"].iloc[:N_ITEMS]) == list(range(N_ITEMS))


def test_replacing_metadata_without_additions_is_unchanged():
    model, features = _with_metadata(
        pd.DataFrame({"score": np.arange(N_ITEMS, dtype="float64")})
    )
    ids = np.array([f"item-{i}" for i in range(N_ITEMS)], dtype=object)

    model.update_candidates(
        item_ids=[ids[2]],
        item_features=features[:1],
        metadata=pd.DataFrame({"score": [42.0]}),
        on_conflict="replace",
    )

    metadata = model.candidates.snapshot().metadata
    assert metadata is not None
    assert len(metadata) == N_ITEMS
    assert metadata["score"].dtype.kind == "f"
    assert metadata["score"].iloc[2] == 42.0
