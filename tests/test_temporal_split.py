from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from compresso_recsys.builder import (
    DEFAULT_TEMPORAL_PERIOD_HOURS,
    _build_args,
    _build_temporal_split,
    _timestamps_in_seconds,
)
from compresso_recsys.checkpoint import load_recsys_split, save_recsys_split


def _temporal_args(**overrides):
    values = {
        "dataset": "ml1m",
        "split_mode": "temporal",
        "temporal_period_hours": 1,
        "min_user_support": 2,
        "item_min_support": 2,
        "min_source_items": 1,
        "min_target_items": 1,
    }
    values.update(overrides)
    return _build_args(**values)


def _timeline() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for user_id in ("u1", "u2"):
        rows.extend(
            [
                {"user_id": user_id, "item_id": "A", "value": 1.0, "timestamp": 1_800},
                {"user_id": user_id, "item_id": "B", "value": 1.0, "timestamp": 5_400},
                {"user_id": user_id, "item_id": "C", "value": 1.0, "timestamp": 9_000},
                {"user_id": user_id, "item_id": "D", "value": 1.0, "timestamp": 14_400},
            ]
        )
    rows.extend(
        [
            {"user_id": "u3", "item_id": "A", "value": 1.0, "timestamp": 1_800},
            {"user_id": "u3", "item_id": "B", "value": 1.0, "timestamp": 5_400},
            {"user_id": "u3", "item_id": "E", "value": 1.0, "timestamp": 9_000},
            {"user_id": "u3", "item_id": "D", "value": 1.0, "timestamp": 14_400},
        ]
    )
    return pd.DataFrame(rows)


def test_temporal_period_defaults_to_official_339_day_scale():
    args = _build_args(dataset="amazon2023")

    assert args.temporal_period_hours == DEFAULT_TEMPORAL_PERIOD_HOURS
    assert args.temporal_period_hours == 339 * 24


def test_temporal_fraction_is_deprecated_in_favor_of_hours():
    with pytest.warns(DeprecationWarning, match="temporal_period_hours"):
        args = _build_args(dataset="amazon2023", temporal_test_frac=0.2)

    assert args.temporal_test_frac == 0.2


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True])
def test_temporal_period_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="temporal_period_hours"):
        _build_args(dataset="amazon2023", temporal_period_hours=value)


def test_timestamp_normalization_accepts_seconds_and_milliseconds():
    seconds = pd.Series([1_700_000_000, 1_700_003_600])
    milliseconds = seconds * 1_000

    np.testing.assert_array_equal(
        _timestamps_in_seconds(seconds),
        _timestamps_in_seconds(milliseconds),
    )


def test_temporal_split_builds_expanding_mixed_catalogs_and_filters_support():
    split = _build_temporal_split(_temporal_args(), _timeline())

    assert split["train_item_ids"].tolist() == ["A", "B"]
    assert split["val_item_ids"].tolist() == ["A", "B", "C"]
    assert split["test_item_ids"].tolist() == ["A", "B", "C", "D"]
    assert split["item_ids"].tolist() == ["A", "B", "C", "D"]

    assert split["train_source_matrix"].shape == (3, 2)
    assert split["train_target_matrix"].shape == (3, 2)
    assert split["val_source_matrix"].shape == (2, 3)
    assert split["val_target_matrix"].shape == (2, 3)
    assert split["test_source_matrix"].shape == (3, 4)
    assert split["test_target_matrix"].shape == (3, 4)

    assert split["val_user_ids"].tolist() == ["u1", "u2"]
    assert split["test_user_ids"].tolist() == ["u1", "u2", "u3"]
    assert "E" not in split["val_item_ids"]
    assert "E" not in split["test_item_ids"]
    assert split["val_item_indices"].tolist() == [2]
    assert split["test_item_indices"].tolist() == [3]

    expected_x_train = split["train_source_matrix"].maximum(
        split["train_target_matrix"]
    )
    assert (split["x_train"] != expected_x_train).nnz == 0
    assert np.all(split["val_source_matrix"].getnnz(axis=1) >= 1)
    assert np.all(split["val_target_matrix"].getnnz(axis=1) >= 1)
    assert np.all(
        split["val_source_matrix"]
        .maximum(split["val_target_matrix"])
        .getnnz(axis=1)
        >= 2
    )


def test_temporal_split_rejects_period_longer_than_available_history():
    with pytest.raises(ValueError, match="three target windows"):
        _build_temporal_split(
            _temporal_args(temporal_period_hours=2),
            _timeline(),
        )


def test_checkpoint_round_trip_preserves_stage_item_spaces_and_training_union(
    tmp_path,
):
    train_source = csr_matrix([[1.0, 0.0]], dtype=np.float32)
    train_target = csr_matrix([[0.0, 1.0]], dtype=np.float32)
    x_train = train_source.maximum(train_target)
    val_source = csr_matrix([[1.0, 1.0, 0.0]], dtype=np.float32)
    val_target = csr_matrix([[0.0, 0.0, 1.0]], dtype=np.float32)
    test_source = csr_matrix([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    test_target = csr_matrix([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)

    save_recsys_split(
        tmp_path,
        item_ids=np.asarray(["A", "B", "C", "D"]),
        train_item_ids=np.asarray(["A", "B"]),
        val_item_ids=np.asarray(["A", "B", "C"]),
        test_item_ids=np.asarray(["A", "B", "C", "D"]),
        x_train=x_train,
        train_source_matrix=train_source,
        train_target_matrix=train_target,
        val_source_matrix=val_source,
        val_target_matrix=val_target,
        test_source_matrix=test_source,
        test_target_matrix=test_target,
        val_source_indices=[np.asarray([0, 1])],
        val_target_indices=[np.asarray([2])],
        test_source_indices=[np.asarray([0, 1, 2])],
        test_target_indices=[np.asarray([3])],
    )

    loaded = load_recsys_split(tmp_path)

    assert loaded["train_item_ids"].tolist() == ["A", "B"]
    assert loaded["val_item_ids"].tolist() == ["A", "B", "C"]
    assert loaded["test_item_ids"].tolist() == ["A", "B", "C", "D"]
    assert (loaded["train_source_matrix"] != train_source).nnz == 0
    assert (loaded["train_target_matrix"] != train_target).nnz == 0
    assert (loaded["x_train"] != x_train).nnz == 0
    assert (loaded["val_source_matrix"] != val_source).nnz == 0
    assert (loaded["test_target_matrix"] != test_target).nnz == 0


def test_checkpoint_rejects_matrix_item_id_mismatch(tmp_path):
    matrix = csr_matrix([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="validation matrix columns"):
        save_recsys_split(
            tmp_path,
            item_ids=np.asarray(["A", "B"]),
            train_item_ids=np.asarray(["A"]),
            val_item_ids=np.asarray(["A", "B"]),
            test_item_ids=np.asarray(["A"]),
            x_train=matrix,
            train_source_matrix=matrix,
            train_target_matrix=matrix,
            val_source_matrix=matrix,
            val_target_matrix=matrix,
            test_source_matrix=matrix,
            test_target_matrix=matrix,
            val_source_indices=[np.asarray([0])],
            val_target_indices=[np.asarray([0])],
            test_source_indices=[np.asarray([0])],
            test_target_indices=[np.asarray([0])],
        )
