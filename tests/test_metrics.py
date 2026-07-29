from __future__ import annotations

import math

import pytest
import torch

from compresso import SRPTensor
from compresso_recsys.metrics import (
    CalibratedRecall,
    HitRate,
    MAP,
    MRR,
    NDCG,
    Precision,
    Recall,
    RankingBatch,
)


def _ranking_batch() -> RankingBatch:
    predictions = SRPTensor(
        cols=torch.tensor(
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1],
                [0, 1, 2, 3],
            ],
            dtype=torch.long,
        ),
        vals=torch.tensor(
            [
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
                [4.0, 3.0, 2.0, 1.0],
            ]
        ),
        shape=(3, 5),
    )
    return RankingBatch(
        predictions=predictions,
        hits=torch.tensor(
            [
                [True, False, True, False],
                [False, False, True, False],
                [False, False, False, False],
            ]
        ),
        target_counts=torch.tensor([2, 1, 0], dtype=torch.long),
    )


def test_calibrated_recall_supports_multiple_cutoffs_and_ignores_empty_targets():
    metric = CalibratedRecall([4, 1, 2, 2])

    metric.update(_ranking_batch())

    assert metric.result_keys == ("recall@1", "recall@2", "recall@4")
    assert metric.compute() == pytest.approx(
        {
            "recall@1": 0.5,
            "recall@2": 0.25,
            "recall@4": 1.0,
        }
    )


def test_ndcg_supports_multiple_cutoffs():
    metric = NDCG([1, 2, 4])

    metric.update(_ranking_batch())

    idcg_two = 1.0 + 1.0 / math.log2(3)
    expected_row_zero_at_two = 1.0 / idcg_two
    expected_row_zero_at_four = (1.0 + 1.0 / math.log2(4)) / idcg_two
    assert metric.compute() == pytest.approx(
        {
            "ndcg@1": 0.5,
            "ndcg@2": expected_row_zero_at_two / 2.0,
            "ndcg@4": (expected_row_zero_at_four + 0.5) / 2.0,
        },
        rel=1e-6,
    )


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        (
            Recall([1, 2, 4]),
            {
                "standard_recall@1": 0.25,
                "standard_recall@2": 0.25,
                "standard_recall@4": 1.0,
            },
        ),
        (
            Precision([1, 2, 4]),
            {
                "precision@1": 0.5,
                "precision@2": 0.25,
                "precision@4": 0.375,
            },
        ),
        (
            HitRate([1, 2, 4]),
            {
                "hit_rate@1": 0.5,
                "hit_rate@2": 0.5,
                "hit_rate@4": 1.0,
            },
        ),
        (
            MRR([1, 2, 4]),
            {
                "mrr@1": 0.5,
                "mrr@2": 0.5,
                "mrr@4": 2.0 / 3.0,
            },
        ),
        (
            MAP([1, 2, 4]),
            {
                "map@1": 0.5,
                "map@2": 0.25,
                "map@4": 7.0 / 12.0,
            },
        ),
    ],
)
def test_optional_metrics_support_multiple_cutoffs(metric, expected):
    metric.update(_ranking_batch())

    assert metric.compute() == pytest.approx(expected, rel=1e-6)


def test_metric_state_streams_and_resets():
    batch = _ranking_batch()
    metric = CalibratedRecall(4)

    metric.update(batch)
    once = metric.compute()
    metric.update(batch)
    assert metric.compute() == once

    metric.reset()
    assert metric.compute() == {"recall@4": 0.0}


@pytest.mark.parametrize("cutoffs", [0, [], [1, -1]])
def test_metric_rejects_invalid_cutoffs(cutoffs):
    with pytest.raises(ValueError, match="positive integer"):
        CalibratedRecall(cutoffs)


def test_ranking_batch_validates_shared_tensor_shapes():
    batch = _ranking_batch()

    with pytest.raises(ValueError, match="hits rows"):
        RankingBatch(
            predictions=batch.predictions,
            hits=batch.hits[:2],
            target_counts=batch.target_counts,
        )
