from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.retrieval import (
    _iter_topk_predictions,
    evaluate_item_embeddings_with_holdout,
)


@pytest.fixture
def embedding_fixture():
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.2, 0.8],
            [-1.0, 0.0],
            [0.0, -1.0],
        ],
        dtype=np.float32,
    )
    source = [np.array([0]), np.array([2]), np.array([0, 2])]
    targets = [np.array([1, 3]), np.array([3]), np.array([1, 3])]
    return embeddings, source, targets


def test_embedding_evaluator_preserves_existing_results(embedding_fixture):
    embeddings, source, targets = embedding_fixture

    result = evaluate_item_embeddings_with_holdout(
        item_embeddings=embeddings,
        source_indices=source,
        target_indices=targets,
        k=3,
        score_batch_size=2,
        debug=True,
    )

    assert result["calibrated_recall@3"] == 1.0
    assert result["ndcg@3"] == 1.0
    assert result["n_scored_rows"] == 3.0
    assert [row["hit_ranks"] for row in result["debug"]] == [[1, 2], [1], [1, 2]]
    assert result["debug"][0]["dcg"] == pytest.approx(1.6309297535714575)


def test_embedding_evaluator_accepts_reusable_multi_cutoff_metrics(embedding_fixture):
    embeddings, source, targets = embedding_fixture

    result = evaluate_item_embeddings_with_holdout(
        item_embeddings=embeddings,
        source_indices=source,
        target_indices=targets,
        k=3,
        metrics=[CalibratedRecall([1, 2]), NDCG([1, 2])],
    )

    assert set(result) == {
        "calibrated_recall@1",
        "calibrated_recall@2",
        "ndcg@1",
        "ndcg@2",
        "n_scored_rows",
        "n_units",
    }
    assert result["calibrated_recall@2"] == 1.0
    assert result["ndcg@2"] == 1.0


def test_embedding_retrieval_masks_seen_items(embedding_fixture):
    embeddings, source, _ = embedding_fixture
    normalized = torch.nn.functional.normalize(torch.from_numpy(embeddings), dim=-1)

    batches = list(
        _iter_topk_predictions(
            normalized,
            source,
            k=3,
            batch_size=2,
        )
    )

    for start, end, predictions in batches:
        for local_row, source_items in enumerate(source[start:end]):
            assert set(predictions.cols[local_row].tolist()).isdisjoint(source_items.tolist())


@pytest.mark.parametrize(
    ("k", "batch_size", "match"),
    [
        (0, 2, "k must be >= 1"),
        (7, 2, "cannot exceed"),
        (3, 0, "batch_size must be >= 1"),
        (3, -1, "batch_size must be >= 1"),
    ],
)
def test_embedding_retrieval_validates_direct_configuration(
    embedding_fixture,
    k,
    batch_size,
    match,
):
    embeddings, source, _ = embedding_fixture
    normalized = torch.nn.functional.normalize(
        torch.from_numpy(embeddings),
        dim=-1,
    )

    with pytest.raises(ValueError, match=match):
        list(
            _iter_topk_predictions(
                normalized,
                source,
                k=k,
                batch_size=batch_size,
            )
        )


def test_embedding_evaluator_validates_configuration(embedding_fixture):
    embeddings, source, targets = embedding_fixture

    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_item_embeddings_with_holdout(
            item_embeddings=embeddings,
            source_indices=source,
            target_indices=targets,
            k=7,
        )
    with pytest.raises(ValueError, match="metrics require top-4"):
        evaluate_item_embeddings_with_holdout(
            item_embeddings=embeddings,
            source_indices=source,
            target_indices=targets,
            k=3,
            metrics=[CalibratedRecall(4)],
        )


@pytest.mark.parametrize(
    "invalid_value",
    [
        np.nan,
        np.inf,
        -np.inf,
        np.finfo(np.float64).max,
    ],
)
def test_embedding_evaluator_rejects_nonfinite_float32_embeddings(
    embedding_fixture,
    invalid_value,
):
    embeddings, source, targets = embedding_fixture
    invalid_embeddings = embeddings.astype(np.float64)
    invalid_embeddings[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite values"):
        evaluate_item_embeddings_with_holdout(
            item_embeddings=invalid_embeddings,
            source_indices=source,
            target_indices=targets,
            k=3,
        )


def test_embedding_evaluator_handles_no_users(embedding_fixture):
    embeddings, _, _ = embedding_fixture

    result = evaluate_item_embeddings_with_holdout(
        item_embeddings=embeddings,
        source_indices=[],
        target_indices=[],
        k=3,
        debug=True,
    )

    assert result == {
        "calibrated_recall@3": 0.0,
        "ndcg@3": 0.0,
        "n_scored_rows": 0,
        "n_units": 0,
        "n_units": 0.0,
        "debug": [],
    }
