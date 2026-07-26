from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import RankingEvaluator, evaluate_ranked_predictions
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall, NDCG


def _predictions() -> SRPTensor:
    return SRPTensor(
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


def _targets() -> csr_matrix:
    return csr_matrix(
        (
            np.ones(3, dtype=np.float32),
            (
                np.array([0, 0, 1]),
                np.array([1, 3, 2]),
            ),
        ),
        shape=(3, 5),
    )


@pytest.mark.parametrize("backend", ["dense", "searchsorted", "auto"])
@pytest.mark.parametrize("batch_size", [1, 2, 20])
def test_evaluate_ranked_predictions_matches_expected_metrics(backend, batch_size):
    result = evaluate_ranked_predictions(
        predictions=_predictions(),
        targets=_targets(),
        metrics=[CalibratedRecall([1, 2, 4]), NDCG([1, 2, 4])],
        batch_size=batch_size,
        match_backend=backend,
    )

    idcg_two = 1.0 + 1.0 / math.log2(3)
    assert result == pytest.approx(
        {
            "recall@1": 0.5,
            "recall@2": 0.25,
            "recall@4": 1.0,
            "ndcg@1": 0.5,
            "ndcg@2": (1.0 / idcg_two) / 2.0,
            "ndcg@4": (((1.0 + 0.5) / idcg_two) + 0.5) / 2.0,
            "n_eval_users": 2.0,
        },
        rel=1e-6,
    )


def test_default_metrics_use_prediction_width():
    result = evaluate_ranked_predictions(
        predictions=_predictions(),
        targets=_targets(),
    )

    assert set(result) == {"recall@4", "ndcg@4", "n_eval_users"}


def test_debug_rows_preserve_global_row_numbers_across_batches():
    result = evaluate_ranked_predictions(
        predictions=_predictions(),
        targets=_targets(),
        batch_size=1,
        debug=True,
        debug_users=3,
    )

    assert [row["user_row"] for row in result["debug"]] == [0, 1]
    assert result["debug"][0]["hit_ranks"] == [1, 3]
    assert result["debug"][1]["hit_ranks"] == [3]


def test_csr_targets_are_canonicalized_as_binary_membership():
    targets = csr_matrix(
        (
            np.array([1.0, 2.0, 0.0, 1.0]),
            np.array([3, 1, 2, 1]),
            np.array([0, 4, 4, 4]),
        ),
        shape=(3, 5),
    )

    result = evaluate_ranked_predictions(
        predictions=_predictions(),
        targets=targets,
        metrics=[CalibratedRecall(4)],
    )

    assert result == pytest.approx({"recall@4": 1.0, "n_eval_users": 1.0})


def test_empty_prediction_and_target_rows_return_zero_metrics():
    predictions = SRPTensor(
        cols=torch.empty((0, 2), dtype=torch.long),
        vals=torch.empty((0, 2)),
        shape=(0, 5),
    )
    targets = csr_matrix((0, 5), dtype=np.float32)

    result = evaluate_ranked_predictions(predictions=predictions, targets=targets)

    assert result == {"recall@2": 0.0, "ndcg@2": 0.0, "n_eval_users": 0.0}


def test_empty_predictions_still_must_cover_metric_cutoff():
    predictions = SRPTensor(
        cols=torch.empty((0, 2), dtype=torch.long),
        vals=torch.empty((0, 2)),
        shape=(0, 5),
    )

    with pytest.raises(ValueError, match="require top-3"):
        evaluate_ranked_predictions(
            predictions=predictions,
            targets=csr_matrix((0, 5)),
            metrics=[CalibratedRecall(3)],
        )


def test_evaluator_can_be_updated_incrementally():
    evaluator = RankingEvaluator([CalibratedRecall(4), NDCG(4)])
    predictions = _predictions()
    targets = _targets()

    evaluator.update(
        SRPTensor(
            cols=predictions.cols[:1],
            vals=predictions.vals[:1],
            shape=(1, 5),
        ),
        targets[:1],
    )
    evaluator.update(
        SRPTensor(
            cols=predictions.cols[1:],
            vals=predictions.vals[1:],
            shape=(2, 5),
        ),
        targets[1:],
    )

    assert evaluator.compute() == evaluate_ranked_predictions(
        predictions=predictions,
        targets=targets,
    )


def test_prediction_and_target_shape_must_match_even_when_empty():
    predictions = SRPTensor(
        cols=torch.empty((0, 2), dtype=torch.long),
        vals=torch.empty((0, 2)),
        shape=(0, 5),
    )

    with pytest.raises(ValueError, match="prediction rows"):
        evaluate_ranked_predictions(
            predictions=predictions,
            targets=csr_matrix((1, 5)),
        )
    with pytest.raises(ValueError, match="prediction items"):
        evaluate_ranked_predictions(
            predictions=predictions,
            targets=csr_matrix((0, 6)),
        )


def test_predictions_must_cover_metric_cutoff():
    with pytest.raises(ValueError, match="require top-5"):
        evaluate_ranked_predictions(
            predictions=_predictions(),
            targets=_targets(),
            metrics=[CalibratedRecall(5)],
        )


@pytest.mark.parametrize(
    ("cols", "vals", "message"),
    [
        ([[1, 1, 2, 3]], [[4.0, 3.0, 2.0, 1.0]], "duplicate"),
        ([[1, 2, 3, 4]], [[4.0, 2.0, 3.0, 1.0]], "highest to lowest"),
    ],
)
def test_invalid_ranked_predictions_are_rejected(cols, vals, message):
    predictions = SRPTensor(
        cols=torch.tensor(cols, dtype=torch.long),
        vals=torch.tensor(vals),
        shape=(1, 5),
    )

    with pytest.raises(ValueError, match=message):
        evaluate_ranked_predictions(
            predictions=predictions,
            targets=_targets()[:1],
            metrics=[CalibratedRecall(4)],
        )


def test_validation_can_be_disabled_for_trusted_predictions():
    predictions = SRPTensor(
        cols=torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        vals=torch.tensor([[1.0, 4.0, 3.0, 2.0]]),
        shape=(1, 5),
    )

    result = evaluate_ranked_predictions(
        predictions=predictions,
        targets=_targets()[:1],
        metrics=[CalibratedRecall(4)],
        validate_predictions=False,
    )

    assert result["recall@4"] == 1.0


def test_vectorized_metrics_match_python_reference_on_random_rows():
    rng = np.random.default_rng(123)
    rows, n_items, prediction_k = 37, 211, 25
    columns = np.stack(
        [rng.choice(n_items, size=prediction_k, replace=False) for _ in range(rows)]
    )
    predictions = SRPTensor(
        cols=torch.from_numpy(columns),
        vals=torch.linspace(1.0, 0.0, prediction_k).expand(rows, -1).clone(),
        shape=(rows, n_items),
    )
    target_rows = [
        rng.choice(n_items, size=int(rng.integers(0, 31)), replace=False)
        for _ in range(rows)
    ]
    target_row_indices = np.concatenate(
        [np.full(len(items), row) for row, items in enumerate(target_rows)]
    )
    target_columns = np.concatenate(target_rows)
    targets = csr_matrix(
        (
            np.ones(len(target_columns), dtype=np.float32),
            (target_row_indices, target_columns),
        ),
        shape=(rows, n_items),
    )

    cutoffs = [5, 10, 25]
    expected: dict[str, float] = {}
    valid_rows = [row for row, items in enumerate(target_rows) if len(items)]
    for cutoff in cutoffs:
        recalls = []
        ndcgs = []
        discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
        for row in valid_rows:
            hits = np.isin(columns[row, :cutoff], target_rows[row])
            recalls.append(hits.sum() / min(cutoff, len(target_rows[row])))
            ideal_length = min(cutoff, len(target_rows[row]))
            ndcgs.append(discounts[hits].sum() / discounts[:ideal_length].sum())
        expected[f"recall@{cutoff}"] = float(np.mean(recalls))
        expected[f"ndcg@{cutoff}"] = float(np.mean(ndcgs))
    expected["n_eval_users"] = float(len(valid_rows))

    for backend in ("dense", "searchsorted"):
        result = evaluate_ranked_predictions(
            predictions=predictions,
            targets=targets,
            metrics=[CalibratedRecall(cutoffs), NDCG(cutoffs)],
            batch_size=7,
            match_backend=backend,
        )
        assert result == pytest.approx(expected, rel=1e-6)


class _RecordingRecommender:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def predict_on_batch(self, source: csr_matrix, *, k: int) -> SRPTensor:
        self.calls.append((source.shape[0], k))
        columns = torch.arange(k, dtype=torch.long).expand(source.shape[0], -1).clone()
        values = torch.arange(k, 0, -1, dtype=torch.float32).expand(source.shape[0], -1).clone()
        return SRPTensor(
            cols=columns,
            vals=values,
            shape=source.shape,
        )


class _InvalidRecommender:
    def predict_on_batch(self, source: csr_matrix, *, k: int) -> SRPTensor:
        return SRPTensor(
            cols=torch.arange(k, dtype=torch.long).expand(source.shape[0], -1).clone(),
            vals=torch.arange(k, dtype=torch.float32).expand(source.shape[0], -1).clone(),
            shape=source.shape,
        )


def test_evaluate_recommender_streams_batches_and_derives_required_k():
    model = _RecordingRecommender()
    source = csr_matrix((5, 6), dtype=np.float32)
    targets = csr_matrix(
        (
            np.ones(5, dtype=np.float32),
            (
                np.arange(5),
                np.array([0, 1, 2, 3, 5]),
            ),
        ),
        shape=source.shape,
    )

    result = evaluate_recommender(
        model,
        source=source,
        targets=targets,
        metrics=[CalibratedRecall([1, 4]), NDCG(4)],
        batch_size=2,
    )

    assert model.calls == [(2, 4), (2, 4), (1, 4)]
    assert result == pytest.approx(
        {
            "recall@1": 0.2,
            "recall@4": 0.8,
            "ndcg@4": (
                1.0
                + 1.0 / np.log2(3)
                + 1.0 / np.log2(4)
                + 1.0 / np.log2(5)
            )
            / 5.0,
            "n_eval_users": 5.0,
        }
    )


def test_evaluate_recommender_validates_predictions_by_default():
    with pytest.raises(ValueError, match="highest to lowest"):
        evaluate_recommender(
            _InvalidRecommender(),
            source=csr_matrix((1, 4)),
            targets=csr_matrix(([1.0], ([0], [0])), shape=(1, 4)),
            metrics=[CalibratedRecall(3)],
        )


def test_evaluate_recommender_handles_empty_input_without_calling_model():
    model = _RecordingRecommender()

    result = evaluate_recommender(
        model,
        source=csr_matrix((0, 6)),
        targets=csr_matrix((0, 6)),
        metrics=[CalibratedRecall(2)],
    )

    assert model.calls == []
    assert result == {"recall@2": 0.0, "n_eval_users": 0.0}


def test_evaluate_recommender_validates_model_and_matrices():
    model = _RecordingRecommender()

    with pytest.raises(TypeError, match="predict_on_batch"):
        evaluate_recommender(
            object(),
            source=csr_matrix((1, 4)),
            targets=csr_matrix((1, 4)),
            metrics=[CalibratedRecall(1)],
        )
    with pytest.raises(TypeError, match="source"):
        evaluate_recommender(
            model,
            source=np.zeros((1, 4)),  # type: ignore[arg-type]
            targets=csr_matrix((1, 4)),
            metrics=[CalibratedRecall(1)],
        )
    with pytest.raises(ValueError, match="source shape"):
        evaluate_recommender(
            model,
            source=csr_matrix((1, 4)),
            targets=csr_matrix((2, 4)),
            metrics=[CalibratedRecall(1)],
        )
    with pytest.raises(ValueError, match="batch_size"):
        evaluate_recommender(
            model,
            source=csr_matrix((1, 4)),
            targets=csr_matrix((1, 4)),
            metrics=[CalibratedRecall(1)],
            batch_size=0,
        )
