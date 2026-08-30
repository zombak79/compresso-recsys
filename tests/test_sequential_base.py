"""The sequential prediction contract, and evaluation accepting either source.

No real model yet — a stub is enough to pin the contract and the dispatch, and it
keeps these tests about the plumbing rather than about an RNN.
"""

from __future__ import annotations

import warnings
from typing import get_overloads, get_type_hints

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models.base import (
    BaseSequentialRecommender,
    Recommender,
    SequentialRecommender,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.stats import compare_models

N_ITEMS = 12


def _stable_unique(values: np.ndarray) -> np.ndarray:
    _, first = np.unique(values, return_index=True)
    return values[np.sort(first)]


class _Recency(BaseSequentialRecommender):
    """Ranks unseen items by index, after the history it refuses to repeat.

    Deliberately trivial. It exists to exercise the contract, and to be a second
    model that scores the same targets differently.
    """

    def __init__(self, n_items: int = N_ITEMS, *, offset: int = 0) -> None:
        self._n_items = n_items
        self._offset = offset

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def n_items(self) -> int:
        return self._n_items

    def predict_on_batch(self, source, *, k, exclude_seen=True):
        columns = np.zeros((source.n_rows, k), dtype=np.int64)
        for row in range(source.n_rows):
            history = _stable_unique(source.row(row))
            candidates = np.setdiff1d(np.arange(self._n_items), history)
            if not exclude_seen:
                candidates = np.concatenate([history[::-1], candidates])
            candidates = np.roll(candidates, self._offset)
            columns[row] = candidates[:k]
        return SRPTensor(
            cols=torch.from_numpy(columns),
            vals=torch.arange(k, 0, -1, dtype=torch.float32)
            .expand(source.n_rows, -1)
            .clone(),
            shape=(source.n_rows, self._n_items),
        )


class _Unfitted(_Recency):
    @property
    def is_fitted(self) -> bool:
        return False


def _sequences() -> ItemSequences:
    return ItemSequences.from_rows(
        [[1, 2, 3], [4, 5], [0, 7, 7, 8], []], n_items=N_ITEMS
    )


def _targets() -> csr_matrix:
    rows, cols = [0, 1, 2, 3], [9, 6, 10, 11]
    return csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(rows), N_ITEMS),
    )


# --------------------------------------------------------------------------
# the contract
# --------------------------------------------------------------------------


def test_a_sequential_model_satisfies_both_protocols():
    """Structural, so having the method is the whole requirement."""
    model = _Recency()

    assert isinstance(model, SequentialRecommender)
    # And the evaluator's own check, which asks the same question.
    assert isinstance(model, Recommender)


def test_predict_batches_reassemble_the_whole():
    model = _Recency()
    sequences = _sequences()

    whole = model.predict(sequences, k=4, batch_size=64)
    batched = model.predict(sequences, k=4, batch_size=2)

    assert whole.cols.tolist() == batched.cols.tolist()
    assert whole.cols.shape == (sequences.n_rows, 4)


def test_predict_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="must be fitted"):
        _Unfitted().predict(_sequences(), k=3)


def test_predict_refuses_a_matrix_source():
    """The two families read different things; silence would be worse."""
    with pytest.raises(TypeError, match="predicts from ItemSequences"):
        _Recency().predict(csr_matrix((4, N_ITEMS), dtype=np.float32), k=3)


@pytest.mark.parametrize("k", [0, N_ITEMS + 1])
def test_predict_refuses_an_impossible_k(k):
    with pytest.raises(ValueError, match="k must be in"):
        _Recency().predict(_sequences(), k=k)


def test_exclude_seen_masks_the_whole_history_including_repeats():
    model = _Recency()
    sequences = _sequences()

    predictions = model.predict(sequences, k=N_ITEMS - 4, exclude_seen=True)

    for row in range(sequences.n_rows):
        history = set(sequences.row(row).tolist())
        assert not history & set(predictions.cols[row].tolist())


# --------------------------------------------------------------------------
# evaluation accepts either source
# --------------------------------------------------------------------------


def test_evaluate_recommender_accepts_sequences():
    result = evaluate_recommender(
        _Recency(),
        source=_sequences(),
        targets=_targets(),
        metrics=[CalibratedRecall(3), NDCG(3)],
        sample_ids=["a", "b", "c", "d"],
    )

    assert result.n_scored_rows == 4
    assert set(result.per_user) == {"calibrated_recall@3", "ndcg@3"}
    assert result.target_fingerprint is not None


def test_evaluate_recommender_advertises_both_source_contracts():
    contracts = [
        get_type_hints(candidate)
        for candidate in get_overloads(evaluate_recommender)
    ]

    assert [(contract["model"], contract["source"]) for contract in contracts] == [
        (Recommender, csr_matrix),
        (SequentialRecommender, ItemSequences),
    ]


def test_evaluate_recommender_batches_sequences_identically():
    kwargs = dict(source=_sequences(), targets=_targets(), metrics=[NDCG(3)])

    whole = evaluate_recommender(_Recency(), batch_size=64, **kwargs)
    batched = evaluate_recommender(_Recency(), batch_size=1, **kwargs)

    assert whole.metrics == batched.metrics
    assert whole.target_fingerprint == batched.target_fingerprint


def test_evaluate_recommender_rejects_an_unknown_source_type():
    with pytest.raises(TypeError, match="csr_matrix or an ItemSequences"):
        evaluate_recommender(
            _Recency(), source=[[1, 2]], targets=_targets(), metrics=[NDCG(3)]
        )


def test_row_count_must_still_match_the_targets():
    with pytest.raises(ValueError, match="source rows .* must match target rows"):
        evaluate_recommender(
            _Recency(),
            source=ItemSequences.from_rows([[1]], n_items=N_ITEMS),
            targets=_targets(),
            metrics=[NDCG(3)],
        )


# --------------------------------------------------------------------------
# the payoff: one comparison over two source representations
# --------------------------------------------------------------------------


def test_a_matrix_model_and_a_sequential_model_compare_directly():
    """The statistics layer needs no sequential logic at all.

    The 0.2.0 fingerprint hashes only targets, so two models scored against one
    target matrix pair regardless of how their sources were expressed. This is
    the test that keeps that true.
    """

    class Popularity:
        """A CSR-source model, standing in for EASE or ELSA."""

        def predict_on_batch(self, source, *, k, exclude_seen=True):
            order = np.arange(N_ITEMS - 1, -1, -1)[:k]
            return SRPTensor(
                cols=torch.from_numpy(np.tile(order, (source.shape[0], 1))),
                vals=torch.arange(k, 0, -1, dtype=torch.float32)
                .expand(source.shape[0], -1)
                .clone(),
                shape=(source.shape[0], N_ITEMS),
            )

    sequences = _sequences()
    matrix_source = csr_matrix(
        (
            np.ones(sequences.values.size, dtype=np.float32),
            (
                np.repeat(np.arange(sequences.n_rows), sequences.row_lengths),
                sequences.values,
            ),
        ),
        shape=(sequences.n_rows, N_ITEMS),
    )
    targets = _targets()
    metrics = [NDCG(3)]
    ids = ["a", "b", "c", "d"]

    matrix_result = evaluate_recommender(
        Popularity(), source=matrix_source, targets=targets,
        metrics=metrics, sample_ids=ids,
    )
    sequence_result = evaluate_recommender(
        _Recency(), source=sequences, targets=targets,
        metrics=metrics, sample_ids=ids,
    )

    # Same targets, so the fingerprints agree and comparison accepts the pair.
    assert matrix_result.target_fingerprint == sequence_result.target_fingerprint

    # Four users is far too few for a real comparison, so the discreteness
    # warning is expected and is not what this test is about.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        report = compare_models(
            {"popularity": matrix_result, "recency": sequence_result},
            metrics=["ndcg@3"],
            reference="popularity",
            n_resamples=99,
        )

    assert len(report.comparisons) == 1
    assert report.comparisons[0].n_samples == 4
