from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall
from compresso_recsys.models import Recommender, WarmCatalogAdapter
from compresso_recsys.sequences import ItemSequences


class _FixedWarmModel:
    def __init__(self, *, n_items: int = 3) -> None:
        self.n_items = n_items
        self.last_exclude_seen: bool | None = None

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        self.last_exclude_seen = exclude_seen
        ranking = torch.tensor([0, 2, 1], dtype=torch.long)[:k]
        columns = ranking.expand(source.shape[0], -1).clone()
        values = torch.arange(k, 0, -1, dtype=torch.float32).expand(
            source.shape[0], -1
        ).clone()
        return SRPTensor(
            cols=columns,
            vals=values,
            shape=(source.shape[0], self.n_items),
        )


def _adapter(model=None) -> WarmCatalogAdapter:
    return WarmCatalogAdapter(
        _FixedWarmModel() if model is None else model,
        train_item_ids=["warm-b", "warm-a", "warm-c"],
        catalog_item_ids=["warm-a", "cold-x", "warm-c", "warm-b"],
    )


def test_adapter_aligns_source_and_remaps_predictions():
    model = _FixedWarmModel()
    adapter = _adapter(model)
    source = csr_matrix(
        np.asarray(
            [
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=np.float32,
        )
    )

    aligned = adapter.align_source(source)
    predictions = adapter.predict_on_batch(
        aligned,
        k=2,
        exclude_seen=False,
    )

    np.testing.assert_array_equal(adapter.train_to_catalog, [3, 0, 2])
    np.testing.assert_array_equal(
        aligned.toarray(),
        [[1, 1, 0], [0, 0, 1]],
    )
    torch.testing.assert_close(
        predictions.cols,
        torch.tensor([[3, 2], [3, 2]]),
    )
    assert predictions.shape == (2, 4)
    assert model.last_exclude_seen is False
    assert isinstance(adapter, Recommender)


def test_adapter_integrates_with_full_catalog_evaluation():
    adapter = _adapter()
    expanded_source = csr_matrix(
        np.asarray([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    )
    targets = csr_matrix(
        np.asarray([[0, 0, 0, 1], [0, 1, 0, 0]], dtype=np.float32)
    )

    result = evaluate_recommender(
        adapter,
        source=adapter.align_source(expanded_source),
        targets=targets,
        metrics=[CalibratedRecall(1)],
        batch_size=2,
    )

    assert result["calibrated_recall@1"] == pytest.approx(0.5)
    assert result["n_scored_rows"] == 2.0


def test_adapter_returns_already_aligned_source_unchanged():
    model = _FixedWarmModel()
    adapter = WarmCatalogAdapter(
        model,
        train_item_ids=["a", "b", "c"],
        catalog_item_ids=["a", "b", "c"],
    )
    source = csr_matrix([[1, 0, 1]], dtype=np.float32)

    assert adapter.align_source(source) is source


def test_adapter_rejects_missing_training_item():
    with pytest.raises(ValueError, match="missing training item ID.*'b'"):
        WarmCatalogAdapter(
            _FixedWarmModel(n_items=2),
            train_item_ids=["a", "b"],
            catalog_item_ids=["a", "cold"],
        )


def test_adapter_validates_source_before_alignment_and_prediction():
    adapter = _adapter()

    with pytest.raises(TypeError, match="source must be.*csr_matrix"):
        adapter.align_source(csc_matrix((1, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="source has 3 items.*catalog_item_ids.*4"):
        adapter.align_source(csr_matrix((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="call align_source"):
        adapter.predict_on_batch(csr_matrix((1, 4), dtype=np.float32), k=1)


def test_adapter_rejects_model_with_different_prediction_vocabulary():
    adapter = _adapter(_FixedWarmModel(n_items=2))

    with pytest.raises(ValueError, match="expected 3, got 2"):
        adapter.predict_on_batch(csr_matrix((1, 3), dtype=np.float32), k=1)


def test_adapter_ids_and_mapping_are_immutable():
    adapter = _adapter()

    with pytest.raises(ValueError, match="read-only"):
        adapter.train_to_catalog[0] = 0
    with pytest.raises(ValueError, match="read-only"):
        adapter.train_item_ids[0] = "changed"
    with pytest.raises(ValueError, match="read-only"):
        adapter.catalog_item_ids[0] = "changed"


# --------------------------------------------------------------------------
# sequence sources
#
# The projection is the same operation on either representation, so these tests
# are mostly about the two paths agreeing. Where they cannot agree exactly -- a
# CSR row is a set, a history is ordered -- the tests say which property holds.
# --------------------------------------------------------------------------


class _FixedWarmSequenceModel:
    """Ranks the fitted item space in a fixed order, ignoring the history."""

    def __init__(self, *, n_items: int = 3) -> None:
        self.n_items = n_items
        self.seen_lengths: list[int] | None = None

    def predict_on_batch(self, source, *, k, exclude_seen=True):
        self.seen_lengths = source.row_lengths.tolist()
        ranking = torch.tensor([0, 2, 1], dtype=torch.long)[:k]
        return SRPTensor(
            cols=ranking.expand(source.n_rows, -1).clone(),
            vals=torch.arange(k, 0, -1, dtype=torch.float32)
            .expand(source.n_rows, -1)
            .clone(),
            shape=(source.n_rows, self.n_items),
        )


def _sequence_adapter(model=None) -> WarmCatalogAdapter:
    return WarmCatalogAdapter(
        _FixedWarmSequenceModel() if model is None else model,
        train_item_ids=["warm-b", "warm-a", "warm-c"],
        catalog_item_ids=["warm-a", "cold-x", "warm-c", "warm-b"],
    )


def _catalog_sequences(rows) -> ItemSequences:
    return ItemSequences.from_rows(rows, n_items=4)


def test_aligning_a_history_keeps_the_fitted_items_in_order():
    """Catalog order is a, cold-x, c, b; fitted order is b, a, c."""
    adapter = _sequence_adapter()

    aligned = adapter.align_source(_catalog_sequences([[0, 1, 2, 3]]))

    assert isinstance(aligned, ItemSequences)
    assert aligned.n_items == 3
    # warm-a -> 1, cold-x dropped, warm-c -> 2, warm-b -> 0.
    assert aligned.row(0).tolist() == [1, 2, 0]


def test_aligning_a_history_preserves_order_and_repeats():
    """The two things the matrix path cannot express, so they are pinned here."""
    adapter = _sequence_adapter()

    aligned = adapter.align_source(_catalog_sequences([[3, 0, 3, 1, 0]]))

    # warm-b, warm-a, warm-b, (cold dropped), warm-a
    assert aligned.row(0).tolist() == [0, 1, 0, 1]


def test_the_two_source_paths_agree_on_rows_width_and_warm_content():
    rows = [[0, 1, 2, 3], [1], [], [3, 3, 0]]
    adapter = _sequence_adapter()
    sequences = _catalog_sequences(rows)
    lengths = [len(row) for row in rows]
    matrix = csr_matrix(
        (
            np.ones(sum(lengths), dtype=np.float32),
            (np.repeat(np.arange(len(rows)), lengths), np.concatenate([*rows, []])),
        ),
        shape=(len(rows), 4),
    )

    aligned_sequences = adapter.align_source(sequences)
    aligned_matrix = adapter.align_source(matrix)

    assert aligned_sequences.n_rows == aligned_matrix.shape[0] == len(rows)
    assert aligned_sequences.n_items == aligned_matrix.shape[1] == 3
    for row in range(len(rows)):
        assert set(aligned_sequences.row(row).tolist()) == set(
            aligned_matrix[row].indices.tolist()
        ), row


def test_an_entirely_cold_history_becomes_an_empty_row_not_a_missing_one():
    """Row alignment against the targets is what makes evaluation valid."""
    adapter = _sequence_adapter()

    aligned = adapter.align_source(_catalog_sequences([[1], [0], [1, 1]]))

    assert aligned.n_rows == 3
    assert aligned.row(0).size == 0
    assert aligned.row(2).size == 0
    assert aligned.row(1).tolist() == [1]


def test_an_already_aligned_sequence_source_is_returned_unchanged():
    adapter = WarmCatalogAdapter(
        _FixedWarmSequenceModel(),
        train_item_ids=["a", "b", "c"],
        catalog_item_ids=["a", "b", "c"],
    )
    sequences = ItemSequences.from_rows([[0, 2, 1]], n_items=3)

    assert adapter.align_source(sequences) is sequences


def test_aligning_a_sequence_over_the_wrong_catalog_is_refused():
    adapter = _sequence_adapter()

    with pytest.raises(ValueError, match="spans 7 items"):
        adapter.align_source(ItemSequences.from_rows([[0]], n_items=7))


def test_predicting_from_sequences_remaps_columns_into_the_catalog():
    model = _FixedWarmSequenceModel()
    adapter = _sequence_adapter(model)
    aligned = adapter.align_source(_catalog_sequences([[0, 1, 2], [3]]))

    predictions = adapter.predict_on_batch(aligned, k=3)

    assert predictions.cols_total == 4
    # Fitted ranking b, c, a maps to catalog rows 3, 2, 0.
    assert predictions.cols[0].tolist() == [3, 2, 0]
    # The cold item cannot be emitted at all.
    assert 1 not in predictions.cols.flatten().tolist()
    # The model saw the projected history, two of three events for row 0.
    assert model.seen_lengths == [2, 1]


def test_predicting_from_an_unaligned_sequence_source_is_refused():
    adapter = _sequence_adapter()

    with pytest.raises(ValueError, match="call align_source"):
        adapter.predict_on_batch(_catalog_sequences([[0, 2]]), k=2)


def test_the_catalog_to_train_mapping_marks_cold_items():
    adapter = _sequence_adapter()

    # Catalog a, cold-x, c, b against fitted b, a, c.
    assert adapter.catalog_to_train.tolist() == [1, -1, 2, 0]
    with pytest.raises(ValueError, match="read-only"):
        adapter.catalog_to_train[0] = 9


def test_a_sequential_model_evaluates_through_the_adapter():
    """The combination that could not run before: temporal plus a sequence model."""
    adapter = _sequence_adapter()
    sequences = _catalog_sequences([[0, 1], [2, 3]])
    # Targets live in the catalog space, cold column included.
    targets = csr_matrix(
        (
            np.ones(2, dtype=np.float32),
            (np.arange(2), np.array([3, 1])),
        ),
        shape=(2, 4),
    )

    result = evaluate_recommender(
        adapter,
        source=adapter.align_source(sequences),
        targets=targets,
        metrics=[CalibratedRecall(2)],
    )

    # Row 0 wants warm-b, which the model ranks first. Row 1 wants the cold item,
    # which is unreachable by construction, so the pair averages to one half.
    assert result["calibrated_recall@2"] == pytest.approx(0.5)
