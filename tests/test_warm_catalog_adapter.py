from __future__ import annotations

from typing import get_type_hints

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall
from compresso_recsys.models import (
    Recommender,
    SequentialRecommender,
    WarmCatalogAdapter,
)
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
        catalog_item_ids=["warm-b", "warm-a", "warm-c", "cold-x"],
    )


def test_adapter_annotation_accepts_both_recommender_contracts():
    model_type = get_type_hints(WarmCatalogAdapter.__init__)["model"]

    assert model_type == Recommender | SequentialRecommender


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

    np.testing.assert_array_equal(adapter.train_to_catalog, [0, 1, 2])
    np.testing.assert_array_equal(
        aligned.toarray(),
        [[1, 1, 0], [0, 1, 1]],
    )
    torch.testing.assert_close(
        predictions.cols,
        torch.tensor([[0, 2], [0, 2]]),
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
        np.asarray([[1, 0, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
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


def test_adapter_rejects_a_training_catalog_that_is_not_an_ordered_prefix():
    with pytest.raises(ValueError, match="exact ordered prefix"):
        WarmCatalogAdapter(
            _FixedWarmModel(n_items=2),
            train_item_ids=["a", "b"],
            catalog_item_ids=["a", "cold", "b"],
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
# The adapter no longer projects histories. A sequential model's tokenizer turns
# an out-of-catalog index into its own unk token, in place, so only the
# prediction columns still need widening.
# --------------------------------------------------------------------------


class _FixedWarmSequenceModel:
    """Ranks the fitted item space in a fixed order, ignoring the history."""

    def __init__(self, *, n_items: int = 3) -> None:
        self.n_items = n_items
        self.seen_lengths: list[int] | None = None
        self.seen_width: int | None = None

    def predict_on_batch(self, source, *, k, exclude_seen=True):
        self.seen_lengths = source.row_lengths.tolist()
        self.seen_width = source.n_items
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
        catalog_item_ids=["warm-b", "warm-a", "warm-c", "cold-x"],
    )


def _catalog_sequences(rows) -> ItemSequences:
    return ItemSequences.from_rows(rows, n_items=4)


def test_aligning_a_history_is_refused_and_says_why():
    """The bug this removal fixes, stated where someone would hit it.

    Projecting a history deleted interior events, which joined their neighbours
    as though they had been consecutive.
    """
    adapter = _sequence_adapter()

    with pytest.raises(TypeError, match="sequences need no alignment"):
        adapter.align_source(_catalog_sequences([[0, 1, 2]]))


def test_a_history_reaches_the_model_whole():
    """Nothing is dropped, so no adjacency is invented."""
    model = _FixedWarmSequenceModel()
    adapter = _sequence_adapter(model)
    rows = [[0, 1, 2, 3], [1], []]

    adapter.predict_on_batch(_catalog_sequences(rows), k=3)

    assert model.seen_lengths == [4, 1, 0]
    assert model.seen_width == 4, "the model reads catalog space, not fitted space"


def test_predicting_from_sequences_still_widens_the_columns():
    """The one job left: the evaluator requires the target width."""
    adapter = _sequence_adapter()

    predictions = adapter.predict_on_batch(_catalog_sequences([[0, 1], [3]]), k=3)

    assert predictions.cols_total == 4
    # The fitted catalog is the full catalog's prefix, so warm rows are stable.
    assert predictions.cols[0].tolist() == [0, 2, 1]
    # The cold item is still unreachable, because the model cannot score it.
    assert 3 not in predictions.cols.flatten().tolist()


def test_a_sequence_over_the_wrong_catalog_is_refused():
    adapter = _sequence_adapter()

    with pytest.raises(ValueError, match="spans 7 items"):
        adapter.predict_on_batch(ItemSequences.from_rows([[0]], n_items=7), k=2)


def test_the_matrix_path_still_projects_because_a_row_is_a_set():
    """Dropping cold columns is correct for a matrix and wrong for a history.

    A CSR row has no adjacency to corrupt, and a set model has no unk to fall
    back on, so the projection stays exactly where it belongs.
    """
    adapter = _sequence_adapter()
    # Catalog order is warm-b, warm-a, warm-c, cold-x; this row holds the first,
    # the second, and the appended cold item.
    matrix = csr_matrix(np.asarray([[1, 1, 0, 1]], dtype=np.float32))

    aligned = adapter.align_source(matrix)

    assert aligned.shape == (1, 3)
    # The two warm items survive in their stable prefix positions, and cold-x is
    # gone with nothing standing in for it.
    assert sorted(aligned[0].indices.tolist()) == [0, 1]


def test_a_sequential_model_evaluates_through_the_adapter():
    """The combination that could not run before: temporal plus a sequence model."""
    adapter = _sequence_adapter()
    targets = csr_matrix(
        (np.ones(2, dtype=np.float32), (np.arange(2), np.array([0, 3]))),
        shape=(2, 4),
    )

    result = evaluate_recommender(
        adapter,
        source=_catalog_sequences([[0, 1], [2, 3]]),
        targets=targets,
        metrics=[CalibratedRecall(2)],
    )

    # Row 0 wants warm-b, which the model ranks first. Row 1 wants the cold item,
    # unreachable by construction, so the pair averages to one half.
    assert result["calibrated_recall@2"] == pytest.approx(0.5)
