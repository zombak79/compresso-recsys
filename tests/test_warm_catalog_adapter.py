from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall
from compresso_recsys.models import Recommender, WarmCatalogAdapter


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
    assert result["n_eval_users"] == 2.0


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
