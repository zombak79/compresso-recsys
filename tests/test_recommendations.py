from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys import ItemSequences
from compresso_recsys.models import (
    BaseCollaborativeRecommender,
    BaseSequentialRecommender,
    ContentRecommender,
    EASE,
    ELSAConfig,
    ELSATrainer,
    Recommendations,
    SimpleGPTConfig,
    SimpleGPTTrainer,
    SimpleRNNConfig,
    SimpleRNNTrainer,
    TransformerConfig,
    WarmCatalogAdapter,
)


@dataclass(frozen=True)
class _RankingConfig:
    n_items: int


class _CollaborativeRankingModel(BaseCollaborativeRecommender):
    """Minimal model proving that recommend() is inherited by new models."""

    checkpoint_type = "test_collaborative_ranking"

    def __init__(self, item_ids: Sequence[Hashable]) -> None:
        self.cfg = _RankingConfig(n_items=len(item_ids))
        self._set_item_ids(item_ids, n_items=len(item_ids))
        self.last_source: csr_matrix | None = None

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def n_items(self) -> int:
        return self.cfg.n_items

    def fit(
        self,
        interactions: csr_matrix,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> "_CollaborativeRankingModel":
        del interactions, item_ids
        return self

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        self.last_source = source.copy()
        candidate_rows = self._candidate_rows(candidate_ids)
        scores = np.broadcast_to(
            np.arange(self.n_items, dtype=np.float32),
            (source.shape[0], self.n_items),
        ).copy()
        selected_scores = scores[:, candidate_rows]
        if exclude_seen:
            candidate_to_local = np.full(self.n_items, -1, dtype=np.int64)
            candidate_to_local[candidate_rows] = np.arange(candidate_rows.size)
            seen_counts = np.diff(source.indptr)
            seen_rows = np.repeat(np.arange(source.shape[0]), seen_counts)
            seen_local = candidate_to_local[source.indices]
            selected = seen_local >= 0
            selected_scores[seen_rows[selected], seen_local[selected]] = -np.inf
        values, local_columns = torch.topk(
            torch.from_numpy(selected_scores),
            k=k,
            dim=1,
        )
        columns = torch.from_numpy(candidate_rows)[local_columns]
        return SRPTensor(
            cols=columns,
            vals=values,
            shape=source.shape,
        )


class _SequentialRankingModel(BaseSequentialRecommender):
    checkpoint_type = "test_sequential_ranking"

    def __init__(self, item_ids: Sequence[Hashable]) -> None:
        self.cfg = _RankingConfig(n_items=len(item_ids))
        self._set_item_ids(item_ids, n_items=len(item_ids))
        self.last_source: ItemSequences | None = None

    @property
    def is_fitted(self) -> bool:
        return True

    @property
    def n_items(self) -> int:
        return self.cfg.n_items

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        self.last_source = source
        candidate_rows = self._candidate_rows(candidate_ids)
        scores = np.broadcast_to(
            np.arange(self.n_items, dtype=np.float32),
            (source.n_rows, self.n_items),
        ).copy()
        selected_scores = scores[:, candidate_rows]
        if exclude_seen:
            candidate_to_local = np.full(self.n_items, -1, dtype=np.int64)
            candidate_to_local[candidate_rows] = np.arange(candidate_rows.size)
            for row in range(source.n_rows):
                seen_local = candidate_to_local[source.row(row)]
                selected_scores[row, seen_local[seen_local >= 0]] = -np.inf
        values, local_columns = torch.topk(
            torch.from_numpy(selected_scores),
            k=k,
            dim=1,
        )
        columns = torch.from_numpy(candidate_rows)[local_columns]
        return SRPTensor(
            cols=columns,
            vals=values,
            shape=(source.n_rows, self.n_items),
        )


def test_recommendations_are_immutable_and_convert_to_ordered_dicts():
    recommendations = Recommendations(
        item_ids=np.array([["c", "a"], ["b", "c"]], dtype=object),
        scores=np.array([[0.9, 0.7], [0.8, 0.4]], dtype=np.float32),
    )

    assert recommendations.item_ids.shape == (2, 2)
    assert recommendations.scores.shape == (2, 2)
    np.testing.assert_array_equal(
        recommendations.valid_mask,
        np.ones((2, 2), dtype=bool),
    )
    np.testing.assert_array_equal(recommendations.valid_counts, [2, 2])
    assert recommendations.to_dicts() == [
        {"c": pytest.approx(0.9), "a": pytest.approx(0.7)},
        {"b": pytest.approx(0.8), "c": pytest.approx(0.4)},
    ]
    assert list(recommendations.to_dicts()[0]) == ["c", "a"]
    assert isinstance(recommendations.to_dicts()[0]["c"], float)
    with pytest.raises(ValueError):
        recommendations.item_ids[0, 0] = "changed"
    with pytest.raises(ValueError):
        recommendations.scores[0, 0] = 0.0
    with pytest.raises(ValueError):
        recommendations.valid_mask[0, 0] = False


@pytest.mark.parametrize(
    ("item_ids", "scores", "error"),
    [
        (np.array(["a"]), np.array([[1.0]]), "must be 2D"),
        (np.array([["a"]]), np.array([[1.0, 2.0]]), "same shape"),
        (np.array([["a"]]), np.array([["bad"]]), "real numeric"),
    ],
)
def test_recommendations_validate_shape_and_scores(item_ids, scores, error):
    with pytest.raises((TypeError, ValueError), match=error):
        Recommendations(item_ids=item_ids, scores=scores)


def test_recommendations_validate_the_validity_mask_shape():
    with pytest.raises(ValueError, match="valid_mask must have the same shape"):
        Recommendations(
            item_ids=np.array([["a"]]),
            scores=np.array([[1.0]]),
            valid_mask=np.ones((1, 2), dtype=bool),
        )


def test_collaborative_recommend_maps_ids_and_collapses_repeated_items():
    model = _CollaborativeRankingModel(["a", "b", "c", "d", "e"])

    result = model.recommend(
        [["a", "a", "c"], ["b"]],
        k=2,
        exclude_seen=True,
    )

    assert result.item_ids.tolist() == [["e", "d"], ["e", "d"]]
    assert model.last_source is not None
    np.testing.assert_array_equal(
        model.last_source.toarray(),
        np.array(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        ),
    )


def test_sequential_recommend_preserves_order_and_repeated_items():
    model = _SequentialRankingModel(["a", "b", "c", "d", "e"])

    model.recommend([["c", "a", "c"]], k=2, exclude_seen=False)

    assert model.last_source is not None
    np.testing.assert_array_equal(model.last_source.row(0), [2, 0, 2])


def test_recommend_applies_allowlist_then_blocklist_before_top_k():
    model = _CollaborativeRankingModel(["a", "b", "c", "d", "e"])

    result = model.recommend(
        [["a"]],
        k=2,
        exclude_seen=False,
        allowlist=["b", "c", "d", "e"],
        blocklist=["d", "e"],
    )

    assert result.item_ids.tolist() == [["c", "b"]]


def test_recommend_keeps_seen_items_by_default_and_blocklist_can_remove_them():
    model = _CollaborativeRankingModel(["a", "b", "c", "d", "e"])

    default = model.recommend([["e"]], k=2)
    blocked = model.recommend([["e"]], k=2, blocklist=["e"])

    assert default.item_ids.tolist() == [["e", "d"]]
    assert blocked.item_ids.tolist() == [["d", "c"]]


@pytest.mark.parametrize(
    ("kwargs", "name"),
    [
        ({"histories": [["missing"]]}, "histories\\[0\\]"),
        ({"histories": [["a"]], "allowlist": ["missing"]}, "allowlist"),
        ({"histories": [["a"]], "blocklist": ["missing"]}, "blocklist"),
    ],
)
def test_recommend_rejects_unknown_ids(kwargs, name):
    model = _CollaborativeRankingModel(["a", "b", "c"])

    with pytest.raises(ValueError, match=name):
        model.recommend(k=1, **kwargs)


def test_recommend_requires_a_batch_of_histories():
    model = _CollaborativeRankingModel(["a", "b", "c"])

    with pytest.raises(TypeError, match="item-ID sequences"):
        model.recommend("a", k=1)
    with pytest.raises(TypeError, match=r"histories\[0\]"):
        model.recommend(["a", "b"], k=1)


def test_recommend_truncates_each_row_without_weakening_filters():
    model = _CollaborativeRankingModel(["a", "b", "c", "d"])

    result = model.recommend(
        [["a", "b", "c"], ["a"]],
        k=3,
        exclude_seen=True,
        allowlist=["a", "b", "c", "d"],
        blocklist=["c"],
    )

    assert result.item_ids.tolist() == [
        ["d", None, None],
        ["d", "b", None],
    ]
    assert result.valid_mask.tolist() == [
        [True, False, False],
        [True, True, False],
    ]
    assert result.valid_counts.tolist() == [1, 2]
    assert np.isneginf(result.scores[0, 1:]).all()
    assert result.to_dicts() == [
        {"d": pytest.approx(3.0)},
        {"d": pytest.approx(3.0), "b": pytest.approx(1.0)},
    ]


def test_recommend_can_return_an_empty_row_when_no_candidate_is_eligible():
    model = _CollaborativeRankingModel(["a", "b", "c"])

    result = model.recommend([["a"]], k=2, allowlist=[])

    assert result.item_ids.tolist() == [[None, None]]
    assert result.valid_counts.tolist() == [0]
    assert result.to_dicts() == [{}]


def test_recommend_strict_mode_rejects_insufficient_candidates():
    model = _CollaborativeRankingModel(["a", "b", "c"])

    with pytest.raises(ValueError, match="k must be in"):
        model.recommend(
            [["a"]],
            k=2,
            allowlist=["b"],
            on_insufficient="raise",
        )
    with pytest.raises(ValueError, match="only 1 unseen candidate"):
        model.recommend(
            [["a", "b"]],
            k=2,
            exclude_seen=True,
            on_insufficient="raise",
        )


@pytest.mark.parametrize("value", ["unknown", None, False])
def test_recommend_validates_the_insufficient_candidate_policy(value):
    model = _CollaborativeRankingModel(["a", "b", "c"])

    with pytest.raises(ValueError, match="on_insufficient"):
        model.recommend([["a"]], k=1, on_insufficient=value)


def test_fixed_model_recommend_uses_positional_ids_by_default():
    interactions = csr_matrix(np.eye(4, dtype=np.float32))
    model = EASE().fit(interactions)

    result = model.recommend([[0]], k=2, exclude_seen=False)

    assert result.item_ids.shape == (1, 2)
    assert set(result.item_ids[0].tolist()) <= {0, 1, 2, 3}


def test_fixed_model_truncates_when_seen_items_leave_too_few_candidates():
    interactions = csr_matrix(np.eye(4, dtype=np.float32))
    model = EASE().fit(interactions, item_ids=["a", "b", "c", "d"])

    result = model.recommend(
        [["a", "b", "c"]],
        k=3,
        exclude_seen=True,
    )

    assert result.item_ids.tolist() == [["d", None, None]]
    assert result.valid_counts.tolist() == [1]
    assert set(result.to_dicts()[0]) == {"d"}


def test_item_identity_and_recommendations_survive_checkpoint(tmp_path):
    interactions = csr_matrix(
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 0],
                [1, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )
    model = EASE().fit(interactions, item_ids=["a", "b", "c", "d"])
    before = model.recommend([["a"]], k=2, exclude_seen=False)
    path = tmp_path / "ease.ckpt"

    model.save(path)
    restored = EASE.load(path)
    after = restored.recommend([["a"]], k=2, exclude_seen=False)

    np.testing.assert_array_equal(restored.source_item_ids, ["a", "b", "c", "d"])
    np.testing.assert_array_equal(after.item_ids, before.item_ids)
    np.testing.assert_allclose(after.scores, before.scores)


def test_cold_start_recommend_tracks_current_candidate_catalog():
    features = np.eye(3, dtype=np.float32)
    model = ContentRecommender().fit(features, item_ids=["a", "b", "c"])
    model.update_candidates(
        item_ids=["d"],
        item_features=np.array([[0.8, 0.2, 0.0]], dtype=np.float32),
    )

    result = model.recommend(
        [["a"]],
        k=2,
        exclude_seen=True,
        allowlist=["b", "c", "d"],
    )

    assert result.item_ids.tolist()[0][0] == "d"
    assert set(result.item_ids[0].tolist()) <= {"b", "c", "d"}


def test_warm_catalog_adapter_recommend_never_returns_cold_items():
    model = _CollaborativeRankingModel(["a", "b", "c"])
    adapter = WarmCatalogAdapter(
        model,
        train_item_ids=["a", "b", "c"],
        catalog_item_ids=["a", "b", "c", "cold"],
    )

    result = adapter.recommend([["a", "cold"]], k=2, exclude_seen=False)

    assert result.item_ids.tolist() == [["c", "b"]]
    assert "cold" not in result.item_ids[0]


@pytest.mark.parametrize("kind", ["ease", "elsa", "rnn", "gpt"])
def test_builtin_fixed_models_apply_identified_candidate_filters(kind):
    item_ids = np.array([f"item-{item}" for item in range(6)], dtype=object)
    interactions = csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )
    sequences = ItemSequences.from_rows(
        [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]],
        n_items=6,
    )
    if kind == "ease":
        model = EASE().fit(interactions, item_ids=item_ids)
    elif kind == "elsa":
        model = ELSATrainer(
            ELSAConfig(
                latent_dim=4,
                epochs=1,
                batch_size=2,
                show_progress=False,
            )
        ).fit(interactions, item_ids=item_ids)
    elif kind == "rnn":
        model = SimpleRNNTrainer(
            SimpleRNNConfig(
                embedding_dim=8,
                hidden_dim=10,
                epochs=1,
                batch_size=2,
                show_progress=False,
            )
        ).fit(sequences, item_ids=item_ids)
    else:
        model = SimpleGPTTrainer(
            SimpleGPTConfig(
                transformer=TransformerConfig(
                    d_model=8,
                    n_heads=2,
                    n_layers=1,
                    dropout=0.0,
                ),
                epochs=1,
                batch_size=2,
                show_progress=False,
            )
        ).fit(sequences, item_ids=item_ids)

    result = model.recommend(
        [["item-0"]],
        k=2,
        exclude_seen=False,
        allowlist=["item-1", "item-2", "item-3"],
        blocklist=["item-3"],
    )

    assert set(result.item_ids[0].tolist()) == {"item-1", "item-2"}
