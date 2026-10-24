"""The user-identity channel: models with a learned embedding per training user."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
import torch
from compresso import SRPTensor
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import NDCG
from compresso_recsys.models import BaseCollaborativeRecommender, MMConcatWrapper, WarmCatalogAdapter


@dataclass(frozen=True)
class _UserConfig:
    unknown_users: Literal["error", "fold_in"] = "error"


class _UserModel(BaseCollaborativeRecommender):
    """Scores item i for training row u as (u + i) % n_items; unknown users get popularity."""

    checkpoint_type = "test_user_model"
    accepts_user_ids = True

    def __init__(self, config: _UserConfig | None = None) -> None:
        self.cfg = config or _UserConfig()
        self.n_items_: int | None = None
        self.popularity_: np.ndarray | None = None
        self.received: list = []

    @property
    def is_fitted(self) -> bool:
        return self.popularity_ is not None

    @property
    def n_items(self) -> int | None:
        return self.n_items_

    def fit(self, interactions: csr_matrix, *, item_ids=None, user_ids=None) -> "_UserModel":
        interactions = interactions.tocsr()
        self._publish_item_vocabulary(self._prepare_item_vocabulary(item_ids, n_items=interactions.shape[1]))
        self._set_user_ids(user_ids, interactions.shape[0])
        self.n_items_ = interactions.shape[1]
        self.popularity_ = np.asarray(interactions.sum(axis=0)).ravel().astype(np.float32)
        return self

    def predict_on_batch(self, source, *, k, exclude_seen=True, candidate_ids=None, user_ids=None):
        source = self._prepare_source(source)
        self.received.append(None if user_ids is None else list(user_ids))
        rows, known = self._resolve_user_rows(user_ids, source.shape[0])
        items = np.arange(self.n_items_, dtype=np.float32)
        scores = np.where(
            known[:, None],
            (rows[:, None] + items[None, :]) % self.n_items_,
            self.popularity_[None, :],
        ).astype(np.float32)
        if exclude_seen:
            seen = np.repeat(np.arange(source.shape[0]), np.diff(source.indptr))
            scores[seen, source.indices] = -np.inf
        return SRPTensor.from_dense(torch.from_numpy(scores), k=k, score_mode="raw")

    @classmethod
    def _from_checkpoint_config(cls, config, reader, *, device):
        return cls(_UserConfig(**config))

    def _save_checkpoint_state(self, writer) -> None:
        writer.write_numpy("state/popularity.npy", self.popularity_)

    def _load_checkpoint_state(self, reader) -> None:
        self.popularity_ = reader.read_numpy("state/popularity.npy")
        self.n_items_ = int(self.popularity_.size)


class _Plain:
    """A model without user identity; records every keyword it is called with."""

    def __init__(self, n_items: int) -> None:
        self.n_items = n_items
        self.keywords: list[set] = []

    def predict_on_batch(self, source, *, k, **kwargs):
        self.keywords.append(set(kwargs))
        scores = torch.arange(self.n_items, 0, -1, dtype=torch.float32).repeat(source.shape[0], 1)
        values, cols = scores.topk(k, dim=1)
        return SRPTensor(cols=cols, vals=values, shape=(source.shape[0], self.n_items))


N_USERS, N_ITEMS = 9, 6


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    seen = rng.random((N_USERS, N_ITEMS)) < 0.3
    seen[:, 0] = True
    relevant = ~seen & (rng.random((N_USERS, N_ITEMS)) < 0.5)
    relevant[np.arange(N_USERS), 1 + np.arange(N_USERS) % (N_ITEMS - 1)] = True
    seen &= ~relevant
    x = csr_matrix(seen.astype(np.float32))
    ids = np.array([f"user{i}" for i in range(N_USERS)])
    return x, csr_matrix(relevant.astype(np.float32)), ids


def test_models_without_identity_are_called_exactly_as_before(data):
    x, targets, ids = data
    model = _Plain(N_ITEMS)
    evaluate_recommender(model, source=x, targets=targets, metrics=[NDCG(3)], sample_ids=ids, batch_size=4)
    assert model.keywords and all(keys == set() for keys in model.keywords)


def test_sample_ids_reach_the_model_batch_by_batch(data):
    x, targets, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    result = evaluate_recommender(model, source=x, targets=targets, metrics=[NDCG(3)],
                                  sample_ids=ids, batch_size=4)  # 4 does not divide 9
    assert [len(batch) for batch in model.received] == [4, 4, 1]
    assert sum(model.received, []) == ids.tolist()
    assert result.metadata["user_identity"] == "passed"


def test_explicit_user_ids_override_sample_ids(data):
    x, targets, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    reordered = ids[::-1]
    evaluate_recommender(model, source=x, targets=targets, metrics=[NDCG(3)],
                         sample_ids=np.arange(N_USERS), user_ids=reordered)
    assert sum(model.received, []) == reordered.tolist()


def test_identity_without_ids_needs_sample_ids(data):
    x, targets, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    with pytest.raises(ValueError, match="uses user identity"):
        evaluate_recommender(model, source=x, targets=targets, metrics=[NDCG(3)])


def test_user_ids_none_applies_the_unknown_user_policy(data):
    x, targets, ids = data
    strict = _UserModel().fit(x, user_ids=ids)
    with pytest.raises(KeyError, match="needs user ids"):
        evaluate_recommender(strict, source=x, targets=targets, metrics=[NDCG(3)],
                             sample_ids=ids, user_ids=None)
    folding = _UserModel(_UserConfig(unknown_users="fold_in")).fit(x, user_ids=ids)
    result = evaluate_recommender(folding, source=x, targets=targets, metrics=[NDCG(3)],
                                  sample_ids=ids, user_ids=None)
    assert result.metadata["user_identity"] == "none"
    assert np.isfinite(result.metrics["ndcg@3"])


def test_user_ids_must_match_the_rows(data):
    x, targets, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    with pytest.raises(ValueError, match="one entry per source row"):
        evaluate_recommender(model, source=x, targets=targets, metrics=[NDCG(3)],
                             sample_ids=ids, user_ids=ids[:3])


def test_resolve_user_rows_maps_known_ids_and_names_unknown_ones(data):
    x, _, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    rows, known = model._user_rows(["user3", "stranger", "user0"])
    assert rows.tolist() == [3, -1, 0]
    assert known.tolist() == [True, False, True]
    with pytest.raises(KeyError, match="stranger"):
        model._resolve_user_rows(["user3", "stranger"], 2)
    folding = _UserModel(_UserConfig(unknown_users="fold_in")).fit(x, user_ids=ids)
    _, known = folding._resolve_user_rows(["user3", "stranger"], 2)
    assert known.tolist() == [True, False]


def test_positional_user_ids_when_fit_gets_none(data):
    x, _, _ = data
    model = _UserModel().fit(x)
    assert model.user_ids.tolist() == list(range(N_USERS))


def test_user_vocabulary_round_trips(tmp_path, data):
    x, targets, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    path = tmp_path / "user_model.zip"
    model.save(path)
    restored = _UserModel.load(path)
    assert restored.user_ids.tolist() == ids.tolist()
    before = model.predict_on_batch(x, k=3, user_ids=ids)
    after = restored.predict_on_batch(x, k=3, user_ids=ids)
    torch.testing.assert_close(after.cols, before.cols)


def test_recommend_forwards_user_ids_only_to_models_that_use_them(data):
    x, _, ids = data
    model = _UserModel().fit(x, user_ids=ids)
    recs = model.recommend([[0], [0, 2]], k=2, user_ids=["user1", "user5"])
    assert recs.item_ids.shape == (2, 2)
    assert sum(model.received, []) == ["user1", "user5"]

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from test_model_bases import _PopularityModel  # a model without user identity

    plain = _PopularityModel().fit(x)
    with pytest.raises(TypeError, match="does not use user identity"):
        plain.recommend([[0]], k=2, user_ids=["user1"])


def test_wrappers_declare_identity_from_the_model_they_wrap(data):
    x, _, ids = data
    inner = _UserModel().fit(x, user_ids=ids)
    adapter = WarmCatalogAdapter(inner, train_item_ids=np.arange(N_ITEMS),
                                 catalog_item_ids=np.arange(N_ITEMS + 2))
    assert adapter.accepts_user_ids is True
    adapter.predict_on_batch(x, k=2, user_ids=ids)
    assert inner.received[-1] == ids.tolist()

    wrapper_property = MMConcatWrapper.accepts_user_ids.fget
    assert wrapper_property(SimpleNamespace(inner_model=inner)) is True
    assert wrapper_property(SimpleNamespace(inner_model=object())) is False
