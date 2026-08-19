"""The catalog lifecycle as an owned object rather than an inherited one.

``CandidateCatalog`` was always a standalone immutable snapshot. What was stuck
inside ``BaseColdStartRecommender`` was the lifecycle around it, which meant
"cold-capable" and "inherits that base" were the same statement. These tests are
about the lifecycle standing on its own, and about the thing that motivated
moving it: a model reading ordered histories can now be cold-capable without
multiple inheritance or a fourth base class.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso import SRPTensor
from compresso_recsys.models import (
    BaseSequentialRecommender,
    CandidateCatalog,
    MutableCandidateCatalog,
    Recommender,
    SequentialRecommender,
)
from compresso_recsys.sequences import ItemSequences

WARM_IDS = np.array(["a", "b", "c"], dtype=object)
# One feature per item, so a profile's score is trivially readable.
WARM_FEATURES = np.eye(3, dtype=np.float32)


def _installed(**overrides) -> MutableCandidateCatalog:
    catalog = MutableCandidateCatalog(**overrides)
    catalog.install(
        source_item_ids=WARM_IDS,
        source_popularity=np.array([3.0, 2.0, 1.0], dtype=np.float32),
        n_input_features=3,
        candidate_features=WARM_FEATURES,
        metadata=None,
        feature_space_id="features@1",
        dtype=np.dtype("float32"),
        include_popularity=False,
    )
    return catalog


# --------------------------------------------------------------------------
# before installation
# --------------------------------------------------------------------------


def test_a_fresh_catalog_holds_nothing():
    catalog = MutableCandidateCatalog()

    assert not catalog.is_installed
    assert catalog.n_items is None
    assert catalog.source_vocabulary is None
    assert catalog.source_item_ids is None
    assert catalog.source_id_to_row is None
    assert catalog.source_popularity is None
    assert catalog.feature_space_id is None
    assert catalog.n_input_features is None


@pytest.mark.parametrize(
    "call",
    [
        lambda c: c.snapshot(),
        lambda c: c.align_source(None, item_ids=["a"]),
        lambda c: c.resolve_selection(None),
        lambda c: c.remove(["a"]),
    ],
)
def test_using_a_catalog_before_installation_is_an_error(call):
    with pytest.raises(RuntimeError, match="not been fitted|not installed"):
        call(MutableCandidateCatalog())


# --------------------------------------------------------------------------
# installation and reading
# --------------------------------------------------------------------------


def test_install_publishes_version_one_and_the_source_surface():
    catalog = _installed()

    snapshot = catalog.snapshot()
    assert isinstance(snapshot, CandidateCatalog)
    assert snapshot.version == 1
    assert catalog.is_installed
    assert catalog.n_items == 3
    assert catalog.source_item_ids.tolist() == ["a", "b", "c"]
    assert catalog.source_id_to_row["b"] == 1
    assert catalog.source_popularity.tolist() == [3.0, 2.0, 1.0]
    assert catalog.feature_space_id == "features@1"
    assert catalog.n_input_features == 3
    assert catalog.source_vocabulary.n_items == 3


def test_the_source_surface_is_read_only():
    """Fitted state, so a caller must not be able to edit it in place."""
    catalog = _installed()

    with pytest.raises(ValueError, match="read-only"):
        catalog.source_popularity[0] = 99.0


def test_a_snapshot_is_a_consistent_view_that_republishing_cannot_move():
    """Why reads go through snapshot() rather than forwarded properties.

    Several reads off one snapshot cannot straddle a concurrent republish.
    Reading n_items and then item_ids off the mutable holder could.
    """
    catalog = _installed()
    held = catalog.snapshot()

    catalog.update(
        item_ids=["d"],
        item_features=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )

    assert held.n_items == 3
    assert held.item_ids.tolist() == ["a", "b", "c"]
    assert held.version == 1
    assert catalog.snapshot().n_items == 4
    assert catalog.snapshot() is not held


# --------------------------------------------------------------------------
# the publish callback
# --------------------------------------------------------------------------


def test_every_publication_notifies_the_owner():
    """How an owner drops caches derived from the previous snapshot."""
    seen: list[tuple[int, int]] = []
    catalog = _installed(on_publish=lambda c: seen.append((c.version, c.n_items)))

    catalog.update(
        item_ids=["d"], item_features=np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    )
    catalog.build(
        item_ids=["x", "y"],
        item_features=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        feature_space_id="features@1",
    )
    catalog.remove(["x"])

    assert seen == [(1, 3), (2, 4), (3, 2), (4, 1)]


def test_a_catalog_without_a_callback_is_usable():
    """The owner hook is optional, so the object stands alone."""
    catalog = _installed()

    assert catalog.remove(["c"]).n_items == 2


def test_the_notification_carries_the_snapshot_that_is_already_live():
    """An owner reading back during the callback must not see the old catalog."""
    observed: list[bool] = []
    owner: dict[str, MutableCandidateCatalog] = {}
    catalog = MutableCandidateCatalog(
        on_publish=lambda c: observed.append(c is owner["catalog"].snapshot())
    )
    owner["catalog"] = catalog
    catalog.install(
        source_item_ids=WARM_IDS,
        source_popularity=np.zeros(3, dtype=np.float32),
        n_input_features=3,
        candidate_features=WARM_FEATURES,
        metadata=None,
        feature_space_id=None,
        dtype=np.dtype("float32"),
        include_popularity=False,
    )

    assert observed == [True]


# --------------------------------------------------------------------------
# lifecycle operations
# --------------------------------------------------------------------------


def test_build_replaces_the_whole_catalog_and_keeps_the_source_vocabulary():
    catalog = _installed()

    catalog.build(
        item_ids=["cold-1", "cold-2"],
        item_features=np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        feature_space_id="features@1",
    )

    assert catalog.snapshot().item_ids.tolist() == ["cold-1", "cold-2"]
    assert catalog.snapshot().version == 2
    # The source side is fitted state and does not move with the candidates.
    assert catalog.source_item_ids.tolist() == ["a", "b", "c"]


def test_update_can_add_and_replace_and_remove_can_shrink():
    catalog = _installed()

    catalog.update(
        item_ids=["d"], item_features=np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
    )
    catalog.update(
        item_ids=["a"],
        item_features=np.array([[9.0, 0.0, 0.0]], dtype=np.float32),
        on_conflict="replace",
    )
    catalog.remove(["b"])

    snapshot = catalog.snapshot()
    assert snapshot.item_ids.tolist() == ["a", "c", "d"]
    assert snapshot.item_features[snapshot.rows_for(["a"])[0]].tolist() == [9.0, 0.0, 0.0]


def test_a_declared_feature_space_is_enforced_on_later_changes():
    catalog = _installed()

    with pytest.raises(ValueError, match="feature_space_id must match"):
        catalog.build(
            item_ids=["z"],
            item_features=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            feature_space_id="a-different-model",
        )


def test_resolve_selection_maps_source_items_onto_candidate_rows():
    catalog = _installed()
    catalog.update(
        item_ids=["cold"],
        item_features=np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
    )

    selection = catalog.resolve_selection(None)

    assert selection.rows.tolist() == [0, 1, 2, 3]
    # Source items a, b, c occupy candidate rows 0, 1, 2; "cold" has no source row.
    assert selection.source_to_candidate.tolist() == [0, 1, 2]
    assert selection.catalog.n_items == 4

    restricted = catalog.resolve_selection(["cold", "a"])
    assert restricted.rows.tolist() == [0, 3]


# --------------------------------------------------------------------------
# the payoff: cold candidates from an ordered history, by composition
# --------------------------------------------------------------------------


class _SequentialContentModel(BaseSequentialRecommender):
    """Cold-capable *and* sequential, with one base class and no mixin.

    This is the class that could not exist before the extraction. It needs an
    ordered history on the input side and a rebuildable candidate catalog on the
    output side, and those used to live on two different base classes.

    The scoring is deliberately trivial: a profile is the feature vector of the
    most recent interaction, and candidates are ranked by dot product with it.
    """

    def __init__(self) -> None:
        self.candidates = MutableCandidateCatalog()

    def fit(self, source_item_ids, item_features) -> "_SequentialContentModel":
        self.candidates.install(
            source_item_ids=np.asarray(source_item_ids, dtype=object),
            source_popularity=np.zeros(len(source_item_ids), dtype=np.float32),
            n_input_features=item_features.shape[1],
            candidate_features=item_features,
            metadata=None,
            feature_space_id=None,
            dtype=np.dtype("float32"),
            include_popularity=False,
        )
        return self

    @property
    def is_fitted(self) -> bool:
        return self.candidates.is_installed

    @property
    def n_items(self) -> int | None:
        return self.candidates.n_items

    def predict_on_batch(self, source, *, k, exclude_seen=True):
        catalog = self.candidates.snapshot()
        features = np.asarray(catalog.item_features, dtype=np.float32)
        source_features = np.asarray(
            self.candidates.resolve_selection(None).features, dtype=np.float32
        )
        scores = np.zeros((source.n_rows, catalog.n_items), dtype=np.float32)
        for row in range(source.n_rows):
            history = source.row(row)
            if history.size:
                # Only the most recent interaction, so order decides the answer.
                scores[row] = features @ source_features[int(history[-1])]
            if exclude_seen:
                scores[row, np.unique(history)] = -np.inf
        order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
        return SRPTensor(
            cols=torch.from_numpy(np.ascontiguousarray(order)),
            vals=torch.from_numpy(
                np.ascontiguousarray(np.take_along_axis(scores, order, axis=1))
            ),
            shape=(source.n_rows, catalog.n_items),
        )


def test_a_sequential_model_can_own_a_catalog_without_multiple_inheritance():
    model = _SequentialContentModel().fit(WARM_IDS, WARM_FEATURES)

    assert isinstance(model, BaseSequentialRecommender)
    assert isinstance(model, SequentialRecommender)
    assert isinstance(model, Recommender)
    # And crucially not the cold-start base, which it never inherited.
    from compresso_recsys.models import BaseColdStartRecommender

    assert not isinstance(model, BaseColdStartRecommender)
    assert model.is_fitted and model.n_items == 3


def test_that_model_recommends_items_absent_from_every_history():
    """The actual cold-start capability, reached from a sequence source."""
    model = _SequentialContentModel().fit(WARM_IDS, WARM_FEATURES)
    # A candidate that shares item "c"'s feature but was never interacted with.
    model.candidates.update(
        item_ids=["cold"], item_features=np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
    )

    predictions = model.predict(
        ItemSequences.from_rows([[0, 2]], n_items=3), k=1, exclude_seen=True
    )

    assert model.candidates.snapshot().ids_for(predictions.cols).tolist() == [["cold"]]


def test_the_owned_catalog_stays_rebuildable_after_fitting():
    model = _SequentialContentModel().fit(WARM_IDS, WARM_FEATURES)

    model.candidates.build(
        item_ids=["new-1", "new-2"],
        item_features=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
    )

    assert model.n_items == 2
    assert model.candidates.snapshot().version == 2
    # The history vocabulary is untouched, which is the property the base
    # deliberately does not tie to the candidate catalog.
    assert model.candidates.source_item_ids.tolist() == ["a", "b", "c"]


def test_the_read_surface_is_not_forwarded():
    """The decision that reads go through a snapshot, pinned.

    Forwarding ``item_ids``, ``rows_for`` and ``ids_for`` would be convenient and
    would give one piece of state two names, each read able to land on a
    different version. ``n_items`` is the one exception, because reporting "how
    many candidates are there" needs an answer before installation, which a
    snapshot cannot give.
    """
    catalog = _installed()
    surface = {name for name in dir(catalog) if not name.startswith("_")}

    assert not surface & {"item_ids", "rows_for", "ids_for", "metadata", "version"}
    assert surface == {
        "align_source",
        "build",
        "feature_space_id",
        "install",
        "is_installed",
        "n_input_features",
        "n_items",
        "remove",
        "resolve_selection",
        "snapshot",
        "source_id_to_row",
        "source_item_ids",
        "source_popularity",
        "source_vocabulary",
        "update",
    }
