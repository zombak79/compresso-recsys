from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from compresso import SRPTensor
from scipy.sparse import csr_matrix, issparse

from compresso_recsys.models import (
    BaseColdStartRecommender,
    BaseMultiModalRecommender,
    MultiModalCandidateCatalog,
    MutableMultiModalCandidateCatalog,
)
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter


def _features(rows=(0, 1, 2)):
    return {
        "text": np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)[list(rows)],
        "image": csr_matrix(np.array([[1], [2], [3]], dtype=np.float32)[list(rows)]),
    }


def _dense(matrix):
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def _installed(**kwargs):
    catalog = MutableMultiModalCandidateCatalog(**kwargs)
    catalog.install(
        source_item_ids=["a", "b", "c"],
        candidate_features=_features(),
        feature_space_id="embeddings@1",
    )
    return catalog


@dataclass
class _Config:
    dtype: str = "float32"


class _MultiModalContent(BaseMultiModalRecommender):
    """Minimal concrete scorer exercising the inherited production workflow."""

    checkpoint_type = "test_multimodal_content"

    def __init__(self, config=None):
        super().__init__()
        self.cfg = config or _Config()
        self.source_features = None

    @property
    def is_fitted(self):
        return self.source_features is not None

    def fit(self, interactions, item_features, *, item_ids):
        if interactions.shape[1] != len(item_ids):
            raise ValueError("history width does not match item IDs")
        snapshot = self.candidates.install(
            source_item_ids=item_ids,
            candidate_features=item_features,
            dtype=self.cfg.dtype,
        )
        self.source_features = {
            name: _dense(matrix).copy()
            for name, matrix in snapshot.item_features.items()
        }
        return self

    def predict_on_batch(self, source, *, k, exclude_seen=True, candidate_ids=None):
        source = self._prepare_source(source)
        selected = self.candidates.resolve_selection(candidate_ids)
        scores = sum(
            (source @ self.source_features[name]) @ _dense(matrix).T
            for name, matrix in selected.features.items()
        )
        if exclude_seen:
            users, history_items = source.nonzero()
            catalog_rows = selected.source_to_candidate[history_items]
            local_rows = np.full(catalog_rows.size, -1, dtype=np.int64)
            present = catalog_rows >= 0
            local_rows[present] = selected.candidate_to_local[catalog_rows[present]]
            present = local_rows >= 0
            scores[users[present], local_rows[present]] = -np.inf
        local = SRPTensor.from_dense(torch.from_numpy(scores), k=k, score_mode="raw")
        return SRPTensor(
            cols=torch.from_numpy(selected.rows)[local.cols],
            vals=local.vals,
            shape=(source.shape[0], selected.catalog.n_items),
        )

    @classmethod
    def _from_checkpoint_config(cls, config, reader, *, device):
        return cls(_Config(**config))

    def _save_checkpoint_state(self, writer):
        super()._save_checkpoint_state(writer)
        for index, name in enumerate(self.candidates.feature_dims):
            writer.write_numpy(f"source/{index}.npy", self.source_features[name])

    def _load_checkpoint_state(self, reader):
        super()._load_checkpoint_state(reader)
        self.source_features = {
            name: reader.read_numpy(f"source/{index}.npy")
            for index, name in enumerate(self.candidates.feature_dims)
        }


def _model():
    return _MultiModalContent().fit(
        csr_matrix(np.eye(3)), _features(), item_ids=["a", "b", "c"]
    )


def test_base_is_abstract_and_reuses_existing_history_workflow():
    with pytest.raises(TypeError, match="abstract"):
        BaseMultiModalRecommender()
    model = _model()
    assert isinstance(model, BaseColdStartRecommender)
    assert BaseMultiModalRecommender.recommend is BaseColdStartRecommender.recommend
    assert BaseMultiModalRecommender.predict is BaseColdStartRecommender.predict
    source = csr_matrix(np.eye(3, dtype=np.float32))
    batched = model.predict(source, k=1, batch_size=1)
    single = model.predict_on_batch(source, k=1)
    torch.testing.assert_close(batched.cols, single.cols)
    torch.testing.assert_close(batched.vals, single.vals)
    recommended = model.recommend([["a"]], k=1, exclude_seen=True)
    assert recommended.item_ids[0, 0] == "c"
    with pytest.raises(ValueError, match="source has 4 items"):
        model.predict(csr_matrix((1, 4)), k=1)


def test_registered_candidate_is_recommendable_but_not_a_history_item():
    model = _model()
    model.update_candidates(
        item_ids=["cold"],
        item_features={"image": np.array([[20]]), "text": np.array([[4, 0]])},
    )
    assert model.recommend([["a"]], k=1).item_ids[0, 0] == "cold"
    with pytest.raises(ValueError, match="unknown item ID.*cold"):
        model.recommend([["cold"]], k=1)
    assert model.source_item_ids.tolist() == ["a", "b", "c"]


def test_removed_candidate_remains_valid_in_histories_and_filters_work():
    model = _model()
    model.remove_candidates(["a"])
    results = model.recommend([["a"]], k=2, exclude_seen=True)
    assert set(results.item_ids[0]) == {"b", "c"}
    assert model.source_item_ids.tolist() == ["a", "b", "c"]
    assert model.recommend([["a"]], k=1, allowlist=["b"]).item_ids[0, 0] == "b"
    assert model.recommend([["a"]], k=1, blocklist=["c"]).item_ids[0, 0] == "b"


def test_default_registration_preserves_separate_raw_modalities():
    catalog = _installed()
    snapshot = catalog.snapshot()
    assert dict(snapshot.feature_dims) == {"text": 2, "image": 1}
    assert issparse(snapshot.item_features["image"])
    for name, expected in _features().items():
        np.testing.assert_array_equal(
            _dense(snapshot.item_features[name]), _dense(expected)
        )


@pytest.mark.parametrize("kind", ["numpy", "csr", "torch", "torch_csr", "srp"])
def test_all_matrix_input_formats_are_canonicalized(kind):
    matrix = np.array([[1, 0], [2, 3]], dtype=np.float64)
    converters = {
        "numpy": lambda x: x,
        "csr": csr_matrix,
        "torch": torch.from_numpy,
        "torch_csr": lambda x: torch.from_numpy(x).to_sparse_csr(),
        "srp": lambda x: SRPTensor.from_dense(
            torch.from_numpy(x), k=2, score_mode="raw"
        ),
    }
    catalog = MutableMultiModalCandidateCatalog()
    snapshot = catalog.install(
        source_item_ids=["a", "b"],
        candidate_features={"only": converters[kind](matrix)},
        dtype="float64",
    )
    np.testing.assert_array_equal(_dense(snapshot.item_features["only"]), matrix)
    assert snapshot.item_features["only"].dtype == np.float64


@pytest.mark.parametrize(
    "features, message",
    [
        ({}, "at least one modality"),
        (np.ones((3, 2)), "mapping"),
        ({"": np.ones((3, 2))}, "non-empty strings"),
        ({1: np.ones((3, 2))}, "non-empty strings"),
        ({"text": np.ones((3, 2))}, "modality names"),
        (
            {
                "text": np.ones((3, 2)),
                "image": np.ones((3, 1)),
                "extra": np.ones((3, 1)),
            },
            "modality names",
        ),
        ({"text": np.ones((2, 2)), "image": np.ones((3, 1))}, "rows"),
        ({"text": np.ones((3, 3)), "image": np.ones((3, 1))}, "columns"),
        ({"text": np.ones(3), "image": np.ones((3, 1))}, "two-dimensional"),
        ({"text": np.full((3, 2), np.nan), "image": np.ones((3, 1))}, "finite"),
    ],
)
def test_failed_build_preserves_snapshot(features, message):
    catalog = _installed()
    before = catalog.snapshot()
    with pytest.raises((ValueError, TypeError), match=message):
        catalog.build(item_ids=["x", "y", "z"], item_features=features)
    assert catalog.snapshot() is before


def test_failed_refit_preserves_source_and_schema():
    catalog = _installed()
    snapshot = catalog.snapshot()
    with pytest.raises(ValueError, match="rows"):
        catalog.install(source_item_ids=["new"], candidate_features=_features())
    assert catalog.snapshot() is snapshot
    assert catalog.source_item_ids.tolist() == ["a", "b", "c"]
    assert dict(catalog.feature_dims) == {"text": 2, "image": 1}


def test_dtype_conversion_cannot_install_nonfinite_features():
    catalog = _installed()
    old = catalog.snapshot()
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        catalog.install(
            source_item_ids=["new"],
            candidate_features={"text": np.array([[1e300]], dtype=np.float64)},
        )
    assert catalog.snapshot() is old
    assert catalog.source_item_ids.tolist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "operation",
    [
        lambda catalog: catalog.snapshot(),
        lambda catalog: catalog.build(item_ids=["a"], item_features=_features((0,))),
        lambda catalog: catalog.update(item_ids=["a"], item_features=_features((0,))),
        lambda catalog: catalog.remove(["a"]),
        lambda catalog: catalog.resolve_selection(None),
        lambda catalog: catalog.align_source(csr_matrix([[1]]), item_ids=["a"]),
    ],
)
def test_catalog_operations_require_installation(operation):
    catalog = MutableMultiModalCandidateCatalog()
    assert not catalog.is_installed
    assert catalog.source_vocabulary is None
    assert catalog.n_items is None
    assert catalog.feature_dims is None
    with pytest.raises(RuntimeError, match="not been fitted"):
        operation(catalog)


def test_models_can_transform_registration_inputs_with_explicit_overrides(tmp_path):
    class ProjectedContent(_MultiModalContent):
        @staticmethod
        def project(features):
            return {
                name: _dense(matrix).sum(axis=1, keepdims=True)
                for name, matrix in features.items()
            }

        def fit(self, interactions, item_features, *, item_ids):
            return super().fit(
                interactions, self.project(item_features), item_ids=item_ids
            )

        def build_candidates(self, *, item_features, **kwargs):
            return super().build_candidates(
                item_features=self.project(item_features), **kwargs
            )

        def update_candidates(self, *, item_features, **kwargs):
            return super().update_candidates(
                item_features=self.project(item_features), **kwargs
            )

    model = ProjectedContent().fit(
        csr_matrix(np.eye(3)), _features(), item_ids=["a", "b", "c"]
    )
    assert dict(model.candidates.feature_dims) == {"text": 1, "image": 1}
    model.build_candidates(item_ids=["a", "b"], item_features=_features((0, 1)))
    model.update_candidates(item_ids=["cold"], item_features=_features((2,)))
    np.testing.assert_array_equal(
        model.candidates.snapshot().item_features["text"], [[1], [1], [2]]
    )
    path = tmp_path / "projected.zip"
    model.save(path)
    restored = ProjectedContent.load(path)
    restored.update_candidates(item_ids=["another"], item_features=_features((0,)))
    assert dict(restored.candidates.feature_dims) == {"text": 1, "image": 1}
    assert restored.recommend([["a"]], k=1).item_ids[0, 0] == "cold"


def test_reordering_modality_keys_does_not_change_predictions():
    model = _model()
    source = csr_matrix([[1, 0, 0]])
    before = model.predict(source, k=2)
    model.build_candidates(
        item_ids=["a", "b", "c"],
        item_features=dict(reversed(list(_features().items()))),
    )
    after = model.predict(source, k=2)
    torch.testing.assert_close(before.cols, after.cols)
    torch.testing.assert_close(before.vals, after.vals)
    assert tuple(model.candidates.snapshot().item_features) == ("text", "image")


def test_mutations_and_selection_keep_modalities_and_metadata_aligned():
    catalog = _installed()
    catalog.build(
        item_ids=["x", "a"],
        item_features=_features((1, 0)),
        metadata=pd.DataFrame({"label": ["X", "A"]}),
    )
    held = catalog.snapshot()
    catalog.update(
        item_ids=["a", "d"],
        item_features=_features((2, 1)),
        metadata=pd.DataFrame({"label": ["A2", "D"], "item_id": ["a", "d"]}),
        on_conflict="replace",
    )
    catalog.update(item_ids=["d"], item_features=_features((0,)), on_conflict="ignore")
    assert catalog.snapshot().version == 3
    catalog.remove(["x"])
    result = catalog.snapshot()
    assert result.item_ids.tolist() == ["a", "d"]
    np.testing.assert_array_equal(
        result.item_features["text"], _features((2, 1))["text"]
    )
    np.testing.assert_array_equal(_dense(result.item_features["image"]), [[3], [2]])
    assert result.metadata["label"].tolist() == ["A2", "D"]
    assert result.metadata["item_id"].tolist() == ["a", "d"]
    assert held.item_ids.tolist() == ["x", "a"]
    np.testing.assert_array_equal(held.item_features["text"], _features((1, 0))["text"])
    selection = catalog.resolve_selection(["d"])
    assert selection.rows.tolist() == [1]
    assert selection.source_to_candidate.tolist() == [0, -1, -1]
    assert selection.candidate_to_local.tolist() == [-1, 0]
    np.testing.assert_array_equal(selection.features["text"], [[0, 1]])
    np.testing.assert_array_equal(_dense(selection.features["image"]), [[2]])
    assert catalog.align_source(
        csr_matrix([[30, 10, 20]]), item_ids=["c", "a", "b"]
    ).toarray().tolist() == [[10, 20, 30]]


def test_snapshots_own_features_and_metadata_and_expose_read_only_containers():
    inputs = _features()
    metadata = pd.DataFrame({"label": ["A", "B", "C"]})
    snapshot = MultiModalCandidateCatalog(
        item_ids=["a", "b", "c"], item_features=inputs, metadata=metadata
    )
    inputs["text"][0, 0] = 99
    inputs["image"].data[:] = 99
    metadata.iloc[0, 0] = "changed"
    assert snapshot.item_features["text"][0, 0] == 1
    assert snapshot.item_features["image"][0, 0] == 1
    assert snapshot.metadata.iloc[0, 0] == "A"
    metadata_copy = snapshot.metadata
    metadata_copy.iloc[0, 0] = "copy"
    assert snapshot.metadata.iloc[0, 0] == "A"
    with pytest.raises(TypeError):
        snapshot.item_features["other"] = np.zeros((3, 1))
    with pytest.raises(TypeError):
        snapshot.feature_dims["text"] = 4
    with pytest.raises(ValueError):
        snapshot.item_features["text"][0, 0] = 5
    with pytest.raises(ValueError):
        snapshot.item_features["image"].data[0] = 5
    assert snapshot.ids_for(torch.tensor([2, 0])).tolist() == ["c", "a"]


def test_callback_failure_is_post_publication_and_validation_failure_is_not():
    observed = []
    catalog = _installed()

    def fail(snapshot):
        assert catalog.snapshot() is snapshot
        observed.append(snapshot)
        raise RuntimeError("cache notification failed")

    catalog._on_publish = fail
    old = catalog.snapshot()
    with pytest.raises(ValueError, match="rows"):
        catalog.update(item_ids=["d"], item_features=_features())
    assert catalog.snapshot() is old
    assert observed == []
    with pytest.raises(RuntimeError, match="notification failed"):
        catalog.update(item_ids=["d"], item_features=_features((2,)))
    assert catalog.snapshot() is observed[0]
    assert catalog.snapshot().item_ids.tolist() == ["a", "b", "c", "d"]


def test_mutation_errors_are_atomic_and_noops_do_not_publish():
    catalog = _installed()
    old = catalog.snapshot()
    with pytest.raises(ValueError, match="already exists"):
        catalog.update(item_ids=["a"], item_features=_features((0,)))
    with pytest.raises(ValueError, match="feature_space_id"):
        catalog.build(
            item_ids=["a", "b", "c"],
            item_features=_features(),
            feature_space_id="different",
        )
    with pytest.raises(ValueError, match="at least one item"):
        catalog.remove(["a", "b", "c"])
    with pytest.raises(KeyError, match="unknown"):
        catalog.remove(["unknown"])
    assert catalog.remove(["unknown"], missing="ignore") is old
    assert (
        catalog.update(
            item_ids=["a"], item_features=_features((0,)), on_conflict="ignore"
        )
        is old
    )
    assert catalog.snapshot() is old


def test_model_checkpoint_preserves_source_catalog_predictions_and_later_updates(
    tmp_path,
):
    model = _model()
    model.update_candidates(
        item_ids=["cold"],
        item_features=_features((2,)),
        metadata=pd.DataFrame({"label": ["new"]}),
    )
    model.remove_candidates(["a"])
    before = model.recommend([["a"], ["b"]], k=2, exclude_seen=True)
    path = tmp_path / "model.zip"
    model.save(path)
    restored = _MultiModalContent.load(path)
    after = restored.recommend([["a"], ["b"]], k=2, exclude_seen=True)
    np.testing.assert_array_equal(before.item_ids, after.item_ids)
    np.testing.assert_array_equal(before.scores, after.scores)
    assert restored.source_item_ids.tolist() == ["a", "b", "c"]
    assert restored.candidate_item_ids.tolist() == ["b", "c", "cold"]
    assert restored.candidates.snapshot().version == model.candidates.snapshot().version
    pd.testing.assert_frame_equal(
        restored.candidates.snapshot().metadata, model.candidates.snapshot().metadata
    )
    assert issparse(restored.candidates.snapshot().item_features["image"])
    restored.update_candidates(
        item_ids=["newer"], item_features=dict(reversed(list(_features((0,)).items())))
    )
    assert restored.candidate_item_ids.tolist() == ["b", "c", "cold", "newer"]
    with pytest.raises(ValueError, match="modality names"):
        restored.update_candidates(
            item_ids=["bad"], item_features={"text": np.ones((1, 2))}
        )
    with pytest.raises(ValueError, match="unknown item ID"):
        restored.recommend([["newer"]], k=1)


def test_catalog_checkpoint_uses_safe_paths_and_validates_schema_before_loading(
    tmp_path,
):
    catalog = MutableMultiModalCandidateCatalog()
    catalog.install(
        source_item_ids=["a"],
        candidate_features={"text/model": np.ones((1, 2))},
        dtype="float64",
    )
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(
        path, model_type="test_catalog", optimizer_included=False
    ) as writer:
        catalog._save_checkpoint(writer)
    restored = MutableMultiModalCandidateCatalog()
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        snapshot = restored._load_checkpoint(reader)
        assert snapshot.item_features["text/model"].dtype == np.float64
        state_path = reader._input_path("catalog/state.json")
        state = json.loads(state_path.read_text())
        state["modalities"][0]["n_features"] = 99
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="dimensions"):
            restored._load_checkpoint(reader)
        assert restored.snapshot() is snapshot
