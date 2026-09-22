from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Barrier, Event, current_thread

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
    multimodal,
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


@pytest.mark.parametrize(
    "candidate_ids",
    [None, ["c", "a", "b"], ["c", "b"]],
    ids=["all", "explicit-all", "subset"],
)
def test_selection_and_predictions_use_int64_with_simulated_int32_default(
    monkeypatch, candidate_ids
):
    model = _model()
    source = csr_matrix([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
    expected = model.predict_on_batch(source, k=1, candidate_ids=candidate_ids)
    expected_recommendations = model.recommend(
        [["a"], ["b"]], k=1, allowlist=candidate_ids
    )

    class Int32DefaultNumpy:
        def __getattr__(self, name):
            return getattr(np, name)

        def arange(self, *args, **kwargs):
            if len(args) < 4 and kwargs.get("dtype") is None:
                kwargs["dtype"] = np.int32
            return np.arange(*args, **kwargs)

    # Simulate the relevant NumPy 1.x Windows default only in the catalog
    # module. Explicit dtypes still work; NumPy/SciPy/Torch are not patched.
    simulated_numpy = Int32DefaultNumpy()
    assert simulated_numpy.arange(3).dtype == np.int32
    monkeypatch.setattr(multimodal, "np", simulated_numpy)

    selected = model.candidates.resolve_selection(candidate_ids)
    for indices in (
        selected.rows,
        selected.source_to_candidate,
        selected.candidate_to_local,
    ):
        assert indices.dtype == np.int64
    for actual in (
        model.predict_on_batch(source, k=1, candidate_ids=candidate_ids),
        model.predict(source, k=1, batch_size=1, candidate_ids=candidate_ids),
    ):
        assert actual.cols.dtype == torch.long
        torch.testing.assert_close(actual.cols, expected.cols)
        torch.testing.assert_close(actual.vals, expected.vals)
    recommendations = model.recommend([["a"], ["b"]], k=1, allowlist=candidate_ids)
    np.testing.assert_array_equal(
        recommendations.item_ids, expected_recommendations.item_ids
    )
    np.testing.assert_array_equal(
        recommendations.scores, expected_recommendations.scores
    )


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


def test_reinstall_advances_version_and_replaces_source_schema_and_candidates():
    published = []
    catalog = _installed(on_publish=published.append)
    original = catalog.snapshot()
    catalog.update(item_ids=["d"], item_features=_features((0,)))
    catalog.update(item_ids=["e"], item_features=_features((1,)))
    catalog.remove(["b"])

    installed = catalog.install(
        source_item_ids=["new-a", "new-b"],
        candidate_features={"audio": np.ones((2, 4))},
        metadata=pd.DataFrame({"label": ["A", "B"]}),
        feature_space_id="embeddings@2",
        dtype="float64",
    )

    assert [snapshot.version for snapshot in published] == [1, 2, 3, 4, 5]
    assert catalog.snapshot() is installed is published[-1]
    assert installed.item_ids.tolist() == ["new-a", "new-b"]
    assert catalog.source_item_ids.tolist() == ["new-a", "new-b"]
    assert dict(installed.feature_dims) == {"audio": 4}
    assert installed.item_features["audio"].dtype == np.float64
    assert installed.feature_space_id == "embeddings@2"
    assert installed.metadata["label"].tolist() == ["A", "B"]
    assert original.version == 1
    assert original.item_ids.tolist() == ["a", "b", "c"]
    np.testing.assert_array_equal(original.item_features["text"], _features()["text"])

    updated = catalog.update(
        item_ids=["cold"], item_features={"audio": np.zeros((1, 4))}
    )
    assert updated.version == 6
    assert updated.item_ids.tolist() == ["new-a", "new-b", "cold"]
    assert updated.item_features["audio"].dtype == np.float64


def test_concurrent_reinstalls_and_updates_follow_publication_order():
    published = []
    catalog = _installed(on_publish=published.append)
    rounds = 8
    start = Barrier(2, timeout=5)

    def reinstall():
        for _ in range(rounds):
            start.wait()
            catalog.install(
                source_item_ids=["a", "b", "c"], candidate_features=_features()
            )

    def update():
        for index in range(rounds):
            start.wait()
            catalog.update(item_ids=[f"cold-{index}"], item_features=_features((0,)))

    with ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(reinstall), workers.submit(update)]
        for future in futures:
            future.result(timeout=10)

    assert [snapshot.version for snapshot in published] == list(
        range(1, 2 * rounds + 2)
    )
    # Each round permits either serial order: an update after installation is
    # retained; an update before installation is superseded by the replacement.
    for index in range(rounds):
        first, second = published[1 + 2 * index : 3 + 2 * index]
        if first.n_items == 3:
            assert second.item_ids.tolist() == ["a", "b", "c", f"cold-{index}"]
        else:
            assert first.item_ids[-1] == f"cold-{index}"
            assert second.item_ids.tolist() == ["a", "b", "c"]
    assert catalog.snapshot() is published[-1]


def test_failed_refit_preserves_source_and_schema():
    published = []
    catalog = _installed(on_publish=published.append)
    snapshot = catalog.snapshot()
    with pytest.raises(ValueError, match="rows"):
        catalog.install(
            source_item_ids=["new"], candidate_features=_features(), dtype="float64"
        )
    assert catalog.snapshot() is snapshot
    assert catalog.source_item_ids.tolist() == ["a", "b", "c"]
    assert dict(catalog.feature_dims) == {"text": 2, "image": 1}
    assert len(published) == 1
    assert published[0] is snapshot

    # Failed validation neither changes precision nor consumes a version.
    updated = catalog.update(item_ids=["d"], item_features=_features((0,)))
    assert updated.version == snapshot.version + 1
    assert updated.item_features["text"].dtype == np.float32


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


def test_none_dtype_rejects_snapshot_and_initial_installation():
    features = _features()
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        MultiModalCandidateCatalog(
            item_ids=["a", "b", "c"], item_features=features, dtype=None
        )
    published = []
    catalog = MutableMultiModalCandidateCatalog(on_publish=published.append)
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        catalog.install(
            source_item_ids=["a", "b", "c"], candidate_features=features, dtype=None
        )
    assert not catalog.is_installed
    assert catalog.source_vocabulary is None
    assert not published

    # Omission keeps the API's explicit float32 default, unlike passing None.
    installed = catalog.install(
        source_item_ids=["a", "b", "c"], candidate_features=features
    )
    assert all(
        matrix.dtype == np.float32 for matrix in installed.item_features.values()
    )


def test_none_dtype_refit_preserves_catalog_source_precision_and_version():
    published = []
    catalog = _installed(on_publish=published.append)
    before = catalog.snapshot()
    source = catalog.source_vocabulary
    with pytest.raises(ValueError, match="dtype must be float32 or float64"):
        catalog.install(
            source_item_ids=["new"],
            candidate_features={"audio": np.ones((1, 4), dtype=np.float64)},
            dtype=None,
        )
    assert catalog.snapshot() is before
    assert catalog.source_vocabulary is source
    assert len(published) == 1
    updated = catalog.update(item_ids=["d"], item_features=_features((0,)))
    assert updated.version == before.version + 1
    assert all(matrix.dtype == np.float32 for matrix in updated.item_features.values())


@pytest.mark.parametrize(
    "dtype",
    [np.float32, np.dtype("float32"), "f4", np.float64, np.dtype("float64"), "f8"],
)
def test_catalog_dtype_aliases_remain_valid_through_checkpoint_roundtrip(
    tmp_path, dtype
):
    catalog = MutableMultiModalCandidateCatalog()
    catalog.install(
        source_item_ids=["a", "b", "c"], candidate_features=_features(), dtype=dtype
    )
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        catalog._save_checkpoint(writer)
    restored = MutableMultiModalCandidateCatalog()
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        snapshot = restored._load_checkpoint(reader)
    for name, matrix in snapshot.item_features.items():
        assert matrix.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(_dense(matrix), _dense(_features()[name]))


@pytest.mark.parametrize("schema_version", [1, 2])
def test_null_checkpoint_dtype_preserves_installed_catalog(tmp_path, schema_version):
    published = []
    catalog = _installed(on_publish=published.append)
    before = catalog.snapshot()
    source = catalog.source_vocabulary
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        catalog._save_checkpoint(writer)
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        state_path = reader._input_path("catalog/state.json")
        state = json.loads(state_path.read_text())
        state.update(version=schema_version, dtype=None)
        if schema_version == 1:
            del state["metadata_columns"]
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="dtype must be float32 or float64"):
            catalog._load_checkpoint(reader)
    assert catalog.snapshot() is before
    assert catalog.source_vocabulary is source
    assert len(published) == 1


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


@pytest.mark.parametrize("candidate_ids", [None, ["cold", "c"]], ids=["all", "subset"])
@pytest.mark.parametrize("mutation", ["update", "reinstall"])
def test_selection_preparation_allows_reads_and_mutations_without_mixing_state(
    monkeypatch, candidate_ids, mutation
):
    catalog = _installed()
    catalog.build(item_ids=["c", "a", "cold"], item_features=_features((2, 0, 1)))
    before = catalog.snapshot()
    paused, resume = Event(), Event()
    selection_thread = None

    def pause_selection():
        paused.set()
        assert resume.wait(timeout=10), "selection was not released"

    if candidate_ids is None:
        # Pause row preparation on the all-candidates path, which shares its
        # feature matrices and therefore never calls take_features.
        n_items = MultiModalCandidateCatalog.n_items.fget

        def paused_n_items(snapshot):
            if snapshot is before and current_thread() is selection_thread:
                pause_selection()
            return n_items(snapshot)

        monkeypatch.setattr(
            MultiModalCandidateCatalog, "n_items", property(paused_n_items)
        )
    else:
        take_features = multimodal.take_features

        def paused_take_features(matrix, rows):
            if (
                matrix is before.item_features["text"]
                and current_thread() is selection_thread
            ):
                pause_selection()
            return take_features(matrix, rows)

        monkeypatch.setattr(multimodal, "take_features", paused_take_features)

    def select():
        nonlocal selection_thread
        selection_thread = current_thread()
        return catalog.resolve_selection(candidate_ids)

    with ThreadPoolExecutor(max_workers=2) as workers:
        selection = workers.submit(select)
        try:
            assert paused.wait(timeout=5), "selection did not reach preparation"
            # Another selection must finish even while the first is paused.
            concurrent = workers.submit(catalog.resolve_selection, None).result(
                timeout=5
            )
            assert concurrent.catalog is before
            if mutation == "update":
                published = workers.submit(
                    catalog.update, item_ids=["new"], item_features=_features((0,))
                ).result(timeout=5)
            else:
                # Change both the source vocabulary and modality schema. The
                # in-flight selection must retain their old counterparts.
                published = workers.submit(
                    catalog.install,
                    source_item_ids=["fresh", "c"],
                    candidate_features={
                        "audio": np.array([[9], [8]], dtype=np.float32)
                    },
                ).result(timeout=5)
            latest = workers.submit(catalog.resolve_selection, None).result(timeout=5)
            assert latest.catalog is published
        finally:
            # Unblock the worker even on failure, so lock regressions fail the
            # test with a bounded timeout instead of hanging the test process.
            resume.set()
        selected = selection.result(timeout=5)

    assert selected.catalog is before
    assert published.version == before.version + 1
    assert selected.source_to_candidate.tolist() == [1, -1, 0]
    expected_rows = [0, 1, 2] if candidate_ids is None else [0, 2]
    assert selected.rows.tolist() == expected_rows
    assert selected.candidate_to_local.tolist() == (
        [0, 1, 2] if candidate_ids is None else [0, -1, 1]
    )
    for name, matrix in before.item_features.items():
        np.testing.assert_array_equal(
            _dense(selected.features[name]), _dense(matrix)[expected_rows]
        )
    assert catalog.snapshot() is published


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


@pytest.mark.parametrize("schema_version", [1, 2])
@pytest.mark.parametrize(
    "corrupt, message",
    [
        (lambda state: state["modalities"][0].pop("storage"), "feature storage"),
        (
            lambda state: state["modalities"][0].update(storage="future-format"),
            "feature storage",
        ),
        (lambda state: state["modalities"].__setitem__(0, None), "modality schema"),
        (lambda state: state.pop("dtype"), "missing.*dtype"),
        (lambda state: state.pop("feature_space_id"), "missing.*feature_space_id"),
        (lambda state: state.pop("catalog_version"), "missing.*catalog_version"),
    ],
)
def test_invalid_catalog_checkpoint_fields_preserve_installed_state(
    tmp_path, schema_version, corrupt, message
):
    source = _installed()
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        source._save_checkpoint(writer)
    published = []
    target = MutableMultiModalCandidateCatalog(on_publish=published.append)
    before = target.install(
        source_item_ids=["old"],
        candidate_features=_features((0,)),
        dtype="float64",
    )
    vocabulary = target.source_vocabulary
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        state_path = reader._input_path("catalog/state.json")
        state = json.loads(state_path.read_text())
        state["version"] = schema_version
        corrupt(state)
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match=message):
            target._load_checkpoint(reader)
    assert target.snapshot() is before
    assert target.source_vocabulary is vocabulary
    assert target._dtype == np.dtype("float64")
    assert len(published) == 1


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


@pytest.mark.parametrize(
    "metadata",
    [
        pytest.param(pd.DataFrame({"label": ["p", "q"], 7: [1, 2]}), id="mixed"),
        pytest.param(pd.DataFrame([[1, 2], [3, 4]]), id="range"),
        pytest.param(
            pd.DataFrame(
                {7: pd.Series([1, 2], dtype="Int32"), "7": [3.5, 4.5]},
            ).rename_axis("fields", axis=1),
            id="colliding-string-representations",
        ),
        pytest.param(
            pd.DataFrame([[1, 2], [3, 4]], columns=pd.RangeIndex(9, 3, -3, name=7)),
            id="named-nondefault-range",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.Index([False, 2.5], dtype=object, name="fields"),
            ),
            id="scalar-label-types",
        ),
        pytest.param(
            pd.concat(
                [
                    pd.Series([1, 2], name=7, dtype="int32"),
                    pd.Series([3.5, 4.5], name=7),
                ],
                axis=1,
            ),
            id="duplicate-labels-with-different-dtypes",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.Index([("rank", 7), None], tupleize_cols=False),
            ),
            id="tuple-and-null-labels",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "item": ["a", "b", "a", "b"],
                    "attr": [None, None, "genre", "genre"],
                    "val": [1, 2, 3, 4],
                }
            )
            .pivot(index="item", columns="attr", values="val")
            .reset_index(),
            id="pivot-with-missing-attribute",
        ),
        pytest.param(
            pd.DataFrame([[1, 2], [3, 4]], index=[np.nan, "genre"]).T,
            id="transpose-with-missing-index",
        ),
        pytest.param(
            pd.DataFrame({None: [1, 2], 7: [3, 4]}),
            id="none-coerced-to-nan",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.Index([None, np.nan], dtype=object, name=np.nan),
            ),
            id="distinct-none-and-nan-with-nan-name",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.Index([("rank", np.nan), None], tupleize_cols=False),
            ),
            id="tuple-with-nan",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.Index([np.float32("nan"), "rank"], dtype=object),
            ),
            id="numpy-nan",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.MultiIndex.from_tuples(
                    [("rank", np.nan), ("rank", 7)], names=[np.nan, "field"]
                ),
            ),
            id="multi-index-with-nan",
        ),
        pytest.param(
            pd.DataFrame(
                [[1, 2], [3, 4]],
                columns=pd.MultiIndex.from_tuples(
                    [("rank", 7), ("rank", "7")], names=["group", "field"]
                ),
            ),
            id="multi-index",
        ),
        pytest.param(
            pd.DataFrame(
                {
                    "item_id": ["a", "b"],
                    7: pd.Series(["p", "q"], dtype="category"),
                    "nullable": pd.Series([1, None], dtype="Int64"),
                }
            ),
            id="item-ids-and-extension-dtypes",
        ),
        pytest.param(pd.DataFrame(index=range(2)), id="zero-columns"),
    ],
)
def test_model_checkpoint_preserves_metadata_labels_order_and_dtypes(
    tmp_path, metadata
):
    original = metadata.copy(deep=True)
    model = _model()
    model.build_candidates(
        item_ids=["a", "b"], item_features=_features((0, 1)), metadata=metadata
    )
    before = model.recommend([["a"]], k=1)
    path = tmp_path / "model.zip"
    model.save(path)
    restored = _MultiModalContent.load(path)

    pd.testing.assert_frame_equal(restored.candidates.snapshot().metadata, original)
    pd.testing.assert_frame_equal(model.candidates.snapshot().metadata, original)
    pd.testing.assert_frame_equal(metadata, original)
    after = restored.recommend([["a"]], k=1)
    np.testing.assert_array_equal(after.item_ids, before.item_ids)
    np.testing.assert_array_equal(after.scores, before.scores)


@pytest.mark.parametrize("label", [np.inf, -np.inf, ("rank", np.inf), 1 + 2j])
def test_unsupported_metadata_label_error_identifies_metadata(tmp_path, label):
    model = _model()
    model.build_candidates(
        item_ids=["a", "b"],
        item_features=_features((0, 1)),
        metadata=pd.DataFrame(
            [[1], [2]], columns=pd.Index([label], tupleize_cols=False)
        ),
    )
    with pytest.raises(ValueError, match="unsupported metadata column/index label"):
        model.save(tmp_path / "model.zip")


@pytest.mark.parametrize("dtypes", [None, [], [None, "int64"], ["str", "invalid"]])
def test_invalid_checkpoint_column_level_dtypes_preserve_catalog(tmp_path, dtypes):
    metadata = pd.DataFrame(
        [[1, 2], [3, 4]],
        columns=pd.MultiIndex.from_tuples([("rank", np.nan), ("rank", 7)]),
    )
    catalog = MutableMultiModalCandidateCatalog()
    before = catalog.install(
        source_item_ids=["a", "b"],
        candidate_features=_features((0, 1)),
        metadata=metadata,
    )
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        catalog._save_checkpoint(writer)
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        state_path = reader._input_path("catalog/state.json")
        state = json.loads(state_path.read_text())
        state["metadata_columns"]["level_dtypes"] = dtypes
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match="invalid checkpoint metadata column"):
            catalog._load_checkpoint(reader)
    assert catalog.snapshot() is before


@pytest.mark.parametrize("with_metadata", [False, True])
def test_legacy_catalog_checkpoint_metadata_still_loads(tmp_path, with_metadata):
    metadata = (
        pd.DataFrame({"label": ["p", "q"], "rank": [1, 2]}) if with_metadata else None
    )
    catalog = MutableMultiModalCandidateCatalog()
    catalog.install(
        source_item_ids=["a", "b"],
        candidate_features=_features((0, 1)),
        metadata=metadata,
    )
    path = tmp_path / "legacy.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        catalog._save_checkpoint(writer)
        # Version 1 stored string labels directly in both Parquet and JSON.
        state = json.loads(writer._path("catalog/state.json").read_text())
        state["version"] = 1
        del state["metadata_columns"]
        if metadata is not None:
            writer.write_dataframe("catalog/metadata.parquet", metadata)
            state["metadata_dtypes"] = {
                name: str(dtype) for name, dtype in metadata.dtypes.items()
            }
        writer.write_json("catalog/state.json", state)
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        restored = MutableMultiModalCandidateCatalog()._load_checkpoint(reader)
    if metadata is None:
        assert restored.metadata is None
    else:
        pd.testing.assert_frame_equal(restored.metadata, metadata)


@pytest.mark.parametrize(
    "corrupt, message",
    [
        pytest.param(
            lambda state: state.pop("metadata_dtypes"),
            "dtype description is missing",
            id="missing-dtypes",
        ),
        pytest.param(
            lambda state: state.update(metadata_dtypes=None),
            "dtype description is missing",
            id="null-dtypes-with-metadata",
        ),
        pytest.param(
            lambda state: state.update(metadata_dtypes=[]),
            "invalid.*dtype description",
            id="invalid-dtype-schema",
        ),
        pytest.param(
            lambda state: state["metadata_dtypes"].update(column_0=42),
            "invalid.*dtype description",
            id="non-string-dtype",
        ),
        pytest.param(
            lambda state: state["metadata_dtypes"].update(column_0="not-a-dtype"),
            "cannot restore.*dtypes",
            id="unknown-dtype",
        ),
        pytest.param(
            lambda state: state["metadata_dtypes"].pop("column_0"),
            "columns do not match",
            id="missing-storage-column",
        ),
        pytest.param(
            lambda state: state.pop("metadata_columns"),
            "invalid.*column description",
            id="missing-column-description",
        ),
        pytest.param(
            lambda state: state["metadata_columns"]["labels"].pop(),
            "invalid.*column description",
            id="wrong-label-count",
        ),
        pytest.param(
            lambda state: state["metadata_columns"]["labels"][0].update(type="invalid"),
            "invalid.*column description",
            id="invalid-label-type",
        ),
        pytest.param(
            lambda state: state["metadata_columns"].update(kind="invalid"),
            "invalid.*column description",
            id="invalid-index-kind",
        ),
    ],
)
def test_invalid_metadata_checkpoint_preserves_installed_catalog(
    tmp_path, corrupt, message
):
    source = MutableMultiModalCandidateCatalog()
    source.install(
        source_item_ids=["x", "y"],
        candidate_features=_features((0, 1)),
        metadata=pd.DataFrame({7: [1, 2], "7": [3.5, 4.5]}),
        dtype="float64",
    )
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        source._save_checkpoint(writer)

    published = []
    target = _installed(on_publish=published.append)
    before = target.snapshot()
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        state_path = reader._input_path("catalog/state.json")
        state = json.loads(state_path.read_text())
        corrupt(state)
        state_path.write_text(json.dumps(state))
        with pytest.raises(ValueError, match=message):
            target._load_checkpoint(reader)
    assert target.snapshot() is before
    assert target.source_item_ids.tolist() == ["a", "b", "c"]
    assert len(published) == 1


def test_absent_metadata_still_requires_dtype_description(tmp_path):
    source = _installed()
    path = tmp_path / "catalog.zip"
    with ModelCheckpointWriter(path, model_type="test_catalog") as writer:
        source._save_checkpoint(writer)
    with ModelCheckpointReader(path, expected_model_type="test_catalog") as reader:
        state_path = reader._input_path("catalog/state.json")
        state = json.loads(state_path.read_text())
        del state["metadata_dtypes"]
        state_path.write_text(json.dumps(state))
        target = MutableMultiModalCandidateCatalog()
        with pytest.raises(ValueError, match="metadata dtype description is missing"):
            target._load_checkpoint(reader)
        assert not target.is_installed
