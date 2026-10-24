from __future__ import annotations

import json
import pickle
import zipfile
from copy import copy, deepcopy
from dataclasses import asdict, replace
from functools import lru_cache

import numpy as np
import pytest
import torch
from compresso import SRPTensor
from scipy.sparse import csr_matrix, isspmatrix_csr

from compresso_recsys.models import (
    TEASER,
    BaseMultiModalRecommender,
    ContentRecommender,
    ContentRecommenderConfig,
    MMConcatWrapper,
    MMConcatWrapperConfig,
    TEASERConfig,
    TEASERGDConfig,
    TEASERGDTrainer,
)
from compresso_recsys.models.mm_concat.model import _impute_csr_rows


@pytest.fixture
def data():
    features = {
        "image": np.array([[2, 1], [0, 1], [1, 3], [1, 0]], dtype=np.float32),
        "text": np.array([[1, 0], [1, 1], [0, 1], [2, 1]], dtype=np.float32),
    }
    interactions = csr_matrix([[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0]])
    return features, interactions


def content(**kwargs):
    return MMConcatWrapper(
        MMConcatWrapperConfig(
            model=ContentRecommender,
            model_config=ContentRecommenderConfig(normalize=False),
            modalities=["text", "image"],
            **kwargs,
        )
    )


def assert_predictions(left, right):
    torch.testing.assert_close(left.cols, right.cols)
    torch.testing.assert_close(left.vals, right.vals)


@pytest.mark.parametrize("keyword", [False, True])
@pytest.mark.parametrize(
    "model_class,config",
    [
        (ContentRecommender, ContentRecommenderConfig()),
        (TEASER, TEASERConfig(max_iterations=3)),
        (
            TEASERGDTrainer,
            TEASERGDConfig(
                epochs=1, batch_size=3, max_output=3, seed=4, show_progress=False
            ),
        ),
    ],
)
def test_equivalent_to_manual_concatenation(data, keyword, model_class, config):
    features, x = data
    cfg = MMConcatWrapperConfig(
        model=model_class, model_config=config, modalities=["text", "image"]
    )
    wrapper = MMConcatWrapper(cfg)
    matrix = np.concatenate([features["text"], features["image"]], axis=1)
    args = () if model_class is ContentRecommender else (x,)
    fit_kwargs = (
        {} if model_class is ContentRecommender else {"train_item_indices": [0, 1, 2]}
    )
    manual = model_class(config).fit(*args, matrix, **fit_kwargs)
    if keyword:
        returned = wrapper.fit(*args, item_features=features, **fit_kwargs)
    else:
        returned = wrapper.fit(*args, features, **fit_kwargs)
    assert returned is wrapper
    assert isinstance(wrapper, BaseMultiModalRecommender)
    assert wrapper.candidates is wrapper.inner_model.candidates
    np.testing.assert_array_equal(wrapper.transform_features(features), matrix)
    assert_predictions(
        wrapper.predict_on_batch(x, k=2), manual.predict_on_batch(x, k=2)
    )
    assert_predictions(
        wrapper.predict(x, k=2, batch_size=1), manual.predict(x, k=2, batch_size=1)
    )
    np.testing.assert_array_equal(
        wrapper.recommend([[0]], k=2).item_ids, manual.recommend([[0]], k=2).item_ids
    )


class RenamedContent(ContentRecommender):
    def fit(self, vectors, /, *, sentinel="default", item_ids=None):
        self.sentinel = sentinel
        return super().fit(vectors, item_ids=item_ids)


class OpaqueContent(ContentRecommender):
    def fit(self, *args, **kwargs):
        return super().fit(kwargs.pop("vectors"), **kwargs)


@pytest.mark.parametrize("model_class", [RenamedContent, OpaqueContent])
def test_configurable_parameter_and_opaque_kwargs(data, model_class):
    features, _ = data
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=model_class,
            modalities=["image", "text"],
            fit_features_parameter="vectors",
        )
    )
    if model_class is RenamedContent:
        model.fit(features, sentinel="forwarded")
        assert model.inner_model.sentinel == "forwarded"
    else:
        model.fit(vectors=features)
        with pytest.raises(TypeError, match="supply features"):
            model.fit(features)
    np.testing.assert_array_equal(
        model.transform_features(features),
        np.concatenate([features["image"], features["text"]], axis=1),
    )


def test_invalid_call_and_failed_refit_preserve_fitted_model(data):
    features, _ = data
    model = content().fit(features)
    inner = model.inner_model
    with pytest.raises(TypeError):
        model.fit(features, item_features=features)
    with pytest.raises(ValueError):
        model.fit(features, item_ids=["wrong"])
    assert model.inner_model is inner
    assert model.is_fitted


class BrokenFit(ContentRecommender):
    calls = 0

    def fit(self, item_features):
        type(self).calls += 1
        raise TypeError("failure inside fit")


def test_inner_typeerror_is_not_retried(data):
    model = MMConcatWrapper(MMConcatWrapperConfig(model=BrokenFit, modalities=["text"]))
    with pytest.raises(TypeError, match="failure inside fit"):
        model.fit({"text": data[0]["text"]})
    assert BrokenFit.calls == 1
    assert not model.is_fitted


@pytest.mark.parametrize(
    "conversion",
    [
        np.asarray,
        csr_matrix,
        torch.tensor,
        lambda x: torch.tensor(x).to_sparse(),
        lambda x: SRPTensor.from_dense(torch.tensor(x), k=2, score_mode="raw"),
    ],
)
def test_matrix_types_and_preprocessing_are_nonmutating(data, conversion):
    features, _ = data
    converted = {n: conversion(v) for n, v in features.items()}
    model = content(normalize=True, weights={"image": 2}, missing="zero")
    mask = np.array([True, False, True, True])
    model.fit(converted, modality_masks={"text": mask})
    actual = model.transform_features(converted, modality_masks={"text": mask})
    actual = actual.toarray() if isspmatrix_csr(actual) else actual
    expected = []
    for name in ["text", "image"]:
        block = features[name].copy()
        if name == "text":
            block[~mask] = 0
        block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        expected.append(block * (2 if name == "image" else 1))
    np.testing.assert_allclose(actual, np.concatenate(expected, axis=1), rtol=1e-6)
    assert features["text"][1, 0] == 1


@pytest.mark.parametrize("storage", ["dense", "sparse", "mixed"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("missing", ["error", "zero", "mean"])
def test_readonly_features_remain_independent_of_transformed_output(
    data, storage, dtype, missing
):
    features, _ = data
    converted = {}
    input_buffers = []
    for name, matrix in features.items():
        matrix = matrix.astype(dtype)
        if storage == "sparse" or (storage == "mixed" and name == "text"):
            matrix = csr_matrix(matrix)
            buffers = (matrix.data, matrix.indices, matrix.indptr)
        else:
            buffers = (matrix,)
        for array in buffers:
            array.setflags(write=False)
        input_buffers.extend(buffers)
        converted[name] = matrix
    masks = (
        None if missing == "error" else {"text": np.array([True, False, True, True])}
    )

    # Exercise both the path reusing blocks and preprocessing that allocates them.
    for preprocessing in ({}, {"normalize": True, "weights": {"text": 0, "image": 2}}):
        model = content(dtype=dtype, missing=missing, **preprocessing).fit(
            converted, modality_masks=masks
        )
        actual = model.transform_features(converted, modality_masks=masks)
        assert actual.dtype == np.dtype(dtype)
        output_buffers = (
            (actual.data, actual.indices, actual.indptr)
            if isspmatrix_csr(actual)
            else (actual,)
        )
        for output in output_buffers:
            assert not any(np.shares_memory(output, source) for source in input_buffers)
        # Callers may modify transformed values without affecting their inputs.
        output_buffers[0][:] = -99
        for name, matrix in converted.items():
            values = matrix.toarray() if isspmatrix_csr(matrix) else matrix
            np.testing.assert_array_equal(values, features[name])
        assert all(not array.flags.writeable for array in input_buffers)


@pytest.mark.parametrize("sparse", [False, True])
def test_mean_uses_only_available_training_rows_and_is_frozen(data, sparse):
    features, _ = data
    features["text"][3] = [10000, 10000]  # held-out item must not influence mean
    if sparse:
        features = {n: csr_matrix(v) for n, v in features.items()}
    masks = {"text": np.array([True, True, False, True])}
    model = content(missing="mean").fit(
        features, modality_masks=masks, feature_fit_indices=[0, 1, 2]
    )
    expected_mean = [1, 0.5]
    model.update_candidates(item_ids=["new"], item_features={"image": np.ones((1, 2))})
    snapshot = model.candidates.snapshot().item_features
    snapshot = snapshot.toarray() if isspmatrix_csr(snapshot) else snapshot
    np.testing.assert_allclose(snapshot[-1, :2], expected_mean)
    np.testing.assert_allclose(snapshot[2, :2], expected_mean)
    if sparse:
        assert isspmatrix_csr(model.transform_features(features, modality_masks=masks))


@pytest.mark.parametrize("sparse", [False, True])
def test_imputed_mean_is_normalized_before_weighting(data, sparse):
    features, _ = data
    features["text"] = np.array([[3, 4], [6, 8], [1, 1], [300, 400]], dtype=np.float32)
    if sparse:
        features = {name: csr_matrix(block) for name, block in features.items()}
    masks = {"text": np.array([True, True, True, False])}
    model = content(missing="mean", normalize=True, weights={"text": 2}).fit(
        features, modality_masks=masks
    )
    # Mean [10/3, 13/3] must become a unit vector, then receive weight 2.
    expected = np.array([20, 26]) / np.sqrt(269)
    transformed = model.transform_features(features, modality_masks=masks)
    transformed = transformed.toarray() if sparse else transformed
    np.testing.assert_allclose(transformed[3, :2], expected, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(transformed[:, :2], axis=1), 2, rtol=1e-6)

    # A newly added cold item with no text must use that same normalized mean.
    model.update_candidates(
        item_ids=["cold"], item_features={"image": features["image"][:1]}
    )
    stored = model.candidates.snapshot().item_features
    stored = stored.toarray() if sparse else stored
    np.testing.assert_allclose(
        stored[[3, -1], :2], np.tile(expected, (2, 1)), rtol=1e-6
    )


@pytest.mark.parametrize("sparse", [False, True])
def test_preprocessing_rejects_overflow_from_finite_weighted_features(data, sparse):
    features, _ = data
    if sparse:
        features = {name: csr_matrix(block) for name, block in features.items()}
    model = content(weights={"text": 1e10}).fit(features)
    large = np.full((4, 2), 1e30, dtype=np.float32)
    assert np.isfinite(large).all()
    features["text"] = csr_matrix(large) if sparse else large
    with (
        np.errstate(over="ignore"),
        pytest.raises(ValueError, match="preprocessed modality 'text' must be finite"),
    ):
        model.transform_features(features)


def test_zero_imputation_removes_explicit_csr_zeros(data):
    features, _ = data
    features = {name: csr_matrix(block) for name, block in features.items()}
    model = content(missing="zero").fit(features)
    masks = {name: np.array([True, False, True, True]) for name in features}
    transformed = model.transform_features(features, modality_masks=masks)
    expected = np.concatenate(
        [features[name].toarray() for name in ("text", "image")], axis=1
    )
    expected[1] = 0
    np.testing.assert_array_equal(transformed.toarray(), expected)
    assert transformed[1].nnz == 0
    assert transformed.nnz == np.count_nonzero(expected)
    assert np.all(transformed.data != 0)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("mean", [[0, 0, 0, 0], [0, 2, -1, 0], [1, 2, -1, 3]])
@pytest.mark.parametrize(
    "missing",
    [
        [False, False, False, False, False, False],
        [True, True, True, True, True, True],
        [True, False, True, False, True, False],
        [False, True, True, False, False, True],
    ],
)
def test_csr_mean_rows_match_dense_replacement_without_mutation(dtype, mean, missing):
    dense = np.array(
        [
            [1, 0, 2, 0],
            [0, 0, 0, 0],
            [-2, 1, 0, 3],
            [0, 4, 0, 0],
            [5, 0, 0, -1],
            [0, 0, 0, 0],
        ],
        dtype=dtype,
    )
    matrix = csr_matrix(dense)
    mean = np.asarray(mean, dtype=dtype)
    missing = np.asarray(missing, dtype=bool)
    for array in (matrix.data, matrix.indices, matrix.indptr, mean, missing):
        array.setflags(write=False)

    actual = _impute_csr_rows(matrix, missing, mean)
    expected = dense.copy()
    expected[missing] = mean

    assert isinstance(actual, csr_matrix)
    assert actual.dtype == np.dtype(dtype)
    assert actual.has_canonical_format
    assert actual.nnz == np.count_nonzero(expected)
    np.testing.assert_array_equal(actual.toarray(), expected)
    np.testing.assert_array_equal(matrix.toarray(), dense)


def test_sparse_mean_imputation_keeps_a_wide_sparse_mean_compact():
    # A repeated dense scan would touch nearly a billion cells. Only two mean
    # entries are nonzero, so the actual output needs fewer than 10,000 values.
    n_rows, n_features = 4096, 250_000
    matrix = csr_matrix(
        (np.ones(n_rows, dtype=np.float32), (np.arange(n_rows), np.full(n_rows, 17))),
        shape=(n_rows, n_features),
    )
    mean = np.zeros(n_features, dtype=np.float32)
    mean[[5, n_features - 1]] = [2, -3]
    missing = np.ones(n_rows, dtype=bool)
    missing[::13] = False

    actual = _impute_csr_rows(matrix, missing, mean)

    assert actual.dtype == np.float32
    assert actual.has_canonical_format
    assert actual.nnz == 2 * missing.sum() + (~missing).sum()
    assert actual.data.nbytes == actual.nnz * np.dtype("float32").itemsize
    np.testing.assert_array_equal(actual[:, 17].toarray().ravel(), ~missing)
    np.testing.assert_array_equal(actual[:, 5].toarray().ravel(), 2 * missing)
    np.testing.assert_array_equal(
        actual[:, n_features - 1].toarray().ravel(), -3 * missing
    )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("normalize", [False, True])
def test_sparse_mean_wrapper_matches_dense_fit_and_candidate_updates(
    data, dtype, normalize
):
    features, _ = data
    features = {n: v.astype(dtype) for n, v in features.items()}
    masks = {
        "text": np.array([True, False, True, True]),
        "image": np.array([True, True, False, True]),
    }
    config = MMConcatWrapperConfig(
        model=ContentRecommender,
        model_config=ContentRecommenderConfig(normalize=False, dtype=dtype),
        modalities=["text", "image"],
        missing="mean",
        dtype=dtype,
        normalize=normalize,
        weights={"image": 0.5},
    )
    dense = MMConcatWrapper(config).fit(features, modality_masks=masks)
    sparse = MMConcatWrapper(config).fit(
        {n: csr_matrix(v) for n, v in features.items()}, modality_masks=masks
    )
    expected = dense.transform_features(features, modality_masks=masks)
    actual = sparse.transform_features(
        {n: csr_matrix(v) for n, v in features.items()}, modality_masks=masks
    )
    assert isinstance(actual, csr_matrix)
    assert actual.dtype == np.dtype(dtype)
    np.testing.assert_allclose(actual.toarray(), expected)

    for model, convert in ((dense, np.asarray), (sparse, csr_matrix)):
        model.update_candidates(
            item_ids=["new-a", "new-b"],
            item_features={"image": convert(features["image"][:2])},
            modality_masks={"image": np.array([True, False])},
        )
    stored = sparse.candidates.snapshot().item_features
    assert isinstance(stored, csr_matrix)
    assert stored.dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        stored.toarray(), dense.candidates.snapshot().item_features
    )
    for histories in ([[0]], [[1, 2]]):
        np.testing.assert_allclose(
            sparse.recommend(histories, k=2, allowlist=["new-a", "new-b"]).scores,
            dense.recommend(histories, k=2, allowlist=["new-a", "new-b"]).scores,
        )


def test_forwarded_train_indices_determine_mean(data):
    features, x = data
    features["text"][3] = [900, 900]
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=TEASER,
            model_config=TEASERConfig(max_iterations=1),
            modalities=["text", "image"],
            missing="mean",
        )
    )
    model.fit(x, features, train_item_indices=[0, 1, 2])
    result = model.transform_features({"image": np.ones((1, 2))})
    assert isinstance(result, np.ndarray)
    np.testing.assert_allclose(result[0, :2], [2 / 3, 2 / 3])


def test_missing_policy_requires_explicit_masks_and_known_fit_widths(data):
    features, _ = data
    features["text"][0] = 0
    assert content().fit(features).is_fitted  # zero is a legitimate observed row
    with pytest.raises(ValueError, match="unavailable"):
        content().fit(features, modality_masks={"image": np.zeros(4, dtype=bool)})
    with pytest.raises(ValueError, match="no available training"):
        content(missing="mean").fit(
            features, modality_masks={"image": np.zeros(4, dtype=bool)}
        )
    with pytest.raises(ValueError, match="establish its width"):
        content(missing="zero").fit({"text": features["text"]})
    model = content(missing="zero").fit(
        features, modality_masks={"image": np.zeros(4, dtype=bool)}
    )
    model.update_candidates(item_ids=["missing-both"], item_features={})
    np.testing.assert_array_equal(
        model.candidates.snapshot().item_features[-1], np.zeros(4)
    )


@pytest.mark.parametrize("missing", ["zero", "mean"])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("operation", ["append", "replace", "build"])
def test_omitted_modality_preserves_candidate_storage(data, missing, sparse, operation):
    features, _ = data
    convert = csr_matrix if sparse else np.asarray
    model = content(missing=missing).fit({n: convert(v) for n, v in features.items()})
    original = np.concatenate([features["text"], features["image"]], axis=1)
    imputed = features["text"].mean(axis=0) if missing == "mean" else np.zeros(2)
    expected_row = np.concatenate([imputed, [9, 8]]).astype(np.float32)
    incoming = {"image": convert(np.array([[9, 8]], dtype=np.float32))}

    if operation == "append":
        model.update_candidates(item_ids=["new"], item_features=incoming)
        expected = np.vstack([original, expected_row])
    elif operation == "replace":
        model.update_candidates(
            item_ids=[0], item_features=incoming, on_conflict="replace"
        )
        expected = original.copy()
        expected[0] = expected_row
    else:
        model.build_candidates(item_ids=["new"], item_features=incoming)
        expected = expected_row[None, :]

    stored = model.candidates.snapshot().item_features
    assert isinstance(stored, csr_matrix if sparse else np.ndarray)
    assert stored.dtype == np.float32
    np.testing.assert_allclose(stored.toarray() if sparse else stored, expected)


@pytest.mark.parametrize("fitted_sparse", [False, True])
@pytest.mark.parametrize("supplied_sparse", [False, True])
def test_partial_omission_uses_supplied_format(data, fitted_sparse, supplied_sparse):
    features, _ = data
    convert_fit = csr_matrix if fitted_sparse else np.asarray
    convert_input = csr_matrix if supplied_sparse else np.asarray
    model = content(missing="zero").fit(
        {n: convert_fit(v) for n, v in features.items()}
    )
    incoming = {"image": convert_input(features["image"][:1])}
    actual = model.transform_features(incoming)
    assert isinstance(actual, csr_matrix if supplied_sparse else np.ndarray)
    np.testing.assert_array_equal(
        actual.toarray() if supplied_sparse else actual,
        np.concatenate([np.zeros((1, 2)), features["image"][:1]], axis=1),
    )


@pytest.mark.parametrize("missing", ["zero", "mean"])
@pytest.mark.parametrize("fitted_sparse", [False, True])
@pytest.mark.parametrize("reload", [False, True])
def test_all_omitted_modalities_reuse_fitted_format(
    data, tmp_path, missing, fitted_sparse, reload
):
    features, _ = data
    convert_fit = csr_matrix if fitted_sparse else np.asarray
    model = content(missing=missing).fit(
        {n: convert_fit(v) for n, v in features.items()}
    )
    # Explicit caller input may change the live catalog's format. The fallback
    # for entirely omitted features still belongs to fitting, not that catalog.
    convert_update = np.asarray if fitted_sparse else csr_matrix
    model.build_candidates(
        item_ids=["changed"],
        item_features={n: convert_update(v[:1]) for n, v in features.items()},
    )
    assert (
        isspmatrix_csr(model.candidates.snapshot().item_features) is not fitted_sparse
    )
    if reload:
        path = tmp_path / "wrapper.zip"
        model.save(path)
        model = MMConcatWrapper.load(path)
    model.build_candidates(item_ids=["missing"], item_features={})
    actual = model.candidates.snapshot().item_features
    assert isinstance(actual, csr_matrix if fitted_sparse else np.ndarray)
    expected = (
        np.concatenate([features["text"].mean(axis=0), features["image"].mean(axis=0)])[
            None, :
        ]
        if missing == "mean"
        else np.zeros((1, 4))
    )
    np.testing.assert_allclose(actual.toarray() if fitted_sparse else actual, expected)


@pytest.mark.parametrize("sparse", [False, True])
def test_legacy_checkpoint_without_fitted_format_uses_catalog(data, tmp_path, sparse):
    convert = csr_matrix if sparse else np.asarray
    model = content(missing="zero").fit({n: convert(v) for n, v in data[0].items()})
    # ContentRecommender.fit stores dense features itself. Install the desired
    # catalog format explicitly to exercise both legacy storage fallbacks.
    model.build_candidates(
        item_ids=np.arange(4),
        item_features={n: convert(v) for n, v in data[0].items()},
    )
    path = tmp_path / "legacy.zip"
    model.save(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    preprocessing = json.loads(members["preprocessing.json"])
    del preprocessing["fit_sparse_output"]
    members["preprocessing.json"] = json.dumps(preprocessing).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    loaded = MMConcatWrapper.load(path)
    loaded.build_candidates(item_ids=["missing"], item_features={})
    actual = loaded.candidates.snapshot().item_features
    assert isinstance(actual, csr_matrix if sparse else np.ndarray)
    np.testing.assert_array_equal(
        actual.toarray() if sparse else actual, np.zeros((1, 4))
    )


def test_failed_refit_preserves_fitted_format(data):
    model = content(missing="zero").fit(data[0])
    with pytest.raises(ValueError, match="item_ids"):
        model.fit({n: csr_matrix(v) for n, v in data[0].items()}, item_ids=["wrong"])
    model.build_candidates(item_ids=["missing"], item_features={})
    assert isinstance(model.candidates.snapshot().item_features, np.ndarray)


@pytest.mark.parametrize(
    "bad",
    [
        {"text": np.ones((3, 2)), "image": np.ones((4, 2))},
        {"text": np.ones((4, 3)), "image": np.ones((4, 2))},
        {"text": np.full((4, 2), np.nan), "image": np.ones((4, 2))},
        {"image": np.ones((4, 2))},
    ],
)
def test_invalid_candidate_update_preserves_snapshot(data, bad):
    model = content().fit(data[0])
    snapshot = model.candidates.snapshot()
    with pytest.raises(ValueError):
        model.update_candidates(item_ids=[4, 5, 6, 7], item_features=bad)
    assert model.candidates.snapshot() is snapshot


@pytest.mark.parametrize(
    "mask", [[1, 0, 1, 1], [True, False], np.ones((4, 1), dtype=bool)]
)
def test_invalid_masks(data, mask):
    with pytest.raises(ValueError, match="boolean vector"):
        content().fit(data[0], modality_masks={"text": mask})


@pytest.mark.parametrize("missing", ["error", "zero", "mean"])
@pytest.mark.parametrize("mapping", ["item_features", "modality_masks"])
@pytest.mark.parametrize(
    "method", ["fit", "transform_features", "build_candidates", "update_candidates"]
)
def test_unknown_modality_keys_reject_call_without_changing_model(
    data, missing, mapping, method
):
    features, _ = data
    model = content(missing=missing).fit(features)
    inner = model.inner_model
    snapshot = model.candidates.snapshot()
    means = {n: v.copy() for n, v in model._means.items()}
    incoming = dict(features)
    masks = None
    if mapping == "item_features":
        incoming["Text"] = incoming.pop("text")
    else:
        masks = {"Text": np.array([True, False, True, True])}
    kwargs = {"item_features": incoming, "modality_masks": masks}
    if method in {"build_candidates", "update_candidates"}:
        kwargs["item_ids"] = np.arange(4)
    if method == "update_candidates":
        kwargs["on_conflict"] = "replace"

    with pytest.raises(
        ValueError, match=f"{mapping} contains unknown modality keys: 'Text'"
    ):
        getattr(model, method)(**kwargs)

    assert model.inner_model is inner
    assert model.candidates.snapshot() is snapshot
    for name, mean in means.items():
        np.testing.assert_array_equal(model._means[name], mean)


@pytest.mark.parametrize("mapping", ["item_features", "modality_masks"])
def test_unknown_keys_reject_initial_fit_even_when_all_features_present(data, mapping):
    model = content(missing="mean")
    features = dict(data[0])
    masks = {}
    inputs = features if mapping == "item_features" else masks
    inputs["txt"] = np.ones(4, dtype=bool)
    inputs[7] = np.ones(4, dtype=bool)
    with pytest.raises(
        ValueError, match=f"{mapping} contains unknown modality keys"
    ) as error:
        model.fit(features, modality_masks=masks)
    assert "'txt'" in str(error.value)
    assert "7" in str(error.value)
    assert not model.is_fitted
    assert not model._means


@pytest.mark.parametrize("extra_modalities", ["error", "ignore"])
@pytest.mark.parametrize(
    "method", ["transform_features", "build_candidates", "update_candidates"]
)
def test_valid_candidate_mask_uses_fitted_mean(data, extra_modalities, method):
    features, _ = data
    mask = np.array([True, False, True, True])
    model = content(missing="mean", extra_modalities=extra_modalities).fit(
        features, modality_masks={"text": mask}
    )
    incoming = {name: values[:2].copy() for name, values in features.items()}
    incoming["text"][0] = [999, 999]
    kwargs = {
        "item_features": incoming,
        "modality_masks": {"text": np.array([False, True])},
    }
    if method != "transform_features":
        kwargs["item_ids"] = ["new-a", "new-b"]
        getattr(model, method)(**kwargs)
        actual = model.candidates.snapshot().item_features[-2:]
    else:
        actual = model.transform_features(**kwargs)
    expected = np.concatenate([features["text"][:2], features["image"][:2]], axis=1)
    expected[0, :2] = features["text"][mask].mean(axis=0)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(incoming["text"][0], [999, 999])


@pytest.mark.parametrize(
    "method", ["transform_features", "build_candidates", "update_candidates"]
)
def test_ignore_policy_selects_features_and_masks_from_shared_dictionaries(
    data, method
):
    features, _ = data
    mask = np.array([True, False, True, True])
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=ContentRecommender,
            model_config=ContentRecommenderConfig(normalize=False),
            modalities=["text"],
            missing="mean",
            extra_modalities="ignore",
        )
    ).fit(features, modality_masks={"text": mask, "image": np.zeros(4, dtype=bool)})
    expected = features["text"].copy()
    expected[~mask] = features["text"][mask].mean(axis=0)
    np.testing.assert_allclose(model.candidates.snapshot().item_features, expected)
    kwargs = {
        "item_features": features,
        "modality_masks": {"text": mask, "image": np.zeros(4, dtype=bool)},
    }
    if method != "transform_features":
        kwargs["item_ids"] = ["a", "b", "c", "d"]
        getattr(model, method)(**kwargs)
        actual = model.candidates.snapshot().item_features[-4:]
    else:
        actual = model.transform_features(**kwargs)
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize("policy", ["error", "ignore"])
@pytest.mark.parametrize("mapping", ["item_features", "modality_masks"])
def test_extra_modality_policy_survives_checkpoint(data, tmp_path, policy, mapping):
    model = content(missing="mean", extra_modalities=policy).fit(data[0])
    path = tmp_path / "wrapper.zip"
    model.save(path)
    loaded = MMConcatWrapper.load(path)
    assert loaded.cfg.extra_modalities == policy
    incoming = dict(data[0])
    masks = {}
    if mapping == "item_features":
        incoming["unused"] = incoming["text"]
    else:
        masks["unused"] = np.zeros(4, dtype=bool)
    if policy == "error":
        with pytest.raises(
            ValueError, match=f"{mapping} contains unknown modality keys"
        ):
            loaded.transform_features(incoming, modality_masks=masks)
    else:
        np.testing.assert_array_equal(
            loaded.transform_features(incoming, modality_masks=masks),
            np.concatenate([data[0]["text"], data[0]["image"]], axis=1),
        )


def test_checkpoint_without_extra_modality_policy_loads_with_strict_default(
    data, tmp_path
):
    model = content().fit(data[0])
    path = tmp_path / "legacy.zip"
    model.save(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    config = json.loads(members["config.json"])
    del config["extra_modalities"]
    members["config.json"] = json.dumps(config).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    loaded = MMConcatWrapper.load(path)
    assert loaded.cfg.extra_modalities == "error"
    with pytest.raises(
        ValueError, match="modality_masks contains unknown modality keys"
    ):
        loaded.transform_features(
            data[0], modality_masks={"Text": np.ones(4, dtype=bool)}
        )


def test_candidate_lifecycle_does_not_change_history_vocabulary(data):
    features, x = data
    model = content().fit(features, item_ids=["a", "b", "c", "d"])
    new = {n: matrix[:1] for n, matrix in features.items()}
    model.update_candidates(item_ids=["new"], item_features=new)
    assert model.recommend([["a"]], k=1, allowlist=["new"]).item_ids[0, 0] == "new"
    with pytest.raises(ValueError, match="unknown"):
        model.recommend([["new"]], k=1)
    model.remove_candidates(["a"])
    assert "a" not in model.recommend([["a"]], k=4).item_ids
    np.testing.assert_array_equal(
        model.align_source(x, item_ids=["a", "b", "c", "d"]).toarray(), x.toarray()
    )
    model.build_candidates(item_ids=["only"], item_features=new)
    assert model.recommend([["a"]], k=1).item_ids[0, 0] == "only"


@pytest.mark.parametrize(
    "model_class,config",
    [
        (ContentRecommender, ContentRecommenderConfig()),
        (TEASER, TEASERConfig(max_iterations=2, include_popularity=True)),
        (
            TEASERGDTrainer,
            TEASERGDConfig(
                epochs=1, batch_size=3, max_output=3, seed=4, show_progress=False
            ),
        ),
    ],
)
def test_save_load_and_continue_candidate_updates(data, tmp_path, model_class, config):
    features, x = data
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=model_class,
            model_config=config,
            modalities=["text", "image"],
            missing="mean",
            normalize=True,
        )
    )
    args = () if model_class is ContentRecommender else (x,)
    model.fit(*args, features, feature_fit_indices=[0, 1, 2])
    model.remove_candidates([3])
    path = tmp_path / "model.zip"
    optimizer = model_class is TEASERGDTrainer
    model.save(path, include_optimizer=optimizer)
    loaded = MMConcatWrapper.load(path, load_optimizer=optimizer)
    if model_class is TEASER:
        assert not hasattr(loaded, "device")
        with pytest.raises(TypeError, match="no device-backed state"):
            loaded.to("cpu")
    else:
        assert loaded.device == loaded.inner_model.device == torch.device("cpu")
        assert loaded.to("cpu") is loaded
    assert_predictions(loaded.predict(x, k=1), model.predict(x, k=1))
    for instance in (model, loaded):
        instance.update_candidates(
            item_ids=["new"], item_features={"image": np.ones((1, 2))}
        )
    assert_predictions(loaded.predict(x, k=2), model.predict(x, k=2))
    assert dict(loaded.feature_dims_) == {"text": 2, "image": 2}


def test_device_move_keeps_refits_on_the_selected_device(data):
    # An unfitted meta model allocates no accelerator tensors; moving it to CPU
    # lets this test check a genuine device change on CPU-only test runners.
    supplied_config = ContentRecommenderConfig(device="meta", normalize=False)
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=ContentRecommender,
            model_config=supplied_config,
            modalities=["text", "image"],
        )
    )
    assert model.device == torch.device("meta")
    assert model.to("cpu") is model
    assert model.device == model.inner_model.device == torch.device("cpu")
    assert model.cfg.model_config.device == "cpu"
    assert supplied_config.device == "meta"

    model.fit(data[0])
    assert model.device == torch.device("cpu")
    assert model.inner_model.source_features_.device == torch.device("cpu")
    before = model.predict(data[1], k=1)
    assert model.to(torch.device("cpu")) is model
    assert_predictions(model.predict(data[1], k=1), before)


class ReplacingContent(ContentRecommender):
    def to(self, device):
        moved = type(self)(replace(self.cfg, device=str(device)))
        if self.is_fitted:
            moved.fit(
                self.source_features_.cpu().numpy(), item_ids=self.source_item_ids
            )
        return moved


def test_device_move_keeps_replacement_inner_model(data):
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=ReplacingContent,
            model_config=ContentRecommenderConfig(normalize=False),
            modalities=["text", "image"],
        )
    ).fit(data[0])
    previous = model.inner_model
    before = model.predict(data[1], k=1)

    assert model.to("cpu") is model
    assert model.inner_model is not previous
    assert model.candidates is model.inner_model.candidates
    assert model.candidates is not previous.candidates
    assert model.device == model.inner_model.device == torch.device("cpu")
    assert model.cfg.model_config is model.inner_model.cfg
    assert model.is_fitted
    assert_predictions(model.predict(data[1], k=1), before)


def test_device_move_propagates_failure_without_replacing_inner_model(
    data, monkeypatch
):
    model = content().fit(data[0])
    previous, config = model.inner_model, model.cfg

    def fail(device):
        raise RuntimeError("device move failed")

    monkeypatch.setattr(previous, "to", fail)
    with pytest.raises(RuntimeError, match="device move failed"):
        model.to("cpu")
    assert model.inner_model is previous
    assert model.cfg is config


def test_custom_checkpoint_classes_require_local_registration(
    data, tmp_path, monkeypatch
):
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=OpaqueContent, modalities=["text"], fit_features_parameter="vectors"
        )
    ).fit(vectors={"text": data[0]["text"]})
    path = tmp_path / "model.zip"
    model.save(path)
    model_id = f"{OpaqueContent.__module__}.{OpaqueContent.__qualname__}"
    with monkeypatch.context() as patch:
        patch.delitem(MMConcatWrapper._registered_models, model_id)
        with pytest.raises(ValueError, match="unregistered"):
            MMConcatWrapper.load(path)
    assert MMConcatWrapper.load(path).is_fitted


@pytest.mark.parametrize(
    "filename, corrupt, message",
    [
        ("config.json", lambda state: state.pop("model"), "checkpoint model"),
        ("config.json", lambda state: state.update(model=[]), "checkpoint model"),
        (
            "config.json",
            lambda state: state.pop("modalities"),
            "checkpoint config.*modalities",
        ),
        (
            "config.json",
            lambda state: state.update(future_option=True),
            "checkpoint config.*future_option",
        ),
        (
            "config.json",
            lambda state: state.update(normalize=None),
            "checkpoint config.*normalize",
        ),
        (
            "preprocessing.json",
            lambda state: state.pop("feature_dims"),
            "checkpoint modality dimensions",
        ),
        (
            "preprocessing.json",
            lambda state: state.update(feature_dims=None),
            "checkpoint modality dimensions",
        ),
        (
            "preprocessing.json",
            lambda state: state.update(feature_dims=["text", "image"]),
            "checkpoint modality dimensions",
        ),
    ],
)
def test_invalid_checkpoint_fields_raise_value_error(
    data, tmp_path, filename, corrupt, message
):
    model = content(missing="mean").fit(data[0])
    path = tmp_path / "model.zip"
    model.save(path)
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    state = json.loads(members[filename])
    corrupt(state)
    members[filename] = json.dumps(state).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, value in members.items():
            archive.writestr(name, value)
    with pytest.raises(ValueError, match=message):
        MMConcatWrapper.load(path)


@pytest.mark.parametrize(
    "change",
    [
        {"modalities": []},
        {"modalities": ["text", "text"]},
        {"modalities": "text"},
        {"modalities": [""]},
        {"missing": "drop"},
        {"extra_modalities": "drop"},
        {"extra_modalities": None},
        {"dtype": "float16"},
        {"weights": {"unknown": 1}},
        {"weights": {"text": float("nan")}},
        {"weights": {"text": -1}},
        {"weights": {"text": 0, "image": 0}},
        {"fit_features_parameter": ""},
        {"fit_features_parameter": "modality_masks"},
        {"model": object},
    ],
)
def test_configuration_validation(change):
    with pytest.raises((TypeError, ValueError)):
        replace(content().cfg, **change)


def test_config_copies_mutable_order_and_weights():
    names, weights = ["text", "image"], {"text": 2}
    cfg = MMConcatWrapperConfig(
        model=ContentRecommender, modalities=names, weights=weights
    )
    names.reverse()
    weights["text"] = 99
    assert cfg.modalities == ("text", "image")
    assert cfg.weights["text"] == 2


@pytest.mark.parametrize("weights", [None, {}, {"text": np.float32(0.5), "image": 2}])
@pytest.mark.parametrize(
    "model_class,model_config",
    [
        (ContentRecommender, None),
        (ContentRecommender, ContentRecommenderConfig()),
        (TEASER, TEASERConfig()),
    ],
)
def test_config_supports_copy_pickle_asdict_and_hash(
    weights, model_class, model_config
):
    cfg = MMConcatWrapperConfig(
        model=model_class,
        model_config=model_config,
        modalities=["text", "image"],
        weights=weights,
    )
    expected_weights = {} if weights is None else dict(weights)
    values = asdict(cfg)
    assert values["model"] is model_class
    assert values["modalities"] == ("text", "image")
    assert values["weights"] == expected_weights
    assert values["model_config"] == (
        None if model_config is None else asdict(model_config)
    )

    for cloned in (copy(cfg), deepcopy(cfg), pickle.loads(pickle.dumps(cfg))):
        assert cloned == cfg
        assert hash(cloned) == hash(cfg)
        assert {cfg: "cached"}[cloned] == "cached"
        assert dict(cloned.weights) == expected_weights
        with pytest.raises(TypeError):
            cloned.weights["text"] = 99
        with pytest.raises(TypeError):
            del cloned.weights["text"]


def test_config_weight_insertion_order_does_not_affect_hash_or_cache():
    left = content(weights={"text": 1, "image": 2}).cfg
    right = content(weights={"image": 2.0, "text": 1.0}).cfg
    assert left == right
    assert hash(left) == hash(right)

    @lru_cache
    def build(config):
        return MMConcatWrapper(config)

    assert build(left) is build(right)
    assert build.cache_info().hits == 1
    changed = replace(left, weights={"text": 3, "image": 2})
    assert changed != left
    assert build(changed) is not build(left)
    assert left.weights == {"text": 1, "image": 2}


def test_config_with_mutable_inner_config_can_copy_and_pickle_but_not_hash():
    cfg = MMConcatWrapperConfig(
        model=ContentRecommender,
        model_config={"custom_option": [1, 2]},
        modalities=["text"],
        weights={"text": 2},
    )
    assert asdict(cfg)["model_config"] == {"custom_option": [1, 2]}
    for cloned in (deepcopy(cfg), pickle.loads(pickle.dumps(cfg))):
        assert cloned == cfg
        cloned.model_config["custom_option"].append(3)
        assert cfg.model_config == {"custom_option": [1, 2]}
        with pytest.raises(TypeError):
            hash(cloned)
    with pytest.raises(TypeError):
        hash(cfg)


def test_weight_storage_cannot_be_reassigned_or_deleted():
    weights = content(weights={"text": 2}).cfg.weights
    with pytest.raises(AttributeError, match="immutable"):
        weights._pairs = (("text", 99),)
    with pytest.raises(AttributeError, match="immutable"):
        del weights._pairs
    assert weights.get("text") == 2
    assert weights.get("image", 1) == 1
    with pytest.raises(KeyError):
        weights["image"]


def test_checkpoint_config_is_json_compatible_and_used_by_save(
    data, tmp_path, monkeypatch
):
    model = content(weights={"text": np.float32(0.5), "image": 2}, missing="mean")
    config = model._checkpoint_config()
    assert json.loads(json.dumps(config, allow_nan=False)) == config
    assert config["weights"] == {"text": 0.5, "image": 2}
    assert config["model"] == "compresso_recsys.models.content.ContentRecommender"
    assert "model_config" not in config  # persisted in the inner checkpoint
    config["weights"]["text"] = 99
    assert model.cfg.weights["text"] == 0.5

    model.fit(data[0])
    serialize = model._checkpoint_config
    calls = []

    def track_serialization():
        payload = serialize()
        calls.append(payload)
        return payload

    monkeypatch.setattr(model, "_checkpoint_config", track_serialization)
    path = tmp_path / "wrapper.zip"
    model.save(path)
    assert len(calls) == 1
    with zipfile.ZipFile(path) as archive:
        assert json.loads(archive.read("config.json")) == calls[0]
    loaded = MMConcatWrapper.load(path)
    assert loaded.cfg == model.cfg
    assert hash(loaded.cfg) == hash(model.cfg)
    assert_predictions(loaded.predict(data[1], k=1), model.predict(data[1], k=1))


def test_publication_callback_failure_keeps_published_snapshot(data):
    model = content().fit(data[0])
    previous = model.candidates.snapshot()
    original = model.candidates._on_publish

    def failing_callback(snapshot):
        original(snapshot)
        raise RuntimeError("callback failed after publication")

    model.candidates._on_publish = failing_callback
    with pytest.raises(RuntimeError, match="after publication"):
        model.update_candidates(
            item_ids=["new"], item_features={n: v[:1] for n, v in data[0].items()}
        )
    assert model.candidates.snapshot() is not previous
    assert model.candidate_item_ids[-1] == "new"
    assert model.recommend([[0]], k=1, allowlist=["new"]).item_ids[0, 0] == "new"


def test_embedded_checkpoint_round_trip(data, tmp_path):
    from compresso_recsys import update_checkpoint

    checkpoint = tmp_path / "data.zip"
    with update_checkpoint(checkpoint):
        pass
    model = content(missing="mean").fit(data[0])
    model.save_to_checkpoint(checkpoint, "multimodal")
    loaded = MMConcatWrapper.load_from_checkpoint(checkpoint, "multimodal")
    assert_predictions(loaded.predict(data[1], k=1), model.predict(data[1], k=1))


def test_unfitted_wrapper_rejects_transform_and_save(data, tmp_path):
    model = content()
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.transform_features(data[0])
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.save(tmp_path / "unfitted.zip")
