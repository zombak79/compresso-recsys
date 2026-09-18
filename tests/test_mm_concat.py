from __future__ import annotations

from dataclasses import replace

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
        model.fit(data[0])
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
    np.testing.assert_allclose(result.toarray()[0, :2], [2 / 3, 2 / 3])


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
        model.candidates.snapshot().item_features.toarray()[-1], np.zeros(4)
    )


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
        with pytest.raises(TypeError, match="no device-backed state"):
            loaded.to("cpu")
    else:
        assert loaded.to("cpu") is loaded
    assert_predictions(loaded.predict(x, k=1), model.predict(x, k=1))
    for instance in (model, loaded):
        instance.update_candidates(
            item_ids=["new"], item_features={"image": np.ones((1, 2))}
        )
    assert_predictions(loaded.predict(x, k=2), model.predict(x, k=2))
    assert dict(loaded.feature_dims_) == {"text": 2, "image": 2}


def test_custom_checkpoint_classes_require_local_registration(data, tmp_path):
    model = MMConcatWrapper(
        MMConcatWrapperConfig(
            model=OpaqueContent, modalities=["text"], fit_features_parameter="vectors"
        )
    ).fit(vectors=data[0])
    path = tmp_path / "model.zip"
    model.save(path)
    model_id = f"{OpaqueContent.__module__}.{OpaqueContent.__qualname__}"
    MMConcatWrapper._registered_models.pop(model_id)
    with pytest.raises(ValueError, match="unregistered"):
        MMConcatWrapper.load(path)
    MMConcatWrapper.register_model(OpaqueContent)
    assert MMConcatWrapper.load(path).is_fitted


@pytest.mark.parametrize(
    "change",
    [
        {"modalities": []},
        {"modalities": ["text", "text"]},
        {"modalities": "text"},
        {"modalities": [""]},
        {"missing": "drop"},
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
