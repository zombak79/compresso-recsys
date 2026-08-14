from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import (
    evaluate_ranked_predictions,
    evaluate_recommender,
)
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import CandidateCatalog, Recommender, TEASER, TEASERConfig


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [1, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 1, 0],
                [0, 1, 0, 0, 1, 1],
            ],
            dtype=np.float64,
        )
    )


@pytest.fixture
def item_features() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def source() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )


def _fit_reference(interactions, item_features) -> TEASER:
    return TEASER(
        TEASERConfig(
            l2_coefficients=0.07,
            l2_encoder=0.11,
            rho=0.13,
            max_iterations=4,
            include_popularity=True,
            dtype="float64",
        )
    ).fit(interactions, item_features)


def test_teaser_defaults_match_reference_configuration():
    config = TEASERConfig()

    assert config.l2_coefficients == 0.05
    assert config.l2_encoder == 0.05
    assert config.rho == 0.05
    assert config.max_iterations == 10
    assert config.include_popularity is False
    assert config.dtype == "float64"


def test_teaser_matches_original_admm_reference(interactions, item_features):
    model = _fit_reference(interactions, csr_matrix(item_features))
    expected_encoder = np.array(
        [
            [0.1579803809345355, 0.0480066008111144, -0.1450553540562592, 0.1921749549048460],
            [0.3157921926125347, -0.0874495160842595, -0.2069244283314512, 0.2597116849950547],
            [-0.3463169503920562, 0.3675186508379312, -0.1366888172213027, 0.2699088046618324],
            [-0.2094474083142827, -0.1752586522311156, 0.1924822531224707, 0.3565141753915905],
            [-0.0677709370182639, -0.0317589899254948, 0.3226744643472819, 0.1157722630403518],
            [0.0047368578388486, -0.1253579468612674, 0.1980604492967310, 0.1269373615277240],
        ],
        dtype=np.float64,
    )
    expected_dual = np.array(
        [
            -1.3054421580102962,
            -2.1268324288336298,
            -2.1016444394991930,
            -1.3036442949370000,
            -1.8910179036321195,
            -1.3471313597477457,
        ]
    )

    np.testing.assert_allclose(model.encoder_, expected_encoder, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model.diagonal_, np.zeros(6), rtol=0, atol=0)
    np.testing.assert_allclose(model.dual_, expected_dual, rtol=1e-12, atol=1e-12)
    assert len(model.admm_history_) == 4
    assert model.admm_history_[-1]["primal_residual"] == pytest.approx(
        0.9511551531376848
    )


def test_supported_item_feature_types_are_equivalent(interactions, item_features, source):
    srp_features = SRPTensor.from_dense(
        torch.from_numpy(item_features),
        k=2,
        score_mode="raw",
    )
    feature_inputs = [
        item_features,
        csr_matrix(item_features),
        torch.from_numpy(item_features),
        srp_features,
    ]

    models = [_fit_reference(interactions, features) for features in feature_inputs]
    expected_encoder = models[0].encoder_
    expected_predictions = models[0].predict_on_batch(source, k=4)

    for model in models[1:]:
        np.testing.assert_allclose(model.encoder_, expected_encoder, rtol=1e-12, atol=1e-12)
        predictions = model.predict_on_batch(source, k=4)
        torch.testing.assert_close(predictions.cols, expected_predictions.cols)
        torch.testing.assert_close(predictions.vals, expected_predictions.vals)


def test_sparse_torch_item_features_are_supported(interactions, item_features):
    sparse_features = torch.from_numpy(item_features).to_sparse()

    model = _fit_reference(interactions, sparse_features)

    expected = _fit_reference(interactions, csr_matrix(item_features))
    np.testing.assert_allclose(model.encoder_, expected.encoder_, rtol=1e-12, atol=1e-12)


def test_real_valued_dense_features_are_supported(interactions, item_features):
    dense_embeddings = item_features * np.array([0.25, -1.5, 2.0])

    model = TEASER(
        TEASERConfig(max_iterations=2, include_popularity=False)
    ).fit(interactions, dense_embeddings)

    assert model.encoder_.shape == dense_embeddings.shape
    assert isinstance(model.decoder_features_, np.ndarray)
    assert np.all(np.isfinite(model.encoder_))


def test_cold_items_are_decoder_only_candidates(item_features):
    train_indices = np.array([0, 2, 4], dtype=np.int64)
    interactions = csr_matrix(
        np.array(
            [
                [1, 0, 1, 0, 0, 0],
                [0, 0, 1, 0, 1, 0],
                [1, 0, 0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    model = TEASER(
        TEASERConfig(max_iterations=3, include_popularity=True)
    ).fit(
        interactions,
        csr_matrix(item_features),
        train_item_indices=train_indices,
        feature_names=["first", "second", "third"],
    )
    source = csr_matrix([[1, 0, 0, 0, 1, 0]], dtype=np.float64)

    predictions = model.predict_on_batch(source, k=source.shape[1], exclude_seen=False)
    profiles = np.asarray(source[:, train_indices] @ model.encoder_)
    expected_scores = np.asarray((model.decoder_features_ @ profiles.T).T)

    np.testing.assert_allclose(
        predictions.to_dense().numpy(),
        expected_scores,
        rtol=1e-12,
        atol=1e-12,
    )
    assert model.encoder_.shape == (len(train_indices), item_features.shape[1] + 1)
    assert model.decoder_features_.shape == (item_features.shape[0], item_features.shape[1] + 1)
    np.testing.assert_array_equal(
        model.decoder_features_[:, -1].toarray().ravel()[[1, 3, 5]],
        0,
    )
    assert model.feature_names_ == ("first", "second", "third", "popularity")


def test_cold_items_cannot_be_used_as_source_history(item_features):
    interactions = csr_matrix(
        np.array(
            [
                [1, 0, 1, 0],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
    )
    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions,
        item_features[:4],
        train_item_indices=[0, 2],
    )

    with pytest.raises(ValueError, match="no fitted encoder row"):
        model.predict_on_batch(csr_matrix([[0, 1, 0, 0]]), k=2)


def test_fit_builds_id_aware_candidate_catalog(interactions, item_features):
    item_ids = np.array([f"book-{index}" for index in range(interactions.shape[1])])
    metadata = pd.DataFrame({"item_id": item_ids, "title": list("ABCDEF")})

    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions,
        item_features,
        item_ids=item_ids,
        metadata=metadata,
        feature_space_id="encoder@revision",
    )

    assert isinstance(model.candidates, CandidateCatalog)
    assert model.candidates.version == 1
    assert model.candidates.feature_space_id == "encoder@revision"
    assert model.n_candidates_ == interactions.shape[1]
    np.testing.assert_array_equal(model.candidates.item_ids, item_ids)
    assert model.candidates.metadata.equals(metadata)
    np.testing.assert_array_equal(
        model.candidates.rows_for(["book-4", "book-1"]),
        [4, 1],
    )
    np.testing.assert_array_equal(
        model.candidates.ids_for(torch.tensor([[4, 1]])),
        [["book-4", "book-1"]],
    )


def test_fit_validates_item_identity_and_metadata(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1))

    with pytest.raises(ValueError, match="duplicate"):
        model.fit(interactions, item_features, item_ids=[0, 1, 2, 3, 4, 4])
    with pytest.raises(ValueError, match="item_features has 6 rows"):
        model.fit(interactions, item_features, item_ids=[0, 1])
    with pytest.raises(ValueError, match="must match item_ids"):
        model.fit(
            interactions,
            item_features,
            item_ids=list("ABCDEF"),
            metadata=pd.DataFrame({"item_id": list("BACDEF")}),
        )
    with pytest.raises(TypeError, match="pandas.DataFrame"):
        model.fit(interactions, item_features, metadata=[{}])
    with pytest.raises(ValueError, match="feature_space_id"):
        model.fit(interactions, item_features, feature_space_id="")


def test_build_candidates_replaces_catalog_but_not_source_vocabulary(
    interactions,
    item_features,
    source,
):
    source_ids = list("ABCDEF")
    model = TEASER(
        TEASERConfig(max_iterations=2, include_popularity=True)
    ).fit(
        interactions,
        item_features,
        item_ids=source_ids,
        feature_space_id="features-v1",
    )
    candidate_ids = ["new-1", "A", "new-2"]
    candidate_features = np.vstack((item_features[0], item_features[0], item_features[5]))

    catalog = model.build_candidates(
        item_ids=candidate_ids,
        item_features=candidate_features,
        metadata=pd.DataFrame({"item_id": candidate_ids, "title": ["N1", "A", "N2"]}),
        feature_space_id="features-v1",
    )
    predictions = model.predict_on_batch(source[:1], k=3, exclude_seen=False)
    profiles = model.user_profiles(source[:1])
    expected = np.asarray(catalog.item_features @ profiles.T).T

    assert catalog.version == 2
    assert predictions.shape == (1, 3)
    np.testing.assert_allclose(predictions.to_dense().numpy(), expected)
    np.testing.assert_array_equal(
        np.asarray(catalog.item_features[:, -1]).ravel(),
        [0.0, model.source_popularity_[0], 0.0],
    )
    unseen = model.predict_on_batch(source[:1], k=2)
    assert 1 not in unseen.cols[0].tolist()  # A moved to candidate row 1.
    assert model.n_items_ == interactions.shape[1]
    with pytest.raises(ValueError, match="fitted with 6 items"):
        model.predict_on_batch(csr_matrix((1, 3)), k=1)


def test_update_candidates_appends_replaces_and_ignores_conflicts(
    interactions,
    item_features,
):
    model = TEASER(TEASERConfig(max_iterations=1, include_popularity=False)).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
        metadata=pd.DataFrame({"item_id": list("ABCDEF"), "title": list("abcdef")}),
    )

    with pytest.raises(ValueError, match="already exists"):
        model.update_candidates(item_ids=["A"], item_features=item_features[[5]])

    replaced = model.update_candidates(
        item_ids=["C", "G"],
        item_features=csr_matrix(np.vstack(([9, 8, 7], [6, 5, 4]))),
        metadata=pd.DataFrame({"item_id": ["C", "G"], "title": ["new-c", "new-g"]}),
        on_conflict="replace",
    )
    assert replaced.item_ids.tolist() == list("ABCDEFG")
    np.testing.assert_array_equal(replaced.item_features[2].toarray(), [[9, 8, 7]])
    np.testing.assert_array_equal(replaced.item_features[6].toarray(), [[6, 5, 4]])
    assert replaced.metadata.loc[2, "title"] == "new-c"
    assert replaced.metadata.loc[6, "title"] == "new-g"

    ignored = model.update_candidates(
        item_ids=["C"],
        item_features=np.array([[1, 1, 1]], dtype=np.float64),
        metadata=pd.DataFrame({"item_id": ["C"], "title": ["ignored"]}),
        on_conflict="ignore",
    )
    assert ignored is replaced
    np.testing.assert_array_equal(ignored.item_features[2].toarray(), [[9, 8, 7]])
    assert ignored.metadata.loc[2, "title"] == "new-c"


def test_candidate_allowlist_scores_only_registered_ids_and_returns_global_rows(
    interactions,
    item_features,
    source,
):
    model = TEASER(TEASERConfig(max_iterations=2)).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
    )
    model.update_candidates(
        item_ids=["G", "H"],
        item_features=item_features[[4, 5]],
    )

    predictions = model.predict_on_batch(
        source[:2],
        k=3,
        candidate_ids=["H", "B", "G", "D"],
    )

    assert predictions.shape == (2, 8)
    assert set(predictions.cols.flatten().tolist()).issubset({1, 3, 6, 7})
    assert 1 not in predictions.cols[1].tolist()  # B is seen in the second row.
    with pytest.raises(KeyError, match="unknown candidate"):
        model.predict_on_batch(source[:1], k=1, candidate_ids=["missing"])
    with pytest.raises(ValueError, match="duplicate"):
        model.predict_on_batch(source[:1], k=1, candidate_ids=["G", "G"])
    with pytest.raises(ValueError, match="unseen items among the selected candidates"):
        model.predict_on_batch(source[:1], k=2, candidate_ids=["A", "B"])


def test_remove_candidates_and_feature_space_validation(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions,
        item_features,
        item_ids=list("ABCDEF"),
        feature_space_id="features-v1",
    )

    catalog = model.remove_candidates(["B", "E"])
    assert catalog.item_ids.tolist() == ["A", "C", "D", "F"]
    assert catalog.version == 2
    with pytest.raises(KeyError, match="unknown candidate"):
        model.remove_candidates(["missing"])
    assert model.remove_candidates(["missing"], missing="ignore") is catalog
    with pytest.raises(ValueError, match="feature_space_id"):
        model.update_candidates(
            item_ids=["G"],
            item_features=item_features[[0]],
            feature_space_id="other-space",
        )


def test_failed_catalog_rebuild_is_atomic(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1)).fit(interactions, item_features)
    original = model.candidates

    with pytest.raises(ValueError, match="input features"):
        model.build_candidates(
            item_ids=["new"],
            item_features=np.ones((1, item_features.shape[1] + 1)),
        )

    assert model.candidates is original


def test_align_source_selects_and_reorders_columns_by_stable_id(
    interactions,
    item_features,
    source,
):
    source_columns = np.array([4, 0, 2], dtype=np.int64)
    source_ids = np.array(list("ABCDEF"), dtype=object)
    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions[:, source_columns],
        item_features[source_columns],
        item_ids=source_ids[source_columns],
    )

    aligned = model.align_source(source, item_ids=source_ids)
    expected = source[:, source_columns]

    assert isinstance(aligned, csr_matrix)
    assert aligned.shape == (source.shape[0], source_columns.size)
    np.testing.assert_array_equal(aligned.toarray(), expected.toarray())
    assert model.align_source(aligned, item_ids=model.source_item_ids_) is aligned


def test_aligned_source_produces_same_predictions_as_manual_projection(
    interactions,
    item_features,
    source,
):
    train_columns = np.array([0, 2, 4], dtype=np.int64)
    item_ids = np.array(list("ABCDEF"), dtype=object)
    model = TEASER(TEASERConfig(max_iterations=2)).fit(
        interactions[:, train_columns],
        item_features[train_columns],
        item_ids=item_ids[train_columns],
    )
    model.build_candidates(item_ids=item_ids, item_features=item_features)

    aligned = model.align_source(source, item_ids=item_ids)
    expected = model.predict_on_batch(source[:, train_columns], k=3)
    actual = model.predict_on_batch(aligned, k=3)

    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_align_source_validates_model_matrix_and_item_ids(
    interactions,
    item_features,
    source,
):
    with pytest.raises(RuntimeError, match="fitted"):
        TEASER().align_source(source, item_ids=list("ABCDEF"))

    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions[:, [0, 2]],
        item_features[[0, 2]],
        item_ids=["A", "C"],
    )
    with pytest.raises(TypeError, match="csr_matrix"):
        model.align_source(csc_matrix(source), item_ids=list("ABCDEF"))
    with pytest.raises(ValueError, match="source has 6 columns"):
        model.align_source(source, item_ids=["A", "B"])
    with pytest.raises(ValueError, match="duplicate"):
        model.align_source(source, item_ids=["A", "B", "C", "D", "E", "E"])
    with pytest.raises(ValueError, match="missing fitted source item ID: 'C'"):
        model.align_source(
            csr_matrix((source.shape[0], 2)),
            item_ids=["A", "B"],
        )


def test_align_source_handles_empty_user_batches(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions[:, [1, 3]],
        item_features[[1, 3]],
        item_ids=["B", "D"],
    )

    aligned = model.align_source(
        csr_matrix((0, interactions.shape[1])),
        item_ids=list("ABCDEF"),
    )

    assert aligned.shape == (0, 2)
    assert aligned.nnz == 0


def test_predict_uses_one_catalog_snapshot_across_batches(
    interactions,
    item_features,
    source,
    monkeypatch,
):
    model = TEASER(TEASERConfig(max_iterations=1)).fit(interactions, item_features)
    original_predict = model._predict_prepared_batch
    versions: list[int] = []

    def recording_predict(source_batch, **kwargs):
        versions.append(kwargs["catalog"].version)
        result = original_predict(source_batch, **kwargs)
        if len(versions) == 1:
            model.build_candidates(
                item_ids=["new-0", "new-1", "new-2"],
                item_features=item_features[:3],
            )
        return result

    monkeypatch.setattr(model, "_predict_prepared_batch", recording_predict)
    predictions = model.predict(source, k=2, batch_size=2, exclude_seen=False)

    assert versions == [1, 1]
    assert predictions.shape == source.shape
    assert model.candidates.version == 2


def test_predict_matches_predict_on_batch(interactions, item_features, source):
    model = TEASER(TEASERConfig(max_iterations=2)).fit(
        interactions,
        item_features,
    )

    expected = model.predict_on_batch(source, k=3)
    actual = model.predict(source, k=3, batch_size=2)

    assert actual.shape == source.shape
    assert actual.k == 3
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_predictions_exclude_seen_items(interactions, item_features, source):
    model = TEASER(TEASERConfig(max_iterations=2)).fit(
        interactions,
        item_features,
    )

    predictions = model.predict(source, k=3, batch_size=2)

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()).isdisjoint(source[row].indices)


def test_seen_item_mask_can_be_disabled(interactions, item_features, source):
    model = TEASER(TEASERConfig(max_iterations=2)).fit(
        interactions,
        item_features,
    )

    predictions = model.predict(
        source,
        k=source.shape[1],
        batch_size=2,
        exclude_seen=False,
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()) == set(range(source.shape[1]))


def test_empty_source_returns_empty_srp(interactions, item_features):
    model = TEASER(TEASERConfig(dtype="float32", max_iterations=1)).fit(
        interactions,
        item_features,
    )
    source = csr_matrix((0, interactions.shape[1]), dtype=np.float32)

    predictions = model.predict(source, k=3)

    assert predictions.shape == source.shape
    assert predictions.cols.shape == (0, 3)
    assert predictions.vals.shape == (0, 3)
    assert predictions.vals.dtype == torch.float32


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("l2_coefficients", 0),
        ("l2_coefficients", np.nan),
        ("l2_encoder", -1),
        ("rho", np.inf),
        ("max_iterations", 0),
        ("max_iterations", 1.5),
        ("include_popularity", "yes"),
        ("dtype", "float16"),
    ],
)
def test_config_rejects_invalid_values(field, value):
    kwargs = {field: value}
    with pytest.raises(ValueError, match=field):
        TEASERConfig(**kwargs)


def test_fit_validates_interactions(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1))

    with pytest.raises(TypeError, match="csr_matrix"):
        model.fit(csc_matrix(interactions), item_features)
    with pytest.raises(ValueError, match="at least one"):
        model.fit(csr_matrix((0, interactions.shape[1])), item_features)
    with pytest.raises(ValueError, match="binary"):
        model.fit(interactions * 2, item_features)
    invalid = interactions.copy()
    invalid.data[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.fit(invalid, item_features)


def test_fit_validates_item_features(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1))

    with pytest.raises(TypeError, match="item_features"):
        model.fit(interactions, item_features.tolist())
    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(interactions, item_features.ravel())
    with pytest.raises(ValueError, match="finite"):
        invalid = item_features.copy()
        invalid[0, 0] = np.nan
        model.fit(interactions, invalid)
    with pytest.raises(TypeError, match="real numeric"):
        model.fit(interactions, item_features.astype(np.complex128) * (1 + 1j))
    with pytest.raises(ValueError, match="two-dimensional"):
        model.fit(interactions, torch.from_numpy(item_features).unsqueeze(0))
    with pytest.raises(ValueError, match="rows"):
        model.fit(interactions, item_features[:-1])
    with pytest.raises(ValueError, match="feature_names"):
        model.fit(interactions, item_features, feature_names=["too", "short"])


def test_fit_validates_train_item_indices(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1))

    with pytest.raises(ValueError, match="one-dimensional"):
        model.fit(interactions, item_features, train_item_indices=[[0, 1]])
    with pytest.raises(TypeError, match="integers"):
        model.fit(interactions, item_features, train_item_indices=[0.0, 1.0])
    with pytest.raises(ValueError, match="duplicates"):
        model.fit(interactions, item_features, train_item_indices=[0, 0])
    with pytest.raises(ValueError, match=r"\[0, 5\]"):
        model.fit(interactions, item_features, train_item_indices=[6])


def test_prediction_requires_fitted_model(source):
    with pytest.raises(RuntimeError, match="fitted"):
        TEASER().predict_on_batch(source, k=2)


def test_prediction_validates_shape_and_parameters(interactions, item_features, source):
    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions,
        item_features,
    )

    with pytest.raises(ValueError, match="fitted with"):
        model.predict_on_batch(csr_matrix((1, interactions.shape[1] + 1)), k=1)
    with pytest.raises(ValueError, match=r"k must be in \[1"):
        model.predict_on_batch(source, k=0)
    with pytest.raises(ValueError, match="batch_size"):
        model.predict(source, k=2, batch_size=0)
    with pytest.raises(TypeError, match="csr_matrix"):
        model.predict_on_batch(csc_matrix(source), k=2)
    with pytest.raises(ValueError, match="binary"):
        model.predict_on_batch(source * 2, k=2)


def test_prediction_rejects_rows_with_too_few_unseen_items(interactions, item_features):
    model = TEASER(TEASERConfig(max_iterations=1)).fit(
        interactions,
        item_features,
    )
    source = csr_matrix(np.array([[1, 1, 1, 1, 1, 0]], dtype=np.float64))

    with pytest.raises(ValueError, match="only 1 unseen items"):
        model.predict_on_batch(source, k=2)


def test_teaser_implements_recommender_protocol():
    assert isinstance(TEASER(), Recommender)


def test_streaming_evaluation_matches_materialized_predictions(
    interactions,
    item_features,
    source,
):
    model = TEASER(TEASERConfig(max_iterations=2)).fit(
        interactions,
        item_features,
    )
    targets = csr_matrix(
        (
            np.ones(source.shape[0], dtype=np.float64),
            (
                np.arange(source.shape[0]),
                np.array([1, 3, 2, 5]),
            ),
        ),
        shape=source.shape,
    )

    streamed = evaluate_recommender(
        model,
        source=source,
        targets=targets,
        metrics=[CalibratedRecall([1, 3]), NDCG(3)],
        batch_size=2,
    )
    materialized = evaluate_ranked_predictions(
        predictions=model.predict(source, k=3, batch_size=2),
        targets=targets,
        metrics=[CalibratedRecall([1, 3]), NDCG(3)],
        batch_size=2,
    )

    assert streamed == materialized
