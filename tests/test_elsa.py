from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from scipy.sparse import csc_matrix, csr_matrix

from compresso import SRPTensor
from compresso_recsys.evaluation import (
    evaluate_ranked_predictions,
    evaluate_recommender,
)
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import (
    CompressedELSA,
    ELSA,
    ELSACompressionConfig,
    ELSAConfig,
    ELSATrainer,
    Recommender,
)
from compresso_recsys.models.elsa import (
    _dense_training_target,
    _ELSAInteractionDataset,
)


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 0, 1, 1, 0, 0],
            ],
            dtype=np.float32,
        )
    )


@pytest.fixture
def source() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )


def _trainer(**overrides) -> ELSATrainer:
    defaults = {
        "latent_dim": 4,
        "batch_size": 2,
        "max_output": 6,
        "epochs": 2,
        "lr": 1e-2,
        "show_progress": False,
        "seed": 7,
    }
    defaults.update(overrides)
    return ELSATrainer(ELSAConfig(**defaults))


def _compressed_trainer(
    *,
    compression: ELSACompressionConfig | None = None,
    **overrides,
) -> ELSATrainer:
    defaults = {
        "latent_dim": 4,
        "batch_size": 2,
        "max_output": 6,
        "epochs": 2,
        "lr": 1e-2,
        "show_progress": False,
        "seed": 7,
    }
    defaults.update(overrides)
    return ELSATrainer(
        ELSAConfig(
            **defaults,
            compression=compression
            or ELSACompressionConfig(
                k_target=2,
                k_schedule=(4, 3, 2),
                stability_window=1,
                change_threshold=100.0,
                mask_update_interval=1,
            ),
        )
    )


def test_elsa_forward_matches_restricted_reference():
    model = ELSA(input_dim=4, latent_dim=2, use_relu=False)
    with torch.no_grad():
        model.A.copy_(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                    [-1.0, 1.0],
                ]
            )
        )
    x = torch.tensor([[2.0, 1.0], [0.0, 3.0]])
    sources = torch.tensor([0, 2])
    candidates = torch.tensor([0, 2, 1, 3])
    x_out = torch.tensor(
        [[2.0, 1.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]]
    )

    actual = model(
        x,
        sources=sources,
        candidates=candidates,
        x_out=x_out,
    )

    embeddings = F.normalize(model.A, dim=-1)
    expected = (x @ embeddings[sources]) @ embeddings[candidates].T - x_out
    torch.testing.assert_close(actual, expected)


def test_candidate_sampler_preserves_positives_and_samples_unique_negatives(
    interactions,
):
    dataset = _ELSAInteractionDataset(
        interactions,
        device=torch.device("cpu"),
        batch_size=2,
        shuffle=False,
        max_output=5,
        seed=3,
    )

    x, sources, candidates = dataset[0]

    assert x.is_sparse
    assert candidates is not None
    assert sources.tolist() == [0, 1, 2]
    assert candidates.shape == (5,)
    assert len(set(candidates.tolist())) == 5
    assert candidates[: len(sources)].tolist() == sources.tolist()
    assert set(candidates[len(sources) :].tolist()).isdisjoint(sources.tolist())
    assert x.shape == (2, 3)
    x_dense = x.to_dense()
    y_dense = _dense_training_target(
        x_dense,
        sources=sources,
        candidates=candidates,
        input_dim=interactions.shape[1],
    )
    assert y_dense.shape == (2, 5)
    torch.testing.assert_close(x_dense, y_dense[:, : len(sources)])
    assert torch.count_nonzero(y_dense[:, len(sources) :]) == 0


def test_candidate_sampler_uses_full_catalog_when_unlimited(interactions):
    dataset = _ELSAInteractionDataset(
        interactions,
        device=torch.device("cpu"),
        batch_size=2,
        shuffle=False,
        max_output=None,
        seed=3,
    )

    x, sources, candidates = dataset[0]

    assert x.is_sparse
    assert candidates is None
    y = _dense_training_target(
        x.to_dense(),
        sources=sources,
        candidates=candidates,
        input_dim=interactions.shape[1],
    )
    assert y.shape == (2, interactions.shape[1])
    torch.testing.assert_close(
        y,
        torch.from_numpy(interactions[:2].toarray()),
    )
    torch.testing.assert_close(
        x.to_dense(),
        y[:, sources],
    )


def test_candidate_sampler_never_drops_positives_to_meet_budget(interactions):
    dataset = _ELSAInteractionDataset(
        interactions,
        device=torch.device("cpu"),
        batch_size=4,
        shuffle=False,
        max_output=2,
        seed=3,
    )

    x, sources, candidates = dataset[0]

    assert candidates is not None
    assert sources.tolist() == [0, 1, 2, 3, 4]
    assert candidates.tolist() == sources.tolist()
    y = _dense_training_target(
        x.to_dense(),
        sources=sources,
        candidates=candidates,
        input_dim=interactions.shape[1],
    )
    assert y.shape == (4, 5)


def test_candidate_sampling_is_reproducible(interactions):
    kwargs = {
        "device": torch.device("cpu"),
        "batch_size": 2,
        "shuffle": False,
        "max_output": 5,
        "seed": 11,
    }
    first = _ELSAInteractionDataset(interactions, **kwargs)
    second = _ELSAInteractionDataset(interactions, **kwargs)

    first_candidates = first[0][2]
    second_candidates = second[0][2]
    assert first_candidates is not None
    assert second_candidates is not None
    torch.testing.assert_close(first_candidates, second_candidates)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("latent_dim", 0),
        ("batch_size", 0),
        ("max_output", 0),
        ("epochs", 0),
        ("lr", 0),
        ("lr", np.nan),
        ("weight_decay", -1),
        ("weight_decay", np.inf),
        ("optimizer", "SGD"),
    ],
)
def test_elsa_config_rejects_invalid_values(name, value):
    with pytest.raises(ValueError, match=name):
        ELSAConfig(**{name: value})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"k_target": 0}, "k_target"),
        ({"k_target": 2, "k_schedule": ()}, "k_schedule"),
        ({"k_target": 2, "k_schedule": (4, 3)}, "last k_schedule"),
        ({"k_target": 2, "k_schedule": (4, 2, 3)}, "non-increasing"),
        ({"k_target": 2, "stability_window": 0}, "stability_window"),
        ({"k_target": 2, "change_threshold": -1}, "change_threshold"),
        ({"k_target": 2, "mask_update_interval": 0}, "mask_update_interval"),
        ({"k_target": 2, "max_epochs_per_stage": 0}, "max_epochs_per_stage"),
        ({"k_target": 2, "ste_alpha": 2}, "ste_alpha"),
        ({"k_target": 2, "sparse_finetune_backend": "csr"}, "finetune"),
        ({"k_target": 2, "sparse_inference_backend": "coo"}, "inference"),
    ],
)
def test_compression_config_rejects_invalid_values(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ELSACompressionConfig(**kwargs)


def test_elsa_config_validates_compression_against_model():
    compression = ELSACompressionConfig(k_target=3)
    with pytest.raises(ValueError, match="k_target"):
        ELSAConfig(latent_dim=2, compression=compression)
    with pytest.raises(ValueError, match="torch.compile"):
        ELSAConfig(latent_dim=3, compile=True, compression=compression)
    with pytest.raises(ValueError, match="start with latent_dim"):
        ELSAConfig(
            latent_dim=4,
            compression=ELSACompressionConfig(
                k_target=2,
                k_schedule=(3, 2),
            ),
        )


def test_fit_records_history_and_returns_trainer(interactions):
    trainer = _trainer()

    returned = trainer.fit(interactions)

    assert returned is trainer
    assert trainer.is_built
    assert trainer.is_fitted
    assert trainer.input_dim == interactions.shape[1]
    assert len(trainer.history) == trainer.cfg.epochs
    assert all(np.isfinite(record["loss"]) for record in trainer.history)
    assert all(np.isfinite(record["cosine_loss"]) for record in trainer.history)


def test_compressed_fit_searches_converts_and_finetunes(interactions):
    trainer = _compressed_trainer().fit(interactions)

    assert isinstance(trainer.elsa, CompressedELSA)
    assert trainer.elsa.phase == "inference"
    assert trainer.elsa.masked_A is None
    assert trainer.elsa.sparse_A is not None
    assert trainer.elsa.sparse_A.k == 2
    assert trainer.sparsity_controller is None
    assert [name for name, _ in trainer.elsa.named_parameters()] == [
        "sparse_A.values"
    ]

    search = [
        record
        for record in trainer.history
        if record.get("phase") == "mask_search"
    ]
    finetune = [
        record
        for record in trainer.history
        if record.get("phase") == "sparse_finetune"
    ]
    assert [record["stage"] for record in search] == [0.0, 1.0]
    assert [record["k"] for record in search] == [4.0, 3.0]
    assert [record["transition"] for record in search] == [
        "stable",
        "stable",
    ]
    assert len(finetune) == trainer.cfg.epochs
    assert all(np.isfinite(record["loss"]) for record in trainer.history)


def test_compressed_mask_search_scores_only_selected_candidate_rows(
    monkeypatch,
):
    model = CompressedELSA(
        input_dim=8,
        latent_dim=4,
        compression=ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 2),
        ),
        use_relu=False,
    )
    assert model.masked_A is not None
    model.masked_A.k_current = 2
    candidates = torch.tensor([0, 3, 5, 7])
    sources = candidates[:2]
    x = torch.tensor([[1.0, 0.5], [0.0, 2.0]])
    full_factors = model.normalized_item_embeddings()
    expected = (
        (x @ full_factors[sources])
        @ full_factors[candidates].T
    )

    def fail_full_materialization():
        raise AssertionError("full MaskedParam materialization was called")

    monkeypatch.setattr(model.masked_A, "forward", fail_full_materialization)

    actual = model(
        x,
        sources=sources,
        candidates=candidates,
    )

    torch.testing.assert_close(actual, expected)


def test_compressed_sparse_finetuning_selects_rows_and_preserves_gradients(
    interactions,
    monkeypatch,
):
    trainer = _compressed_trainer(
        epochs=1,
        use_relu=False,
    ).fit(interactions)
    assert isinstance(trainer.elsa, CompressedELSA)
    model = trainer.elsa
    assert model.sparse_A is not None
    model.train()
    model.zero_grad(set_to_none=True)
    candidates = torch.tensor([0, 2, 4, 6])
    sources = candidates[:2]
    x = torch.tensor([[1.0, 0.5], [0.25, 1.5]])
    full_factors = model.normalized_item_embeddings()
    expected = (
        (x @ full_factors[sources])
        @ full_factors[candidates].T
    )

    def fail_full_materialization():
        raise AssertionError("full SRPParam materialization was called")

    monkeypatch.setattr(model.sparse_A, "forward", fail_full_materialization)

    actual = model(
        x,
        sources=sources,
        candidates=candidates,
    )
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()

    assert model.sparse_A.values.grad is not None
    unselected = torch.ones(model.input_dim, dtype=torch.bool)
    unselected[candidates] = False
    assert torch.count_nonzero(model.sparse_A.values.grad[unselected]) == 0
    assert torch.count_nonzero(model.sparse_A.values.grad[candidates]) > 0


def test_compressed_coo_finetuning_matches_dense_outputs_and_gradients(
    interactions,
    monkeypatch,
):
    trainer = _compressed_trainer(
        epochs=1,
        use_relu=False,
    ).fit(interactions)
    assert isinstance(trainer.elsa, CompressedELSA)
    model = trainer.elsa
    assert model.sparse_A is not None
    model.train()
    candidates = torch.tensor([0, 2, 4, 6])
    sources = candidates[:2]
    x = torch.tensor([[1.0, 0.5], [0.25, 1.5]])
    x_out = torch.tensor(
        [[1.0, 0.5, 0.0, 0.0], [0.25, 1.5, 0.0, 0.0]]
    )

    model.compression = replace(
        model.compression,
        sparse_finetune_backend="dense",
    )
    model.zero_grad(set_to_none=True)
    dense_output = model(
        x,
        sources=sources,
        candidates=candidates,
        x_out=x_out,
    )
    dense_output.square().sum().backward()
    dense_gradient = model.sparse_A.values.grad.detach().clone()
    model.zero_grad(set_to_none=True)

    model.compression = replace(
        model.compression,
        sparse_finetune_backend="coo",
    )

    def fail_densification(self):
        raise AssertionError("COO fine-tuning densified SRP factors")

    monkeypatch.setattr(SRPTensor, "to_dense", fail_densification)
    coo_output = model(
        x,
        sources=sources,
        candidates=candidates,
        x_out=x_out,
    )
    coo_output.square().sum().backward()

    torch.testing.assert_close(coo_output, dense_output)
    torch.testing.assert_close(model.sparse_A.values.grad, dense_gradient)


def test_compressed_coo_finetuning_supports_full_catalog(
    interactions,
):
    trainer = _compressed_trainer(
        epochs=1,
        use_relu=False,
    ).fit(interactions)
    assert isinstance(trainer.elsa, CompressedELSA)
    model = trainer.elsa
    assert model.sparse_A is not None
    model.train()
    source_columns = torch.tensor([0, 2, 4])
    x = torch.tensor([[1.0, 0.5, 0.25], [0.25, 1.5, 0.0]])

    model.compression = replace(
        model.compression,
        sparse_finetune_backend="dense",
    )
    dense_output = model(x, sources=source_columns)
    model.compression = replace(
        model.compression,
        sparse_finetune_backend="coo",
    )
    coo_output = model(x, sources=source_columns)

    torch.testing.assert_close(coo_output, dense_output)


def test_compressed_fit_accepts_coo_finetuning(interactions):
    trainer = _compressed_trainer(
        epochs=1,
        compression=ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 2),
            stability_window=1,
            change_threshold=100.0,
            mask_update_interval=1,
            sparse_finetune_backend="coo",
        ),
    ).fit(interactions)

    assert isinstance(trainer.elsa, CompressedELSA)
    assert trainer.elsa.phase == "inference"
    assert any(
        record.get("phase") == "sparse_finetune"
        for record in trainer.history
    )


def test_compressed_stage_can_be_forced_after_epoch_limit(
    interactions,
    capsys,
):
    trainer = _compressed_trainer(
        epochs=1,
        compression=ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 3, 2),
            stability_window=100,
            change_threshold=0.0,
            mask_update_interval=1,
            max_epochs_per_stage=1,
        ),
    ).fit(interactions)

    search = [
        record
        for record in trainer.history
        if record.get("phase") == "mask_search"
    ]

    assert [record["stage"] for record in search] == [0.0, 1.0]
    assert [record["transition"] for record in search] == [
        "forced",
        "forced",
    ]
    assert isinstance(trainer.elsa, CompressedELSA)
    assert trainer.elsa.phase == "inference"
    assert trainer.elsa.sparse_A is not None
    assert trainer.elsa.sparse_A.k == 2
    output = capsys.readouterr().out
    assert output.count("[ELSATrainer] Forced rewind") == 2
    assert "max_epochs_per_stage=1" in output
    assert "'schedule_done': True" in output


def test_compressed_export_is_l2_normalized(interactions):
    trainer = _compressed_trainer(
        epochs=1,
        compression=ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 2),
            stability_window=1,
            change_threshold=100.0,
            mask_update_interval=1,
        ),
    ).fit(interactions)
    assert isinstance(trainer.elsa, CompressedELSA)

    factors = trainer.elsa.export_item_embeddings()

    torch.testing.assert_close(
        factors.vals.norm(p=2, dim=1),
        torch.ones(interactions.shape[1]),
    )


def test_compressed_sparse_inference_matches_dense_scoring(
    interactions,
    source,
):
    trainer = _compressed_trainer(
        epochs=1,
        compression=ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 2),
            stability_window=1,
            change_threshold=100.0,
            mask_update_interval=1,
        ),
    ).fit(interactions)
    assert isinstance(trainer.elsa, CompressedELSA)
    source_columns = np.unique(source.indices).astype(np.int64, copy=False)
    x = torch.from_numpy(
        source[:, source_columns].toarray().astype(np.float32, copy=False)
    )
    sources = torch.from_numpy(source_columns)

    sparse_scores = trainer.elsa.score_all_items(
        x,
        sources=sources,
        backend="csr",
    )
    dense_backend_scores = trainer.elsa.score_all_items(
        x,
        sources=sources,
        backend="dense",
    )
    dense_reference_scores = trainer.elsa(x, sources=sources)

    torch.testing.assert_close(sparse_scores, dense_reference_scores)
    torch.testing.assert_close(dense_backend_scores, dense_reference_scores)


def test_compressed_inference_builds_backend_caches_lazily(
    interactions,
    source,
):
    trainer = _compressed_trainer(
        epochs=1,
        compression=ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 2),
            stability_window=1,
            change_threshold=100.0,
            mask_update_interval=1,
            sparse_inference_backend="dense",
        ),
    ).fit(interactions)
    assert isinstance(trainer.elsa, CompressedELSA)
    model = trainer.elsa
    assert model._inference_srp is not None
    assert model._inference_dense is not None
    assert model._inference_csr is None
    dense_cache = model._inference_dense

    trainer.predict_on_batch(
        source,
        k=3,
        sparse_inference_backend="dense",
    )
    assert model._inference_dense is dense_cache

    trainer.predict_on_batch(
        source,
        k=3,
        sparse_inference_backend="csr",
    )
    assert model._inference_csr is not None
    assert model._inference_dense is dense_cache


def test_compressed_decay_starts_only_after_mask_search(interactions):
    trainer = _compressed_trainer(decay=True).fit(interactions)
    search_lrs = [
        record["lr"]
        for record in trainer.history
        if record.get("phase") == "mask_search"
    ]
    finetune_lrs = [
        record["lr"]
        for record in trainer.history
        if record.get("phase") == "sparse_finetune"
    ]

    assert search_lrs == [trainer.cfg.lr, trainer.cfg.lr]
    assert finetune_lrs == [trainer.cfg.lr, trainer.cfg.lr / 2]


def test_refit_compressed_model_reuses_ticket(interactions):
    trainer = _compressed_trainer(epochs=1).fit(interactions)
    model = trainer.elsa
    search_records = sum(
        record.get("phase") == "mask_search"
        for record in trainer.history
    )

    trainer.fit(interactions)

    assert trainer.elsa is model
    assert isinstance(trainer.elsa, CompressedELSA)
    assert trainer.elsa.phase == "inference"
    assert (
        sum(
            record.get("phase") == "mask_search"
            for record in trainer.history
        )
        == search_records
    )
    assert (
        sum(
            record.get("phase") == "sparse_finetune"
            for record in trainer.history
        )
        == 2
    )


def test_build_is_idempotent_only_for_same_input_dimension():
    trainer = _trainer().build(8)
    model = trainer.elsa

    assert trainer.build(8) is trainer
    assert trainer.elsa is model
    with pytest.raises(ValueError, match="already built"):
        trainer.build(9)


def test_prediction_requires_fitted_model(source):
    with pytest.raises(RuntimeError, match="fitted"):
        _trainer().predict_on_batch(source, k=2)


def test_fit_and_prediction_validate_inputs(interactions, source):
    with pytest.raises(TypeError, match="csr_matrix"):
        _trainer().fit(csc_matrix(interactions))
    with pytest.raises(ValueError, match="at least one"):
        _trainer().fit(csr_matrix((0, interactions.shape[1])))
    with pytest.raises(ValueError, match="finite"):
        _trainer().fit(csr_matrix(np.array([[1.0, np.nan]])))

    trainer = _trainer(epochs=1).fit(interactions)
    with pytest.raises(TypeError, match="csr_matrix"):
        trainer.predict_on_batch(csc_matrix(source), k=2)
    with pytest.raises(ValueError, match="expected"):
        trainer.predict_on_batch(csr_matrix((1, source.shape[1] + 1)), k=1)
    with pytest.raises(ValueError, match=r"k must be in \[1"):
        trainer.predict_on_batch(source, k=0)
    with pytest.raises(ValueError, match="batch_size"):
        trainer.predict(source, k=2, batch_size=0)


def test_predict_matches_predict_on_batch_across_batch_sizes(interactions, source):
    trainer = _trainer(epochs=1).fit(interactions)

    expected = trainer.predict_on_batch(source, k=3)
    actual = trainer.predict(source, k=3, batch_size=2, show_progress=False)

    assert actual.shape == source.shape
    torch.testing.assert_close(actual.cols, expected.cols)
    torch.testing.assert_close(actual.vals, expected.vals)


def test_compressed_prediction_backend_override_preserves_rankings(
    interactions,
    source,
):
    trainer = _compressed_trainer(epochs=1).fit(interactions)

    csr_predictions = trainer.predict_on_batch(
        source,
        k=3,
        sparse_inference_backend="csr",
    )
    dense_predictions = trainer.predict_on_batch(
        source,
        k=3,
        sparse_inference_backend="dense",
    )
    streamed_dense_predictions = trainer.predict(
        source,
        k=3,
        batch_size=2,
        show_progress=False,
        sparse_inference_backend="dense",
    )

    torch.testing.assert_close(dense_predictions.cols, csr_predictions.cols)
    torch.testing.assert_close(dense_predictions.vals, csr_predictions.vals)
    torch.testing.assert_close(
        streamed_dense_predictions.cols,
        dense_predictions.cols,
    )
    torch.testing.assert_close(
        streamed_dense_predictions.vals,
        dense_predictions.vals,
    )


def test_dense_elsa_rejects_sparse_inference_backend(interactions, source):
    trainer = _trainer(epochs=1).fit(interactions)

    with pytest.raises(ValueError, match="only available for compressed ELSA"):
        trainer.predict_on_batch(
            source,
            k=3,
            sparse_inference_backend="dense",
        )
    with pytest.raises(ValueError, match="only available for compressed ELSA"):
        trainer.predict(
            source,
            k=3,
            show_progress=False,
            sparse_inference_backend="csr",
        )


def test_predictions_exclude_seen_items_by_default(interactions, source):
    predictions = _trainer(epochs=1).fit(interactions).predict(
        source,
        k=3,
        show_progress=False,
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()).isdisjoint(source[row].indices)


def test_seen_item_mask_can_be_disabled(interactions, source):
    trainer = _trainer(epochs=1).fit(interactions)

    predictions = trainer.predict(
        source,
        k=source.shape[1],
        batch_size=2,
        exclude_seen=False,
        show_progress=False,
    )

    for row in range(source.shape[0]):
        assert set(predictions.cols[row].tolist()) == set(range(source.shape[1]))


def test_prediction_rejects_rows_with_too_few_unseen_items(interactions):
    trainer = _trainer(epochs=1).fit(interactions)
    source = csr_matrix(
        np.array([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=np.float32)
    )

    with pytest.raises(ValueError, match="only 1 unseen"):
        trainer.predict_on_batch(source, k=2)


def test_empty_source_returns_empty_srp(interactions):
    trainer = _trainer(epochs=1).fit(interactions)
    source = csr_matrix((0, interactions.shape[1]), dtype=np.float32)

    predictions = trainer.predict(source, k=3, show_progress=False)

    assert predictions.shape == source.shape
    assert predictions.cols.shape == (0, 3)
    assert predictions.vals.shape == (0, 3)
    assert predictions.vals.dtype == torch.float32


def test_elsa_trainer_implements_recommender_protocol():
    assert isinstance(_trainer(), Recommender)


def test_streaming_evaluation_matches_materialized_predictions(
    interactions,
    source,
):
    trainer = _trainer(epochs=1).fit(interactions)
    targets = csr_matrix(
        (
            np.ones(source.shape[0], dtype=np.float32),
            (
                np.arange(source.shape[0]),
                np.array([1, 3, 2, 5]),
            ),
        ),
        shape=source.shape,
    )
    metrics = [CalibratedRecall([1, 3]), NDCG(3)]

    streamed = evaluate_recommender(
        trainer,
        source=source,
        targets=targets,
        metrics=metrics,
        batch_size=2,
    )
    materialized = evaluate_ranked_predictions(
        predictions=trainer.predict(
            source,
            k=3,
            batch_size=2,
            show_progress=False,
        ),
        targets=targets,
        metrics=metrics,
        batch_size=2,
    )

    assert streamed == materialized
