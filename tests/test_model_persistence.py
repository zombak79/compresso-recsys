from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from scipy.sparse import csr_matrix, issparse

from compresso import SRPTensor
from compresso_recsys import ItemSequences
from compresso_recsys.checkpoint import (
    load_manifest,
    read_checkpoint,
    update_checkpoint,
)
from compresso_recsys.models import (
    ContentRecommender,
    BaseCollaborativeRecommender,
    EASE,
    ELSACompressionConfig,
    ELSAConfig,
    ELSATrainer,
    ItemTokenizer,
    SequenceBatcher,
    SimpleGPTConfig,
    SimpleGPTTrainer,
    SimpleRNNConfig,
    SimpleRNNTrainer,
    TEASER,
    TEASERConfig,
    TEASERGDConfig,
    TEASERGDTrainer,
    TransformerConfig,
    WarmCatalogAdapter,
)
from compresso_recsys.persistence import (
    MODEL_CHECKPOINT_FORMAT,
    MODEL_CHECKPOINT_VERSION,
    ModelCheckpointReader,
)


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
            dtype=np.float32,
        )
    )


@pytest.fixture
def source() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
            dtype=np.float32,
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
        dtype=np.float32,
    )


def _assert_same_predictions(before, after) -> None:
    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)
    assert after.shape == before.shape


def _assert_same_features(left, right) -> None:
    if issparse(left) or issparse(right):
        np.testing.assert_array_equal(csr_matrix(left).toarray(), csr_matrix(right).toarray())
    else:
        np.testing.assert_array_equal(left, right)


def _sequence_data() -> ItemSequences:
    return ItemSequences.from_rows(
        [[0, 1, 2], [1, 3], [2, 4, 5], [0, 5]],
        n_items=6,
    )


@dataclass(frozen=True)
class _PureTorchConfig:
    n_items: int


class _PureTorchRecommender(nn.Module, BaseCollaborativeRecommender):
    """Test model using inherited config and Torch-state persistence only."""

    checkpoint_type = "test_pure_torch"

    def __init__(self, config: _PureTorchConfig) -> None:
        nn.Module.__init__(self)
        self.cfg = config
        self.weight = nn.Parameter(torch.arange(config.n_items, dtype=torch.float32))
        self._is_fitted = False

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def n_items(self) -> int:
        return self.cfg.n_items

    def fit(self, interactions: csr_matrix) -> "_PureTorchRecommender":
        if interactions.shape[1] != self.n_items:
            raise ValueError("interaction width must match n_items")
        self._is_fitted = True
        return self

    def predict_on_batch(
        self,
        source: csr_matrix,
        *,
        k: int,
        exclude_seen: bool = True,
    ) -> SRPTensor:
        del exclude_seen
        scores = self.weight.detach().expand(source.shape[0], -1)
        values, columns = torch.topk(scores, k=k, dim=1)
        return SRPTensor(cols=columns, vals=values, shape=source.shape)

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> "_PureTorchRecommender":
        del reader
        model = cls(_PureTorchConfig(**config)).to(device)
        model._is_fitted = True
        return model


def test_pure_torch_model_needs_no_custom_state_serializer(tmp_path):
    interactions = csr_matrix(np.eye(4, dtype=np.float32))
    model = _PureTorchRecommender(_PureTorchConfig(n_items=4)).fit(interactions)
    with torch.no_grad():
        model.weight.add_(3.5)
    path = tmp_path / "pure-torch.ckpt"

    model.save(path)
    restored = _PureTorchRecommender.load(path)

    torch.testing.assert_close(restored.weight, model.weight)
    _assert_same_predictions(
        model.predict_on_batch(interactions, k=2),
        restored.predict_on_batch(interactions, k=2),
    )


def test_ease_round_trip_and_manifest(tmp_path, interactions, source):
    model = EASE().fit(interactions)
    before = model.predict_on_batch(source, k=2)
    path = tmp_path / "ease.ckpt"

    model.save(path)
    restored = EASE.load(path)

    _assert_same_predictions(before, restored.predict_on_batch(source, k=2))
    with ModelCheckpointReader(path, expected_model_type="ease") as reader:
        assert reader.manifest == {
            "format": MODEL_CHECKPOINT_FORMAT,
            "version": MODEL_CHECKPOINT_VERSION,
            "model_type": "ease",
            "optimizer_included": False,
        }


def test_job_logger_is_not_persisted(tmp_path, interactions):
    class RecordingLogger:
        def info(self, message: str) -> None:
            del message

    model = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            batch_size=2,
            max_output=4,
            epochs=1,
            show_progress=True,
            log_prefix="job-17",
            log_every_n_steps=7,
        ),
        logger=RecordingLogger(),
    ).fit(interactions)
    path = tmp_path / "logger.ckpt"

    model.save(path)
    restored = ELSATrainer.load(path)

    assert restored.logger is None
    assert restored.cfg.log_prefix == "job-17"
    assert restored.cfg.log_every_n_steps == 7


@pytest.mark.parametrize("kind", ["rnn", "gpt"])
def test_sequential_round_trip_preserves_vocabulary_and_context(tmp_path, kind):
    sequences = _sequence_data()
    tokenizer = ItemTokenizer(
        sequences.n_items,
        item_ids=np.array([f"item-{i}" for i in range(sequences.n_items)]),
    )
    batcher = SequenceBatcher(tokenizer, max_length=3)
    if kind == "rnn":
        model = SimpleRNNTrainer(
            SimpleRNNConfig(
                embedding_dim=8,
                hidden_dim=12,
                epochs=1,
                batch_size=2,
                show_progress=False,
            ),
            batcher,
        ).fit(sequences)
        model_class = SimpleRNNTrainer
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
            ),
            batcher,
        ).fit(sequences)
        model_class = SimpleGPTTrainer
    before = model.predict_on_batch(sequences, k=3, exclude_seen=False)
    before_recommendations = model.recommend(
        [["item-0", "item-1"]],
        k=3,
        exclude_seen=False,
    )
    path = tmp_path / f"{kind}.ckpt"

    model.save(path)
    restored = model_class.load(path)

    _assert_same_predictions(
        before,
        restored.predict_on_batch(sequences, k=3, exclude_seen=False),
    )
    later_sequences = ItemSequences.from_rows(
        [[0, sequences.n_items, 1]],
        n_items=sequences.n_items + 1,
    )
    _assert_same_predictions(
        model.predict_on_batch(later_sequences, k=3, exclude_seen=False),
        restored.predict_on_batch(later_sequences, k=3, exclude_seen=False),
    )
    assert restored.history == model.history
    assert restored.batcher.max_length == 3
    np.testing.assert_array_equal(restored.batcher.tokenizer.item_ids, tokenizer.item_ids)
    after_recommendations = restored.recommend(
        [["item-0", "item-1"]],
        k=3,
        exclude_seen=False,
    )
    np.testing.assert_array_equal(
        after_recommendations.item_ids,
        before_recommendations.item_ids,
    )
    np.testing.assert_allclose(
        after_recommendations.scores,
        before_recommendations.scores,
    )


def test_optimizer_state_is_explicitly_optional(tmp_path):
    sequences = _sequence_data()
    model = SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=8,
            hidden_dim=12,
            epochs=1,
            batch_size=2,
            show_progress=False,
        )
    ).fit(sequences)
    with_optimizer = tmp_path / "with-optimizer.ckpt"
    without_optimizer = tmp_path / "without-optimizer.ckpt"

    model.save(with_optimizer, include_optimizer=True)
    model.save(without_optimizer)

    inference_model = SimpleRNNTrainer.load(with_optimizer)
    resumed_model = SimpleRNNTrainer.load(
        with_optimizer,
        load_optimizer=True,
    )
    assert inference_model.optimizer is None
    assert resumed_model.optimizer is not None
    assert resumed_model.optimizer.state_dict()["state"]
    with pytest.raises(ValueError, match="does not contain optimizer"):
        SimpleRNNTrainer.load(without_optimizer, load_optimizer=True)
    with pytest.raises(ValueError, match="has no optimizer"):
        EASE().fit(
            csr_matrix(np.eye(2, dtype=np.float32))
        ).save(tmp_path / "ease.ckpt", include_optimizer=True)


def test_model_round_trips_inside_a_data_checkpoint(tmp_path):
    sequences = _sequence_data()
    model = SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=8,
            hidden_dim=12,
            epochs=1,
            batch_size=2,
            show_progress=False,
        )
    ).fit(sequences)
    before = model.predict_on_batch(sequences, k=3, exclude_seen=False)
    checkpoint = tmp_path / "dataset.zip"
    with update_checkpoint(checkpoint) as root:
        (root / "sentinel.txt").write_text("dataset", encoding="utf-8")

    model.save_to_checkpoint(checkpoint, "gru", include_optimizer=True)

    with read_checkpoint(checkpoint) as root:
        assert (root / "sentinel.txt").read_text(encoding="utf-8") == "dataset"
        assert (root / "models" / "gru.zip").is_file()
        manifest = load_manifest(root)
    assert manifest["models"]["gru"] == {
        "path": "models/gru.zip",
        "model_type": "simple_rnn_trainer",
        "optimizer_included": True,
    }

    restored = SimpleRNNTrainer.load_from_checkpoint(
        checkpoint,
        "gru",
        load_optimizer=True,
    )
    assert restored.optimizer is not None
    _assert_same_predictions(
        before,
        restored.predict_on_batch(sequences, k=3, exclude_seen=False),
    )


def test_embedded_model_can_be_replaced_by_the_same_type(tmp_path):
    sequences = _sequence_data()
    model = SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=8,
            hidden_dim=12,
            epochs=1,
            batch_size=2,
            show_progress=False,
        )
    ).fit(sequences)
    checkpoint = tmp_path / "dataset.zip"
    with update_checkpoint(checkpoint):
        pass

    model.save_to_checkpoint(checkpoint, "primary", include_optimizer=True)
    model.save_to_checkpoint(checkpoint, "primary")

    restored = SimpleRNNTrainer.load_from_checkpoint(checkpoint, "primary")
    assert restored.optimizer is None
    with pytest.raises(ValueError, match="does not contain optimizer"):
        SimpleRNNTrainer.load_from_checkpoint(
            checkpoint,
            "primary",
            load_optimizer=True,
        )


def test_embedded_model_name_cannot_change_model_type(tmp_path, interactions):
    checkpoint = tmp_path / "dataset.zip"
    with update_checkpoint(checkpoint):
        pass
    ease = EASE().fit(interactions)
    ease.save_to_checkpoint(checkpoint, "primary")

    rnn = SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=8,
            hidden_dim=12,
            epochs=1,
            batch_size=2,
            show_progress=False,
        )
    ).fit(_sequence_data())
    with pytest.raises(ValueError, match="not 'simple_rnn_trainer'"):
        rnn.save_to_checkpoint(checkpoint, "primary")
    with pytest.raises(ValueError, match="not 'simple_rnn_trainer'"):
        SimpleRNNTrainer.load_from_checkpoint(checkpoint, "primary")


@pytest.mark.parametrize("name", ["", "../gru", "models/gru", "gru.zip", "gru model"])
def test_embedded_model_names_are_safe(tmp_path, interactions, name):
    checkpoint = tmp_path / "dataset.zip"
    with update_checkpoint(checkpoint):
        pass
    model = EASE().fit(interactions)

    with pytest.raises(ValueError, match="model name"):
        model.save_to_checkpoint(checkpoint, name)
    with pytest.raises(ValueError, match="model name"):
        EASE.load_from_checkpoint(checkpoint, name)


def test_embedded_model_requires_an_existing_data_checkpoint(tmp_path, interactions):
    checkpoint = tmp_path / "missing.zip"
    model = EASE().fit(interactions)

    with pytest.raises(FileNotFoundError):
        model.save_to_checkpoint(checkpoint, "ease")
    with pytest.raises(FileNotFoundError):
        EASE.load_from_checkpoint(checkpoint, "ease")


def test_loading_defaults_to_cpu_even_if_saved_config_says_cuda(tmp_path):
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
    ).fit(_sequence_data())
    path = tmp_path / "cuda-config.ckpt"
    model.save(path)
    with update_checkpoint(path) as root:
        config_path = root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["device"] = "cuda"
        config_path.write_text(json.dumps(config), encoding="utf-8")

    restored = SimpleGPTTrainer.load(path)

    assert restored.device == torch.device("cpu")
    assert restored.cfg.device == "cpu"
    assert all(parameter.device.type == "cpu" for parameter in restored.model.parameters())


@pytest.mark.parametrize(
    "model",
    [
        ContentRecommender(),
        ELSATrainer(),
        TEASERGDTrainer(),
        SimpleRNNTrainer(),
        SimpleGPTTrainer(),
    ],
    ids=["content", "elsa", "teaser-gd", "simple-rnn", "simple-gpt"],
)
def test_torch_backed_recommenders_share_the_to_contract(model):
    assert model.to("cpu") is model
    assert model.device == torch.device("cpu")
    assert model.cfg.device == "cpu"


def test_numpy_only_recommenders_reject_device_transfer():
    with pytest.raises(TypeError, match="no device-backed state"):
        EASE().to("cuda")


@pytest.mark.parametrize("compressed", [False, True])
def test_dense_and_compressed_elsa_round_trip(
    tmp_path,
    interactions,
    source,
    compressed,
):
    compression = (
        ELSACompressionConfig(
            k_target=2,
            k_schedule=(4, 2),
            stability_window=1,
            change_threshold=100.0,
            mask_update_interval=1,
        )
        if compressed
        else None
    )
    model = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            epochs=1,
            batch_size=2,
            max_output=6,
            show_progress=False,
            compression=compression,
        )
    ).fit(interactions)
    before = model.predict_on_batch(source, k=2)
    path = tmp_path / f"elsa-{compressed}.ckpt"

    model.save(path, include_optimizer=True)
    restored = ELSATrainer.load(path, load_optimizer=True)

    _assert_same_predictions(before, restored.predict_on_batch(source, k=2))
    assert restored.history == model.history
    assert restored.optimizer is not None


def _mutate_candidates(model, item_features):
    model.update_candidates(
        item_ids=["item-6"],
        item_features=np.array([[0.25, 0.5, 1.0]], dtype=np.float32),
        metadata=pd.DataFrame({"item_id": ["item-6"], "label": ["new"]}),
    )
    model.remove_candidates(["item-5"])
    return model


@pytest.mark.parametrize("kind", ["content", "teaser", "teaser_gd"])
def test_cold_start_round_trip_preserves_mutated_catalog(
    tmp_path,
    interactions,
    source,
    item_features,
    kind,
):
    item_ids = np.array([f"item-{i}" for i in range(item_features.shape[0])])
    if kind == "content":
        model = ContentRecommender().fit(item_features, item_ids=item_ids)
        model_class = ContentRecommender
    elif kind == "teaser":
        model = TEASER(
            TEASERConfig(max_iterations=2, dtype="float32")
        ).fit(interactions, item_features, item_ids=item_ids)
        model_class = TEASER
    else:
        model = TEASERGDTrainer(
            TEASERGDConfig(
                epochs=1,
                batch_size=2,
                coefficient_regularization_samples=4,
                show_progress=False,
            )
        ).fit(interactions, item_features, item_ids=item_ids)
        model_class = TEASERGDTrainer
    _mutate_candidates(model, item_features)
    before_predictions = model.predict_on_batch(source, k=2)
    before_catalog = model.candidates.snapshot()
    path = tmp_path / f"{kind}.ckpt"

    model.save(path)
    restored = model_class.load(path)
    after_catalog = restored.candidates.snapshot()
    if kind == "content":
        assert restored._candidate_cache is None
    elif kind == "teaser_gd":
        assert restored._candidate_tensor_cache is None
        assert restored._training_tensor_cache is None

    _assert_same_predictions(
        before_predictions,
        restored.predict_on_batch(source, k=2),
    )
    np.testing.assert_array_equal(after_catalog.item_ids, before_catalog.item_ids)
    _assert_same_features(after_catalog.item_features, before_catalog.item_features)
    pd.testing.assert_frame_equal(after_catalog.metadata, before_catalog.metadata)
    assert after_catalog.version == before_catalog.version


def test_loading_through_the_wrong_class_is_refused(tmp_path, interactions):
    path = tmp_path / "ease.ckpt"
    EASE().fit(interactions).save(path)

    with pytest.raises(ValueError, match="model type 'ease'"):
        SimpleRNNTrainer.load(path)


def test_an_unknown_checkpoint_format_is_refused(tmp_path, interactions):
    path = tmp_path / "ease.ckpt"
    EASE().fit(interactions).save(path)
    with update_checkpoint(path) as root:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["format"] = "some.other.format"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="not a Compresso Recsys model checkpoint"):
        EASE.load(path)


def test_custom_object_item_ids_are_refused_without_pickle(tmp_path, item_features):
    class CustomID:
        pass

    ids = np.empty(item_features.shape[0], dtype=object)
    ids[:] = [CustomID() for _ in range(item_features.shape[0])]
    model = ContentRecommender().fit(item_features, item_ids=ids)

    with pytest.raises(TypeError, match="strings, integers"):
        model.save(tmp_path / "custom-ids.ckpt")


def test_warm_catalog_adapter_is_rebuilt_instead_of_persisted(interactions):
    model = EASE().fit(interactions)
    adapter = WarmCatalogAdapter(
        model,
        train_item_ids=np.arange(interactions.shape[1]),
        catalog_item_ids=np.arange(interactions.shape[1]),
    )

    assert not hasattr(adapter, "save")
