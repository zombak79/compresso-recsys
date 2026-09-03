from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.models import (
    MultDAEConfig,
    MultDAETrainer,
    MultVAEConfig,
    MultVAETrainer,
)
from compresso_recsys.models._autoencoder_batching import (
    prepare_dense_training_data,
)


class _RecordingLogger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)


def _interactions(*, dtype=np.float32) -> csr_matrix:
    return csr_matrix(np.eye(3, dtype=dtype))


def test_preload_casts_sparse_data_before_densifying(monkeypatch):
    dtypes: list[np.dtype] = []
    original_toarray = csr_matrix.toarray

    def recording_toarray(self, *args, **kwargs):
        dtypes.append(self.dtype)
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", recording_toarray)

    result = prepare_dense_training_data(
        _interactions(dtype=np.float64),
        device=torch.device("cpu"),
        preload=True,
    )

    assert dtypes == [np.dtype(np.float32)]
    assert result is not None
    assert result.dtype == torch.float32


@pytest.mark.parametrize(
    ("trainer_class", "config", "variational"),
    [
        (
            MultDAETrainer,
            MultDAEConfig(
                latent_dim=2,
                epochs=1,
                batch_size=2,
                show_progress=False,
            ),
            False,
        ),
        (
            MultVAETrainer,
            MultVAEConfig(
                latent_dim=2,
                hidden_dim=3,
                epochs=1,
                batch_size=2,
                show_progress=False,
            ),
            True,
        ),
    ],
)
def test_prediction_avoids_catalog_wide_conversion_copies(
    monkeypatch,
    trainer_class,
    config,
    variational,
):
    trainer = trainer_class(config).fit(_interactions())
    logits = torch.arange(9, dtype=torch.float32).reshape(3, 3)
    assert trainer.model is not None

    def dae_forward(inputs):
        del inputs
        return logits

    def vae_forward(inputs, *, sample=True):
        del inputs, sample
        return logits, torch.empty(0), torch.empty(0)

    model_forward = vae_forward if variational else dae_forward
    monkeypatch.setattr(trainer.model, "forward", model_forward)

    dense_dtypes: list[np.dtype] = []
    original_toarray = csr_matrix.toarray

    def recording_toarray(self, *args, **kwargs):
        dense_dtypes.append(self.dtype)
        return original_toarray(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", recording_toarray)

    topk_inputs: list[torch.Tensor] = []
    original_topk = torch.topk

    def recording_topk(tensor, *args, **kwargs):
        topk_inputs.append(tensor)
        return original_topk(tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", recording_topk)

    source = _interactions(dtype=np.float64)
    trainer.predict_on_batch(source, k=1, exclude_seen=False)

    assert dense_dtypes == [np.dtype(np.float32)]
    assert len(topk_inputs) == 1
    assert topk_inputs[0] is logits


@pytest.mark.parametrize(
    ("trainer_class", "config"),
    [
        (
            MultDAETrainer,
            MultDAEConfig(
                latent_dim=2,
                epochs=1,
                batch_size=2,
                show_progress=False,
            ),
        ),
        (
            MultVAETrainer,
            MultVAEConfig(
                latent_dim=2,
                hidden_dim=3,
                epochs=1,
                batch_size=2,
                show_progress=False,
            ),
        ),
    ],
)
def test_host_preload_oom_is_contextual_and_follows_fit_start(
    monkeypatch,
    trainer_class,
    config,
):
    logger = _RecordingLogger()
    trainer = trainer_class(config, logger=logger)

    def fail_toarray(self, *args, **kwargs):
        del self, args, kwargs
        raise MemoryError("injected host allocation failure")

    monkeypatch.setattr(csr_matrix, "toarray", fail_toarray)

    with pytest.raises(
        MemoryError,
        match=r"host memory; set preload_training_data=False",
    ) as raised:
        trainer.fit(_interactions())

    assert isinstance(raised.value.__cause__, MemoryError)
    assert len(logger.lines) == 1
    assert logger.lines[0].startswith(f"[{config.log_prefix}] fit started:")
    assert not trainer.is_fitted


def test_device_preload_oom_keeps_streaming_guidance(monkeypatch):
    def fail_from_numpy(dense):
        del dense
        raise torch.OutOfMemoryError("injected device allocation failure")

    monkeypatch.setattr(torch, "from_numpy", fail_from_numpy)

    with pytest.raises(
        torch.OutOfMemoryError,
        match=r"set preload_training_data=False",
    ) as raised:
        prepare_dense_training_data(
            _interactions(),
            device=torch.device("cpu"),
            preload=True,
        )

    assert isinstance(raised.value.__cause__, torch.OutOfMemoryError)


def test_mps_runtime_oom_keeps_streaming_guidance(monkeypatch):
    class FailedTransfer:
        def to(self, device):
            del device
            raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(torch, "from_numpy", lambda dense: FailedTransfer())

    with pytest.raises(
        torch.OutOfMemoryError,
        match=r"set preload_training_data=False",
    ) as raised:
        prepare_dense_training_data(
            _interactions(),
            device=torch.device("mps"),
            preload=True,
        )

    assert isinstance(raised.value.__cause__, RuntimeError)


def test_unrelated_mps_runtime_error_is_not_reclassified(monkeypatch):
    failure = RuntimeError("MPS operation is unsupported")

    class FailedTransfer:
        def to(self, device):
            del device
            raise failure

    monkeypatch.setattr(torch, "from_numpy", lambda dense: FailedTransfer())

    with pytest.raises(RuntimeError, match="MPS operation is unsupported") as raised:
        prepare_dense_training_data(
            _interactions(),
            device=torch.device("mps"),
            preload=True,
        )

    assert raised.value is failure
