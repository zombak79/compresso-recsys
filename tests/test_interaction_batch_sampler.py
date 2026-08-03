from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.sparse import csc_matrix, csr_matrix

from compresso_recsys.models import InteractionBatch, InteractionBatchSampler


@pytest.fixture
def interactions() -> csr_matrix:
    return csr_matrix(
        np.asarray(
            [
                [1, 0, 1, 0, 0],
                [0, 1, 1, 0, 0],
                [0, 0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )


def test_sampler_packs_sources_as_candidate_prefix(interactions):
    sampler = InteractionBatchSampler(
        interactions,
        device="cpu",
        batch_size=2,
        shuffle=False,
        max_output=4,
        seed=7,
    )

    batch = sampler[0]

    assert isinstance(batch, InteractionBatch)
    assert batch.x.layout == torch.sparse_coo
    torch.testing.assert_close(
        batch.x.to_dense(),
        torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32),
    )
    torch.testing.assert_close(batch.sources, torch.tensor([0, 1, 2]))
    assert batch.candidates is not None
    torch.testing.assert_close(batch.candidates[:3], batch.sources)
    assert batch.candidates.numel() == 4
    assert len(sampler) == 2


def test_sampler_uses_none_to_request_full_output_catalog(interactions):
    sampler = InteractionBatchSampler(
        interactions,
        device=torch.device("cpu"),
        batch_size=3,
        shuffle=False,
        max_output=None,
        seed=0,
    )

    batch = sampler[0]

    assert batch.candidates is None
    torch.testing.assert_close(batch.sources, torch.arange(5))


def test_sampler_is_reproducible_across_epochs(interactions):
    kwargs = {
        "device": "cpu",
        "batch_size": 2,
        "shuffle": True,
        "max_output": 4,
        "seed": 13,
    }
    first = InteractionBatchSampler(interactions, **kwargs)
    second = InteractionBatchSampler(interactions, **kwargs)

    for _ in range(2):
        torch.testing.assert_close(first[0].sources, second[0].sources)
        torch.testing.assert_close(first[0].candidates, second[0].candidates)
        first.on_epoch_end()
        second.on_epoch_end()


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_sampler_rejects_invalid_batch_size(interactions, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        InteractionBatchSampler(
            interactions,
            device="cpu",
            batch_size=batch_size,
            shuffle=False,
            max_output=None,
            seed=0,
        )


@pytest.mark.parametrize("max_output", [0, -1, 1.5, True])
def test_sampler_rejects_invalid_max_output(interactions, max_output):
    with pytest.raises(ValueError, match="max_output"):
        InteractionBatchSampler(
            interactions,
            device="cpu",
            batch_size=2,
            shuffle=False,
            max_output=max_output,
            seed=0,
        )


def test_sampler_rejects_non_csr_and_empty_interactions(interactions):
    with pytest.raises(TypeError, match="csr_matrix"):
        InteractionBatchSampler(
            csc_matrix(interactions),
            device="cpu",
            batch_size=2,
            shuffle=False,
            max_output=None,
            seed=0,
        )
    with pytest.raises(ValueError, match="at least one user and one item"):
        InteractionBatchSampler(
            csr_matrix((0, 5), dtype=np.float32),
            device="cpu",
            batch_size=2,
            shuffle=False,
            max_output=None,
            seed=0,
        )
