from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import NDCG
from compresso_recsys.models import (
    ItemTokenizer,
    SequenceBatcher,
    SimpleBidirectionalTransformer,
    SimpleBidirectionalTransformerConfig,
    SimpleBidirectionalTransformerTrainer,
    SimpleGPTTrainer,
    SimpleRNNTrainer,
    TransformerConfig,
)
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 6
PAD_ID = 0
VOCAB_SIZE = N_ITEMS + 2


def _backbone(**overrides) -> TransformerConfig:
    defaults = dict(d_model=16, n_heads=4, n_layers=2, dropout=0.0)
    return TransformerConfig(**{**defaults, **overrides})


def _config(**overrides) -> SimpleBidirectionalTransformerConfig:
    defaults = dict(
        transformer=_backbone(),
        tie_embeddings=False,
        epochs=1,
        batch_size=8,
        lr_schedule="constant",
        unk_dropout=0.0,
        show_progress=False,
        seed=0,
    )
    return SimpleBidirectionalTransformerConfig(**{**defaults, **overrides})


def _batcher(n_items=N_ITEMS, max_length=8) -> SequenceBatcher:
    return SequenceBatcher(ItemTokenizer(n_items), max_length=max_length)


def _sequences(rows, n_items=N_ITEMS) -> ItemSequences:
    return ItemSequences.from_rows(rows, n_items=n_items)


def _targets(rows, n_items=N_ITEMS) -> csr_matrix:
    return csr_matrix(np.asarray(rows, dtype=np.float32).reshape(-1, n_items))


def _model() -> SimpleBidirectionalTransformer:
    return SimpleBidirectionalTransformer(
        vocab_size=VOCAB_SIZE,
        n_items=N_ITEMS,
        max_positions=9,
        pad_id=PAD_ID,
        config=_backbone(),
        tie_embeddings=False,
    )


def _trainer(**config_overrides) -> SimpleBidirectionalTransformerTrainer:
    return SimpleBidirectionalTransformerTrainer(
        _config(**config_overrides), _batcher()
    )


def _force_item_to_the_top(
    trainer: SimpleBidirectionalTransformerTrainer, item: int
) -> None:
    assert trainer.model is not None and trainer.model.head is not None
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.zero_()
        trainer.model.head.bias[item] = 10.0


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


def test_cls_and_earlier_positions_can_read_a_later_token():
    model = _model().eval()
    mask = torch.ones((1, 3), dtype=torch.bool)
    tokens = torch.tensor([[2, 3, 4]])

    with torch.no_grad():
        before = model(tokens, mask)
        changed = tokens.clone()
        changed[0, -1] = 5
        after = model(changed, mask)

    assert not torch.allclose(before[:, 0], after[:, 0])
    assert not torch.allclose(before[:, 1], after[:, 1])


def test_padding_is_not_visible_to_real_positions():
    model = _model().eval()

    with torch.no_grad():
        short = model(
            torch.tensor([[2, 3]]),
            torch.tensor([[True, True]]),
        )
        padded = model(
            torch.tensor([[2, 3, PAD_ID, PAD_ID]]),
            torch.tensor([[True, True, False, False]]),
        )

    torch.testing.assert_close(short, padded[:, :3])


def test_model_shapes_and_catalog_head():
    model = _model()
    states = model(
        torch.tensor([[2, 3], [4, PAD_ID]]),
        torch.tensor([[True, True], [True, False]]),
    )

    assert states.shape == (2, 3, 16)
    assert model.score(states[:, 0]).shape == (2, N_ITEMS)
    assert model.head is not None and model.head.out_features == N_ITEMS


def test_tokens_and_mask_must_have_the_same_shape():
    with pytest.raises(ValueError, match="mask shape"):
        _model()(
            torch.tensor([[2, 3]]),
            torch.tensor([[True]]),
        )


def test_mask_must_be_boolean():
    with pytest.raises(TypeError, match="boolean dtype"):
        _model()(
            torch.tensor([[2, 3]]),
            torch.tensor([[1, 1]]),
        )


# --------------------------------------------------------------------------
# target contract and batching
# --------------------------------------------------------------------------


def test_only_the_target_capable_trainer_declares_targets():
    assert "targets" in inspect.signature(
        SimpleBidirectionalTransformerTrainer.fit
    ).parameters
    assert "targets" not in inspect.signature(SimpleGPTTrainer.fit).parameters
    assert "targets" not in inspect.signature(SimpleRNNTrainer.fit).parameters


def test_targets_must_be_csr():
    sequences = _sequences([[0], [1]])

    with pytest.raises(TypeError, match="targets must be a scipy.sparse.csr_matrix"):
        _trainer().fit(sequences, targets=np.ones((2, N_ITEMS)))


@pytest.mark.parametrize(
    "shape",
    [(1, N_ITEMS), (2, N_ITEMS - 1)],
)
def test_target_shape_must_match_both_sequence_dimensions(shape):
    sequences = _sequences([[0], [1]])

    with pytest.raises(ValueError) as error:
        _trainer().fit(sequences, targets=csr_matrix(shape))

    assert f"targets shape {shape}" in str(error.value)
    assert f"sequences shape {(2, N_ITEMS)}" in str(error.value)


def test_all_zero_rows_stay_aligned_but_do_not_contribute_to_the_loss():
    sequences = _sequences([[], [1], [2]])
    targets = _targets(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )

    model = _trainer().fit(sequences, targets=targets)

    assert model.history[0]["target_rows"] == 2.0


def test_a_dataset_without_any_training_target_is_refused():
    with pytest.raises(ValueError, match="no positive items"):
        _trainer().fit(
            _sequences([[], []]),
            targets=csr_matrix((2, N_ITEMS)),
        )


def test_sequence_and_target_rows_share_one_shuffle():
    sequences = _sequences([[0], [1], [2], [3]])
    target_rows = np.zeros((4, N_ITEMS), dtype=np.float32)
    target_rows[np.arange(4), np.arange(1, 5)] = 1
    trainer = _trainer(batch_size=2, seed=7)
    observed: list[tuple[int, int]] = []

    def capture(batch: ItemSequences, targets: csr_matrix):
        for row in range(batch.n_rows):
            observed.append(
                (int(batch.row(row)[0]), int(targets[row].indices[0]))
            )
        return 0.0, batch.n_rows

    trainer._train_step = capture  # type: ignore[method-assign]
    trainer.fit(sequences, targets=csr_matrix(target_rows))

    assert sorted(observed) == [(0, 1), (1, 2), (2, 3), (3, 4)]


def test_targets_are_binary_membership_not_weights():
    sequences = _sequences([[0], [1]])
    one = _targets([[0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0]])
    weighted = one.copy()
    weighted.data[:] = [19.0, -3.0]
    first = _trainer().fit(sequences, targets=one)
    second = _trainer().fit(sequences, targets=weighted)

    assert first.history == second.history
    assert first.model is not None and second.model is not None
    for left, right in zip(first.model.parameters(), second.model.parameters()):
        torch.testing.assert_close(left, right)


def test_none_targets_reconstruct_source_membership():
    trainer = _trainer()
    observed: list[list[int]] = []

    def capture(batch: ItemSequences, targets: csr_matrix):
        observed.extend(targets[row].indices.tolist() for row in range(targets.shape[0]))
        return 0.0, sum(bool(targets[row].nnz) for row in range(targets.shape[0]))

    trainer._train_step = capture  # type: ignore[method-assign]
    trainer.fit(_sequences([[2, 1, 2], [4]]))

    assert observed == [[1, 2], [4]]
    assert not trainer.trained_with_explicit_targets


# --------------------------------------------------------------------------
# learning and masking semantics
# --------------------------------------------------------------------------


def test_the_model_learns_an_explicit_source_to_target_mapping():
    rows = [[item] for _ in range(12) for item in range(N_ITEMS)]
    dense_targets = np.zeros((len(rows), N_ITEMS), dtype=np.float32)
    dense_targets[np.arange(len(rows)), [(row[0] + 1) % N_ITEMS for row in rows]] = 1
    trainer = SimpleBidirectionalTransformerTrainer(
        _config(
            transformer=_backbone(d_model=24, n_layers=1),
            epochs=50,
            batch_size=len(rows),
            lr=0.03,
            optimizer="AdamW",
        ),
        _batcher(),
    ).fit(_sequences(rows), targets=csr_matrix(dense_targets))

    probe = _sequences([[item] for item in range(N_ITEMS)])
    predictions = trainer.predict_on_batch(probe, k=1)

    assert predictions.cols[:, 0].tolist() == [1, 2, 3, 4, 5, 0]


def test_self_supervised_prediction_masks_source_items():
    source = _sequences([[1]])
    trainer = _trainer().fit(source)
    _force_item_to_the_top(trainer, 1)

    masked = trainer.predict_on_batch(source, k=1)
    unmasked = trainer.predict_on_batch(source, k=1, exclude_seen=False)

    assert masked.cols.item() != 1
    assert unmasked.cols.item() == 1


def test_explicit_target_training_keeps_source_items_eligible():
    source = _sequences([[1]])
    trainer = _trainer().fit(
        source,
        targets=_targets([[0, 1, 0, 0, 0, 0]]),
    )
    _force_item_to_the_top(trainer, 1)

    prediction = trainer.predict_on_batch(source, k=1)

    assert prediction.cols.item() == 1


def test_explicit_target_mode_does_not_apply_unseen_capacity_checks():
    source = _sequences([list(range(N_ITEMS))])
    trainer = _trainer().fit(
        source,
        targets=_targets([[1, 0, 0, 0, 0, 0]]),
    )

    assert trainer.predict_on_batch(source, k=N_ITEMS).cols.shape == (1, N_ITEMS)


# --------------------------------------------------------------------------
# persistence and evaluation
# --------------------------------------------------------------------------


def test_explicit_target_mode_and_predictions_survive_a_checkpoint(tmp_path):
    source = _sequences([[1], [2]])
    targets = _targets(
        [[0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]
    )
    trainer = _trainer().fit(source, targets=targets)
    _force_item_to_the_top(trainer, 1)
    before = trainer.predict_on_batch(source, k=3)
    path = tmp_path / "simple-bidirectional.zip"

    trainer.save(path, include_optimizer=True)
    restored = SimpleBidirectionalTransformerTrainer.load(
        path, load_optimizer=True
    )
    after = restored.predict_on_batch(source, k=3)

    assert restored.trained_with_explicit_targets
    assert restored.history == trainer.history
    assert restored.optimizer is not None
    torch.testing.assert_close(after.cols, before.cols)
    torch.testing.assert_close(after.vals, before.vals)


def test_self_supervised_mode_survives_a_checkpoint(tmp_path):
    trainer = _trainer().fit(_sequences([[0, 1], [2]]))
    path = tmp_path / "self-supervised.zip"

    trainer.save(path)
    restored = SimpleBidirectionalTransformerTrainer.load(path)

    assert not restored.trained_with_explicit_targets


def test_evaluation_needs_no_target_trained_model_special_case():
    source = _sequences([[1]])
    targets = _targets([[0, 1, 0, 0, 0, 0]])
    trainer = _trainer().fit(source, targets=targets)
    _force_item_to_the_top(trainer, 1)

    result = evaluate_recommender(
        trainer,
        source=source,
        targets=targets,
        metrics=[NDCG(1)],
        sample_ids=["user-1"],
    )

    assert result["ndcg@1"] == 1.0
