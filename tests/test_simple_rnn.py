from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso_recsys.models.simple_rnn import (
    SimpleRNN,
    SimpleRNNConfig,
    SimpleRNNTrainer,
)
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 8


def _seqs(rows, n_items=N_ITEMS):
    return ItemSequences.from_rows(rows, n_items=n_items)


def _cycle_rows(n_rows=48, length=4, n_items=N_ITEMS):
    """Histories drawn from a single cycle, so ``next(i) == (i + 1) % n_items``.

    Learnable from the last item alone, which is the point: it isolates whether
    the wiring works from whether the architecture is any good.
    """
    return [
        [(start + step) % n_items for step in range(length)]
        for start in range(n_rows)
    ]


def _fast_config(**overrides):
    defaults = dict(
        embedding_dim=16,
        hidden_dim=32,
        epochs=1,
        batch_size=16,
        show_progress=False,
        seed=0,
    )
    return SimpleRNNConfig(**{**defaults, **overrides})


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rnn_type": "transformer"}, "rnn_type must be"),
        ({"embedding_dim": 0}, "embedding_dim must be >= 1"),
        ({"hidden_dim": -1}, "hidden_dim must be >= 1"),
        ({"num_layers": 0}, "num_layers must be >= 1"),
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"epochs": 0}, "epochs must be >= 1"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
        ({"lr": 0.0}, "lr must be > 0"),
        ({"max_length": 1}, "max_length must be >= 2"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SimpleRNNConfig(**kwargs)


def test_max_length_may_be_disabled():
    assert SimpleRNNConfig(max_length=None).max_length is None


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
def test_both_recurrences_produce_states_of_the_configured_width(rnn_type):
    net = SimpleRNN(
        vocab_size=N_ITEMS + 1,
        n_items=N_ITEMS,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=1,
        dropout=0.0,
        rnn_type=rnn_type,
        pad_id=N_ITEMS,
    )

    states = net(torch.tensor([[1, 2, 3]]))

    assert states.shape == (1, 3, 6)
    assert net.score(states).shape == (1, 3, N_ITEMS)


def test_the_head_scores_the_catalog_not_the_vocabulary():
    """A special token is never a target, so an output column for it can only
    ever learn to be wrong — and would let a padding bug score plausibly."""
    net = SimpleRNN(
        vocab_size=N_ITEMS + 2,
        n_items=N_ITEMS,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=1,
        dropout=0.0,
        rnn_type="gru",
        pad_id=N_ITEMS,
    )

    assert net.head.out_features == N_ITEMS
    assert net.embedding.num_embeddings == N_ITEMS + 2
    assert net.embedding.padding_idx == N_ITEMS


def test_padding_embeds_as_zero_and_stays_there():
    net = SimpleRNN(
        vocab_size=N_ITEMS + 1,
        n_items=N_ITEMS,
        embedding_dim=4,
        hidden_dim=6,
        num_layers=1,
        dropout=0.0,
        rnn_type="gru",
        pad_id=N_ITEMS,
    )

    assert torch.all(net.embedding.weight[N_ITEMS] == 0)
    net.score(net(torch.tensor([[1, N_ITEMS]]))).sum().backward()
    assert torch.all(net.embedding.weight.grad[N_ITEMS] == 0)


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def test_fit_returns_the_trainer_and_reports_the_catalog():
    trainer = SimpleRNNTrainer(_fast_config())

    fitted = trainer.fit(_seqs(_cycle_rows(16)))

    assert fitted is trainer
    assert trainer.is_fitted
    assert trainer.n_items == N_ITEMS
    assert trainer.batcher is not None and trainer.batcher.n_items == N_ITEMS


def test_before_fitting_nothing_is_claimed():
    trainer = SimpleRNNTrainer(_fast_config())

    assert not trainer.is_fitted
    assert trainer.n_items is None


def test_fit_refuses_a_matrix_source():
    from scipy.sparse import csr_matrix

    with pytest.raises(TypeError, match="trains on ItemSequences"):
        SimpleRNNTrainer(_fast_config()).fit(csr_matrix((3, N_ITEMS)))


def test_fit_refuses_an_empty_training_set():
    with pytest.raises(ValueError, match="zero sequences"):
        SimpleRNNTrainer(_fast_config()).fit(_seqs([]))


def test_fit_refuses_data_with_no_next_item_example():
    """Every history of length one: nothing to predict from anything."""
    with pytest.raises(ValueError, match="two or more interactions"):
        SimpleRNNTrainer(_fast_config()).fit(_seqs([[1], [2], [3]]))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64])
def test_the_number_of_training_positions_is_batching_invariant(batch_size):
    """Positions come from the shift, so how rows are grouped cannot change it.

    Also the cheapest guard that padding never becomes a target: the head has no
    column for ``pad_id``, so a wrong shift would raise here rather than train.
    """
    rows = [[1, 2, 3], [4, 5], [6], []]  # 2 + 1 + 0 + 0 examples

    trainer = SimpleRNNTrainer(_fast_config(batch_size=batch_size)).fit(_seqs(rows))

    assert trainer.history[0]["positions"] == 3.0


def test_training_records_a_loss_per_epoch_and_reduces_it():
    trainer = SimpleRNNTrainer(
        _fast_config(epochs=40, lr=0.02, batch_size=48)
    ).fit(_seqs(_cycle_rows()))

    losses = [record["loss"] for record in trainer.history]

    assert len(losses) == 40
    assert [record["epoch"] for record in trainer.history] == [
        float(i) for i in range(40)
    ]
    assert losses[-1] < losses[0], losses[:3] + losses[-3:]
    # A cycle is fully determined by its last item, so the objective is
    # learnable to near zero and a plateau would mean the wiring is broken.
    assert losses[-1] < 0.5, losses[-1]


def test_refitting_starts_the_history_over():
    trainer = SimpleRNNTrainer(_fast_config(epochs=2))
    trainer.fit(_seqs(_cycle_rows(16)))

    trainer.fit(_seqs(_cycle_rows(16)))

    assert len(trainer.history) == 2


# --------------------------------------------------------------------------
# what it learns
# --------------------------------------------------------------------------


def _fitted_on_cycle(**overrides):
    config = _fast_config(epochs=60, lr=0.02, batch_size=48, **overrides)
    return SimpleRNNTrainer(config).fit(_seqs(_cycle_rows()))


@pytest.mark.parametrize("rnn_type", ["gru", "lstm"])
def test_it_predicts_the_successor_of_the_last_item(rnn_type):
    """End to end: encoding, the final state, the head and top-k together."""
    model = _fitted_on_cycle(rnn_type=rnn_type)
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0], [5]])

    predictions = model.predict(sources, k=1)

    expected = [(int(sources.row(i)[-1]) + 1) % N_ITEMS for i in range(4)]
    assert predictions.cols[:, 0].tolist() == expected


def test_prediction_is_batching_invariant():
    """The test that catches reading the padded last column.

    Batching changes the padded width, so a model scoring from ``states[:, -1]``
    agrees with itself at ``batch_size=1`` — where every row fills its own batch
    — and disagrees once short rows sit beside long ones.
    """
    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 2, 3, 4], [5], [6, 7], [2, 3, 4]])

    whole = model.predict(sources, k=4, batch_size=64)
    one_at_a_time = model.predict(sources, k=4, batch_size=1)

    assert whole.cols.tolist() == one_at_a_time.cols.tolist()
    torch.testing.assert_close(whole.vals, one_at_a_time.vals)


def test_order_changes_the_prediction():
    """Otherwise nothing here needed sequences in the first place."""
    model = _fitted_on_cycle()

    forward = model.predict(_seqs([[1, 2, 3]]), k=1, exclude_seen=False)
    reversed_ = model.predict(_seqs([[3, 2, 1]]), k=1, exclude_seen=False)

    assert forward.cols[0, 0].item() != reversed_.cols[0, 0].item()


def test_the_same_seed_gives_the_same_model():
    sources = _seqs([[0, 1, 2], [4, 5]])
    kwargs = dict(epochs=3, lr=0.02)

    first = SimpleRNNTrainer(_fast_config(seed=7, **kwargs)).fit(_seqs(_cycle_rows(16)))
    second = SimpleRNNTrainer(_fast_config(seed=7, **kwargs)).fit(_seqs(_cycle_rows(16)))

    assert first.history == second.history
    assert (
        first.predict(sources, k=3).cols.tolist()
        == second.predict(sources, k=3).cols.tolist()
    )


# --------------------------------------------------------------------------
# prediction contract
# --------------------------------------------------------------------------


def test_predict_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="must be fitted"):
        SimpleRNNTrainer(_fast_config()).predict_on_batch(_seqs([[1]]), k=1)


def test_predict_refuses_a_matrix_source():
    from scipy.sparse import csr_matrix

    model = SimpleRNNTrainer(_fast_config()).fit(_seqs(_cycle_rows(16)))

    with pytest.raises(TypeError, match="predicts from ItemSequences"):
        model.predict(csr_matrix((2, N_ITEMS)), k=2)


@pytest.mark.parametrize("k", [0, N_ITEMS + 1])
def test_predict_refuses_an_impossible_k(k):
    model = SimpleRNNTrainer(_fast_config()).fit(_seqs(_cycle_rows(16)))

    with pytest.raises(ValueError, match="k must be in"):
        model.predict_on_batch(_seqs([[1, 2]]), k=k)


def test_predictions_are_shaped_and_ordered():
    model = SimpleRNNTrainer(_fast_config()).fit(_seqs(_cycle_rows(16)))

    predictions = model.predict(_seqs([[1, 2], [3]]), k=4)

    assert predictions.cols.shape == (2, 4)
    assert predictions.vals.shape == (2, 4)
    assert predictions.shape == (2, N_ITEMS)
    for row in range(2):
        values = predictions.vals[row].tolist()
        assert values == sorted(values, reverse=True)


def test_no_rows_at_all():
    model = SimpleRNNTrainer(_fast_config()).fit(_seqs(_cycle_rows(16)))

    predictions = model.predict_on_batch(_seqs([]), k=3)

    assert predictions.cols.shape == (0, 3)


def test_an_empty_history_is_predictable_and_gives_every_row_the_same_prior():
    """No state to read, so the answer is whatever the model believes a priori."""
    model = _fitted_on_cycle()

    predictions = model.predict(_seqs([[], []]), k=N_ITEMS)

    assert predictions.cols[0].tolist() == predictions.cols[1].tolist()


# --------------------------------------------------------------------------
# exclude_seen
# --------------------------------------------------------------------------


def test_exclude_seen_masks_the_whole_history_including_repeats():
    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 1, 2], [4, 5]])

    predictions = model.predict(sources, k=N_ITEMS - 4, exclude_seen=True)

    for row in range(sources.n_rows):
        seen = set(sources.row(row).tolist())
        assert not seen & set(predictions.cols[row].tolist())


def test_exclude_seen_masks_what_truncation_dropped():
    """Truncation bounds the model's memory, not what it is allowed to return.

    With ``max_length=2`` the encoder reads only ``[3, 4]``, but the earlier
    items are still part of this user's history and must not come back as
    recommendations.
    """
    model = _fitted_on_cycle(max_length=2)
    history = [0, 1, 2, 3, 4]

    predictions = model.predict(_seqs([history]), k=N_ITEMS - len(history))

    assert not set(history) & set(predictions.cols[0].tolist())
    assert model.batcher is not None
    assert model.batcher.truncated_lengths(_seqs([history])).tolist() == [2]


def test_exclude_seen_false_may_return_the_history():
    model = _fitted_on_cycle()

    predictions = model.predict(_seqs([[0, 1, 2, 3, 4, 5, 6]]), k=N_ITEMS)

    assert sorted(predictions.cols[0].tolist()) == list(range(N_ITEMS))


def test_masking_does_not_mutate_the_source():
    model = _fitted_on_cycle()
    sources = _seqs([[1, 2, 3]])
    before = sources.values.copy()

    model.predict(sources, k=2, exclude_seen=True)

    assert np.array_equal(sources.values, before)


# --------------------------------------------------------------------------
# integration with evaluation
# --------------------------------------------------------------------------


def test_it_evaluates_through_the_standard_entry_point():
    from scipy.sparse import csr_matrix

    from compresso_recsys.evaluation import evaluate_recommender
    from compresso_recsys.metrics import NDCG

    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
    # The true successor of each history's last item.
    wanted = [3, 6, 1]
    targets = csr_matrix(
        (
            np.ones(len(wanted), dtype=np.float32),
            (np.arange(len(wanted)), np.array(wanted)),
        ),
        shape=(len(wanted), N_ITEMS),
    )

    result = evaluate_recommender(
        model, source=sources, targets=targets, metrics=[NDCG(1)]
    )

    assert result.n_scored_rows == 3
    assert result["ndcg@1"] == pytest.approx(1.0)
