from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
from torch import nn

from compresso_recsys.models.sasrec import (
    LAYER_NORM_EPS,
    SASRec,
    SASRecConfig,
    SASRecTrainer,
)
from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 8
PAD_ID = 0
N_RESERVED = 1
MAX_POSITIONS = 8


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


def _batcher(max_length=None, **kwargs):
    return SequenceBatcher(ItemTokenizer(N_ITEMS), max_length=max_length, **kwargs)


def _config(**overrides):
    defaults = dict(
        d_model=16,
        n_blocks=2,
        n_heads=2,
        max_history_length=6,
        epochs=1,
        batch_size=16,
        show_progress=False,
        seed=0,
    )
    return SASRecConfig(**{**defaults, **overrides})


def _fitted_on_cycle(epochs=60, **overrides):
    config = _config(epochs=epochs, lr=0.01, batch_size=48, **overrides)
    return SASRecTrainer(config).fit(_seqs(_cycle_rows()))


def _model(**overrides):
    defaults = dict(
        n_items=N_ITEMS,
        n_reserved=N_RESERVED,
        max_history_length=MAX_POSITIONS,
        pad_id=PAD_ID,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
    )
    return SASRec(**{**defaults, **overrides})


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(d_model=0), "d_model must be >= 1"),
        (dict(n_blocks=0), "n_blocks must be >= 1"),
        (dict(n_negatives=0), "n_negatives must be >= 1"),
        (dict(max_history_length=0), "max_history_length must be >= 1"),
        (dict(d_model=10, n_heads=4), "divisible by n_heads"),
        (dict(epochs=0), "epochs must be >= 1"),
        (dict(dropout=1.0), r"dropout must be in \[0, 1\)"),
        (dict(unk_dropout=-0.1), r"unk_dropout must be in \[0, 1\)"),
        (dict(lr=0.0), "lr must be > 0"),
        (dict(optimizer="NAdam"), "optimizer must be 'Adam'"),
        (dict(betas=(0.9,)), "betas must be two values"),
        (dict(betas=(0.9, 1.0)), r"betas must each be in \[0, 1\)"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_the_defaults_are_the_papers_movielens_settings():
    """The whole point of this model, so the numbers are pinned rather than left
    to drift back toward whatever a modern default would be."""
    cfg = SASRecConfig()

    assert (cfg.d_model, cfg.n_blocks, cfg.n_heads) == (50, 2, 1)
    assert (cfg.dropout, cfg.max_history_length) == (0.2, 200)
    assert (cfg.batch_size, cfg.epochs, cfg.lr) == (128, 201, 0.001)
    assert (cfg.optimizer, cfg.betas) == ("Adam", (0.9, 0.98))
    assert cfg.n_negatives == 1


def test_betas_belong_to_adam_and_reach_the_optimizer_through_the_config():
    """Not every optimizer takes betas, so they are selected by name rather than
    handed to whatever ``torch.optim`` resolves to."""
    cfg = SASRecConfig()

    assert cfg.optimizer_kwargs() == {"betas": (0.9, 0.98)}

    built = getattr(torch.optim, cfg.optimizer)(
        [torch.nn.Parameter(torch.zeros(1))],
        lr=cfg.lr,
        **cfg.optimizer_kwargs(),
    )
    assert built.param_groups[0]["betas"] == (0.9, 0.98)


def test_betas_survive_a_json_round_trip_as_a_tuple():
    """``asdict`` writes an array and reading it back gives a list, so a reloaded
    config would compare unequal to a fresh one without normalisation."""
    cfg = SASRecConfig()

    restored = SASRecConfig(**json.loads(json.dumps(dataclasses.asdict(cfg))))

    assert isinstance(restored.betas, tuple)
    assert restored == cfg


# --------------------------------------------------------------------------
# the module: shapes and scoring
# --------------------------------------------------------------------------


def test_the_positional_table_has_one_row_per_position_plus_padding():
    """Positions are numbered from one so index 0 can stay reserved for padding
    steps, which is where the extra row goes."""
    model = _model(max_history_length=5)

    assert model.position_embedding.weight.shape[0] == 6
    assert torch.all(model.position_embedding.weight[0] == 0)


def test_a_history_longer_than_the_table_is_refused():
    model = _model(max_history_length=3)

    with pytest.raises(ValueError, match="needs 4 positions"):
        model(torch.ones((1, 4), dtype=torch.long))


def test_item_history_must_be_two_dimensional():
    with pytest.raises(ValueError, match="must be .rows, length."):
        _model()(torch.ones(4, dtype=torch.long))


def test_scoring_is_the_item_embedding_itself_not_a_separate_head():
    """SASRec scores by the dot product of a state with the item's *input*
    embedding, so the tie is structural: an untied SASRec is a different model
    and there is no head matrix to find."""
    model = _model()

    assert not hasattr(model, "head")
    names = {name for name, _ in model.named_parameters()}
    assert not any("head" in name for name in names)

    states = torch.randn(2, 3, 16)
    expected = states @ model.item_embedding.weight[N_RESERVED:].T
    torch.testing.assert_close(model.score(states), expected)


def test_the_score_covers_the_catalog_not_the_reserved_ids():
    """Padding is never a recommendation, so it has no column to be ranked in."""
    model = _model()

    scores = model.score(torch.randn(2, 16))

    assert scores.shape == (2, N_ITEMS)


def test_padding_embeds_as_zero_and_stays_there():
    """``padding_idx`` holds the gradient at zero, so whatever sits in that row
    at the start is permanent -- which is only correct if it starts at zero."""
    model = _model()

    assert torch.all(model.item_embedding.weight[PAD_ID] == 0)

    model.score(model(torch.tensor([[3, PAD_ID]]))).sum().backward()

    assert torch.all(model.item_embedding.weight.grad[PAD_ID] == 0)


# --------------------------------------------------------------------------
# the context window has one owner
# --------------------------------------------------------------------------


def test_fit_builds_a_batcher_from_the_config_when_none_is_passed():
    trainer = SASRecTrainer(_config(max_history_length=5)).fit(
        _seqs(_cycle_rows(16))
    )

    assert trainer.batcher.max_length == 5
    assert trainer.model.max_history_length == 5


def test_a_batcher_naming_no_window_inherits_the_configs():
    """Passing a batcher is how you supply a vocabulary, not how you re-declare
    the window, so the common case states nothing and stays in sync."""
    batcher = _batcher()
    trainer = SASRecTrainer(_config(max_history_length=5), batcher).fit(
        _seqs(_cycle_rows(16))
    )

    assert trainer.batcher.max_length == 5
    # Frozen, so the caller's own object was not mutated on the way through.
    assert batcher.max_length is None


def test_a_batcher_naming_a_different_window_is_refused():
    """Silently overruling either side is how the positional table and the
    truncation drift apart, and the drift only shows up as a wrong answer."""
    trainer = SASRecTrainer(_config(max_history_length=6), _batcher(max_length=3))

    with pytest.raises(ValueError, match="batcher max_length is 3"):
        trainer.fit(_seqs(_cycle_rows(16)))


def test_a_batcher_naming_the_same_window_is_accepted():
    trainer = SASRecTrainer(
        _config(max_history_length=6), _batcher(max_length=6)
    ).fit(_seqs(_cycle_rows(16)))

    assert trainer.model.max_history_length == 6


def test_fit_puts_the_batcher_on_the_left():
    """Left padding is the architecture rather than a preference: it is what
    anchors position n to the newest interaction for every history length."""
    trainer = SASRecTrainer(_config(), _batcher()).fit(_seqs(_cycle_rows(16)))

    assert trainer.batcher.padding == "left"
    assert trainer._train_batcher.padding == "left"


def test_training_reads_one_interaction_more_than_the_model_has_positions():
    """The next-item shift spends a step. Paying for it out of the window would
    leave the highest position with no input ever standing on it, while
    prediction -- which does not shift -- reads exactly that position."""
    trainer = SASRecTrainer(_config(max_history_length=6)).fit(
        _seqs(_cycle_rows(16))
    )

    assert trainer.batcher.max_length == 6
    assert trainer._train_batcher.max_length == 7
    assert trainer.model.max_history_length == 6


# --------------------------------------------------------------------------
# the fit contract
# --------------------------------------------------------------------------


def test_fit_returns_the_trainer_and_reports_the_catalog():
    trainer = SASRecTrainer(_config())

    fitted = trainer.fit(_seqs(_cycle_rows(16)))

    assert fitted is trainer
    assert trainer.is_fitted
    assert trainer.n_items == N_ITEMS


def test_before_fitting_nothing_is_claimed():
    trainer = SASRecTrainer(_config())

    assert not trainer.is_fitted
    assert trainer.n_items is None
    assert trainer.model is None
    assert trainer.history == []


def test_fit_refuses_a_matrix_source():
    from scipy.sparse import csr_matrix

    with pytest.raises(TypeError, match="trains on ItemSequences"):
        SASRecTrainer(_config()).fit(csr_matrix((3, N_ITEMS)))


def test_fit_refuses_an_empty_training_set():
    with pytest.raises(ValueError, match="zero sequences"):
        SASRecTrainer(_config()).fit(_seqs([]))


def test_fit_refuses_a_catalog_too_small_to_draw_a_negative_from():
    """The sampled objective needs somewhere to draw from, which a one-item
    catalog does not provide -- unlike the siblings' full-catalog softmax."""
    with pytest.raises(ValueError, match="needs at least two items"):
        SASRecTrainer(_config()).fit(_seqs([[0], [0]], n_items=1))


def test_fit_refuses_data_with_no_next_item_example():
    with pytest.raises(ValueError, match="two or more interactions"):
        SASRecTrainer(_config()).fit(_seqs([[1], [2], [3]]))


def test_fit_rejects_a_supplied_batcher_for_a_different_catalog():
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS + 3), max_length=6)
    trainer = SASRecTrainer(_config(), batcher)

    with pytest.raises(ValueError, match="batcher tokenizer has"):
        trainer.fit(_seqs(_cycle_rows(16)))


def test_training_records_a_loss_per_epoch():
    trainer = SASRecTrainer(_config(epochs=3)).fit(_seqs(_cycle_rows()))

    assert [entry["epoch"] for entry in trainer.history] == [1.0, 2.0, 3.0]
    assert all(entry["positions"] > 0 for entry in trainer.history)
    assert all(np.isfinite(entry["loss"]) for entry in trainer.history)


def test_training_reduces_the_loss_on_a_learnable_cycle():
    trainer = SASRecTrainer(_config(epochs=40, lr=0.01)).fit(
        _seqs(_cycle_rows())
    )

    assert trainer.history[-1]["loss"] < trainer.history[0]["loss"]


def test_refitting_starts_the_history_over():
    trainer = SASRecTrainer(_config(epochs=2))
    trainer.fit(_seqs(_cycle_rows(16)))

    trainer.fit(_seqs(_cycle_rows(16)))

    assert len(trainer.history) == 2


def test_the_number_of_training_positions_is_batching_invariant():
    """Positions come from the shift, so how rows are grouped cannot change how
    many there are."""
    counts = []
    for batch_size in (1, 2, 3, 64):
        trainer = SASRecTrainer(_config(batch_size=batch_size)).fit(
            _seqs(_cycle_rows(16))
        )
        counts.append(trainer.history[-1]["positions"])

    assert len(set(counts)) == 1, counts


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def test_predict_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="must be fitted"):
        SASRecTrainer(_config()).predict_on_batch(_seqs([[1, 2]]), k=2)


def test_predict_refuses_a_matrix_source():
    from scipy.sparse import csr_matrix

    model = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))

    with pytest.raises(TypeError, match="predicts from ItemSequences"):
        model.predict_on_batch(csr_matrix((2, N_ITEMS)), k=2)


@pytest.mark.parametrize("k", [0, N_ITEMS + 1])
def test_predict_refuses_an_impossible_k(k):
    model = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))

    with pytest.raises(ValueError, match="k must be in"):
        model.predict_on_batch(_seqs([[1, 2]]), k=k)


def test_predictions_are_shaped_and_ordered():
    model = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))

    predictions = model.predict(_seqs([[1, 2], [3]]), k=4)

    assert predictions.cols.shape == (2, 4)
    assert predictions.vals.shape == (2, 4)
    assert predictions.shape == (2, N_ITEMS)
    for row in range(2):
        values = predictions.vals[row].tolist()
        assert values == sorted(values, reverse=True)


def test_no_rows_at_all():
    model = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))

    predictions = model.predict_on_batch(_seqs([]), k=3)

    assert predictions.cols.shape == (0, 3)


def test_an_empty_history_is_predictable():
    """Left padding makes an empty row all padding, which the attention mask has
    to survive rather than divide by zero on."""
    model = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))

    predictions = model.predict(_seqs([[], [1, 2]]), k=3)

    assert predictions.cols.shape == (2, 3)
    assert bool(torch.isfinite(predictions.vals).all())


def test_it_predicts_the_successor_of_the_last_item():
    """The end-to-end wiring check: shift, positions, padding, negatives and the
    objective all have to be right together for this to come out."""
    model = _fitted_on_cycle()

    predictions = model.predict(_seqs([[0, 1, 2], [4, 5, 6]]), k=1)

    assert predictions.cols[0].item() == 3
    assert predictions.cols[1].item() == 7


def test_prediction_is_batching_invariant():
    model = _fitted_on_cycle(epochs=5)
    sources = _seqs([[0, 1], [2, 3, 4], [5], [6, 7, 0, 1]])

    whole = model.predict(sources, k=3, batch_size=64)
    split = model.predict(sources, k=3, batch_size=1)

    assert torch.equal(whole.cols, split.cols)
    torch.testing.assert_close(whole.vals, split.vals)


def test_the_same_seed_gives_the_same_model():
    rows = _seqs(_cycle_rows(16))

    first = SASRecTrainer(_config(seed=7, epochs=2)).fit(rows)
    second = SASRecTrainer(_config(seed=7, epochs=2)).fit(rows)
    other = SASRecTrainer(_config(seed=8, epochs=2)).fit(rows)

    # The whole score vector, so this compares the model and not just its top
    # few -- which means the history may not be masked out of it.
    def scores(model):
        return model.predict(probe, k=N_ITEMS, exclude_seen=False).vals

    probe = _seqs([[1, 2, 3]])
    torch.testing.assert_close(scores(first), scores(second))
    assert not torch.allclose(scores(first), scores(other))


# --------------------------------------------------------------------------
# exclude_seen
# --------------------------------------------------------------------------


def test_exclude_seen_masks_the_whole_history_including_repeats():
    model = _fitted_on_cycle(epochs=5)
    sources = _seqs([[0, 1, 1, 2], [4, 5]])

    predictions = model.predict(sources, k=N_ITEMS - 4, exclude_seen=True)

    for row in range(sources.n_rows):
        seen = set(sources.row(row).tolist())
        assert not seen & set(predictions.cols[row].tolist())


def test_exclude_seen_masks_what_truncation_dropped():
    """Truncation bounds what the model reads, not what it may return: the
    earlier items are still this user's history."""
    model = _fitted_on_cycle(epochs=5, max_history_length=2)
    history = [0, 1, 2, 3, 4]

    predictions = model.predict(_seqs([history]), k=N_ITEMS - len(history))

    assert not set(history) & set(predictions.cols[0].tolist())
    assert model.batcher.truncated_lengths(_seqs([history])).tolist() == [2]


def test_exclude_seen_refuses_when_fewer_than_k_unseen_items_remain():
    model = _fitted_on_cycle(epochs=5)
    sources = _seqs([[0, 1, 2, 3, 4, 5, 6]])

    with pytest.raises(ValueError):
        model.predict(sources, k=3, exclude_seen=True)


def test_exclude_seen_false_may_return_the_history():
    """Shape alone would hold even if the mask were applied regardless, so this
    pins that history items actually come back: seven of eight items are seen,
    so at least two of any three results must be among them."""
    model = _fitted_on_cycle(epochs=5)
    history = [0, 1, 2, 3, 4, 5, 6]
    sources = _seqs([history])

    predictions = model.predict(sources, k=3, exclude_seen=False)

    assert predictions.cols.shape == (1, 3)
    # The load-bearing assertion. Overlap alone is not enough: masking sets
    # -inf, and topk still has to return k columns, so history items come back
    # anyway -- carrying -inf scores. Finite values are what says nothing was
    # masked. The capacity check is gated on exclude_seen too, so it does not
    # catch this either.
    assert bool(torch.isfinite(predictions.vals).all()), predictions.vals
    returned = set(predictions.cols[0].tolist())
    assert len(returned & set(history)) >= 2, returned


def test_masking_does_not_mutate_the_source():
    model = _fitted_on_cycle(epochs=5)
    sources = _seqs([[0, 1, 2], [3, 4]])
    before = sources.values.copy()

    model.predict(sources, k=2, exclude_seen=True)

    assert np.array_equal(sources.values, before)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_a_reloaded_model_predicts_identically(tmp_path):
    model = SASRecTrainer(_config(epochs=2)).fit(_seqs(_cycle_rows(16)))
    sources = _seqs([[0, 1, 2], [4, 5]])
    expected = model.predict(sources, k=4)

    path = tmp_path / "sasrec.zip"
    model.save(path)
    restored = SASRecTrainer.load(path)

    got = restored.predict(sources, k=4)
    assert torch.equal(expected.cols, got.cols)
    torch.testing.assert_close(expected.vals, got.vals)


def test_a_reloaded_model_keeps_the_window_and_the_padding_side(tmp_path):
    """The checkpoint records the model's window, not the wider one training
    used, and the padding side has to come back with it or every position
    shifts."""
    model = SASRecTrainer(_config(max_history_length=5, epochs=1)).fit(
        _seqs(_cycle_rows(16))
    )

    path = tmp_path / "sasrec.zip"
    model.save(path)
    restored = SASRecTrainer.load(path)

    assert restored.batcher.max_length == 5
    assert restored.batcher.padding == "left"
    assert restored.model.max_history_length == 5


def test_a_reloaded_model_can_be_fitted_again(tmp_path):
    """Load hands back a batcher with an explicit window, so a second fit has to
    agree with the config it was saved beside rather than trip its own check."""
    model = SASRecTrainer(_config(max_history_length=5, epochs=1)).fit(
        _seqs(_cycle_rows(16))
    )
    path = tmp_path / "sasrec.zip"
    model.save(path)

    restored = SASRecTrainer.load(path)
    restored.fit(_seqs(_cycle_rows(16)))

    assert restored.batcher.max_length == 5
    assert restored._train_batcher.max_length == 6


# --------------------------------------------------------------------------
# the shift, and the positions it used to leave behind
# --------------------------------------------------------------------------


def _position_grads(trainer):
    """Gradient norm per row of the positional table, from the last backward."""
    grad = trainer.model.position_embedding.weight.grad
    assert grad is not None, "fit leaves the final backward's gradient in place"
    return [float(grad[row].norm()) for row in range(grad.shape[0])]


def test_every_position_receives_gradient():
    """The off-by-one this closes.

    The shift used to be paid for out of the window, so the highest position
    had no input ever standing on it -- while prediction, which does not shift,
    reads exactly that position for any history that fills the window.
    """
    trainer = SASRecTrainer(_config(max_history_length=5, epochs=2)).fit(
        _seqs(_cycle_rows(length=6))
    )

    norms = _position_grads(trainer)

    assert norms[0] == 0.0, "row 0 is padding and padding_idx pins its gradient"
    untrained = [i for i in range(1, len(norms)) if norms[i] == 0.0]
    assert untrained == [], f"positions {untrained} received no gradient"


def test_the_highest_position_moves_off_its_initialisation():
    """A gradient of zero is unreachable, not merely small: without the extra
    interaction the row stays byte-identical however long you train."""
    cfg = _config(max_history_length=5, epochs=3)
    trainer = SASRecTrainer(cfg).fit(_seqs(_cycle_rows(length=6)))

    # fit seeds torch immediately before building, so repeating both reproduces
    # the tensor training started from.
    torch.manual_seed(cfg.seed)
    initial = trainer._build_model().position_embedding.weight.detach().clone()
    trained = trainer.model.position_embedding.weight.detach()

    assert torch.equal(initial[0], trained[0]), "the pad row is pinned"
    for row in range(1, trained.shape[0]):
        assert not torch.equal(initial[row], trained[row]), f"row {row}"


def test_the_effective_targets_match_the_papers_expected_output():
    """The paper's o_t: ``<pad>`` where s_t is padding, ``s_{t+1}`` while
    ``t < n``, and the held-out final interaction at ``t = n``."""
    n = 5
    history = [1, 2, 3, 4]
    trainer = SASRecTrainer(_config(max_history_length=n)).fit(_seqs(_cycle_rows()))
    offset = trainer.batcher.tokenizer.n_reserved

    tokens, mask = trainer._train_batcher.encode(_seqs([history]))
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    valid = mask[:, :-1] & mask[:, 1:] & (targets >= offset)

    def catalog(row):
        return [None if int(v) < offset else int(v) - offset for v in row]

    s = catalog(inputs[0])
    assert s == [None, None, 1, 2, 3], "left padded, final item held out"

    paper = [
        None if s[t] is None else (s[t + 1] if t < n - 1 else history[-1])
        for t in range(n)
    ]
    ours = [
        int(targets[0, t]) - offset if valid[0, t] else None for t in range(n)
    ]

    assert ours == paper == [None, None, 2, 3, 4]


def test_a_padding_input_is_never_trained_to_predict():
    """Under left padding the step before the first real item has a pad input
    and a real target. The paper makes o_t ``<pad>`` whenever s_t is padding,
    which the target's own mask cannot express: real tokens are a suffix here,
    so a real target no longer implies a real input.
    """
    trainer = SASRecTrainer(_config(max_history_length=5)).fit(_seqs(_cycle_rows()))
    tokens, mask = trainer._train_batcher.encode(_seqs([[1, 2]]))

    boundary = int(mask[0].nonzero()[0]) - 1
    valid = mask[:, :-1] & mask[:, 1:]

    assert not mask[0, boundary], "the step before the history is padding"
    assert mask[0, boundary + 1], "and its target is a real item"
    assert not valid[0, boundary], "so the pair must not reach the loss"


def test_the_newest_interaction_is_always_the_last_position():
    """What left padding buys: position n means "n from the end" for a user
    with one interaction and a user with a full window alike."""
    trainer = SASRecTrainer(_config(max_history_length=6)).fit(_seqs(_cycle_rows()))
    sources = _seqs([[1], [1, 2, 3], [0, 1, 2, 3, 4, 5]])
    offset = trainer.batcher.tokenizer.n_reserved

    tokens, mask = trainer.batcher.encode(sources)

    assert tokens.shape[1] == 6, "left padding fills to the window"
    assert trainer.batcher.final_positions(mask).tolist() == [5, 5, 5]
    assert (tokens[:, -1] - offset).tolist() == [1, 3, 5]


def test_short_histories_train_the_high_positions():
    """Under right padding position 1 meant "oldest retained item", so the high
    rows were reachable only by the longest histories -- while prediction reads
    them for exactly those users."""
    trainer = SASRecTrainer(_config(max_history_length=20, epochs=2)).fit(
        _seqs(_cycle_rows(length=4))
    )

    norms = _position_grads(trainer)
    trained = [i for i in range(1, len(norms)) if norms[i] > 0.0]

    # Four interactions give three trained positions, and they sit at the end.
    assert trained == [18, 19, 20], trained


def test_the_untrimmed_training_batch_would_overflow_the_table():
    """Why forward keeps a length check of its own: the training batch is one
    wider than the positional table, so dropping the shift is caught on the
    first step rather than quietly training a misaligned objective."""
    trainer = SASRecTrainer(_config(max_history_length=5)).fit(_seqs(_cycle_rows()))
    tokens, _ = trainer._train_batcher.encode(_seqs([[1, 2, 3]]))

    assert tokens.shape[1] == 6
    trainer.model(tokens[:, :-1])  # what _train_step feeds: fits exactly

    with pytest.raises(ValueError, match="needs 6 positions"):
        trainer.model(tokens)


def test_a_history_longer_than_the_window_is_truncated_not_refused():
    """A context window is a claim about recency. The module's length check
    guards a hand-built tensor; nothing arriving through the batcher trips it."""
    trainer = SASRecTrainer(_config(max_history_length=4)).fit(_seqs(_cycle_rows()))
    history = [0, 1, 2, 3, 4, 5, 6, 7]
    offset = trainer.batcher.tokenizer.n_reserved

    tokens, _ = trainer.batcher.encode(_seqs([history]))

    assert tokens.shape[1] == 4
    assert (tokens[0] - offset).tolist() == history[-4:]


# --------------------------------------------------------------------------
# negative sampling
# --------------------------------------------------------------------------


def _draws(trainer, sources, seed=0):
    """Sampled negatives as catalog positions, shaped (rows, length, n)."""
    tokens, _ = trainer._train_batcher.encode(sources)
    negatives = trainer._sample_negatives(
        sources, tokens[:, 1:], np.random.default_rng(seed)
    )
    return negatives - trainer.batcher.tokenizer.n_reserved


def test_a_negative_is_never_an_item_from_that_users_history():
    """The paper draws from ``I \\ S_u``. An item they interacted with earlier
    -- or later, which the shift makes just as reachable -- is one they did
    engage with, so ranking it below the target teaches the opposite."""
    trainer = SASRecTrainer(_config(n_negatives=4, epochs=1)).fit(_seqs(_cycle_rows()))
    sources = _seqs([[0, 1, 2], [3, 4], [5, 6, 7, 0]])

    catalog = _draws(trainer, sources)

    for row in range(sources.n_rows):
        seen = set(sources.row(row).tolist())
        drawn = set(catalog[row].flatten().tolist())
        assert not seen & drawn, f"row {row}: {sorted(seen & drawn)}"


def test_exclusion_reads_the_whole_history_not_the_window():
    """S_u is the user's sequence, not the part that survived truncation."""
    trainer = SASRecTrainer(
        _config(max_history_length=2, n_negatives=3, epochs=1)
    ).fit(_seqs(_cycle_rows()))
    history = [0, 1, 2, 3, 4]
    sources = _seqs([history])

    catalog = _draws(trainer, sources)

    assert trainer.batcher.truncated_lengths(sources).tolist() == [2]
    assert not set(history) & set(catalog.flatten().tolist())


def test_a_repeated_item_counts_as_one_exclusion():
    """Counting a duplicate twice would shrink the range the draw is uniform
    over and step the mapping past items that were never excluded."""
    trainer = SASRecTrainer(_config(epochs=1)).fit(_seqs(_cycle_rows()))

    _, once = trainer._excluded_items(_seqs([[1, 2, 3]]), N_ITEMS)
    _, twice = trainer._excluded_items(_seqs([[1, 2, 2, 3, 3, 3]]), N_ITEMS)

    assert once.tolist() == [3]
    assert twice.tolist() == [3]


def test_negatives_are_uniform_over_the_complement():
    """Not merely "outside the history" -- evenly outside it, which is what the
    searchsorted mapping is for."""
    trainer = SASRecTrainer(_config(n_negatives=8, epochs=1)).fit(_seqs(_cycle_rows()))
    history = [1, 4]
    sources = _seqs([history])
    tokens, _ = trainer._train_batcher.encode(sources)
    offset = trainer.batcher.tokenizer.n_reserved

    counts = np.zeros(N_ITEMS, dtype=int)
    rng = np.random.default_rng(0)
    for _ in range(400):
        drawn = trainer._sample_negatives(sources, tokens[:, 1:], rng)
        np.add.at(counts, (drawn - offset).numpy().ravel(), 1)

    allowed = [j for j in range(N_ITEMS) if j not in history]
    assert counts[history].sum() == 0
    expected = counts.sum() / len(allowed)
    assert counts[allowed].min() > expected * 0.8
    assert counts[allowed].max() < expected * 1.2


def test_a_reserved_id_is_never_drawn():
    """Padding and unk are not items, and scoring them as negatives would train
    the model to reject its own filler."""
    trainer = SASRecTrainer(_config(n_negatives=6, epochs=1)).fit(_seqs(_cycle_rows()))
    sources = _seqs([[0, 1, 2], [4, 5]])
    tokens, _ = trainer._train_batcher.encode(sources)

    drawn = trainer._sample_negatives(sources, tokens[:, 1:], np.random.default_rng(1))

    assert int(drawn.min()) >= trainer.model.n_reserved
    assert int(drawn.max()) < trainer.model.n_reserved + trainer.model.n_items


def test_the_same_seed_draws_the_same_negatives():
    trainer = SASRecTrainer(_config(n_negatives=3, epochs=1)).fit(_seqs(_cycle_rows()))
    sources = _seqs([[0, 1, 2], [4, 5]])
    tokens, _ = trainer._train_batcher.encode(sources)
    positives = tokens[:, 1:]

    first = trainer._sample_negatives(sources, positives, np.random.default_rng(3))
    again = trainer._sample_negatives(sources, positives, np.random.default_rng(3))
    other = trainer._sample_negatives(sources, positives, np.random.default_rng(4))

    assert torch.equal(first, again)
    assert not torch.equal(first, other)


def test_a_history_covering_the_catalog_leaves_nothing_to_draw():
    trainer = SASRecTrainer(_config(epochs=1)).fit(_seqs(_cycle_rows()))
    everything = _seqs([list(range(N_ITEMS))])
    tokens, _ = trainer._train_batcher.encode(everything)

    with pytest.raises(ValueError, match="no item outside it"):
        trainer._sample_negatives(
            everything, tokens[:, 1:], np.random.default_rng(0)
        )


@pytest.mark.parametrize("n_negatives", [1, 2, 5])
def test_the_draw_count_follows_the_config(n_negatives):
    trainer = SASRecTrainer(
        _config(n_negatives=n_negatives, max_history_length=4, epochs=1)
    ).fit(_seqs(_cycle_rows()))
    sources = _seqs([[0, 1, 2], [4, 5]])
    tokens, _ = trainer._train_batcher.encode(sources)
    positives = tokens[:, 1:]

    drawn = trainer._sample_negatives(sources, positives, np.random.default_rng(0))

    assert drawn.shape == positives.shape + (n_negatives,)


def test_negatives_are_drawn_per_position_not_once_per_row():
    """Sharing one draw across a history would make the gradient far poorer
    than the position count suggests."""
    trainer = SASRecTrainer(_config(max_history_length=6, epochs=1)).fit(
        _seqs(_cycle_rows())
    )
    sources = _seqs([[0, 1]])

    catalog = _draws(trainer, sources)

    assert len(set(catalog.flatten().tolist())) > 1


# --------------------------------------------------------------------------
# what a state may read
# --------------------------------------------------------------------------


def test_a_step_cannot_see_a_later_step():
    """Perturb one token: states before it must not move, and the state that
    should read it must -- or the first assertion passes vacuously."""
    model = _model().eval()
    tokens = torch.tensor([[2, 3, 4, 5, 6]])

    with torch.no_grad():
        before = model(tokens)
        for j in range(tokens.shape[1]):
            changed = tokens.clone()
            changed[0, j] = 7
            after = model(changed)
            torch.testing.assert_close(
                after[:, :j], before[:, :j], msg=f"leak at token {j}"
            )
            assert not torch.allclose(after[:, j], before[:, j]), (
                f"token {j} had no effect on its own state"
            )


def test_a_real_step_cannot_see_padding():
    """Left padding puts the pad steps inside every causal window, so causal
    masking no longer excludes them for free and the mask says so explicitly."""
    model = _model().eval()
    tokens = torch.tensor([[PAD_ID, PAD_ID, 3, 4]])
    real = tokens != PAD_ID

    with torch.no_grad():
        before = model(tokens).clone()
        model.item_embedding.weight[PAD_ID] = (
            torch.randn(model.item_embedding.embedding_dim) * 50.0
        )
        after = model(tokens)

    assert float((before - after)[real].abs().max()) == 0.0
    assert float((before - after)[~real].abs().max()) > 0.0, (
        "the poison must have changed something, or this proves nothing"
    )


def test_an_all_padding_row_does_not_produce_nan():
    """A pad step's causal window is entirely padding, and a query masked
    everywhere softmaxes over nothing. Freeing the diagonal keeps it finite."""
    model = _model().eval()

    with torch.no_grad():
        states = model(torch.full((1, 4), PAD_ID, dtype=torch.long))

    assert bool(torch.isfinite(states).all())


def test_every_padding_step_produces_the_same_state():
    """Their input is zero and they read only themselves, so there is exactly
    one pad state -- non-zero, because a layer norm maps zero to its bias."""
    model = _model().eval()

    with torch.no_grad():
        states = model(torch.tensor([[PAD_ID, PAD_ID, PAD_ID, 5]]))

    pads = states[0, :3]
    assert float((pads - pads[0]).abs().max()) == 0.0


def test_the_position_of_a_step_does_not_depend_on_its_item():
    """Only the layout reaches the position index. Catalog item 0 is the case
    that would collide with padding if the reserved offset were dropped."""
    model = _model()
    a = torch.tensor([[PAD_ID, 2, 3, 4]])
    b = torch.tensor([[PAD_ID, 8, 5, 9]])

    positions_a = torch.arange(1, 5) * (a != model.pad_id)
    positions_b = torch.arange(1, 5) * (b != model.pad_id)

    assert torch.equal(positions_a, positions_b)
    assert positions_a[0].tolist() == [0, 2, 3, 4]


def test_dropout_is_inactive_in_eval_mode():
    model = _model(dropout=0.5).eval()
    tokens = torch.tensor([[2, 3, 4]])

    with torch.no_grad():
        assert torch.equal(model(tokens), model(tokens))

    model.train()
    with torch.no_grad():
        assert not torch.equal(model(tokens), model(tokens))


def test_the_same_seed_builds_the_same_module():
    torch.manual_seed(11)
    first = _model()
    torch.manual_seed(11)
    second = _model()

    for (name, a), (_, b) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert torch.equal(a, b), name


# --------------------------------------------------------------------------
# the objective and the optimizer
# --------------------------------------------------------------------------


def test_the_loss_is_binary_cross_entropy_on_the_logits():
    """The paper's objective, term for term: log sigma(positive) plus
    log(1 - sigma(negative)). The sigmoid is fused into the loss, which is why
    score_items returns raw logits."""
    positives = torch.tensor([2.0, -1.0, 0.5])
    negatives = torch.tensor([-0.5, 1.5, 0.0])
    objective = nn.BCEWithLogitsLoss()

    loss = objective(positives, torch.ones_like(positives)) + objective(
        negatives, torch.zeros_like(negatives)
    )

    manual = -torch.log(torch.sigmoid(positives)).mean() - torch.log(
        1 - torch.sigmoid(negatives)
    ).mean()
    torch.testing.assert_close(loss, manual)


def test_only_valid_positions_reach_the_loss():
    """Padded steps carry no lesson, and a pad input with a real target carries
    the wrong one. Count what the objective is actually handed."""
    seen = []
    rows = [[1, 2, 3], [4], [5, 6]]
    trainer = SASRecTrainer(_config(max_history_length=4, batch_size=4))
    trainer.fit(_seqs(rows))

    def spy(scores, labels):
        seen.append(tuple(scores.shape))
        return nn.BCEWithLogitsLoss()(scores, labels)

    trainer._rng = np.random.default_rng(0)
    trainer._train_step(
        _seqs(rows), torch.optim.SGD(trainer.model.parameters(), lr=0.0), spy
    )

    # Two positions from [1,2,3], none from [4] (one item predicts nothing),
    # one from [5,6] -- and one negative each.
    assert seen == [(3,), (3, 1)]


def test_training_never_scores_the_whole_catalog():
    """The sampled objective exists to avoid a rows x length x catalog tensor:
    3.5 GB of activations at batch 128, length 200 and a 34k catalog, against
    0.2 MB for the two columns the loss reads."""
    calls = []
    rows = [[1, 2, 3], [4, 5]]
    trainer = SASRecTrainer(_config(max_history_length=4, epochs=1))
    trainer.fit(_seqs(rows))
    original = trainer.model.score
    trainer.model.score = lambda states: calls.append(states.shape) or original(
        states
    )

    trainer._rng = np.random.default_rng(0)
    trainer._train_step(
        _seqs(rows),
        torch.optim.SGD(trainer.model.parameters(), lr=0.0),
        nn.BCEWithLogitsLoss(),
    )

    assert calls == [], "training goes through score_items, never score"


def test_the_optimizer_is_adam_with_the_papers_betas():
    trainer = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))

    assert isinstance(trainer.optimizer, torch.optim.Adam)
    assert trainer.optimizer.param_groups[0]["betas"] == (0.9, 0.98)


def test_a_resumed_optimizer_keeps_its_betas(tmp_path):
    """Momentum is not in the weights, so a run resumed without the optimizer
    restarts it cold however exact the model."""
    trainer = SASRecTrainer(_config(epochs=1)).fit(_seqs(_cycle_rows(16)))
    path = tmp_path / "sasrec.zip"
    trainer.save(path, include_optimizer=True)

    resumed = SASRecTrainer.load(path, load_optimizer=True)

    assert resumed.optimizer is not None
    assert resumed.optimizer.param_groups[0]["betas"] == (0.9, 0.98)
    assert SASRecTrainer.load(path).optimizer is None


# --------------------------------------------------------------------------
# through the package's own entry points
# --------------------------------------------------------------------------


def test_it_evaluates_through_the_standard_entry_point():
    from scipy.sparse import csr_matrix

    from compresso_recsys.evaluation import evaluate_recommender
    from compresso_recsys.metrics import NDCG

    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
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


def test_recommend_round_trips_real_item_ids():
    """Giving the tokenizer the real IDs is what lets recommend() take and
    return them instead of catalog offsets."""
    ids = [f"item-{i}" for i in range(N_ITEMS)]
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS, item_ids=ids))
    model = SASRecTrainer(
        _config(epochs=60, lr=0.01, batch_size=48), batcher
    ).fit(_seqs(_cycle_rows()), item_ids=ids)

    out = model.recommend([["item-0", "item-1", "item-2"]], k=3, exclude_seen=True)

    assert list(out.item_ids[0])[0] == "item-3"
    assert not {"item-0", "item-1", "item-2"} & set(out.item_ids[0])


def test_candidate_ids_restrict_the_ranking():
    """Ranking within a shortlist, which a serving layer needs and which the
    full-catalog path would otherwise hide."""
    ids = [f"item-{i}" for i in range(N_ITEMS)]
    batcher = SequenceBatcher(ItemTokenizer(N_ITEMS, item_ids=ids))
    model = SASRecTrainer(_config(epochs=2), batcher).fit(
        _seqs(_cycle_rows()), item_ids=ids
    )

    predictions = model.predict(
        _seqs([[0, 1, 2]]),
        k=2,
        exclude_seen=False,
        candidate_ids=["item-5", "item-6", "item-7"],
    )

    assert set(predictions.cols[0].tolist()) <= {5, 6, 7}
    assert predictions.cols.shape == (1, 2)


# --------------------------------------------------------------------------
# the gaps mutation testing found
# --------------------------------------------------------------------------


def test_the_negatives_influence_the_loss():
    """That the negative term is *present* is not the same as it mattering.

    The objective can still be called with both tensors while its negative half
    contributes nothing — a coefficient of zero, a detached branch. Changing the
    drawn negatives must change the loss.
    """
    trainer = SASRecTrainer(_config(dropout=0.0, epochs=1)).fit(_seqs(_cycle_rows()))
    rows = _seqs([[0, 1, 2], [3, 4]])
    offset = trainer.batcher.tokenizer.n_reserved
    frozen = torch.optim.SGD(trainer.model.parameters(), lr=0.0)

    def always(item):
        return lambda batch, positives, rng: torch.full(
            tuple(positives.shape) + (trainer.cfg.n_negatives,),
            offset + item,
            dtype=torch.long,
        )

    original = trainer._sample_negatives
    try:
        # Items 5 and 6 appear in neither history, so both are legal negatives.
        trainer._sample_negatives = always(5)
        first, _ = trainer._train_step(rows, frozen, nn.BCEWithLogitsLoss())
        trainer._sample_negatives = always(6)
        second, _ = trainer._train_step(rows, frozen, nn.BCEWithLogitsLoss())
    finally:
        trainer._sample_negatives = original

    assert first != second, "the loss ignored which items were the negatives"


def test_the_reported_loss_is_the_two_bce_terms_added():
    """Recomputed from the scores the step actually produced, rather than from
    tensors a test invented -- which is what makes it able to notice a term
    being dropped, halved or handed the wrong labels."""
    captured = []
    trainer = SASRecTrainer(_config(dropout=0.0, epochs=1)).fit(_seqs(_cycle_rows()))
    rows = _seqs([[0, 1, 2], [3, 4]])

    def spy(scores, labels):
        captured.append((scores.detach().clone(), labels.detach().clone()))
        return nn.BCEWithLogitsLoss()(scores, labels)

    trainer._rng = np.random.default_rng(0)
    loss, _ = trainer._train_step(
        rows, torch.optim.SGD(trainer.model.parameters(), lr=0.0), spy
    )

    (positive_scores, positive_labels), (negative_scores, negative_labels) = captured
    assert bool((positive_labels == 1).all()), "positives are labelled 1"
    assert bool((negative_labels == 0).all()), "negatives are labelled 0"

    manual = float(
        -torch.log(torch.sigmoid(positive_scores)).mean()
        - torch.log(1 - torch.sigmoid(negative_scores)).mean()
    )
    assert loss == pytest.approx(manual, rel=1e-5)


def test_masking_drops_items_beyond_the_fitted_catalog():
    """A later split stage hands histories over a wider catalog. Those items were
    never scoreable, so there is nothing to forbid -- and clipping them into range
    instead would mask a real item that should still be recommendable."""
    model = SASRecTrainer(_config(epochs=2)).fit(_seqs(_cycle_rows()))
    # Item 8 is outside the fitted catalog of 8; clipping would fold it onto 7.
    wider = _seqs([[0, N_ITEMS, 1]], n_items=N_ITEMS + 1)

    predictions = model.predict(wider, k=6, exclude_seen=True)

    assert set(predictions.cols[0].tolist()) == {2, 3, 4, 5, 6, 7}


# --------------------------------------------------------------------------
# unk dropout
# --------------------------------------------------------------------------


def _corrupted(trainer, history, mask, rate):
    trainer.cfg = dataclasses.replace(trainer.cfg, unk_dropout=rate)
    return trainer._with_unk_dropout(history, mask)


def test_unk_dropout_never_touches_padding():
    """Corrupting padding would teach the model that unk and pad mean the same
    thing, and the whole point of unk is that they do not.

    The rate is 0.99 rather than 1.0 because the config bounds it to [0, 1) --
    the assertion is about padding, which holds whatever the draw selects.
    """
    trainer = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))
    pad = trainer.batcher.tokenizer.pad_id
    history = torch.tensor([[pad, pad, 3, 4, 5]])
    mask = history != pad

    torch.manual_seed(0)
    got = _corrupted(trainer, history, mask, rate=0.99)

    assert got[0, :2].tolist() == [pad, pad]
    assert bool((got[0, 2:] != history[0, 2:]).any()), "nothing was corrupted"


def test_a_corrupted_position_becomes_unk_and_nothing_else():
    """Stated as a rule over whichever positions the draw picked, so it does not
    depend on a rate of 1.0 -- which the config forbids -- or on a lucky seed."""
    trainer = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))
    unk = trainer.batcher.tokenizer.unk_id
    history = torch.tensor([[3, 4, 5, 6]])
    mask = torch.ones_like(history, dtype=torch.bool)

    torch.manual_seed(0)
    got = _corrupted(trainer, history, mask, rate=0.99)

    changed = got != history
    assert bool(changed.any()), "nothing was corrupted at a rate of 0.99"
    assert bool((got[changed] == unk).all()), "a changed position must hold unk"
    assert bool((got[~changed] == history[~changed]).all()), "the rest is intact"


def test_zero_unk_dropout_changes_nothing():
    trainer = SASRecTrainer(_config()).fit(_seqs(_cycle_rows(16)))
    history = torch.tensor([[3, 4, 5, 6]])
    mask = torch.ones_like(history, dtype=torch.bool)

    got = _corrupted(trainer, history, mask, rate=0.0)

    assert torch.equal(got, history)


def test_a_vocabulary_without_unk_ignores_the_rate():
    """``unk`` is optional, so the rate has to degrade to a no-op rather than
    substituting some other reserved id."""
    tokenizer = ItemTokenizer(N_ITEMS, special_tokens={"pad": 0})
    trainer = SASRecTrainer(
        _config(unk_dropout=0.9), SequenceBatcher(tokenizer)
    ).fit(_seqs(_cycle_rows(16)))
    history = torch.tensor([[1, 2, 3]])
    mask = torch.ones_like(history, dtype=torch.bool)

    torch.manual_seed(0)
    got = trainer._with_unk_dropout(history, mask)

    assert torch.equal(got, history)


def test_unk_dropout_corrupts_the_inputs_and_leaves_the_targets_alone():
    """A corrupted position teaches 'an item was here you cannot identify,
    predict the next one anyway' rather than costing a training example -- so it
    is applied after the shift, and only to one side of it."""
    trainer = SASRecTrainer(_config(unk_dropout=0.99, epochs=1)).fit(
        _seqs(_cycle_rows(16))
    )
    unk = trainer.batcher.tokenizer.unk_id
    seen = []
    original = trainer.model.forward
    trainer.model.forward = lambda h: seen.append(h.clone()) or original(h)

    torch.manual_seed(0)
    trainer._rng = np.random.default_rng(0)
    trainer._train_step(
        _seqs([[0, 1, 2]]),
        torch.optim.SGD(trainer.model.parameters(), lr=0.0),
        nn.BCEWithLogitsLoss(),
    )
    trainer.model.forward = original

    tokens, mask = trainer._train_batcher.encode(_seqs([[0, 1, 2]]))
    inputs_seen = seen[0]
    corrupted = inputs_seen == unk

    assert bool(corrupted.any()), "no input was corrupted at a rate of 0.99"
    # Corruption lands only on real positions, never on padding.
    assert bool(mask[:, :-1][corrupted].all()), "padding was corrupted"
    # And the targets come from the clean tokens, so unk never becomes one.
    assert not bool((tokens[:, 1:] == unk).any())


# --------------------------------------------------------------------------
# architecture constants the paper fixes
# --------------------------------------------------------------------------


def test_every_layer_norm_uses_the_published_epsilon():
    """1e-8, against PyTorch's 1e-5 default. Nothing in the paper tunes it, and
    nothing here should either."""
    model = _model()

    assert LAYER_NORM_EPS == 1e-8
    norms = [*model.attention_norms, *model.forward_norms, model.last_norm]
    assert len(norms) == 5, "two blocks give two of each, plus the final one"
    for norm in norms:
        assert norm.eps == LAYER_NORM_EPS


def test_the_item_embedding_is_rescaled_by_sqrt_d_model():
    """Xavier gives a fan-based scale rather than the unit-ish one the norms
    downstream expect, and the reference rescales here to compensate."""
    model = _model().eval()
    captured = []
    model.embedding_dropout.register_forward_hook(
        lambda module, inputs, output: captured.append(inputs[0].detach().clone())
    )

    with torch.no_grad():
        # Zero the positional half so what reaches the dropout is the item half.
        model.position_embedding.weight.zero_()
        model(torch.tensor([[3]]))

    expected = model.item_embedding.weight[3] * (model.item_embedding.embedding_dim ** 0.5)
    torch.testing.assert_close(captured[0][0, 0], expected)


def test_the_configured_dropout_reaches_every_sublayer():
    """One rate for the embedding sum, inside attention, and between the
    feed-forward layers -- one knob because the reference exposes one."""
    model = _model(dropout=0.3)

    assert model.embedding_dropout.p == 0.3
    for attention in model.attention_layers:
        assert attention.dropout == 0.3
    for forward in model.forward_layers:
        assert forward.dropout1.p == 0.3
        assert forward.dropout2.p == 0.3
