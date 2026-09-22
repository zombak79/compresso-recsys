"""BERT4Rec: the Cloze objective, bidirectional attention, and ranking at [mask].

Organized like ``tests/test_sasrec.py``, because the two models differ in
exactly the places worth testing: SASRec is causal and trained on sampled
negatives, BERT4Rec is bidirectional and trained by masking. Where a test here
mirrors one there, the assertion is usually inverted -- a SASRec step must not
see a later step, while a BERT4Rec masked position must.

The Cloze sections have no SASRec counterpart at all. They cover the machinery
that turns one history into many training samples (sliding windows, the
duplication factor, the last-item samples) and the masking distribution itself,
which is the part of §3.6 the paper specifies as a rate rather than a procedure.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from compresso_recsys.models.bert4rec import (
    Bert4Rec,
    Bert4RecConfig,
    Bert4RecTrainer,
)
from compresso_recsys.models.bert4rec.trainer import (
    SPECIAL_TOKENS,
    _sliding_windows,
)
from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 8
PAD_ID = 0
MASK_ID = 1
N_RESERVED = 3
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


def _tokenizer(n_items=N_ITEMS, **kwargs):
    return ItemTokenizer(n_items, special_tokens=SPECIAL_TOKENS, **kwargs)


def _batcher(max_length=None, n_items=N_ITEMS, **kwargs):
    return SequenceBatcher(_tokenizer(n_items), max_length=max_length, **kwargs)


def _config(**overrides):
    defaults = dict(
        d_model=16,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
        att_dropout=0.0,
        max_history_length=6,
        epochs=1,
        batch_size=16,
        duplication_factor=1,
        show_progress=False,
        seed=0,
    )
    return Bert4RecConfig(**{**defaults, **overrides})


def _trainer(config=None, batcher=None, **overrides):
    """A trainer whose batcher is already built, so the Cloze helpers can run.

    ``fit`` builds one from the training catalog; the sample-construction tests
    call ``_training_windows`` and ``_cloze_batch`` directly and need the
    vocabulary without a fit.
    """
    config = config or _config(**overrides)
    if batcher is None:
        batcher = _batcher(max_length=config.max_history_length)
    return Bert4RecTrainer(config, batcher)


def _fitted_on_cycle(epochs=60, **overrides):
    config = _config(epochs=epochs, lr=0.01, batch_size=48, **overrides)
    return Bert4RecTrainer(config).fit(_seqs(_cycle_rows()))


def _model(**overrides):
    defaults = dict(
        n_items=N_ITEMS,
        max_history_length=MAX_POSITIONS,
        d_model=16,
        n_blocks=2,
        n_heads=2,
        dropout=0.0,
        att_dropout=0.0,
        pad=PAD_ID,
        mask=MASK_ID,
        n_reserved=N_RESERVED,
    )
    return Bert4Rec(**{**defaults, **overrides})


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(d_model=0), "d_model must be >= 1"),
        (dict(n_heads=0), "n_heads must be >= 1"),
        (dict(d_model=10, n_heads=4), "divisible by n_heads"),
        (dict(n_blocks=0), "n_blocks must be >= 1"),
        (dict(dropout=1.0), r"dropout must be in \[0, 1\)"),
        (dict(att_dropout=-0.1), r"att_dropout must be in \[0, 1\)"),
        (dict(max_history_length=1), "max_history_length must be >= 2"),
        (dict(initializer_range=0.0), "initializer_range must be > 0"),
        (dict(mask_proportion=0.0), r"mask_proportion .* must be in \(0, 1\]"),
        (dict(mask_proportion=1.1), r"mask_proportion .* must be in \(0, 1\]"),
        (dict(max_predictions=0), "max_predictions must be >= 1"),
        (dict(mask_token_probability=1.1), r"mask_token_probability must be in \[0, 1\]"),
        (dict(duplication_factor=0), "duplication_factor must be >= 1"),
        (dict(sliding_window_step=0.0), r"sliding_window_step .* must be in \(0, 1\]"),
        (dict(batch_size=0), "batch_size must be >= 1"),
        (dict(epochs=0), "epochs must be >= 1"),
        (dict(lr=0.0), "lr must be > 0"),
        (dict(weight_decay=-0.1), "weight_decay must be >= 0"),
        (dict(epsilon=0.0), "epsilon must be > 0"),
        (dict(max_grad_norm=0.0), "max_grad_norm must be > 0"),
        (dict(warmup_steps=-1), "warmup_steps must be >= 0"),
        (dict(optimizer="Adam"), "PyTorch spells AdamW"),
        (dict(betas=(0.9,)), "betas must be two values"),
        (dict(betas=(0.9, 1.0)), r"betas must each be in \[0, 1\)"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_the_defaults_are_the_published_settings():
    """Every default is the paper's, or a run_*.sh figure where it is silent.

    The config docstring cites a provenance per field; this pins the numbers so
    the citation cannot quietly stop describing the code.

    "The reference" means the four run_*.sh scripts, which produced the paper's
    tables -- not the flag defaults in run.py and gen_data_fin.py, which the
    repository inherited from BERT's pretraining code and which no published run
    used. The two differ on batch_size (256 against 32), learning_rate (1e-4
    against 5e-5), warmup (100 steps against 10000) and the sliding window (0.5
    against 0.1), so citing the wrong tier is worth a test rather than a
    comment. Where the scripts disagree with each other these follow ml-20m,
    the long-sequence run the default max_history_length commits to -- except
    dropout, which is ml-1m's 0.2 against ml-20m's 0.1, so the defaults are no
    single published run. Dropout also comes from a third file again, the
    released bert_config_*.json, rather than from a script.
    """
    cfg = Bert4RecConfig()

    # Architecture: L = 2, h = 2 (§4.3); d_model 64 and the positional table at
    # 200 are the released config's hidden_size and max_position_embeddings.
    assert (cfg.d_model, cfg.n_blocks, cfg.n_heads) == (64, 2, 2)
    assert cfg.max_history_length == 200
    # ml-1m's hidden_dropout_prob and attention_probs_dropout_prob, the one
    # place these defaults leave ml-20m (which sets both to 0.1).
    assert (cfg.dropout, cfg.att_dropout) == (0.2, 0.2)
    assert cfg.initializer_range == 0.02

    # Cloze: rho 0.2 and max_predictions_per_seq 20 are ml-1m's and ml-20m's
    # rho and ml-20m's cap; mask_prob 1.0 disables BERT's 80/10/10;
    # dupe_factor 10 is every script's; prop_sliding_window 0.5 is ml-1m's,
    # ml-20m's and steam's, against beauty's 0.1 at an N of 50.
    assert cfg.mask_proportion == 0.2
    assert cfg.max_predictions == 20
    assert cfg.mask_token_probability == 1.0
    assert cfg.duplication_factor == 10
    assert cfg.sliding_window_step == 0.5
    assert cfg.last_item_samples is True

    # Optimization: §4.3's batch of 256 and lr 1e-4, which every script passes
    # too; decoupled weight decay 0.01; gradients clipped at l2 norm 5; the
    # reference's eps of 1e-6; and 100 warmup steps, the scripts' num_warmup_
    # steps rather than run.py's 10000 flag default. It is a count and not a
    # fraction because warmup is a fixed cost, not a share of the run.
    assert cfg.batch_size == 256
    # epochs has no counterpart: the reference trains a fixed num_train_steps
    # over a corpus generated once, so this is a package default, not published.
    assert cfg.epochs == 10
    assert cfg.lr == 1e-4
    assert cfg.weight_decay == 0.01
    assert cfg.betas == (0.9, 0.999)
    assert cfg.epsilon == 1e-6
    assert cfg.max_grad_norm == 5.0
    assert cfg.warmup_steps == 100
    assert cfg.optimizer == "AdamW"


def test_the_feed_forward_width_is_not_configurable():
    """Eq. 3 fixes the inner width at 4 * d_model, so it is not a field."""
    assert not hasattr(Bert4RecConfig(), "d_ff")
    model = _model(d_model=16)
    assert model.feed_forward_layers[0].linear1.out_features == 64


def test_betas_survive_a_json_round_trip_as_a_tuple():
    """A reloaded config must compare equal to the one that was saved.

    ``asdict`` writes a JSON array, which reads back as a list. Without the
    normalization in ``__post_init__`` the two configs differ by the type of one
    field and nothing else, which is the kind of inequality that only shows up
    in a checkpoint test.
    """
    config = _config()
    restored = Bert4RecConfig(
        **json.loads(json.dumps(dataclasses.asdict(config), default=str))
    )
    assert isinstance(restored.betas, tuple)
    assert restored == config


def test_betas_given_as_a_list_are_normalized():
    assert Bert4RecConfig(betas=[0.9, 0.999]).betas == (0.9, 0.999)


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


def test_the_vocabulary_is_the_catalog_above_the_reserved_ids():
    model = _model()
    assert model.vocab_size == N_RESERVED + N_ITEMS
    assert model.item_embedding.num_embeddings == N_RESERVED + N_ITEMS
    assert model.output_bias.shape == (N_RESERVED + N_ITEMS,)


def test_the_positional_table_has_one_row_per_position():
    model = _model(max_history_length=5)
    assert model.positional_embedding.num_embeddings == 5


def test_the_model_constructs_with_its_own_defaults():
    """Regression: n_heads defaulted to 5, which does not divide d_model 64.

    The trainer always passes n_heads explicitly, so the broken default was
    invisible until someone built the module directly.
    """
    model = Bert4Rec(n_items=N_ITEMS, max_history_length=6, d_model=64)
    assert model.d_model == 64


def test_padding_embeds_as_zero():
    model = _model()
    assert torch.equal(
        model.item_embedding.weight[PAD_ID], torch.zeros(model.d_model)
    )


def test_the_output_is_logits_not_a_distribution():
    """Eq. 8 is evaluated by cross_entropy, which needs logits.

    A softmax here would leave the trainer taking the log of a probability that
    can underflow; see ``test_the_loss_is_cross_entropy_on_the_masked_logits``.
    """
    model = _model()
    model.eval()
    scores = model(torch.tensor([[3, 4, MASK_ID, 6]]))
    assert not torch.allclose(scores.exp().sum(dim=-1), torch.ones(1))


def test_a_batch_with_no_mask_is_refused():
    """The output is read at [mask]; a batch with none would score nothing.

    Returning an empty matrix instead would push the mistake downstream, where
    it surfaces as a shape error far from its cause.
    """
    model = _model()
    with pytest.raises(ValueError, match=r"no token equals the mask id"):
        model(torch.tensor([[3, 4, 5, 6]]))


def test_an_index_selecting_nothing_is_refused():
    """The same guard on the training path, where the index and not the token
    says where to read."""
    model = _model()
    with pytest.raises(ValueError, match=r"mask_positions selects nothing"):
        model(
            torch.tensor([[3, 4, MASK_ID, 6]]),
            None,
            torch.zeros((1, 4), dtype=torch.bool),
        )


def test_one_row_of_scores_per_masked_position():
    model = _model()
    model.eval()
    history = torch.tensor([[MASK_ID, 4, MASK_ID, 6], [3, 4, 5, MASK_ID]])
    assert model(history).shape == (3, N_RESERVED + N_ITEMS)


def test_masked_positions_are_scored_in_row_major_order():
    """The trainer builds its labels in this order and passes no index along.

    Two rows whose masked positions hold different items must come back in the
    order the rows appear, with the earlier position of a row first.
    """
    model = _model()
    model.eval()
    with torch.no_grad():
        both = model(torch.tensor([[MASK_ID, 4, MASK_ID, 6], [3, MASK_ID, 5, 6]]))
        first = model(torch.tensor([[MASK_ID, 4, MASK_ID, 6]]))
        second = model(torch.tensor([[3, MASK_ID, 5, 6]]))
    assert torch.allclose(both[:2], first, atol=1e-5)
    assert torch.allclose(both[2:], second, atol=1e-5)


def test_a_masked_position_reads_the_items_after_it():
    """Attention is bidirectional -- the property that makes the Cloze objective
    necessary and separates this model from a causal one.

    ``tests/test_sasrec.py`` asserts the opposite for SASRec: there a step
    cannot see a later step. Here changing a *later* item must change the score
    at an earlier masked position, or the model is not reading the future at all
    and the masking was pointless.
    """
    model = _model()
    model.eval()
    with torch.no_grad():
        before = model(torch.tensor([[3, MASK_ID, 5, 6]]))
        after = model(torch.tensor([[3, MASK_ID, 5, 7]]))
    assert not torch.allclose(before, after, atol=1e-6)


def test_a_masked_position_also_reads_the_items_before_it():
    model = _model()
    model.eval()
    with torch.no_grad():
        before = model(torch.tensor([[3, 4, 5, MASK_ID]]))
        after = model(torch.tensor([[7, 4, 5, MASK_ID]]))
    assert not torch.allclose(before, after, atol=1e-6)


def test_padding_is_not_visible_to_a_masked_position():
    """A row's score must not depend on how much padding the batch gave it.

    Without the key padding mask, a real position attends to filler embeddings
    and the amount mixed in varies with the width of the batch rectangle, which
    is an artefact of who else was in the batch.
    """
    model = _model()
    model.eval()
    history = torch.tensor([[3, 4, MASK_ID]])
    padded = torch.tensor([[3, 4, MASK_ID, PAD_ID, PAD_ID]])
    mask = torch.tensor([[True, True, True]])
    padded_mask = torch.tensor([[True, True, True, False, False]])
    with torch.no_grad():
        tight = model(history, mask)
        loose = model(padded, padded_mask)
    assert torch.allclose(tight, loose, atol=1e-5)


def test_a_history_longer_than_the_table_keeps_its_most_recent_positions():
    model = _model(max_history_length=4)
    model.eval()
    long_history = torch.tensor([[3, 4, 5, 6, 7, MASK_ID]])
    with torch.no_grad():
        truncated = model(long_history)
        equivalent = model(torch.tensor([[5, 6, 7, MASK_ID]]))
    assert torch.allclose(truncated, equivalent, atol=1e-5)


# --------------------------------------------------------------------------
# Cloze sample construction
# --------------------------------------------------------------------------


def test_a_history_no_longer_than_the_window_is_one_sample():
    history = np.arange(4)
    assert [w.tolist() for w in _sliding_windows(history, 6, 2)] == [[0, 1, 2, 3]]


def test_sliding_windows_cover_the_whole_history():
    """Everything before the last N interactions would otherwise never train.

    The windows walk backwards from the most recent interaction and the oldest
    is pinned to the start, so the union is the entire history.
    """
    history = np.arange(10)
    windows = _sliding_windows(history, 4, 2)
    assert set(np.concatenate(windows).tolist()) == set(range(10))
    assert all(len(w) == 4 for w in windows)


def test_sliding_windows_are_chronological():
    """The reference builds them back-to-front and reverses; so does this."""
    windows = _sliding_windows(np.arange(10), 4, 2)
    starts = [int(w[0]) for w in windows]
    assert starts == sorted(starts)
    assert int(windows[-1][-1]) == 9


def test_the_duplication_factor_repeats_every_window():
    """§3.6 turns one history into many samples; this is how many."""
    rows = [[0, 1, 2, 3], [1, 2, 3, 4]]
    one = _trainer(_config(duplication_factor=1, last_item_samples=False))
    ten = _trainer(_config(duplication_factor=10, last_item_samples=False))
    single = one._training_windows(_seqs(rows))
    repeated = ten._training_windows(_seqs(rows))
    assert len(repeated) == 10 * len(single)


def test_each_history_contributes_one_last_item_sample():
    """The paper's fine-tuning for the task the Cloze objective does not match."""
    rows = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]
    without = _trainer(_config(last_item_samples=False))._training_windows(_seqs(rows))
    with_tail = _trainer(_config(last_item_samples=True))._training_windows(_seqs(rows))
    assert len(with_tail) == len(without) + len(rows)
    assert sum(1 for _, force_last in with_tail if force_last) == len(rows)


def test_a_long_history_contributes_one_last_item_sample_per_window():
    """mask_last iterates the user's whole document, which is the window list.

    One per user would leave everything before the final window with no
    fine-tuning sample at all -- the same gap the sliding window exists to
    close for the Cloze samples.
    """
    config = _config(
        max_history_length=6, last_item_samples=True, sliding_window_step=1 / 6
    )
    history = [step % N_ITEMS for step in range(10)]
    windows = _sliding_windows(np.asarray(history), 6, 1)
    samples = _trainer(config)._training_windows(_seqs([history]))
    assert len(windows) > 1
    assert sum(1 for _, force_last in samples if force_last) == len(windows)


def test_the_last_item_samples_are_the_windows_themselves():
    """Each is masked at its own final position, so each leaves width - 1 of
    context -- what predict_on_batch encodes before appending [mask]."""
    config = _config(
        max_history_length=6, last_item_samples=True, sliding_window_step=1 / 6
    )
    history = [step % N_ITEMS for step in range(10)]
    samples = _trainer(config)._training_windows(_seqs([history]))
    tails = [window.tolist() for window, force_last in samples if force_last]
    expected = _sliding_windows(np.asarray(history), 6, 1)
    assert tails == [window.tolist() for window in expected]


def test_a_single_item_history_yields_no_last_item_sample():
    """Masking the only position leaves no context to predict it from.

    The reference asserts the position is non-zero rather than handling it;
    dropping the sample is the same requirement without the crash.
    """
    samples = _trainer(_config())._training_windows(_seqs([[0], [1, 2, 3]]))
    assert [len(window) for window, force_last in samples if force_last] == [3]


def test_an_empty_history_contributes_nothing():
    samples = _trainer(_config())._training_windows(_seqs([[], [1, 2, 3]]))
    assert all(window.size > 0 for window, _ in samples)


def _masked_batch(trainer, rows, seed=0, force_last=False):
    windows = [(np.asarray(row, dtype=np.int64), force_last) for row in rows]
    return trainer._cloze_batch(windows, np.random.default_rng(seed))


def test_the_masked_count_is_the_proportion_of_the_length():
    """§3.6 masks round(len * rho) positions per sequence."""
    trainer = _trainer(_config(mask_proportion=0.5, max_predictions=100))
    tokens, _, _, labels = _masked_batch(trainer, [list(range(8))])
    assert int((tokens == MASK_ID).sum()) == 4
    assert labels.numel() == 4


def test_at_least_one_position_is_always_masked():
    """A sample with nothing masked carries no gradient at all."""
    trainer = _trainer(_config(mask_proportion=0.01, max_predictions=100))
    tokens, _, _, labels = _masked_batch(trainer, [[0, 1]])
    assert int((tokens == MASK_ID).sum()) == 1
    assert labels.numel() == 1


def test_the_masked_count_is_capped_by_max_predictions():
    """The reference's max_predictions_per_seq binds before rho on a long row."""
    trainer = _trainer(
        _config(mask_proportion=1.0, max_predictions=3, max_history_length=20)
    )
    tokens, _, _, _ = _masked_batch(trainer, [list(range(8))])
    assert int((tokens == MASK_ID).sum()) == 3


def test_a_chosen_position_always_becomes_mask_at_the_default_probability():
    """mask_prob 1.0 disables BERT's 80/10/10, as §3.6 describes."""
    trainer = _trainer(_config(mask_proportion=0.5, mask_token_probability=1.0))
    tokens, _, _, labels = _masked_batch(trainer, [list(range(8))])
    assert int((tokens == MASK_ID).sum()) == labels.numel()


def test_below_one_some_chosen_positions_keep_or_replace_the_item():
    trainer = _trainer(
        _config(mask_proportion=1.0, max_predictions=100, mask_token_probability=0.0)
    )
    tokens, _, _, labels = _masked_batch(trainer, [list(range(8))] * 8, seed=3)
    assert int((tokens == MASK_ID).sum()) == 0
    assert labels.numel() == 64


def test_a_random_replacement_is_never_a_reserved_id():
    """A reserved id in an item position would teach the model that pad or mask
    can appear as data."""
    trainer = _trainer(
        _config(mask_proportion=1.0, max_predictions=100, mask_token_probability=0.0)
    )
    for seed in range(5):
        tokens, padding, _, _ = _masked_batch(trainer, [list(range(8))] * 8, seed=seed)
        assert int((tokens[padding] < N_RESERVED).sum()) == 0


def test_the_masked_index_covers_positions_that_kept_their_item():
    """Regression: the loss is taken at every *chosen* position, and below a
    mask_token_probability of 1.0 most of those no longer hold [MASK]. The
    index is what tells the model where to read; recovering it from the token
    would find nothing here while the labels still number 64.
    """
    trainer = _trainer(
        _config(mask_proportion=1.0, max_predictions=100, mask_token_probability=0.0)
    )
    tokens, _, masked, labels = _masked_batch(trainer, [list(range(8))] * 8, seed=3)
    assert int((tokens == MASK_ID).sum()) == 0
    assert int(masked.sum()) == labels.numel() == 64


def test_training_runs_below_a_mask_token_probability_of_one():
    """Regression: forward used to gather at ``item_history == mask``, so any
    corruption at all left fewer logit rows than labels and cross_entropy
    refused the batch.
    """
    trainer = Bert4RecTrainer(
        _config(mask_token_probability=0.5, mask_proportion=0.5)
    ).fit(_seqs(_cycle_rows()))
    assert trainer.history[-1]["masked_items"] > 0


def test_the_read_positions_come_from_the_index_not_the_token():
    """Given an index, the model reads exactly those positions -- including
    ones holding an ordinary item, which is what corruption produces."""
    model = _model()
    tokens = torch.tensor([[3, 4, 5, 6]])
    index = torch.tensor([[False, True, False, True]])
    assert model(tokens, None, index).shape[0] == 2


def test_the_labels_are_the_true_items_of_the_masked_positions():
    """This is the contract the model's row-major gather relies on."""
    trainer = _trainer(_config(mask_proportion=0.5, max_predictions=100))
    rows = [[0, 1, 2, 3], [4, 5, 6, 7]]
    tokens, _, _, labels = _masked_batch(trainer, rows)
    tokenizer = trainer.batcher.tokenizer
    expected = []
    for row_index, row in enumerate(rows):
        encoded = tokenizer.encode_indices(np.asarray(row, dtype=np.int64))
        expected.extend(
            int(encoded[position])
            for position in torch.where(tokens[row_index] == MASK_ID)[0].tolist()
        )
    assert labels.tolist() == expected


def test_the_last_item_sample_masks_only_its_final_position():
    trainer = _trainer(_config())
    tokens, _, _, labels = _masked_batch(trainer, [[0, 1, 2, 3]], force_last=True)
    assert tokens[0, -1] == MASK_ID
    assert int((tokens == MASK_ID).sum()) == 1
    assert labels.numel() == 1


def test_short_rows_are_right_padded_to_the_batch_width():
    """Position 0 must mean 'oldest retained interaction' for every row."""
    trainer = _trainer(_config())
    tokens, padding, _, _ = _masked_batch(trainer, [[0, 1, 2, 3], [4, 5]])
    assert tokens.shape == (2, 4)
    assert padding[1].tolist() == [True, True, False, False]
    assert tokens[1, 2:].tolist() == [PAD_ID, PAD_ID]


# --------------------------------------------------------------------------
# fit and the batcher contract
# --------------------------------------------------------------------------


def test_fit_builds_a_batcher_from_the_config_when_none_is_passed():
    trainer = Bert4RecTrainer(_config()).fit(_seqs(_cycle_rows()))
    assert trainer.batcher is not None
    assert trainer.batcher.max_length == trainer.cfg.max_history_length
    assert trainer.batcher.tokenizer.n_items == N_ITEMS


def test_the_built_vocabulary_carries_a_mask_token():
    trainer = Bert4RecTrainer(_config()).fit(_seqs(_cycle_rows()))
    assert trainer.batcher.tokenizer.token_id("mask") == MASK_ID


def test_a_tokenizer_without_a_mask_token_is_refused():
    """The Cloze objective is not expressible without one: masking with an item
    id would teach the model that that item means 'predict here'."""
    plain = SequenceBatcher(ItemTokenizer(N_ITEMS), max_length=6)
    with pytest.raises(ValueError, match="needs a 'mask' special token"):
        Bert4RecTrainer(_config(), plain).fit(_seqs(_cycle_rows()))


def test_a_batcher_naming_no_window_inherits_the_configs():
    """The usual case: a batcher is handed over for its vocabulary, not its
    window, and the window is what sized the positional table."""
    trainer = Bert4RecTrainer(_config(max_history_length=6), _batcher()).fit(
        _seqs(_cycle_rows())
    )
    assert trainer.batcher.max_length == 6


def test_a_batcher_naming_a_different_window_is_refused():
    trainer = Bert4RecTrainer(_config(max_history_length=6), _batcher(max_length=5))
    with pytest.raises(ValueError, match="cannot be two numbers"):
        trainer.fit(_seqs(_cycle_rows()))


def test_a_batcher_naming_the_same_window_is_accepted():
    trainer = Bert4RecTrainer(_config(max_history_length=6), _batcher(max_length=6))
    assert trainer.fit(_seqs(_cycle_rows())).batcher.max_length == 6


def test_fit_puts_the_batcher_on_the_right():
    """Right padding keeps position 0 meaning 'oldest retained interaction' for
    every row, which is what a learned positional table reads under Cloze."""
    trainer = Bert4RecTrainer(
        _config(max_history_length=6), _batcher(max_length=6, padding="left")
    ).fit(_seqs(_cycle_rows()))
    assert trainer.batcher.padding == "right"


def test_the_prediction_batcher_leaves_room_for_the_appended_mask():
    trainer = Bert4RecTrainer(_config(max_history_length=6)).fit(_seqs(_cycle_rows()))
    assert trainer._predict_batcher.max_length == 5


def test_an_unfitted_trainer_has_no_prediction_batcher():
    """The attribute exists before fit so that nothing has to guess at it."""
    assert Bert4RecTrainer(_config())._predict_batcher is None


def test_fit_returns_the_trainer_and_reports_the_catalog():
    trainer = Bert4RecTrainer(_config())
    assert trainer.fit(_seqs(_cycle_rows())) is trainer
    assert trainer.is_fitted
    assert trainer.n_items == N_ITEMS


def test_before_fitting_nothing_is_claimed():
    trainer = Bert4RecTrainer(_config())
    assert not trainer.is_fitted
    assert trainer.n_items is None


def test_fit_refuses_a_matrix_source():
    with pytest.raises(TypeError, match="trains on ItemSequences"):
        Bert4RecTrainer(_config()).fit(np.zeros((4, N_ITEMS)))


def test_fit_refuses_an_empty_training_set():
    with pytest.raises(ValueError, match="cannot train on zero sequences"):
        Bert4RecTrainer(_config()).fit(_seqs([]))


def test_fit_refuses_data_with_no_cloze_example():
    with pytest.raises(ValueError, match="no Cloze example"):
        Bert4RecTrainer(_config()).fit(_seqs([[], []]))


def test_fit_rejects_a_supplied_batcher_for_a_different_catalog():
    trainer = Bert4RecTrainer(_config(), _batcher(n_items=N_ITEMS + 3))
    with pytest.raises(ValueError, match="but training"):
        trainer.fit(_seqs(_cycle_rows()))


def test_training_records_a_loss_per_epoch():
    trainer = _fitted_on_cycle(epochs=3)
    assert [record["epoch"] for record in trainer.history] == [1.0, 2.0, 3.0]
    assert all(record["masked_items"] > 0 for record in trainer.history)


def test_training_reduces_the_loss_on_a_learnable_cycle():
    trainer = _fitted_on_cycle(epochs=40)
    assert trainer.history[-1]["loss"] < trainer.history[0]["loss"]


def test_refitting_starts_the_history_over():
    trainer = _fitted_on_cycle(epochs=3)
    trainer.fit(_seqs(_cycle_rows()))
    assert len(trainer.history) == 3


def test_the_same_seed_gives_the_same_model():
    first = _fitted_on_cycle(epochs=2, seed=7)
    second = _fitted_on_cycle(epochs=2, seed=7)
    for left, right in zip(
        first.model.state_dict().values(), second.model.state_dict().values()
    ):
        assert torch.equal(left, right)


def test_a_different_seed_gives_a_different_model():
    first = _fitted_on_cycle(epochs=2, seed=1)
    second = _fitted_on_cycle(epochs=2, seed=2)
    assert not torch.equal(
        first.model.item_embedding.weight, second.model.item_embedding.weight
    )


# --------------------------------------------------------------------------
# the objective
# --------------------------------------------------------------------------


def test_the_loss_is_cross_entropy_on_the_masked_logits():
    """Eq. 8 is the negative log-likelihood of Eq. 7's softmax, which is what
    cross_entropy computes from logits."""
    trainer = Bert4RecTrainer(_config()).fit(_seqs(_cycle_rows()))
    rng = np.random.default_rng(0)
    windows = [(np.array([0, 1, 2, 3], dtype=np.int64), True)]
    tokens, padding, masked, labels = trainer._cloze_batch(windows, rng)
    trainer.model.eval()
    with torch.no_grad():
        expected = float(
            F.cross_entropy(trainer.model(tokens, padding, masked), labels)
        )
    # _train_step redraws the batch from a generator in the same state, so it
    # masks the same positions, and it backpropagates, so it runs outside
    # no_grad. Dropout is off in this config, leaving train/eval equivalent.
    loss, count = trainer._train_step(windows, np.random.default_rng(0))
    assert count == int(labels.numel())
    assert loss == pytest.approx(expected, abs=1e-5)


def test_a_confidently_wrong_example_still_carries_gradient():
    """Regression: the loss used to be -log(softmax(x).clamp_min(tiny)).

    Clamping makes the loss locally constant where a probability underflows, so
    its derivative is exactly zero -- the model stopped learning from precisely
    the masked positions it was most wrong about, silently and with no NaN.
    """
    model = _model()
    tokens = torch.tensor([[3, 4, MASK_ID, 6]])
    with torch.no_grad():
        # Drive one item's score far above the target's.
        model.output_bias[5] = 400.0
    labels = torch.tensor([4])
    loss = F.cross_entropy(model(tokens), labels)
    loss.backward()
    total = sum(
        float(p.grad.abs().sum())
        for p in model.parameters()
        if p.grad is not None
    )
    assert total > 0.0


def test_only_masked_positions_contribute_to_the_loss():
    """An unmasked position is context, not a target: the model can read the
    item there, so scoring it would be free."""
    trainer = Bert4RecTrainer(_config(mask_proportion=0.25, max_predictions=1)).fit(
        _seqs(_cycle_rows())
    )
    windows = [(np.array([0, 1, 2, 3], dtype=np.int64), False)]
    _, _, _, labels = trainer._cloze_batch(windows, np.random.default_rng(0))
    assert labels.numel() == 1


def test_every_epoch_redraws_the_masking():
    """The reference materializes dupe_factor maskings to a file and reads it
    each epoch; drawing per epoch is the same distribution, more varied."""
    trainer = _trainer(_config(mask_proportion=0.5, max_predictions=100))
    rows = [list(range(8))]
    first, _, _, _ = _masked_batch(trainer, rows, seed=1)
    second, _, _, _ = _masked_batch(trainer, rows, seed=2)
    assert not torch.equal(first, second)


def test_weight_decay_spares_biases_and_layer_norms():
    """Decaying a normalization scale pulls it toward zero, shrinking the
    activations it exists to standardize."""
    trainer = _fitted_on_cycle(epochs=1)
    decayed, spared = trainer.optimizer.param_groups
    assert decayed["weight_decay"] == trainer.cfg.weight_decay
    assert spared["weight_decay"] == 0.0
    assert all(parameter.ndim > 1 for parameter in decayed["params"])
    assert all(parameter.ndim == 1 for parameter in spared["params"])


def test_no_training_step_runs_at_a_zero_learning_rate():
    """Both ends of the schedule stay off zero: a step whose rate is zero
    computes a gradient, clips it, and applies nothing."""
    trainer = _trainer(_config())
    trainer.model = _model()
    trainer._build_checkpoint_optimizer()
    scheduler = trainer._build_scheduler(trainer.optimizer, 20)
    rates = []
    for _ in range(20):
        rates.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.optimizer.step()
        scheduler.step()
    assert all(rate > 0.0 for rate in rates)


def test_the_schedule_warms_up_and_then_decays():
    trainer = _trainer(_config(warmup_steps=5))
    trainer.model = _model()
    trainer._build_checkpoint_optimizer()
    scheduler = trainer._build_scheduler(trainer.optimizer, 20)
    rates = []
    for _ in range(20):
        rates.append(trainer.optimizer.param_groups[0]["lr"])
        trainer.optimizer.step()
        scheduler.step()
    peak = rates.index(max(rates))
    assert 0 < peak < len(rates) - 1
    assert rates[:peak] == sorted(rates[:peak])
    assert rates[peak:] == sorted(rates[peak:], reverse=True)


def test_the_padding_row_is_zeroed_at_initialisation():
    """So a padded position starts out embedding to nothing.

    It does not stay zero: the embedding matrix is tied to the output head, so
    the pad row is also a scoring row and cross entropy pushes its logit down at
    every step. ``padding_idx`` suppresses only the input-side gradient. That is
    harmless on both paths -- the key padding mask keeps attention off padded
    positions, and prediction slices the reserved ids away before ranking -- but
    it is why this asserts about initialisation rather than about every step.
    """
    trainer = _trainer(_config())
    model = trainer._build_model()
    assert torch.equal(
        model.item_embedding.weight[model.pad], torch.zeros(model.d_model)
    )


def test_a_reserved_id_is_never_recommended():
    """pad, mask and unk are scoreable in the softmax but are not catalog items.

    They are sliced off before ranking, so no amount of probability mass landing
    on them can put one in front of a user.
    """
    trainer = _fitted_on_cycle(epochs=2)
    predictions = trainer.predict_on_batch(
        _seqs(_cycle_rows(n_rows=4)), k=3, exclude_seen=False
    )
    assert int(predictions.cols.min()) >= 0
    assert int(predictions.cols.max()) < N_ITEMS


# --------------------------------------------------------------------------
# prediction
# --------------------------------------------------------------------------


def test_predict_refuses_before_fitting():
    with pytest.raises(RuntimeError, match="must be fitted before predicting"):
        Bert4RecTrainer(_config()).predict_on_batch(_seqs([[0, 1]]), k=2)


def test_predict_refuses_a_matrix_source():
    trainer = _fitted_on_cycle(epochs=1)
    with pytest.raises(TypeError, match="predicts from ItemSequences"):
        trainer.predict_on_batch(np.zeros((2, N_ITEMS)), k=2)


@pytest.mark.parametrize("k", [0, N_ITEMS + 1])
def test_predict_refuses_an_impossible_k(k):
    trainer = _fitted_on_cycle(epochs=1)
    with pytest.raises(ValueError, match="k must be in"):
        trainer.predict_on_batch(_seqs([[0, 1]]), k=k, exclude_seen=False)


def test_predictions_are_shaped_and_ordered():
    trainer = _fitted_on_cycle(epochs=2)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2], [3, 4, 5]]), k=3, exclude_seen=False
    )
    assert predictions.cols.shape == (2, 3)
    assert predictions.shape == (2, N_ITEMS)
    for row in predictions.vals:
        assert torch.equal(row, torch.sort(row, descending=True).values)


def test_no_rows_at_all():
    trainer = _fitted_on_cycle(epochs=1)
    predictions = trainer.predict_on_batch(_seqs([]), k=2, exclude_seen=False)
    assert predictions.cols.shape == (0, 2)


def test_an_empty_history_is_predictable():
    """Nothing to read but the appended [mask]; the model still has to answer."""
    trainer = _fitted_on_cycle(epochs=1)
    predictions = trainer.predict_on_batch(
        _seqs([[], [0, 1]]), k=2, exclude_seen=False
    )
    assert predictions.cols.shape == (2, 2)


def test_it_predicts_the_successor_of_the_last_item():
    """On a pure cycle the answer is determined by the final interaction.

    The source rows are training histories with their last interaction held
    out, which is the split this model is trained for: ``fit`` sees the whole
    history and masks its final position, and prediction sees the same context
    with ``[mask]`` appended. A source of ``[0, 1, 2]`` therefore asks the same
    question the last-item sample for ``[0, 1, 2, 3]`` was trained on.
    """
    trainer = _fitted_on_cycle(epochs=200)
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2], [2, 3, 4]]), k=1, exclude_seen=False
    )
    assert predictions.cols[:, 0].tolist() == [3, 5]


def test_prediction_is_batching_invariant():
    trainer = _fitted_on_cycle(epochs=2)
    rows = [[0, 1, 2], [3, 4, 5], [2, 3, 4]]
    together = trainer.predict_on_batch(_seqs(rows), k=2, exclude_seen=False)
    apart = [
        trainer.predict_on_batch(_seqs([row]), k=2, exclude_seen=False)
        for row in rows
    ]
    for index, single in enumerate(apart):
        assert torch.equal(together.cols[index], single.cols[0])
        assert torch.allclose(together.vals[index], single.vals[0], atol=1e-5)


def test_the_appended_mask_sits_after_the_last_real_interaction():
    """Not at the end of the padded rectangle: the position of the token is what
    the model reads the recommendation from."""
    trainer = _fitted_on_cycle(epochs=1)
    tokens, padding = trainer._encode_with_mask(_seqs([[0, 1, 2], [3]]))
    assert int(torch.where(tokens[0] == MASK_ID)[0]) == 3
    assert int(torch.where(tokens[1] == MASK_ID)[0]) == 1
    assert padding[0].tolist()[:4] == [True, True, True, True]
    assert padding[1].tolist()[:2] == [True, True]


def test_a_history_longer_than_the_window_is_truncated_not_refused():
    trainer = _fitted_on_cycle(epochs=1)
    long_row = [(index % N_ITEMS) for index in range(30)]
    predictions = trainer.predict_on_batch(
        _seqs([long_row]), k=2, exclude_seen=False
    )
    assert predictions.cols.shape == (1, 2)


def test_candidate_ids_restrict_the_ranking():
    trainer = Bert4RecTrainer(_config(epochs=2)).fit(
        _seqs(_cycle_rows()), item_ids=[f"item-{index}" for index in range(N_ITEMS)]
    )
    predictions = trainer.predict_on_batch(
        _seqs([[0, 1, 2]]),
        k=2,
        exclude_seen=False,
        candidate_ids=["item-4", "item-5", "item-6"],
    )
    assert set(predictions.cols[0].tolist()) <= {4, 5, 6}


# --------------------------------------------------------------------------
# exclude_seen
# --------------------------------------------------------------------------


def test_exclude_seen_masks_the_whole_history_including_repeats():
    trainer = _fitted_on_cycle(epochs=2)
    rows = [[0, 1, 1, 2]]
    predictions = trainer.predict_on_batch(_seqs(rows), k=5, exclude_seen=True)
    assert not ({0, 1, 2} & set(predictions.cols[0].tolist()))


def test_exclude_seen_masks_what_truncation_dropped():
    """A model that attends to the last N interactions must still refuse to
    recommend the N+1th."""
    trainer = _fitted_on_cycle(epochs=2, max_history_length=4)
    row = [0, 1, 2, 3, 4, 5]
    predictions = trainer.predict_on_batch(_seqs([row]), k=2, exclude_seen=True)
    assert not (set(row) & set(predictions.cols[0].tolist()))


def test_exclude_seen_refuses_when_fewer_than_k_unseen_items_remain():
    trainer = _fitted_on_cycle(epochs=1)
    row = list(range(N_ITEMS - 1))
    with pytest.raises(ValueError):
        trainer.predict_on_batch(_seqs([row]), k=3, exclude_seen=True)


def test_exclude_seen_false_may_return_the_history():
    trainer = _fitted_on_cycle(epochs=2)
    rows = [[0, 1, 2]]
    predictions = trainer.predict_on_batch(_seqs(rows), k=N_ITEMS, exclude_seen=False)
    assert {0, 1, 2} <= set(predictions.cols[0].tolist())


def test_masking_does_not_mutate_the_source():
    trainer = _fitted_on_cycle(epochs=1)
    source = _seqs([[0, 1, 2]])
    before = np.array(source.values, copy=True)
    trainer.predict_on_batch(source, k=2, exclude_seen=True)
    assert np.array_equal(np.array(source.values), before)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_a_reloaded_model_predicts_identically(tmp_path):
    trainer = _fitted_on_cycle(epochs=2)
    source = _seqs([[0, 1, 2], [3, 4, 5]])
    before = trainer.predict_on_batch(source, k=3, exclude_seen=False)
    path = tmp_path / "bert4rec.ckpt"
    trainer.save(path)
    restored = Bert4RecTrainer.load(path)
    after = restored.predict_on_batch(source, k=3, exclude_seen=False)
    assert torch.equal(before.cols, after.cols)
    assert torch.allclose(before.vals, after.vals)


def test_a_reloaded_model_keeps_the_window_and_the_padding_side(tmp_path):
    trainer = _fitted_on_cycle(epochs=1, max_history_length=5)
    path = tmp_path / "bert4rec.ckpt"
    trainer.save(path)
    restored = Bert4RecTrainer.load(path)
    assert restored.batcher.max_length == 5
    assert restored.batcher.padding == "right"
    assert restored._predict_batcher.max_length == 4
    assert restored.cfg.max_history_length == 5


def test_a_reloaded_config_equals_the_saved_one(tmp_path):
    trainer = _fitted_on_cycle(epochs=1)
    path = tmp_path / "bert4rec.ckpt"
    trainer.save(path)
    assert Bert4RecTrainer.load(path).cfg == trainer.cfg


def test_a_reloaded_model_keeps_the_history(tmp_path):
    trainer = _fitted_on_cycle(epochs=2)
    path = tmp_path / "bert4rec.ckpt"
    trainer.save(path)
    assert Bert4RecTrainer.load(path).history == trainer.history


def test_a_reloaded_model_can_be_fitted_again(tmp_path):
    trainer = _fitted_on_cycle(epochs=1)
    path = tmp_path / "bert4rec.ckpt"
    trainer.save(path)
    restored = Bert4RecTrainer.load(path)
    restored.fit(_seqs(_cycle_rows()))
    assert len(restored.history) == 1
    assert restored.is_fitted


def test_the_vocabulary_survives_the_round_trip(tmp_path):
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    batcher = SequenceBatcher(_tokenizer(item_ids=item_ids), max_length=6)
    trainer = Bert4RecTrainer(_config(max_history_length=6), batcher).fit(
        _seqs(_cycle_rows()), item_ids=item_ids
    )
    path = tmp_path / "bert4rec.ckpt"
    trainer.save(path)
    restored = Bert4RecTrainer.load(path)
    assert list(restored.batcher.tokenizer.item_ids) == item_ids


# --------------------------------------------------------------------------
# the package's entry points
# --------------------------------------------------------------------------


def test_it_evaluates_through_the_standard_entry_point():
    """A sequential model and a matrix model meet the same evaluation API."""
    from scipy.sparse import csr_matrix

    from compresso_recsys.evaluation import evaluate_recommender
    from compresso_recsys.metrics import NDCG

    trainer = _fitted_on_cycle(epochs=200)
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
        trainer, source=sources, targets=targets, metrics=[NDCG(1)]
    )
    assert result.n_scored_rows == 3
    assert result["ndcg@1"] == pytest.approx(1.0)


def test_recommend_round_trips_real_item_ids():
    """Giving the tokenizer the real IDs is what lets recommend() take and
    return them instead of catalog offsets."""
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    batcher = SequenceBatcher(_tokenizer(item_ids=item_ids), max_length=6)
    trainer = Bert4RecTrainer(
        _config(epochs=200, lr=0.01, batch_size=48, max_history_length=6), batcher
    ).fit(_seqs(_cycle_rows()), item_ids=item_ids)

    recommendations = trainer.recommend(
        [["item-0", "item-1", "item-2"]], k=3, exclude_seen=True
    )

    assert list(recommendations.item_ids[0])[0] == "item-3"
    assert not {"item-0", "item-1", "item-2"} & set(recommendations.item_ids[0])


def test_item_ids_disagreeing_with_the_tokenizer_are_refused():
    """Two vocabularies for one catalog is an error rather than a silent win."""
    item_ids = [f"item-{index}" for index in range(N_ITEMS)]
    batcher = SequenceBatcher(_tokenizer(item_ids=item_ids), max_length=6)
    trainer = Bert4RecTrainer(_config(max_history_length=6), batcher)
    with pytest.raises(ValueError, match="must match the batcher tokenizer"):
        trainer.fit(_seqs(_cycle_rows()), item_ids=[f"x{i}" for i in range(N_ITEMS)])
