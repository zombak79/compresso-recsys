from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.simple_gpt import (
    Block,
    CausalSelfAttention,
    LayerNorm,
    SimpleGPT,
    SimpleGPTConfig,
    SimpleGPTTrainer,
    TransformerConfig,
    load_simple_gpt,
    save_simple_gpt,
)
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 8
PAD_ID = 0
VOCAB = N_ITEMS + 2
MAX_POSITIONS = 8


def _config(**overrides) -> TransformerConfig:
    defaults = dict(d_model=16, n_heads=4, n_layers=2, dropout=0.0)
    return TransformerConfig(**{**defaults, **overrides})


def _seqs(rows, n_items=N_ITEMS):
    return ItemSequences.from_rows(rows, n_items=n_items)


def _cycle_rows(n_rows=48, length=4, n_items=N_ITEMS):
    """Histories drawn from a single cycle, so ``next(i) == (i + 1) % n_items``."""
    return [
        [(start + step) % n_items for step in range(length)] for start in range(n_rows)
    ]


def _batcher(max_length=12, **kwargs):
    return SequenceBatcher(ItemTokenizer(N_ITEMS), max_length=max_length, **kwargs)


def _trainer_config(**overrides):
    defaults = dict(
        transformer=_config(d_model=32),
        epochs=1,
        batch_size=16,
        unk_dropout=0.0,
        show_progress=False,
        seed=0,
    )
    return SimpleGPTConfig(**{**defaults, **overrides})


def _fitted_on_cycle(epochs=60, max_length=12, **overrides):
    config = _trainer_config(epochs=epochs, lr=0.01, batch_size=48, **overrides)
    return SimpleGPTTrainer(config, _batcher(max_length=max_length)).fit(
        _seqs(_cycle_rows())
    )


def _model(**overrides) -> SimpleGPT:
    return SimpleGPT(
        vocab_size=VOCAB,
        n_items=N_ITEMS,
        max_positions=MAX_POSITIONS,
        pad_id=PAD_ID,
        config=_config(**overrides),
        tie_embeddings=False,
    )


def _model_tied(**overrides) -> SimpleGPT:
    return SimpleGPT(
        vocab_size=VOCAB,
        n_items=N_ITEMS,
        max_positions=MAX_POSITIONS,
        pad_id=PAD_ID,
        config=_config(**overrides),
        tie_embeddings=True,
    )


# --------------------------------------------------------------------------
# backbone configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"d_model": 0}, "d_model must be >= 1"),
        ({"n_heads": 0}, "n_heads must be >= 1"),
        ({"n_layers": -1}, "n_layers must be >= 1"),
        ({"d_model": 18, "n_heads": 4}, "d_model must be divisible by n_heads"),
        ({"dropout": 1.0}, r"dropout must be in \[0, 1\)"),
    ],
)
def test_invalid_backbone_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _config(**kwargs)


def test_one_width_not_two():
    """A transformer's residual stream forces the embedding and hidden widths to
    agree, which is why there is no `hidden_dim` as there is for the RNN."""
    config = _config(d_model=16, n_heads=4)

    assert config.head_dim == 4
    assert not hasattr(config, "hidden_dim")
    assert not hasattr(config, "embedding_dim")


def test_bias_is_one_flag_across_every_sublayer():
    with_bias, without = _config(bias=True), _config(bias=False)

    assert Block(with_bias).ln_1.bias is not None
    assert Block(without).ln_1.bias is None
    assert Block(with_bias).attn.proj.bias is not None
    assert Block(without).attn.proj.bias is None
    assert Block(with_bias).mlp.up.bias is not None
    assert Block(without).mlp.up.bias is None


def test_layer_norm_without_bias_still_normalises():
    norm = LayerNorm(4, bias=False)

    out = norm(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))

    assert out.mean().abs() < 1e-5
    assert norm.bias is None


# --------------------------------------------------------------------------
# causality
#
# The property recurrence gave SimpleRNN for free and a transformer has to be
# told. Getting it wrong leaks the target into its own input while every shape
# check still passes, so it needs its own tests.
# --------------------------------------------------------------------------


def test_a_position_cannot_see_a_later_token():
    """Perturb one token; every state at or before it must be unchanged.

    States are one wider than tokens because of the CLS prefix, so
    ``states[:, i]`` has read tokens ``[:i]`` -- changing ``tokens[:, j]`` may
    move states from index ``j + 1`` onwards and nothing earlier.
    """
    model = _model().eval()
    tokens = torch.tensor([[2, 3, 4, 5, 6]])

    with torch.no_grad():
        before = model(tokens)
        for j in range(tokens.shape[1]):
            changed = tokens.clone()
            changed[0, j] = 7
            after = model(changed)
            torch.testing.assert_close(
                after[:, : j + 1], before[:, : j + 1], msg=f"leak at token {j}"
            )
            assert not torch.allclose(after[:, j + 1], before[:, j + 1]), (
                f"token {j} had no effect on the state that should read it"
            )


def test_attention_is_causal_by_construction_not_by_mask():
    """No mask is built or accepted because batching always pads on the right."""
    attention = CausalSelfAttention(_config())

    import inspect

    parameters = list(inspect.signature(attention.forward).parameters)
    assert parameters == ["x"], "forward must take no mask argument"


def test_the_first_state_reads_only_cls():
    """Which is what gives an empty history a defined input."""
    model = _model().eval()

    with torch.no_grad():
        one = model(torch.tensor([[2, 3, 4]]))
        other = model(torch.tensor([[5, 6, 7]]))

    torch.testing.assert_close(one[:, 0], other[:, 0])


# --------------------------------------------------------------------------
# shapes and the head
# --------------------------------------------------------------------------


def test_states_are_one_wider_than_the_tokens():
    model = _model()

    states = model(torch.tensor([[2, 3, 4], [5, 6, 0]]))

    assert states.shape == (2, 4, 16)
    assert model.score(states).shape == (2, 4, N_ITEMS)


def test_the_head_scores_the_catalog_not_the_vocabulary():
    """A special is never a target, so a column for one could only learn to be
    wrong -- and would let a misaligned objective score plausibly."""
    model = _model()

    assert model.head.out_features == N_ITEMS
    assert model.embedding.num_embeddings == VOCAB
    assert model.embedding.padding_idx == PAD_ID


def test_a_single_item_history_works():
    """Width one plus CLS is two, which the causal path must still handle."""
    model = _model()

    assert model(torch.tensor([[3]])).shape == (1, 2, 16)


def test_a_history_longer_than_the_position_table_is_refused():
    model = _model()

    with pytest.raises(ValueError, match="was built for 8"):
        model(torch.tensor([[2] * MAX_POSITIONS]))


def test_a_position_table_too_small_for_cls_and_an_item_is_refused():
    with pytest.raises(ValueError, match="max_positions must be >= 2"):
        SimpleGPT(
            vocab_size=VOCAB,
            n_items=N_ITEMS,
            max_positions=1,
            pad_id=PAD_ID,
            config=_config(),
        )


def test_tokens_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"tokens must be \(rows, length\)"):
        _model()(torch.tensor([2, 3, 4]))


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------


def test_padding_embeds_as_zero_and_stays_there():
    """`nn.Embedding` zeroes `padding_idx` at construction and the explicit
    initialisation overwrites it, so it has to be re-zeroed -- and because
    `padding_idx` holds the gradient at zero, whatever sits there is permanent."""
    model = _model()

    assert torch.all(model.embedding.weight[PAD_ID] == 0)

    model.score(model(torch.tensor([[3, PAD_ID]]))).sum().backward()

    assert torch.all(model.embedding.weight.grad[PAD_ID] == 0)


def test_a_pad_position_cannot_affect_an_earlier_real_one():
    """Right padding is what lets the causal mask stand in for a padding mask."""
    model = _model().eval()

    with torch.no_grad():
        short = model(torch.tensor([[2, 3, PAD_ID, PAD_ID]]))
        padded = model(torch.tensor([[2, 3, PAD_ID, PAD_ID, PAD_ID]]))

    # The two real tokens sit at states 1 and 2 either way.
    torch.testing.assert_close(short[:, :3], padded[:, :3])


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_the_same_seed_builds_the_same_model():
    torch.manual_seed(0)
    first = _model()
    torch.manual_seed(0)
    second = _model()

    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)


def test_dropout_is_inactive_in_eval_mode():
    model = _model(dropout=0.5).eval()
    tokens = torch.tensor([[2, 3, 4]])

    with torch.no_grad():
        torch.testing.assert_close(model(tokens), model(tokens))


# --------------------------------------------------------------------------
# trainer configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_size": 0}, "batch_size must be >= 1"),
        ({"epochs": 0}, "epochs must be >= 1"),
        ({"unk_dropout": 1.0}, r"unk_dropout must be in \[0, 1\)"),
        ({"lr": 0.0}, "lr must be > 0"),
    ],
)
def test_invalid_trainer_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SimpleGPTConfig(**kwargs)


def test_the_context_window_is_the_batchers_business():
    """It describes what the encoder reads, not the shape of the network.

    ``rstar`` carries it in both its tokenizer and its transformer config and
    needs a runtime check to keep them equal; deriving it removes the question.
    """
    assert not hasattr(SimpleGPTConfig(), "max_length")
    assert not hasattr(SimpleGPTConfig(), "block_size")


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def test_fit_returns_the_trainer_and_sizes_positions_from_the_batcher():
    trainer = SimpleGPTTrainer(_trainer_config(), _batcher(max_length=12))

    fitted = trainer.fit(_seqs(_cycle_rows(16)))

    assert fitted is trainer
    assert trainer.is_fitted and trainer.n_items == N_ITEMS
    # One slot for CLS on top of the longest history the batcher will emit.
    assert trainer.model.max_positions == 13


def test_before_fitting_nothing_is_claimed():
    trainer = SimpleGPTTrainer(_trainer_config())

    assert not trainer.is_fitted
    assert trainer.n_items is None


def test_fit_refuses_a_matrix_source():
    from scipy.sparse import csr_matrix

    with pytest.raises(TypeError, match="trains on ItemSequences"):
        SimpleGPTTrainer(_trainer_config()).fit(csr_matrix((3, N_ITEMS)))


def test_fit_refuses_an_empty_training_set():
    with pytest.raises(ValueError, match="zero sequences"):
        SimpleGPTTrainer(_trainer_config()).fit(_seqs([]))


def test_fit_refuses_data_with_no_interactions_at_all():
    with pytest.raises(ValueError, match="every history is empty"):
        SimpleGPTTrainer(_trainer_config()).fit(_seqs([[], []]))


def test_a_single_item_history_is_a_usable_example():
    """The difference CLS makes. SimpleRNN needs two items to form one example;
    here the prefix supplies the context, so one item is one target."""
    trainer = SimpleGPTTrainer(_trainer_config(), _batcher()).fit(_seqs([[3], [5]]))

    assert trainer.history[0]["positions"] == 2.0


def test_fit_refuses_an_unbounded_batcher():
    """A learned positional table cannot be extended at prediction time."""
    unbounded = SequenceBatcher(ItemTokenizer(N_ITEMS), max_length=None)

    with pytest.raises(ValueError, match="needs a bounded context"):
        SimpleGPTTrainer(_trainer_config(), unbounded).fit(_seqs(_cycle_rows(16)))


@pytest.mark.parametrize("batch_size", [1, 2, 3, 64])
def test_the_number_of_training_positions_is_batching_invariant(batch_size):
    """Every real position is a target, so grouping cannot change the count.

    Also the cheapest guard that no special becomes a target: the head has no
    column for one, so a misaligned objective would raise here rather than train.
    """
    rows = [[1, 2, 3], [4, 5], [6], []]  # 3 + 2 + 1 + 0 targets

    trainer = SimpleGPTTrainer(
        _trainer_config(batch_size=batch_size), _batcher()
    ).fit(_seqs(rows))

    assert trainer.history[0]["positions"] == 6.0


def test_training_records_a_loss_per_epoch_and_reduces_it():
    trainer = _fitted_on_cycle(epochs=40)

    losses = [record["loss"] for record in trainer.history]

    assert len(losses) == 40
    # One-based, as ELSA's history is, so it agrees with the "epoch N" bar.
    assert [r["epoch"] for r in trainer.history] == [float(i) for i in range(1, 41)]
    assert losses[-1] < losses[0], losses[:3] + losses[-3:]


def test_refitting_starts_the_history_over():
    trainer = SimpleGPTTrainer(_trainer_config(epochs=2), _batcher())
    trainer.fit(_seqs(_cycle_rows(16)))

    trainer.fit(_seqs(_cycle_rows(16)))

    assert len(trainer.history) == 2


def test_refitting_rebuilds_an_automatic_batcher_for_the_new_catalog():
    trainer = SimpleGPTTrainer(_trainer_config())
    trainer.fit(_seqs(_cycle_rows(16)))
    first_batcher = trainer.batcher
    expanded_n_items = N_ITEMS + 2

    trainer.fit(
        _seqs(
            _cycle_rows(16, n_items=expanded_n_items),
            n_items=expanded_n_items,
        )
    )

    assert trainer.batcher is not first_batcher
    assert trainer.batcher.tokenizer.n_items == expanded_n_items
    assert trainer.n_items == expanded_n_items


def test_fit_rejects_a_supplied_batcher_for_a_different_catalog():
    trainer = SimpleGPTTrainer(_trainer_config(), _batcher())

    with pytest.raises(ValueError, match=r"batcher tokenizer has 8 items.*have 10"):
        trainer.fit(_seqs([[8], [9]], n_items=N_ITEMS + 2))

    assert not trainer.is_fitted


# --------------------------------------------------------------------------
# what it learns
# --------------------------------------------------------------------------


def test_it_predicts_the_successor_of_the_last_item():
    """End to end: encoding, CLS, the causal stack, the final state and top-k."""
    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0], [5]])

    predictions = model.predict(sources, k=1)

    expected = [(int(sources.row(i)[-1]) + 1) % N_ITEMS for i in range(4)]
    assert predictions.cols[:, 0].tolist() == expected


def test_prediction_is_batching_invariant():
    """The test that catches reading the wrong position.

    Batching changes the padded width, so a model scoring from the last column
    agrees with itself at ``batch_size=1`` -- where every row fills its own
    batch -- and disagrees once short rows sit beside long ones.
    """
    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 2, 3, 4], [5], [6, 7], [2, 3, 4]])

    whole = model.predict(sources, k=4, batch_size=64, exclude_seen=False)
    one_at_a_time = model.predict(sources, k=4, batch_size=1, exclude_seen=False)

    assert whole.cols.tolist() == one_at_a_time.cols.tolist()
    torch.testing.assert_close(whole.vals, one_at_a_time.vals)


def test_order_changes_the_prediction():
    model = _fitted_on_cycle()

    forward = model.predict(_seqs([[1, 2, 3]]), k=1, exclude_seen=False)
    backward = model.predict(_seqs([[3, 2, 1]]), k=1, exclude_seen=False)

    assert forward.cols[0, 0].item() != backward.cols[0, 0].item()


def test_the_same_seed_gives_the_same_model():
    sources = _seqs([[0, 1, 2], [4, 5]])

    first = _fitted_on_cycle(epochs=3, seed=7)
    second = _fitted_on_cycle(epochs=3, seed=7)

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
        SimpleGPTTrainer(_trainer_config()).predict_on_batch(_seqs([[1]]), k=1)


def test_predict_refuses_a_matrix_source():
    from scipy.sparse import csr_matrix

    model = SimpleGPTTrainer(_trainer_config(), _batcher()).fit(_seqs(_cycle_rows(16)))

    with pytest.raises(TypeError, match="predicts from ItemSequences"):
        model.predict(csr_matrix((2, N_ITEMS)), k=2)


@pytest.mark.parametrize("k", [0, N_ITEMS + 1])
def test_predict_refuses_an_impossible_k(k):
    model = SimpleGPTTrainer(_trainer_config(), _batcher()).fit(_seqs(_cycle_rows(16)))

    with pytest.raises(ValueError, match="k must be in"):
        model.predict_on_batch(_seqs([[1, 2]]), k=k)


def test_predictions_are_shaped_and_ordered():
    model = SimpleGPTTrainer(_trainer_config(), _batcher()).fit(_seqs(_cycle_rows(16)))

    predictions = model.predict(_seqs([[1, 2], [3]]), k=4)

    assert predictions.cols.shape == (2, 4)
    assert predictions.shape == (2, N_ITEMS)
    for row in range(2):
        values = predictions.vals[row].tolist()
        assert values == sorted(values, reverse=True)


def test_no_rows_at_all():
    model = SimpleGPTTrainer(_trainer_config(), _batcher()).fit(_seqs(_cycle_rows(16)))

    assert model.predict_on_batch(_seqs([]), k=3).cols.shape == (0, 3)


def test_an_empty_history_scores_from_cls_not_from_padding():
    """The improvement over SimpleRNN's "state after one pad".

    A row with no items reads position 0, which is CLS -- a defined prior, and
    the same one for every empty row.
    """
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
    """Truncation bounds the model's memory, not what it may return."""
    model = _fitted_on_cycle(max_length=2)
    history = [0, 1, 2, 3, 4]

    predictions = model.predict(_seqs([history]), k=N_ITEMS - len(history))

    assert not set(history) & set(predictions.cols[0].tolist())
    assert model.batcher.truncated_lengths(_seqs([history])).tolist() == [2]


def test_exclude_seen_refuses_when_fewer_than_k_unseen_items_remain():
    model = _fitted_on_cycle()
    # Repeats and out-of-catalog items are not additional scoreable seen items.
    sources = _seqs([[0, 0, 1, 2, 3, 4, 5, 6, 8]], n_items=N_ITEMS + 1)

    with pytest.raises(
        ValueError,
        match="source row 0 has only 1 unseen items, fewer than k=2",
    ):
        model.predict_on_batch(sources, k=2, exclude_seen=True)


def test_masking_does_not_mutate_the_source():
    model = _fitted_on_cycle()
    sources = _seqs([[1, 2, 3]])
    before = sources.values.copy()

    model.predict(sources, k=2, exclude_seen=True)

    assert np.array_equal(sources.values, before)


# --------------------------------------------------------------------------
# unk dropout
# --------------------------------------------------------------------------


def test_unk_dropout_never_touches_padding():
    trainer = SimpleGPTTrainer(_trainer_config(unk_dropout=0.9), _batcher())
    trainer.fit(_seqs([[1, 2, 3], [4], []]))
    tokens, mask = trainer.batcher.encode(_seqs([[1, 2, 3], [4], []]))

    corrupted = trainer._with_unk_dropout(tokens, mask)

    assert corrupted[~mask].tolist() == tokens[~mask].tolist()


def test_unk_dropout_leaves_the_targets_alone():
    """A corrupted position costs no training example, because targets come from
    the clean tokens rather than from the model's input."""
    rows = [[1, 2, 3], [4, 5]]
    clean = SimpleGPTTrainer(_trainer_config(unk_dropout=0.0), _batcher()).fit(_seqs(rows))
    noisy = SimpleGPTTrainer(_trainer_config(unk_dropout=0.9), _batcher()).fit(_seqs(rows))

    assert noisy.history[0]["positions"] == clean.history[0]["positions"] == 5.0


def test_a_vocabulary_without_unk_ignores_the_rate():
    batcher = SequenceBatcher(
        ItemTokenizer(N_ITEMS, special_tokens={"pad": 0}), max_length=12
    )
    trainer = SimpleGPTTrainer(_trainer_config(unk_dropout=0.9), batcher)
    trainer.fit(_seqs(_cycle_rows(16)))
    tokens, mask = batcher.encode(_seqs([[1, 2, 3]]))

    assert batcher.tokenizer.unk_id is None
    assert trainer._with_unk_dropout(tokens, mask).tolist() == tokens.tolist()


# --------------------------------------------------------------------------
# integration with evaluation
# --------------------------------------------------------------------------


def test_it_evaluates_through_the_standard_entry_point():
    from scipy.sparse import csr_matrix

    from compresso_recsys.evaluation import evaluate_recommender
    from compresso_recsys.metrics import NDCG

    model = _fitted_on_cycle()
    sources = _seqs([[0, 1, 2], [3, 4, 5], [6, 7, 0]])
    wanted = [3, 6, 1]
    targets = csr_matrix(
        (np.ones(len(wanted), dtype=np.float32), (np.arange(len(wanted)), np.array(wanted))),
        shape=(len(wanted), N_ITEMS),
    )

    result = evaluate_recommender(
        model, source=sources, targets=targets, metrics=[NDCG(1)]
    )

    assert result.n_scored_rows == 3
    assert result["ndcg@1"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_a_reloaded_model_predicts_identically(tmp_path):
    model = _fitted_on_cycle(epochs=20)
    sources = _seqs([[0, 1, 2], [3, 4, 5], [], [7]])
    before = model.predict(sources, k=4)
    path = tmp_path / "gpt.pt"

    save_simple_gpt(path, model)
    reloaded = load_simple_gpt(path)
    after = reloaded.predict(sources, k=4)

    assert after.cols.tolist() == before.cols.tolist()
    torch.testing.assert_close(after.vals, before.vals)


def test_loading_needs_nothing_but_the_file(tmp_path):
    """The point of carrying the tokenizer: a checkpoint is self-describing."""
    path = tmp_path / "gpt.pt"
    save_simple_gpt(path, _fitted_on_cycle(epochs=2))

    reloaded = load_simple_gpt(path)

    assert reloaded.is_fitted
    assert reloaded.n_items == N_ITEMS
    assert reloaded.batcher.tokenizer.n_items == N_ITEMS
    assert reloaded.batcher.max_length == 12
    assert reloaded.cfg.transformer.d_model == 32


def test_the_vocabulary_survives_including_item_ids(tmp_path):
    """Without the ids a served model cannot say what a predicted column means."""
    ids = np.array([f"item-{j}" for j in range(N_ITEMS)], dtype=object)
    batcher = SequenceBatcher(
        ItemTokenizer(N_ITEMS, item_ids=ids), max_length=12
    )
    model = SimpleGPTTrainer(_trainer_config(epochs=2), batcher).fit(
        _seqs(_cycle_rows(16))
    )
    path = tmp_path / "gpt.pt"

    save_simple_gpt(path, model)
    reloaded = load_simple_gpt(path)

    predictions = reloaded.predict(_seqs([[0, 1, 2]]), k=2)
    resolved = reloaded.batcher.tokenizer.decode_ids(
        predictions.cols[0] + reloaded.batcher.tokenizer.n_reserved
    )
    assert all(str(name).startswith("item-") for name in resolved.tolist())


def test_history_travels_with_the_weights(tmp_path):
    model = _fitted_on_cycle(epochs=3)
    path = tmp_path / "gpt.pt"

    save_simple_gpt(path, model)

    assert load_simple_gpt(path).history == model.history


def test_saving_an_unfitted_trainer_is_refused(tmp_path):
    with pytest.raises(RuntimeError, match="must be fitted before saving"):
        save_simple_gpt(tmp_path / "gpt.pt", SimpleGPTTrainer(_trainer_config()))


def test_an_unknown_format_is_refused(tmp_path):
    path = tmp_path / "gpt.pt"
    save_simple_gpt(path, _fitted_on_cycle(epochs=1))
    state = torch.load(path, weights_only=True)
    state["format"] = 99
    torch.save(state, path)

    with pytest.raises(ValueError, match="unsupported SimpleGPT checkpoint format"):
        load_simple_gpt(path)


def test_the_device_can_be_overridden_on_load(tmp_path):
    path = tmp_path / "gpt.pt"
    save_simple_gpt(path, _fitted_on_cycle(epochs=1))

    reloaded = load_simple_gpt(path, device="cpu")

    assert reloaded.device == torch.device("cpu")
    assert reloaded.cfg.device == "cpu"


def test_the_file_reads_as_data_not_as_a_pickle(tmp_path):
    """``weights_only=True`` means loading cannot execute what the file says."""
    path = tmp_path / "gpt.pt"
    save_simple_gpt(path, _fitted_on_cycle(epochs=1))

    state = torch.load(path, weights_only=True)

    assert set(state) == {
        "format",
        "model_state",
        "config",
        "tokenizer",
        "max_length",
        "history",
    }
    assert isinstance(state["config"]["device"], str)


def test_loading_accepts_legacy_right_padding_metadata(tmp_path):
    path = tmp_path / "gpt.pt"
    save_simple_gpt(path, _fitted_on_cycle(epochs=1))
    state = torch.load(path, weights_only=True)
    state["pad_side"] = "right"
    torch.save(state, path)

    reloaded = load_simple_gpt(path)

    assert reloaded.is_fitted


def test_loading_refuses_legacy_left_padding_metadata(tmp_path):
    path = tmp_path / "gpt.pt"
    save_simple_gpt(path, _fitted_on_cycle(epochs=1))
    state = torch.load(path, weights_only=True)
    state["pad_side"] = "left"
    torch.save(state, path)

    with pytest.raises(ValueError, match="only right padding is valid"):
        load_simple_gpt(path)


def test_only_the_scored_positions_reach_the_head():
    """The head is n_items wide, so scoring padding is the dominant memory cost.

    On a 34k-item catalog at batch 128 the difference is 3.5 GB against 0.16 GB,
    because a median history of 7 sits in a batch padded to its longest row.
    Prediction has always gathered before scoring; this pins that training does
    too, by counting the rows the head actually sees.
    """
    seen = []
    trainer = SimpleGPTTrainer(_trainer_config(batch_size=4), _batcher())
    rows = [[1, 2, 3], [4], [5, 6]]  # 6 real positions in a width-3 batch

    trainer.fit(_seqs(rows))
    original = trainer.model.score
    trainer.model.score = lambda states: seen.append(states.shape) or original(states)
    trainer._train_step(
        _seqs(rows), torch.optim.SGD(trainer.model.parameters(), lr=0.0), nn.CrossEntropyLoss()
    )

    # (n_valid, d_model), not (rows, width, d_model).
    assert len(seen) == 1
    assert seen[0] == (6, 32), seen


# --------------------------------------------------------------------------
# tied embeddings
# --------------------------------------------------------------------------


def test_tying_removes_exactly_the_head_matrix():
    """Halving the parameters is the whole point, so pin the arithmetic."""
    untied, tied = _model(), _model_tied()

    saved = sum(p.numel() for p in untied.parameters()) - sum(
        p.numel() for p in tied.parameters()
    )

    assert saved == N_ITEMS * untied.config.d_model
    assert untied.head is not None and tied.head is None
    # The bias survives tying: it is a claim about the weight alone.
    assert tied.head_bias is not None and tied.head_bias.shape == (N_ITEMS,)


def test_the_tied_head_is_the_embedding_item_rows():
    """Not a copy -- the same storage, or the two would drift apart."""
    tied = _model_tied()
    tied.eval()
    states = torch.randn(3, tied.config.d_model)

    expected = states @ tied.embedding.weight[tied.item_offset :].T + tied.head_bias

    assert torch.allclose(tied.score(states), expected, atol=1e-6)
    assert tied.embedding.weight[tied.item_offset :].data_ptr() == (
        tied.embedding.weight.data_ptr()
        + tied.item_offset * tied.config.d_model * tied.embedding.weight.element_size()
    )


def test_item_offset_is_the_reserved_count():
    """Derived, not passed, so it cannot disagree with the objective's offset.

    The trainer decodes targets with ``tokens - n_reserved``; the head slices at
    ``vocab_size - n_items``. Front loading makes those the same number, and a
    mismatch would silently shift every logit by one item.
    """
    tokenizer = ItemTokenizer(N_ITEMS)

    assert _model_tied().item_offset == tokenizer.n_reserved


def test_a_head_wider_than_the_vocabulary_is_refused():
    with pytest.raises(ValueError, match="cannot exceed vocab_size"):
        SimpleGPT(
            vocab_size=N_ITEMS,
            n_items=N_ITEMS + 1,
            max_positions=MAX_POSITIONS,
            pad_id=PAD_ID,
            config=_config(),
        )


def test_the_output_side_trains_the_item_rows_and_leaves_the_specials_alone():
    """A tied embedding learns from both directions, but not pad and not unk.

    Both sit below ``item_offset`` and so never enter the head. That is correct
    rather than incidental: neither is ever a prediction target, and a pad row
    that moved would stop being the zero vector ``padding_idx`` promises.
    """
    tied = _model_tied()

    tied.score(torch.randn(4, tied.config.d_model)).sum().backward()
    grad = tied.embedding.weight.grad

    assert float(grad[PAD_ID].abs().sum()) == 0.0
    assert float(grad[1].abs().sum()) == 0.0  # unk
    assert float(grad[tied.item_offset :].abs().sum()) > 0.0
    assert bool((tied.embedding.weight[PAD_ID] == 0).all())


@pytest.mark.parametrize("tie", [False, True])
def test_both_heads_learn_the_cycle(tie):
    """Tying is a parameter claim, not a capability one."""
    model = _fitted_on_cycle(tie_embeddings=tie)

    predictions = model.predict(_seqs([[0, 1, 2], [4, 5, 6]]), k=1)

    assert predictions.cols.flatten().tolist() == [3, 7]


def test_tying_round_trips_through_a_checkpoint(tmp_path):
    """The flag lives in the config, so loading must rebuild the same head."""
    model = _fitted_on_cycle(epochs=2, tie_embeddings=True)
    sources = _seqs([[0, 1, 2], [5], [3, 4]])
    path = tmp_path / "tied.pt"

    save_simple_gpt(path, model)
    reloaded = load_simple_gpt(path)

    assert reloaded.cfg.tie_embeddings is True
    assert reloaded.model.head is None
    before = model.predict(sources, k=N_ITEMS, exclude_seen=False)
    after = reloaded.predict(sources, k=N_ITEMS, exclude_seen=False)
    assert torch.equal(before.cols, after.cols)
    assert torch.allclose(before.vals, after.vals, atol=1e-6)


def test_an_untied_checkpoint_still_loads_untied(tmp_path):
    """Tying is the default, so a file saved without it must not acquire it.

    The flag has to travel with the weights: a tied checkpoint reloaded untied
    would silently gain a head of fresh random numbers, and an untied one
    reloaded tied would drop the head it was trained with.
    """
    path = tmp_path / "untied.pt"

    save_simple_gpt(path, _fitted_on_cycle(epochs=2, tie_embeddings=False))
    reloaded = load_simple_gpt(path)

    assert reloaded.cfg.tie_embeddings is False
    assert isinstance(reloaded.model.head, nn.Linear)


def test_tying_is_the_default():
    """Measured better on every split tried, so an unconfigured model gets it."""
    model = SimpleGPT(
        vocab_size=VOCAB,
        n_items=N_ITEMS,
        max_positions=MAX_POSITIONS,
        pad_id=PAD_ID,
        config=_config(),
    )

    assert model.tie_embeddings is True
    assert model.head is None
    assert SimpleGPTConfig().tie_embeddings is True
    assert SimpleGPTConfig(tie_embeddings=False).tie_embeddings is False


# --------------------------------------------------------------------------
# initialisation
# --------------------------------------------------------------------------


def test_every_weight_starts_at_the_gpt2_scale():
    """PyTorch's nn.Linear default is ~2.5x wider at d_model=128, and leaving it
    there is a silent departure from the architecture this model claims."""
    model = _model(d_model=128, n_heads=4, n_layers=2)

    for name, param in model.named_parameters():
        if "ln" in name or param.dim() < 2:
            continue
        if name.endswith(("attn.proj.weight", "mlp.down.weight")):
            continue  # scaled separately, asserted below
        assert 0.015 < param.std().item() < 0.025, f"{name} std {param.std().item()}"


def test_residual_projections_are_scaled_by_depth():
    """GPT-2's 1/sqrt(2 * n_layers) on whatever writes into the residual stream.

    Each block adds twice, so without this the stream's variance grows with
    depth and a deeper model starts further from usable.
    """
    for n_layers in (1, 2, 4):
        model = _model(d_model=64, n_heads=4, n_layers=n_layers)
        expected = 0.02 / math.sqrt(2 * n_layers)
        for block in model.blocks:
            assert block.attn.proj.weight.std().item() == pytest.approx(
                expected, rel=0.25
            )
            assert block.mlp.down.weight.std().item() == pytest.approx(
                expected, rel=0.25
            )


def test_the_pad_row_is_still_zero_after_the_new_init():
    """self.apply() walks the embedding too, so the re-zero has to come last."""
    model = _model_tied()

    assert bool((model.embedding.weight[PAD_ID] == 0).all())


# --------------------------------------------------------------------------
# learning-rate schedule
# --------------------------------------------------------------------------


def test_cosine_is_the_default_and_constant_builds_no_scheduler():
    """Cosine won on all three splits, but only in combination with the GPT-2
    init -- under the old init it lost on Office. The two are one change."""
    assert SimpleGPTConfig().lr_schedule == "cosine"

    flat = SimpleGPTTrainer(_trainer_config(lr_schedule="constant"), _batcher())
    assert flat._build_scheduler(
        torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1), 100
    ) is None


def test_cosine_warms_up_then_decays_to_the_floor():
    trainer = SimpleGPTTrainer(
        _trainer_config(lr=0.1, lr_schedule="cosine", warmup_fraction=0.1,
                        min_lr_ratio=0.05),
        _batcher(),
    )
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.1)
    scheduler = trainer._build_scheduler(optimizer, 100)

    rates = []
    for _ in range(100):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()  # order fit uses: the optimizer moves, then the schedule
        scheduler.step()

    warm = rates[:11]
    assert warm == sorted(warm), "warmup must not decrease"
    assert rates[0] > 0.0, "step zero must still train"
    tail = rates[10:]
    assert tail == sorted(tail, reverse=True), "post-warmup must not increase"
    assert rates[10] == pytest.approx(0.1, rel=0.02), "peak is the configured lr"
    assert rates[-1] == pytest.approx(0.1 * 0.05), "final update uses the floor"


def test_a_schedule_shorter_than_its_warmup_does_not_divide_by_zero():
    """A tiny run is a real case: one batch, one epoch."""
    trainer = SimpleGPTTrainer(
        _trainer_config(lr_schedule="cosine", warmup_fraction=0.9), _batcher()
    )
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    scheduler = trainer._build_scheduler(optimizer, 1)

    used_lr = optimizer.param_groups[0]["lr"]
    optimizer.step()
    scheduler.step()

    assert used_lr == pytest.approx(0.1)


def test_fit_records_the_rate_used_for_the_epochs_final_update():
    """The scheduler advances after the update, so history must not report it."""
    model = SimpleGPTTrainer(
        _trainer_config(
            epochs=2,
            batch_size=8,
            lr=0.1,
            lr_schedule="cosine",
            warmup_fraction=0.5,
            min_lr_ratio=0.1,
        ),
        _batcher(),
    ).fit(_seqs([[0], [1]]))

    # One batch per epoch: the first update warms up at 0.05, and the second is
    # the final update, so it uses the configured 0.01 floor.
    assert [entry["lr"] for entry in model.history] == pytest.approx([0.05, 0.01])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"lr_schedule": "linear"}, "lr_schedule must be"),
        ({"warmup_fraction": 1.0}, r"warmup_fraction must be in \[0, 1\)"),
        ({"warmup_fraction": -0.1}, r"warmup_fraction must be in \[0, 1\)"),
        ({"min_lr_ratio": 0.0}, r"min_lr_ratio must be in \(0, 1\]"),
        ({"min_lr_ratio": 1.5}, r"min_lr_ratio must be in \(0, 1\]"),
    ],
)
def test_invalid_schedule_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _trainer_config(**kwargs)


def test_the_schedule_survives_a_checkpoint(tmp_path):
    model = _fitted_on_cycle(epochs=2, lr_schedule="cosine", min_lr_ratio=0.25)
    path = tmp_path / "scheduled.pt"

    save_simple_gpt(path, model)
    reloaded = load_simple_gpt(path)

    assert reloaded.cfg.lr_schedule == "cosine"
    assert reloaded.cfg.min_lr_ratio == 0.25
