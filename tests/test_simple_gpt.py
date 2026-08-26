from __future__ import annotations

import pytest
import torch

from compresso_recsys.models.simple_gpt import (
    Block,
    CausalSelfAttention,
    LayerNorm,
    SimpleGPT,
    TransformerConfig,
)

N_ITEMS = 8
PAD_ID = 0
VOCAB = N_ITEMS + 2
MAX_POSITIONS = 8


def _config(**overrides) -> TransformerConfig:
    defaults = dict(d_model=16, n_heads=4, n_layers=2, dropout=0.0)
    return TransformerConfig(**{**defaults, **overrides})


def _model(**overrides) -> SimpleGPT:
    return SimpleGPT(
        vocab_size=VOCAB,
        n_items=N_ITEMS,
        max_positions=MAX_POSITIONS,
        pad_id=PAD_ID,
        config=_config(**overrides),
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
    """No mask is built or accepted -- the padding side is what makes that safe."""
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
