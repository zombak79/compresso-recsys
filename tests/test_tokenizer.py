from __future__ import annotations

import numpy as np
import pytest

from compresso_recsys.models.tokenizer import ItemTokenizer, Tokenizer

N_ITEMS = 10


def _tok(**overrides) -> ItemTokenizer:
    kwargs = dict(n_items=N_ITEMS)
    kwargs.update(overrides)
    return ItemTokenizer(**kwargs)


# --------------------------------------------------------------------------
# layout
# --------------------------------------------------------------------------


def test_specials_come_first_and_the_catalog_is_offset_after_them():
    tok = _tok()

    assert tok.pad_id == 0
    assert tok.unk_id == 1
    assert tok.n_reserved == 2
    assert tok.vocab_size == N_ITEMS + 2
    assert tok.encode_indices([0, 9]).tolist() == [2, 11]


def test_growing_the_catalog_moves_no_token():
    """The reason specials are first: catalog growth is what actually happens.

    Stage catalogs nest by prefix, cold-start catalogs append, and a partial fit
    extends the embedding table at the end -- where a `cat` splices optimizer
    state correctly instead of permuting it.
    """
    before = _tok(item_ids=np.array([f"i{j}" for j in range(N_ITEMS)], dtype=object))

    after = before.extended(item_ids=["new-a", "new-b"])

    assert after.n_items == N_ITEMS + 2
    assert after.vocab_size == before.vocab_size + 2
    assert after.pad_id == before.pad_id
    assert after.unk_id == before.unk_id
    assert (
        after.encode_indices(np.arange(N_ITEMS)).tolist()
        == before.encode_indices(np.arange(N_ITEMS)).tolist()
    )


def test_a_reserve_lets_a_special_be_added_later_without_moving_items():
    """Front-loading's one cost, and the thing that removes it."""
    early = _tok(special_tokens={"pad": 0, "unk": 1}, n_reserved=4)
    later = _tok(special_tokens={"pad": 0, "unk": 1, "mask": 2}, n_reserved=4)

    assert early.n_reserved == later.n_reserved == 4
    assert early.vocab_size == later.vocab_size
    assert (
        early.encode_indices([0, 5]).tolist() == later.encode_indices([0, 5]).tolist()
    )
    assert later.token_id("mask") == 2


def test_without_a_reserve_adding_a_special_does_move_items():
    """Stated so the reserve is a decision rather than an accident."""
    two = _tok(special_tokens={"pad": 0, "unk": 1})
    three = _tok(special_tokens={"pad": 0, "unk": 1, "mask": 2})

    assert two.encode_indices([0]).tolist() != three.encode_indices([0]).tolist()


def test_a_name_carries_no_meaning_beyond_pad():
    """cls may be named and simply gets an id; the model decides what it means."""
    tok = _tok(special_tokens={"pad": 0, "unk": 1, "cls": 2, "mask": 3})

    assert tok.token_id("cls") == 2
    assert tok.n_reserved == 4
    assert tok.encode_indices([0]).tolist() == [4]


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_items": 0}, "n_items must be >= 1"),
        ({"special_tokens": {"unk": 0}}, "must include 'pad'"),
        ({"special_tokens": {"pad": 0, "unk": 2}}, "exactly 0..n-1 with no gaps"),
        ({"special_tokens": {"pad": 1, "unk": 0}, "n_reserved": 1}, "must be >= the 2"),
    ],
)
def test_invalid_construction_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _tok(**kwargs)


def test_reordered_special_ids_are_fine_as_long_as_they_are_dense():
    tok = _tok(special_tokens={"unk": 0, "pad": 1})

    assert tok.pad_id == 1 and tok.unk_id == 0
    assert tok.n_reserved == 2


@pytest.mark.parametrize(
    ("item_ids", "message"),
    [
        (np.arange(N_ITEMS), "must not be an integer dtype"),
        (np.array(["a", "b"], dtype=object), f"with {N_ITEMS} entries"),
        (np.array(["a"] * N_ITEMS, dtype=object), "must be unique"),
    ],
)
def test_invalid_item_ids_are_refused(item_ids, message):
    with pytest.raises(ValueError, match=message):
        _tok(item_ids=item_ids)


def test_the_integer_id_refusal_names_the_fix():
    """Integer IDs would make encode()'s dtype dispatch ambiguous."""
    with pytest.raises(ValueError, match=r"astype\(str\)"):
        _tok(item_ids=np.arange(N_ITEMS))


def test_an_unknown_special_name_is_a_key_error():
    with pytest.raises(KeyError, match="'mask' is not a special token"):
        _tok().token_id("mask")


# --------------------------------------------------------------------------
# encoding indices
# --------------------------------------------------------------------------


def test_an_item_beyond_the_catalog_becomes_unk_and_keeps_its_position():
    """Why UNK exists: dropping the item would assert a transition that never
    happened, and this is the normal case for a later split stage."""
    tok = _tok()

    tokens = tok.encode_indices([3, 40, 4])

    assert tokens.tolist() == [5, tok.unk_id, 6]
    assert len(tokens) == 3


def test_without_unk_an_unrepresentable_item_is_an_error():
    """A model with no UNK slot genuinely cannot express one, so it says so."""
    tok = _tok(special_tokens={"pad": 0})

    assert tok.unk_id is None
    assert tok.encode_indices([0, 9]).tolist() == [1, 10]
    with pytest.raises(ValueError, match="has no 'unk' token"):
        tok.encode_indices([0, 40])


def test_a_negative_index_is_never_valid():
    with pytest.raises(ValueError, match="must be non-negative"):
        _tok().encode_indices([-1])


def test_encoding_an_empty_batch_of_values():
    assert _tok().encode_indices([]).tolist() == []


# --------------------------------------------------------------------------
# encoding IDs
# --------------------------------------------------------------------------


def _with_ids() -> ItemTokenizer:
    return _tok(item_ids=np.array([f"i{j:02d}" for j in range(N_ITEMS)], dtype=object))


def test_ids_encode_to_the_same_tokens_as_their_indices():
    tok = _with_ids()

    assert (
        tok.encode_ids(["i00", "i09"]).tolist()
        == tok.encode_indices([0, 9]).tolist()
    )


def test_an_unknown_id_becomes_unk():
    tok = _with_ids()

    assert tok.encode_ids(["i00", "nope"]).tolist() == [2, tok.unk_id]


def test_the_id_path_needs_item_ids():
    tok = _tok()

    assert not tok.has_ids
    with pytest.raises(RuntimeError, match="without item_ids"):
        tok.encode_ids(["i00"])


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_encode_reads_integers_as_indices_and_anything_else_as_ids():
    tok = _with_ids()

    assert tok.encode(np.array([0, 9])).tolist() == [2, 11]
    assert tok.encode(np.array(["i00", "i09"], dtype=object)).tolist() == [2, 11]


def test_decode_cannot_dispatch_so_it_picks_ids():
    """Both decoders take tokens, so the input carries no signal.

    Choosing by whether item_ids happens to be present would make the return
    type depend on construction, silently. So it picks IDs and fails loudly.
    """
    with_ids, without = _with_ids(), _tok()

    assert with_ids.decode([2]).tolist() == ["i00"]
    with pytest.raises(RuntimeError, match="without item_ids"):
        without.decode([2])
    # The index decoder always works.
    assert without.decode_indices([2]).tolist() == [0]


# --------------------------------------------------------------------------
# decoding
# --------------------------------------------------------------------------


def test_decoding_round_trips_every_catalog_item():
    tok = _tok()
    indices = np.arange(N_ITEMS)

    assert tok.decode_indices(tok.encode_indices(indices)).tolist() == indices.tolist()


def test_a_special_decodes_to_minus_one_or_to_its_name():
    tok = _with_ids()

    assert tok.decode_indices([tok.pad_id, tok.unk_id, 2]).tolist() == [-1, -1, 0]
    assert tok.decode_ids([tok.pad_id, tok.unk_id, 2]).tolist() == [
        "pad",
        "unk",
        "i00",
    ]


def test_an_unnamed_reserved_slot_decodes_as_reserved():
    tok = _with_ids()
    tok = ItemTokenizer(
        N_ITEMS, n_reserved=4, item_ids=tok.item_ids, special_tokens={"pad": 0}
    )

    assert tok.decode_ids([3]).tolist() == ["reserved"]


def test_decode_preserves_shape():
    tok = _tok()

    assert tok.decode_indices(np.array([[2, 3], [4, 0]])).tolist() == [[0, 1], [2, -1]]


# --------------------------------------------------------------------------
# growth
# --------------------------------------------------------------------------


def test_extending_by_count_needs_a_tokenizer_without_ids():
    grown = _tok().extended(n_items=N_ITEMS + 5)

    assert grown.n_items == N_ITEMS + 5
    with pytest.raises(RuntimeError, match="has item_ids"):
        _with_ids().extended(n_items=N_ITEMS + 5)


def test_extending_by_id_needs_a_tokenizer_with_ids():
    with pytest.raises(RuntimeError, match="no item_ids"):
        _tok().extended(item_ids=["x"])


def test_a_catalog_may_not_shrink():
    with pytest.raises(ValueError, match="may only grow"):
        _tok().extended(n_items=N_ITEMS - 1)


def test_extending_needs_exactly_one_of_the_two_arguments():
    with pytest.raises(ValueError, match="exactly one"):
        _tok().extended()
    with pytest.raises(ValueError, match="exactly one"):
        _with_ids().extended(n_items=11, item_ids=["x"])


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_round_tripping_through_a_dict_preserves_everything():
    tok = _tok(
        special_tokens={"pad": 0, "unk": 1, "mask": 2},
        n_reserved=5,
        item_ids=np.array([f"i{j:02d}" for j in range(N_ITEMS)], dtype=object),
    )

    restored = ItemTokenizer.from_dict(tok.to_dict())

    assert restored.n_items == tok.n_items
    assert restored.n_reserved == tok.n_reserved
    assert dict(restored.special_tokens) == dict(tok.special_tokens)
    assert restored.item_ids.tolist() == tok.item_ids.tolist()
    assert restored.encode_ids(["i03"]).tolist() == tok.encode_ids(["i03"]).tolist()


def test_item_ids_are_saved_by_default_because_serving_is_why_they_exist():
    tok = _with_ids()

    assert "item_ids" in tok.to_dict()
    assert "item_ids" not in tok.to_dict(include_item_ids=False)
    assert not ItemTokenizer.from_dict(tok.to_dict(include_item_ids=False)).has_ids


def test_the_state_is_json_serialisable():
    import json

    tok = _with_ids()

    assert ItemTokenizer.from_dict(json.loads(json.dumps(tok.to_dict()))).has_ids


# --------------------------------------------------------------------------
# immutability and the contract
# --------------------------------------------------------------------------


def test_the_tokenizer_satisfies_the_protocol_structurally():
    assert isinstance(_tok(), Tokenizer)

    class Minimal:
        pad_id = 0

        def encode_indices(self, values):
            return np.asarray(values)

    assert isinstance(Minimal(), Tokenizer)


def test_item_ids_cannot_be_edited_through_the_tokenizer():
    tok = _with_ids()

    with pytest.raises(ValueError, match="read-only"):
        tok.item_ids[0] = "changed"


def test_constructing_does_not_freeze_the_callers_array():
    ids = np.array([f"i{j:02d}" for j in range(N_ITEMS)], dtype=object)

    _tok(item_ids=ids)

    ids[0] = "still writable"


def test_the_special_mapping_cannot_be_edited():
    with pytest.raises(TypeError):
        _tok().special_tokens["pad"] = 7
