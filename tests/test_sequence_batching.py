from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 10


def _seqs(rows, n_items=N_ITEMS):
    return ItemSequences.from_rows(rows, n_items=n_items)


def _batcher(n_items=N_ITEMS, **kwargs):
    """A batcher over a plain pad+unk vocabulary."""
    return SequenceBatcher(ItemTokenizer(n_items), **kwargs)


def _items(batcher, indices):
    """The tokens `indices` encode to, so tests state relationships not numbers."""
    return batcher.tokenizer.encode_indices(indices).tolist()


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------


def test_right_padding_puts_content_first():
    batcher = _batcher(pad_side="right")

    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4]]))

    pad = batcher.pad_id
    assert tokens.tolist() == [_items(batcher, [1, 2, 3]), _items(batcher, [4]) + [pad, pad]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_left_padding_puts_the_newest_item_last():
    """What a causal transformer needs: prediction always reads position -1."""
    batcher = _batcher(pad_side="left")

    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4]]))

    pad = batcher.pad_id
    assert tokens.tolist() == [_items(batcher, [1, 2, 3]), [pad, pad] + _items(batcher, [4])]
    assert mask.tolist() == [[True, True, True], [False, False, True]]
    assert tokens[:, -1].tolist() == _items(batcher, [3, 4])


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_the_mask_recovers_every_history_exactly(pad_side):
    rows = [[1, 2, 3], [], [4, 4, 5], [9]]
    batcher = _batcher(pad_side=pad_side)
    sequences = _seqs(rows)

    tokens, mask = batcher.encode(sequences)

    for row, expected in enumerate(rows):
        # Decoding is how a history is recovered now that tokens are offset.
        recovered = batcher.tokenizer.decode_indices(tokens[row][mask[row]])
        assert recovered.tolist() == expected, (pad_side, row)


def test_padding_never_collides_with_a_catalog_item():
    """Item 0 is the case that would collide if the offset were forgotten."""
    batcher = _batcher()
    pad = batcher.pad_id

    tokens, mask = batcher.encode(_seqs([[0], [0, 0, 0]]))

    assert tokens[0].tolist() == _items(batcher, [0]) + [pad, pad]
    assert not mask[0, 1:].any()
    # pad_id is itself a small integer, so the guarantee is that no catalog item
    # encodes to it -- not that it sits outside some numeric range.
    assert pad not in _items(batcher, range(N_ITEMS))


def test_dtypes_suit_an_embedding_lookup():
    tokens, mask = _batcher().encode(_seqs([[1, 2]]))

    assert tokens.dtype == torch.int64
    assert mask.dtype == torch.bool


# --------------------------------------------------------------------------
# empty and degenerate batches
# --------------------------------------------------------------------------


def test_an_empty_history_is_all_padding():
    batcher = _batcher()

    tokens, mask = batcher.encode(_seqs([[], [7]]))

    assert not mask[0].any()
    assert batcher.has_history(mask).tolist() == [False, True]


def test_a_batch_of_only_empty_histories_still_has_a_usable_shape():
    """Width is floored at one so an embedding lookup does not see a zero axis."""
    tokens, mask = _batcher().encode(_seqs([[], []]))

    assert tokens.shape == (2, 1)
    assert not mask.any()


def test_no_rows_at_all():
    tokens, mask = _batcher().encode(_seqs([]))

    assert tokens.shape == (0, 1)
    assert mask.shape == (0, 1)


def test_a_wider_source_catalog_becomes_unk_rather_than_an_error():
    """The normal case for a later split stage, and the fix for a real bug.

    Dropping the unknown item instead would join items 1 and 3 as if they had
    been consecutive, which they were not.
    """
    batcher = _batcher(n_items=4)
    unk = batcher.tokenizer.unk_id

    tokens, mask = batcher.encode(_seqs([[1, 7, 3]], n_items=9))

    assert tokens[0].tolist() == [_items(batcher, [1])[0], unk, _items(batcher, [3])[0]]
    assert mask[0].tolist() == [True, True, True]


def test_a_wider_source_is_refused_when_the_vocabulary_has_no_unk():
    batcher = SequenceBatcher(ItemTokenizer(4, special_tokens={"pad": 0}))

    with pytest.raises(ValueError, match="has no 'unk' token"):
        batcher.encode(_seqs([[1, 7]], n_items=9))


# --------------------------------------------------------------------------
# truncation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_truncation_keeps_the_most_recent_items(pad_side):
    """A context window is a claim about recency, not about where a history began."""
    batcher = _batcher(max_length=3, pad_side=pad_side)

    tokens, mask = batcher.encode(_seqs([[1, 2, 3, 4, 5, 6]]))

    assert batcher.tokenizer.decode_indices(tokens[0][mask[0]]).tolist() == [4, 5, 6]


def test_truncation_does_not_pad_shorter_rows_to_the_limit():
    batcher = _batcher(max_length=8)

    tokens, _ = batcher.encode(_seqs([[1, 2], [3]]))

    assert tokens.shape[1] == 2, "width follows the batch, not max_length"


def test_truncated_lengths_reports_what_encode_will_use():
    batcher = _batcher(max_length=2)

    lengths = batcher.truncated_lengths(_seqs([[1, 2, 3, 4], [5], []]))

    assert lengths.tolist() == [2, 1, 0]


# --------------------------------------------------------------------------
# reading the final state
# --------------------------------------------------------------------------


def test_final_positions_point_at_real_tokens_not_padding():
    """The bug this method exists to prevent: reading [:, -1] under right padding."""
    batcher = _batcher(pad_side="right")
    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4], [5, 6]]))

    positions = batcher.final_positions(mask)

    assert positions.tolist() == [2, 0, 1]
    last_tokens = tokens.gather(1, positions.view(-1, 1)).squeeze(1)
    assert last_tokens.tolist() == _items(batcher, [3, 4, 6])
    # Reading the last column instead: two of the three rows would have scored
    # from a pad embedding, and only the longest row would be right.
    pad = batcher.pad_id
    assert tokens[:, -1].tolist() == [_items(batcher, [3])[0], pad, pad]


def test_final_positions_are_the_last_column_under_left_padding():
    batcher = _batcher(pad_side="left")
    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4], [5, 6]]))

    positions = batcher.final_positions(mask)

    assert positions.tolist() == [2, 2, 2]
    assert tokens[:, -1].tolist() == _items(batcher, [3, 4, 6])


def test_final_position_of_an_empty_row_is_zero_and_flagged():
    batcher = _batcher()
    _, mask = batcher.encode(_seqs([[], [1, 2]]))

    assert batcher.final_positions(mask).tolist() == [0, 1]
    assert batcher.has_history(mask).tolist() == [False, True]


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_gather_final_selects_the_right_state(pad_side):
    batcher = _batcher(pad_side=pad_side)
    _, mask = batcher.encode(_seqs([[1, 2, 3], [4]]))
    # A state whose value encodes its own position, so a wrong pick is visible.
    states = (
        torch.arange(mask.shape[1], dtype=torch.float32)
        .view(1, -1, 1)
        .expand(mask.shape[0], -1, 2)
        .contiguous()
    )

    final = batcher.gather_final(states, mask)

    expected = batcher.final_positions(mask).to(torch.float32)
    assert final.shape == (2, 2)
    assert final[:, 0].tolist() == expected.tolist()


def test_gather_final_checks_its_shapes():
    batcher = _batcher()
    _, mask = batcher.encode(_seqs([[1, 2]]))

    with pytest.raises(ValueError, match="must be \\(rows, length, dim\\)"):
        batcher.gather_final(torch.zeros(1, 2), mask)
    with pytest.raises(ValueError, match="must agree on rows and length"):
        batcher.gather_final(torch.zeros(1, 5, 3), mask)


# --------------------------------------------------------------------------
# what the batcher deliberately does not do
# --------------------------------------------------------------------------


def test_the_batcher_has_no_objective():
    """Shift, masking and negatives differ per architecture and stay in trainers."""
    # The instance surface, since required dataclass fields are not class
    # attributes and would otherwise be missed.
    batcher = _batcher()
    surface = {name for name in dir(batcher) if not name.startswith("_")}

    assert not surface & {"shift", "shifted_targets", "mask_positions", "negatives"}
    assert surface == {
        "encode",
        "final_positions",
        "gather_final",
        "has_history",
        "max_length",
        "pad_id",
        "pad_side",
        "tokenizer",
        "truncated_lengths",
    }
    # The vocabulary lives on the tokenizer, not here.
    assert not surface & {"n_items", "vocab_size", "special_tokens", "token_id"}
