from __future__ import annotations

import numpy as np
import pytest
import torch

from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.sequences import ItemSequences

N_ITEMS = 10


def _seqs(rows):
    return ItemSequences.from_rows(rows, n_items=N_ITEMS)


# --------------------------------------------------------------------------
# vocabulary layout
# --------------------------------------------------------------------------


def test_catalog_ids_are_the_identity_and_specials_come_after():
    batcher = SequenceBatcher(n_items=N_ITEMS)

    assert batcher.pad_id == N_ITEMS
    assert batcher.vocab_size == N_ITEMS + 1


def test_adding_a_special_token_does_not_move_any_catalog_id():
    """The reason specials are appended rather than reserved at the front.

    Front-loading would shift every item by one, invalidating any model already
    trained the moment a second special token is introduced.
    """
    before = SequenceBatcher(n_items=N_ITEMS, special_tokens=("pad",))
    after = SequenceBatcher(n_items=N_ITEMS, special_tokens=("pad", "mask"))

    tokens_before, _ = before.encode(_seqs([[0, 5, 9]]))
    tokens_after, _ = after.encode(_seqs([[0, 5, 9]]))

    assert tokens_before[0, :3].tolist() == [0, 5, 9]
    assert tokens_after[0, :3].tolist() == [0, 5, 9]
    assert before.pad_id == after.pad_id == N_ITEMS
    assert after.token_id("mask") == N_ITEMS + 1
    assert after.vocab_size == N_ITEMS + 2


def test_catalog_logits_drops_only_the_special_columns():
    batcher = SequenceBatcher(n_items=N_ITEMS, special_tokens=("pad", "mask"))
    logits = torch.arange(batcher.vocab_size, dtype=torch.float32).expand(3, -1)

    catalog = batcher.catalog_logits(logits)

    assert catalog.shape == (3, N_ITEMS)
    assert catalog[0].tolist() == list(range(N_ITEMS))


def test_unknown_special_token_is_an_error():
    with pytest.raises(KeyError, match="'mask' is not a special token"):
        SequenceBatcher(n_items=N_ITEMS).token_id("mask")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_items": 0}, "n_items must be >= 1"),
        ({"n_items": 5, "max_length": 0}, "max_length must be >= 1"),
        ({"n_items": 5, "pad_side": "middle"}, "pad_side must be"),
        ({"n_items": 5, "special_tokens": ("mask",)}, "must include 'pad'"),
        ({"n_items": 5, "special_tokens": ("pad", "pad")}, "must be unique"),
    ],
)
def test_invalid_configuration_is_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        SequenceBatcher(**kwargs)


# --------------------------------------------------------------------------
# padding
# --------------------------------------------------------------------------


def test_right_padding_puts_content_first():
    batcher = SequenceBatcher(n_items=N_ITEMS, pad_side="right")

    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4]]))

    assert tokens.tolist() == [[1, 2, 3], [4, N_ITEMS, N_ITEMS]]
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_left_padding_puts_the_newest_item_last():
    """What a causal transformer needs: prediction always reads position -1."""
    batcher = SequenceBatcher(n_items=N_ITEMS, pad_side="left")

    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4]]))

    assert tokens.tolist() == [[1, 2, 3], [N_ITEMS, N_ITEMS, 4]]
    assert mask.tolist() == [[True, True, True], [False, False, True]]
    assert tokens[:, -1].tolist() == [3, 4]


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_the_mask_recovers_every_history_exactly(pad_side):
    rows = [[1, 2, 3], [], [4, 4, 5], [9]]
    batcher = SequenceBatcher(n_items=N_ITEMS, pad_side=pad_side)
    sequences = _seqs(rows)

    tokens, mask = batcher.encode(sequences)

    for row, expected in enumerate(rows):
        recovered = tokens[row][mask[row]].tolist()
        assert recovered == expected, (pad_side, row)


def test_padding_never_collides_with_a_catalog_item():
    """pad_id sits outside the catalog, so a pad cannot be mistaken for item 0."""
    batcher = SequenceBatcher(n_items=N_ITEMS)

    tokens, mask = batcher.encode(_seqs([[0], [0, 0, 0]]))

    assert tokens[0].tolist() == [0, N_ITEMS, N_ITEMS]
    assert not mask[0, 1:].any()
    assert batcher.pad_id not in range(N_ITEMS)


def test_dtypes_suit_an_embedding_lookup():
    tokens, mask = SequenceBatcher(n_items=N_ITEMS).encode(_seqs([[1, 2]]))

    assert tokens.dtype == torch.int64
    assert mask.dtype == torch.bool


# --------------------------------------------------------------------------
# empty and degenerate batches
# --------------------------------------------------------------------------


def test_an_empty_history_is_all_padding():
    batcher = SequenceBatcher(n_items=N_ITEMS)

    tokens, mask = batcher.encode(_seqs([[], [7]]))

    assert not mask[0].any()
    assert batcher.has_history(mask).tolist() == [False, True]


def test_a_batch_of_only_empty_histories_still_has_a_usable_shape():
    """Width is floored at one so an embedding lookup does not see a zero axis."""
    tokens, mask = SequenceBatcher(n_items=N_ITEMS).encode(_seqs([[], []]))

    assert tokens.shape == (2, 1)
    assert not mask.any()


def test_no_rows_at_all():
    tokens, mask = SequenceBatcher(n_items=N_ITEMS).encode(_seqs([]))

    assert tokens.shape == (0, 1)
    assert mask.shape == (0, 1)


def test_a_wider_catalog_than_the_batcher_is_refused():
    batcher = SequenceBatcher(n_items=4)

    with pytest.raises(ValueError, match="built for 4"):
        batcher.encode(ItemSequences.from_rows([[1]], n_items=9))


# --------------------------------------------------------------------------
# truncation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_truncation_keeps_the_most_recent_items(pad_side):
    """A context window is a claim about recency, not about where a history began."""
    batcher = SequenceBatcher(n_items=N_ITEMS, max_length=3, pad_side=pad_side)

    tokens, mask = batcher.encode(_seqs([[1, 2, 3, 4, 5, 6]]))

    assert tokens[0][mask[0]].tolist() == [4, 5, 6]


def test_truncation_does_not_pad_shorter_rows_to_the_limit():
    batcher = SequenceBatcher(n_items=N_ITEMS, max_length=8)

    tokens, _ = batcher.encode(_seqs([[1, 2], [3]]))

    assert tokens.shape[1] == 2, "width follows the batch, not max_length"


def test_truncated_lengths_reports_what_encode_will_use():
    batcher = SequenceBatcher(n_items=N_ITEMS, max_length=2)

    lengths = batcher.truncated_lengths(_seqs([[1, 2, 3, 4], [5], []]))

    assert lengths.tolist() == [2, 1, 0]


# --------------------------------------------------------------------------
# reading the final state
# --------------------------------------------------------------------------


def test_final_positions_point_at_real_tokens_not_padding():
    """The bug this method exists to prevent: reading [:, -1] under right padding."""
    batcher = SequenceBatcher(n_items=N_ITEMS, pad_side="right")
    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4], [5, 6]]))

    positions = batcher.final_positions(mask)

    assert positions.tolist() == [2, 0, 1]
    last_tokens = tokens.gather(1, positions.view(-1, 1)).squeeze(1)
    assert last_tokens.tolist() == [3, 4, 6]
    # Reading the last column instead: two of the three rows would have scored
    # from a pad embedding, and only the longest row would be right.
    assert tokens[:, -1].tolist() == [3, N_ITEMS, N_ITEMS]


def test_final_positions_are_the_last_column_under_left_padding():
    batcher = SequenceBatcher(n_items=N_ITEMS, pad_side="left")
    tokens, mask = batcher.encode(_seqs([[1, 2, 3], [4], [5, 6]]))

    positions = batcher.final_positions(mask)

    assert positions.tolist() == [2, 2, 2]
    assert tokens[:, -1].tolist() == [3, 4, 6]


def test_final_position_of_an_empty_row_is_zero_and_flagged():
    batcher = SequenceBatcher(n_items=N_ITEMS)
    _, mask = batcher.encode(_seqs([[], [1, 2]]))

    assert batcher.final_positions(mask).tolist() == [0, 1]
    assert batcher.has_history(mask).tolist() == [False, True]


@pytest.mark.parametrize("pad_side", ["right", "left"])
def test_gather_final_selects_the_right_state(pad_side):
    batcher = SequenceBatcher(n_items=N_ITEMS, pad_side=pad_side)
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
    batcher = SequenceBatcher(n_items=N_ITEMS)
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
    batcher = SequenceBatcher(n_items=N_ITEMS)
    surface = {name for name in dir(batcher) if not name.startswith("_")}

    assert not surface & {"shift", "shifted_targets", "mask_positions", "negatives"}
    assert surface == {
        "catalog_logits",
        "encode",
        "final_positions",
        "gather_final",
        "has_history",
        "max_length",
        "n_items",
        "pad_id",
        "pad_side",
        "special_tokens",
        "token_id",
        "truncated_lengths",
        "vocab_size",
    }
