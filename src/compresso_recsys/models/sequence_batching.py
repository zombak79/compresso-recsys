"""Turning histories into tensors, without deciding what a model does with them.

:class:`~compresso_recsys.sequences.ItemSequences` deliberately holds no padding,
no special tokens and no length limit, because those are modelling decisions. But
several of those decisions are the *same* across sequential architectures, and
re-deriving them per model is how off-by-one bugs get in.

:class:`SequenceBatcher` owns exactly the shared part: where special tokens live
in the vocabulary, how a ragged batch becomes a dense tensor, which positions are
real, and how far back to look. What it deliberately does not own is the training
objective — a next-item shift, a masked-position target, sampled negatives — since
those differ between architectures and a component with three mutually exclusive
modes is not an abstraction.

Vocabulary layout puts the catalog first and special tokens after it::

    catalog index i  ->  token i          (the identity)
    "pad"            ->  token n_items
    "mask"           ->  token n_items + 1

That ordering is chosen so **catalog token ids never move**. Reserving ids at the
front, as text models do, would mean adding a second special token shifts every
item by one and invalidates any model already trained. Appending instead keeps
``logits[..., :n_items]`` the catalog scores under any future vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from compresso_recsys.sequences import ItemSequences

__all__ = ["SequenceBatcher"]

PadSide = Literal["right", "left"]


@dataclass(frozen=True)
class SequenceBatcher:
    """Encodes ragged histories into dense token tensors.

    ``pad_side`` says where the padding goes, which is the setting architectures
    actually disagree about:

    * ``"right"`` — content first, padding after. An RNN reads to each row's own
      final position, so trailing padding costs nothing.
    * ``"left"`` — padding first, content last. A causal transformer wants the
      newest interaction at a fixed index, so that prediction always reads
      position ``-1``.

    ``max_length`` truncates to the **most recent** interactions, which is the
    only sensible direction: a context window is a claim about recency, not about
    where a history happened to start.
    """

    n_items: int
    max_length: int | None = None
    pad_side: PadSide = "right"
    special_tokens: tuple[str, ...] = ("pad",)

    def __post_init__(self) -> None:
        if self.n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {self.n_items}")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError(
                f"max_length must be >= 1 when set, got {self.max_length}"
            )
        if self.pad_side not in ("right", "left"):
            raise ValueError(
                f"pad_side must be 'right' or 'left', got {self.pad_side!r}"
            )
        if "pad" not in self.special_tokens:
            raise ValueError("special_tokens must include 'pad'")
        if len(set(self.special_tokens)) != len(self.special_tokens):
            raise ValueError("special_tokens must be unique")

    # -- vocabulary ---------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Embedding rows needed: the catalog plus the special tokens."""
        return self.n_items + len(self.special_tokens)

    def token_id(self, name: str) -> int:
        """Token id of a special token."""
        try:
            return self.n_items + self.special_tokens.index(name)
        except ValueError:
            raise KeyError(
                f"{name!r} is not a special token; have {self.special_tokens}"
            ) from None

    @property
    def pad_id(self) -> int:
        """Token id used for padding, suitable for ``Embedding(padding_idx=...)``."""
        return self.token_id("pad")

    def catalog_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Drop the special-token columns, leaving one score per catalog item.

        A no-op slice, named so the invariant that catalog ids are the identity
        is stated where it is relied on rather than assumed.
        """
        if logits.shape[-1] < self.n_items:
            raise ValueError(
                f"logits last dimension is {logits.shape[-1]}, which is smaller "
                f"than the catalog ({self.n_items})"
            )
        return logits[..., : self.n_items]

    # -- batching -----------------------------------------------------------

    def encode(
        self,
        sequences: ItemSequences,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens, mask)`` for a batch of histories.

        ``tokens`` is ``(rows, length)`` of ``int64``, padded with :attr:`pad_id`.
        ``mask`` is ``(rows, length)`` of ``bool``, true where a token came from a
        history rather than from padding.

        ``length`` is the longest history in the batch after truncation, floored
        at one so the shape stays usable when every row is empty.
        """
        if sequences.n_items > self.n_items:
            raise ValueError(
                f"sequences span {sequences.n_items} items but this batcher was "
                f"built for {self.n_items}"
            )

        lengths = self.truncated_lengths(sequences)
        rows = sequences.n_rows
        width = max(1, int(lengths.max()) if lengths.size else 1)

        tokens = np.full((rows, width), self.pad_id, dtype=np.int64)
        mask = np.zeros((rows, width), dtype=bool)
        for row in range(rows):
            length = int(lengths[row])
            if length == 0:
                continue
            # Truncation keeps the tail, so a context window means "the most
            # recent N" rather than "the first N".
            history = sequences.row(row)[-length:]
            if self.pad_side == "right":
                tokens[row, :length] = history
                mask[row, :length] = True
            else:
                tokens[row, width - length :] = history
                mask[row, width - length :] = True

        return (
            torch.from_numpy(tokens).to(device),
            torch.from_numpy(mask).to(device),
        )

    def truncated_lengths(self, sequences: ItemSequences) -> np.ndarray:
        """History lengths after ``max_length`` is applied."""
        lengths = sequences.row_lengths
        if self.max_length is None:
            return lengths
        return np.minimum(lengths, self.max_length)

    def final_positions(self, mask: torch.Tensor) -> torch.Tensor:
        """Index of each row's last real token.

        The state a model reads for prediction, and the single easiest thing to
        get wrong: with right padding the last column is padding for every row
        shorter than the batch maximum, so reading ``[:, -1]`` silently scores
        from a pad embedding.

        Empty rows report 0, which is padding. Pair this with :meth:`has_history`
        rather than trusting the position alone.
        """
        if mask.ndim != 2:
            raise ValueError("mask must be two-dimensional")
        lengths = mask.sum(dim=1)
        if self.pad_side == "right":
            return (lengths - 1).clamp_min(0)
        width = mask.shape[1]
        return torch.full_like(lengths, width - 1)

    @staticmethod
    def has_history(mask: torch.Tensor) -> torch.Tensor:
        """Whether each row carries any real token at all."""
        return mask.any(dim=1)

    def gather_final(
        self, states: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Select each row's last real state from ``(rows, length, dim)``."""
        if states.ndim != 3:
            raise ValueError(
                f"states must be (rows, length, dim), got {tuple(states.shape)}"
            )
        if states.shape[:2] != mask.shape:
            raise ValueError(
                f"states {tuple(states.shape[:2])} and mask {tuple(mask.shape)} "
                "must agree on rows and length"
            )
        index = self.final_positions(mask)
        index = index.view(-1, 1, 1).expand(-1, 1, states.shape[-1])
        return states.gather(dim=1, index=index).squeeze(1)
