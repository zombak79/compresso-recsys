"""Ragged histories into dense tensors, without deciding what a model does next.

:class:`~compresso_recsys.models.ItemTokenizer` owns the vocabulary — which token
an item is, and what an unknown item becomes. :class:`SequenceBatcher` owns the
other half: how far back to read, where the padding goes, and which positions of
the resulting rectangle are real.

The two are separate because they have different lifetimes. A vocabulary is a
property of the dataset and outlives any particular model. ``max_length`` is a
property of the *model* — it is ``block_size`` under another name, and it sizes a
transformer's positional embedding — so a single tokenizer can serve two models
that read different amounts of history. Fusing them means one object with two
owners, which shows up as the same number written in two configs and a runtime
check to keep them honest.

What this deliberately does not own is the training objective. A next-item shift,
a masked-position target and sampled negatives differ between architectures, and
a component with three mutually exclusive modes is not an abstraction. Those live
in trainers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch

from compresso_recsys.sequences import ItemSequences

from .tokenizer import Tokenizer

__all__ = ["SequenceBatcher"]

PadSide = Literal["right", "left"]


@dataclass(frozen=True)
class SequenceBatcher:
    """Encodes ragged histories into dense token tensors.

    The tokenizer is positional because it is a collaborator rather than an
    option: without a vocabulary there is nothing to encode.

    ``pad_side`` defaults to ``"right"``, which is the better default for a
    causal model and not only for an RNN. With padding after the content, a
    causal mask already excludes every pad, so training needs no attention mask
    at all — only a loss mask. Left padding puts pads *before* real tokens, so
    causal attention would read them unless masked explicitly, and it buys a
    fixed read position that :meth:`final_positions` and :meth:`gather_final`
    already provide.

    ``max_length`` truncates to the **most recent** interactions, the only
    sensible direction: a context window is a claim about recency, not about
    where a history happened to start.
    """

    tokenizer: Tokenizer
    max_length: int | None = None
    pad_side: PadSide = "right"

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer, Tokenizer):
            raise TypeError(
                "tokenizer must provide pad_id and encode_indices, got "
                f"{type(self.tokenizer).__name__}"
            )
        if self.max_length is not None and self.max_length < 1:
            raise ValueError(
                f"max_length must be >= 1 when set, got {self.max_length}"
            )
        if self.pad_side not in ("right", "left"):
            raise ValueError(
                f"pad_side must be 'right' or 'left', got {self.pad_side!r}"
            )

    @property
    def pad_id(self) -> int:
        """Convenience passthrough, since every caller of :meth:`encode` wants it."""
        return self.tokenizer.pad_id

    # -- batching -----------------------------------------------------------

    def encode(
        self,
        sequences: ItemSequences,
        *,
        device: torch.device | str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(tokens, mask)`` for a batch of histories.

        ``tokens`` is ``(rows, length)`` of ``int64`` holding token ids, padded
        with the tokenizer's ``pad_id``. ``mask`` is ``(rows, length)`` of
        ``bool``, true where a token came from a history rather than from
        padding.

        ``length`` is the longest history in the batch after truncation, floored
        at one so the shape stays usable when every row is empty.

        Nothing here inspects the catalog. A history may span a wider item space
        than the tokenizer covers — which is the normal case for a later split
        stage — and the tokenizer decides what those values become.
        """
        lengths = self.truncated_lengths(sequences)
        rows = sequences.n_rows
        width = max(1, int(lengths.max()) if lengths.size else 1)

        tokens = np.full((rows, width), self.tokenizer.pad_id, dtype=np.int64)
        mask = np.zeros((rows, width), dtype=bool)
        # A row at a time. Measured at 0.06% of a training step on MovieLens-1M,
        # against 1.7x for a fully vectorised gather, so the readable form wins.
        for row in range(rows):
            length = int(lengths[row])
            if length == 0:
                continue
            # Truncation keeps the tail, so a context window means "the most
            # recent N" rather than "the first N".
            history = self.tokenizer.encode_indices(sequences.row(row)[-length:])
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

    # -- reading the result -------------------------------------------------

    def final_positions(self, mask: torch.Tensor) -> torch.Tensor:
        """Index of each row's last real token.

        The state a model reads for prediction, and the single easiest thing to
        get wrong: with right padding the last column is padding for every row
        shorter than the batch maximum, so reading ``[:, -1]`` silently scores
        from a pad embedding — and agrees with itself at ``batch_size=1``, where
        every row fills its own batch, which is what lets the bug survive.

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
