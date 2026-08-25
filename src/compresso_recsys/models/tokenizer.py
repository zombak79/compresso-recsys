"""Vocabulary for sequential models: catalog indices in, token ids out.

A checkpoint stores catalog indices, which is the right thing for it to store —
:class:`~compresso_recsys.ItemSequences` deliberately holds no padding, no
special tokens and no length limit. A model needs something else: a vocabulary
where padding has a row, where an item it has never seen is representable, and
where an index it can embed is distinguishable from one it cannot.

:class:`ItemTokenizer` is that translation and nothing more. It maps values, one
for one, and leaves structure alone: no padding, no truncation, no batching, no
tensors. :class:`~compresso_recsys.models.SequenceBatcher` does those.

Layout puts the specials first and the catalog after::

    0 .. n_reserved-1        specials -- named, or reserved for later
    n_reserved .. vocab-1    catalog item i  ->  token  i + n_reserved

That ordering is chosen because **catalog growth appends**. A checkpoint's stage
catalogs nest by prefix, a cold-start catalog grows by appending, and a
``partial_fit`` extends the embedding table at the end — where a `cat` splices
the optimizer state correctly. Reserving ids at the back instead would put each
new item exactly where the specials sit, turning every extension into a
permutation of the parameter *and* its momentum, and a wrong permutation
attaches one item's history to a special token without raising.

Front-loading has one cost: introducing a special later would shift every item.
``n_reserved`` removes it. Name the specials you use, reserve a few more, and a
token added later lands in the reserve while every trained id stays put. The
price is a handful of embedding rows that never receive a gradient.

The offset stops here. A model's output head is ``n_items`` wide and indexed by
catalog position, so predictions leave in the same space as the target matrix,
the metrics and the item IDs. Nothing downstream of a model ever sees a token id.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Hashable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

__all__ = ["ItemTokenizer", "Tokenizer"]

_TOKEN_DTYPE = np.int64
DEFAULT_SPECIAL_TOKENS: Mapping[str, int] = MappingProxyType({"pad": 0, "unk": 1})


@runtime_checkable
class Tokenizer(Protocol):
    """What :class:`SequenceBatcher` requires of a vocabulary.

    Two members, because that is all the batcher reads: a per-value mapping, and
    something to fill the gaps between rows with. Structural, like
    :class:`~compresso_recsys.models.Recommender`, so a custom vocabulary
    satisfies it by having the members rather than by inheriting anything.

    A *model* asks for more — ``vocab_size`` for its embedding rows, ``n_items``
    for its head width, and ``unk_id`` or ``token_id`` if it injects corruption —
    but that is a per-model requirement rather than a library-wide one, so it is
    documented rather than declared.

    ``encode_indices`` must be **one token per value**, in order.
    :meth:`SequenceBatcher.encode` computes each destination from ``indptr``
    before it maps anything, so a vocabulary that expands one item into several
    tokens — semantic IDs from a residual-quantised autoencoder, say — cannot be
    used with it and needs its own batcher.
    """

    @property
    def pad_id(self) -> int: ...

    def encode_indices(self, values: np.ndarray) -> np.ndarray: ...


class ItemTokenizer:
    """Maps catalog indices to token ids, and back.

    Immutable. :meth:`extended` returns a new tokenizer rather than growing this
    one, which is what makes "every existing token id is unchanged" checkable
    rather than promised.

    ``special_tokens`` names the ids explicitly and they must be exactly
    ``0 .. len - 1``: no gaps, so ``n_reserved`` is unambiguous. ``"pad"`` is
    required because every batching path needs something to fill with. ``"unk"``
    is optional, and its absence is a real choice — without it, an index the
    tokenizer cannot embed is an error rather than a token.

    Nothing here interprets a name. ``"cls"`` may be named and receives an id
    like any other; whether a model emits it, or keeps its CLS purely as a
    prepended parameter, is the model's business.
    """

    def __init__(
        self,
        n_items: int,
        *,
        special_tokens: Mapping[str, int] | None = None,
        n_reserved: int | None = None,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> None:
        specials = dict(
            DEFAULT_SPECIAL_TOKENS if special_tokens is None else special_tokens
        )
        if int(n_items) < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        if "pad" not in specials:
            raise ValueError(
                "special_tokens must include 'pad'; every batching path needs a "
                "token to fill unused positions with"
            )
        assigned = sorted(specials.values())
        if assigned != list(range(len(specials))):
            raise ValueError(
                "special token ids must be exactly 0..n-1 with no gaps, so that "
                f"n_reserved is unambiguous; got {sorted(specials.items())}. Use "
                "n_reserved to leave room for tokens added later"
            )
        reserved = len(specials) if n_reserved is None else int(n_reserved)
        if reserved < len(specials):
            raise ValueError(
                f"n_reserved must be >= the {len(specials)} named special "
                f"tokens, got {reserved}"
            )

        self._n_items = int(n_items)
        self._n_reserved = reserved
        self._special_tokens = MappingProxyType(dict(specials))
        self._name_by_id = {token: name for name, token in specials.items()}
        self._item_ids = self._checked_item_ids(item_ids, self._n_items)
        self._id_to_row: Mapping[Hashable, int] | None = (
            None
            if self._item_ids is None
            else MappingProxyType(
                {value: row for row, value in enumerate(self._item_ids.tolist())}
            )
        )

    @staticmethod
    def _checked_item_ids(
        item_ids: Sequence[Hashable] | np.ndarray | None, n_items: int
    ) -> np.ndarray | None:
        if item_ids is None:
            return None
        ids = np.asarray(item_ids)
        if np.issubdtype(ids.dtype, np.integer):
            raise ValueError(
                "item_ids must not be an integer dtype, because encode() tells "
                "indices from IDs by dtype and integer IDs make that ambiguous: "
                "encode([1, 2]) could mean either. Pass ids.astype(str), which "
                "is what save_recsys_split already stores"
            )
        if ids.ndim != 1 or ids.size != n_items:
            raise ValueError(
                f"item_ids must be one-dimensional with {n_items} entries, got "
                f"shape {ids.shape}"
            )
        if len(set(ids.tolist())) != ids.size:
            raise ValueError("item_ids must be unique")
        ids = ids.copy()
        ids.setflags(write=False)
        return ids

    def __repr__(self) -> str:
        unnamed = self._n_reserved - len(self._special_tokens)
        return (
            f"{type(self).__name__}(n_items={self._n_items}, "
            f"vocab_size={self.vocab_size}, "
            f"specials={dict(self._special_tokens)}"
            + (f", +{unnamed} reserved" if unnamed else "")
            + (", with item_ids" if self.has_ids else "")
            + ")"
        )

    # -- vocabulary ---------------------------------------------------------

    @property
    def n_items(self) -> int:
        """Catalog size: how many items can be embedded and scored."""
        return self._n_items

    @property
    def n_reserved(self) -> int:
        """Token ids held for specials, and so the offset applied to items."""
        return self._n_reserved

    @property
    def vocab_size(self) -> int:
        """Embedding rows needed: the reserve plus the catalog."""
        return self._n_reserved + self._n_items

    @property
    def special_tokens(self) -> Mapping[str, int]:
        """Named special tokens and their ids, read-only."""
        return self._special_tokens

    def token_id(self, name: str) -> int:
        """Token id of a named special."""
        try:
            return self._special_tokens[name]
        except KeyError:
            raise KeyError(
                f"{name!r} is not a special token; have "
                f"{sorted(self._special_tokens)}"
            ) from None

    @property
    def pad_id(self) -> int:
        """Token used to fill unused positions. Always present."""
        return self._special_tokens["pad"]

    @property
    def unk_id(self) -> int | None:
        """Token for an item outside the catalog, or ``None`` if not named.

        ``None`` rather than an exception so a caller can branch on capability:
        a model without an ``unk`` slot genuinely cannot represent an unknown
        item, and should say so rather than guess.
        """
        return self._special_tokens.get("unk")

    @property
    def item_ids(self) -> np.ndarray | None:
        """Stable IDs in catalog order, when the ID path is available."""
        return self._item_ids

    @property
    def has_ids(self) -> bool:
        """Whether :meth:`encode_ids` and :meth:`decode_ids` can be used."""
        return self._item_ids is not None

    # -- encoding -----------------------------------------------------------

    def encode_indices(self, values: np.ndarray | Sequence[int]) -> np.ndarray:
        """Catalog indices to token ids, one for one, in order.

        An index at or above :attr:`n_items` is an item this vocabulary cannot
        embed. That is the normal case rather than an error: checkpoint stages
        nest by prefix, so a model fitted on the training catalog reads a later
        stage's indices directly and anything past its own count is simply an
        item that did not exist when it was fitted. Such values become
        :attr:`unk_id`, which keeps the history's length and adjacency intact —
        dropping them instead would assert transitions that never happened.
        """
        indices = np.asarray(values, dtype=_TOKEN_DTYPE)
        if indices.size and int(indices.min()) < 0:
            raise ValueError(
                f"catalog indices must be non-negative, got {int(indices.min())}"
            )
        unknown = indices >= self._n_items
        if not unknown.any():
            return indices + self._n_reserved
        unk = self.unk_id
        if unk is None:
            raise ValueError(
                f"{int(unknown.sum())} of {indices.size} values are outside the "
                f"catalog of {self._n_items} items and this tokenizer has no "
                "'unk' token, so they cannot be represented. Add 'unk' to "
                "special_tokens, or narrow the source to the fitted catalog"
            )
        return np.where(unknown, unk, indices + self._n_reserved)

    def encode_ids(self, ids: Sequence[Hashable] | np.ndarray) -> np.ndarray:
        """Stable item IDs to token ids, one for one, in order."""
        if self._id_to_row is None:
            raise RuntimeError(
                "this tokenizer was built without item_ids, so it cannot encode "
                "IDs; pass item_ids to enable the ID path"
            )
        wanted = np.asarray(ids).ravel()
        unk = self.unk_id
        rows = np.fromiter(
            (self._id_to_row.get(value, -1) for value in wanted.tolist()),
            dtype=_TOKEN_DTYPE,
            count=wanted.size,
        )
        unknown = rows < 0
        if not unknown.any():
            return rows + self._n_reserved
        if unk is None:
            first = wanted[int(np.flatnonzero(unknown)[0])]
            raise ValueError(
                f"item ID {first!r} is not in the catalog and this tokenizer has "
                "no 'unk' token to represent it"
            )
        return np.where(unknown, unk, rows + self._n_reserved)

    def encode(self, values: np.ndarray | Sequence[Hashable]) -> np.ndarray:
        """Encode indices or IDs, choosing by dtype.

        Integer input is catalog indices, anything else is stable IDs. The two
        are distinguishable because ``ItemSequences.values`` is always integral
        and stored item IDs are always strings, and construction refuses integer
        ``item_ids`` so the ambiguous case cannot arise.
        """
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.integer):
            return self.encode_indices(array)
        return self.encode_ids(array)

    # -- decoding -----------------------------------------------------------

    def decode_indices(self, tokens: np.ndarray | Sequence[int]) -> np.ndarray:
        """Token ids to catalog indices; every special becomes ``-1``."""
        array = np.asarray(tokens, dtype=_TOKEN_DTYPE)
        return np.where(array < self._n_reserved, -1, array - self._n_reserved)

    def decode_ids(self, tokens: np.ndarray | Sequence[int]) -> np.ndarray:
        """Token ids to stable item IDs; a special becomes its own name."""
        if self._item_ids is None:
            raise RuntimeError(
                "this tokenizer was built without item_ids, so it cannot decode "
                "to IDs; use decode_indices, or pass item_ids"
            )
        array = np.asarray(tokens, dtype=_TOKEN_DTYPE)
        out = np.empty(array.shape, dtype=object)
        special = array < self._n_reserved
        if special.any():
            out[special] = [
                self._name_by_id.get(int(token), "reserved")
                for token in array[special].ravel().tolist()
            ]
        if (~special).any():
            out[~special] = self._item_ids[array[~special] - self._n_reserved]
        return out

    def decode(self, tokens: np.ndarray | Sequence[int]) -> np.ndarray:
        """Alias for :meth:`decode_ids`.

        Unlike :meth:`encode` this cannot choose: both decoders take token ids,
        so the input carries no signal and only the return type differs. Picking
        by whether ``item_ids`` happens to be present would make the return type
        depend on construction, silently, so this picks IDs and fails loudly
        when it has none.
        """
        return self.decode_ids(tokens)

    # -- growth and persistence ---------------------------------------------

    def extended(
        self,
        *,
        n_items: int | None = None,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> "ItemTokenizer":
        """A tokenizer over a larger catalog, with every token id unchanged.

        Give ``item_ids`` to grow by appending IDs, or ``n_items`` to grow by
        count alone. The catalog may only grow: shrinking would move nothing but
        would leave a trained model scoring items the vocabulary denies.
        """
        if (n_items is None) == (item_ids is None):
            raise ValueError("give exactly one of n_items or item_ids")
        if item_ids is not None:
            if self._item_ids is None:
                raise RuntimeError(
                    "this tokenizer has no item_ids, so it cannot be extended by "
                    "ID; extend by n_items instead"
                )
            appended = np.asarray(item_ids).ravel()
            grown_ids = np.concatenate([self._item_ids, appended])
            grown_n = int(grown_ids.size)
        else:
            grown_n = int(n_items)
            grown_ids = None
            if self._item_ids is not None:
                raise RuntimeError(
                    "this tokenizer has item_ids, so extending it by count alone "
                    "would leave the new items unnamed; pass item_ids instead"
                )
        if grown_n < self._n_items:
            raise ValueError(
                f"a catalog may only grow: {grown_n} is smaller than the current "
                f"{self._n_items}"
            )
        return ItemTokenizer(
            grown_n,
            special_tokens=dict(self._special_tokens),
            n_reserved=self._n_reserved,
            item_ids=grown_ids,
        )

    def to_dict(self, *, include_item_ids: bool = True) -> dict:
        """A JSON-serialisable description, sufficient to rebuild this exactly.

        ``item_ids`` are included by default because serving is the whole reason
        they exist, and a saved tokenizer that cannot resolve an ID defeats it.
        Pass ``include_item_ids=False`` when only the index path matters and the
        catalog is large enough for the size to.
        """
        state: dict = {
            "n_items": self._n_items,
            "n_reserved": self._n_reserved,
            "special_tokens": dict(self._special_tokens),
        }
        if include_item_ids and self._item_ids is not None:
            state["item_ids"] = self._item_ids.tolist()
        return state

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "ItemTokenizer":
        """Rebuild from :meth:`to_dict`."""
        return cls(
            int(state["n_items"]),  # type: ignore[arg-type]
            special_tokens=dict(state["special_tokens"]),  # type: ignore[arg-type]
            n_reserved=int(state["n_reserved"]),  # type: ignore[arg-type]
            item_ids=state.get("item_ids"),  # type: ignore[arg-type]
        )
