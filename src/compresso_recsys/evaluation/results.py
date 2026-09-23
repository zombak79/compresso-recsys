"""What an evaluation returns, and the fingerprint that keeps runs comparable."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation.arrays import _owned


class _TargetFingerprint:
    """Canonical, batch-size independent digest of the target matrix.

    Two evaluations can only be paired when they scored the same users against
    the same relevant items. Identifiers catch a different user set; this
    catches the same identifiers against different targets, which no identifier
    check can see.

    The evaluator receives targets one batch at a time and never holds the
    whole matrix, so the digest has to be accumulated. Three properties make
    the result independent of how the rows were divided:

    * Row counts and column indices go into two separate streams. Interleaving
      them per batch would put the bytes in a different order for a different
      ``batch_size``, while each stream on its own is simply the same sequence
      of rows however it was chunked, and blake2b digests a stream in pieces
      exactly as it digests it whole.
    * Neither stream contains ``indptr``, whose values are rebased to zero in
      every slice. Row lengths carry the same information and survive slicing.
    * Both are cast to fixed width and byte order, so the same logical matrix
      agrees across platforms and across int32 and int64 index dtypes.

    Values are not hashed. Targets are binary relevance and only nonzero
    locations matter, so two matrices that differ solely in stored values are
    genuinely the same evaluation.

    Canonical form -- sorted, deduplicated, no stored zeros -- is a
    precondition, supplied by :func:`_canonical_csr` before every update.
    """

    __slots__ = ("_row_lengths", "_indices", "_n_items", "_n_rows")

    def __init__(self) -> None:
        self._row_lengths = hashlib.blake2b(digest_size=16)
        self._indices = hashlib.blake2b(digest_size=16)
        self._n_items: int | None = None
        self._n_rows = 0

    def update(self, targets: csr_matrix) -> None:
        """Fold one canonical batch in, in global row order."""
        if self._n_items is None:
            self._n_items = int(targets.shape[1])
        self._row_lengths.update(np.diff(targets.indptr).astype("<u8").tobytes())
        self._indices.update(targets.indices.astype("<i8", copy=False).tobytes())
        self._n_rows += int(targets.shape[0])

    def digest(self) -> str:
        """Hex digest binding both streams to the matrix shape."""
        final = hashlib.blake2b(digest_size=16)
        final.update(int(self._n_items or 0).to_bytes(8, "big"))
        final.update(self._n_rows.to_bytes(8, "big"))
        final.update(self._row_lengths.digest())
        final.update(self._indices.digest())
        return final.hexdigest()


@dataclass(eq=False)
class EvaluationResult(Mapping[str, Any]):
    """Aggregate metrics plus the per-user observations behind them.

    The mapping view carries the aggregates and ``n_scored_rows``, so existing
    code that treats an evaluation as a dictionary keeps working::

        result["ndcg@20"]
        dict(result)

    Per-user values, sample identifiers and metadata are attributes rather than
    mapping keys, because they are large and because a caller reaching for them
    is doing something other than reading a headline number.

    ``per_user`` and ``sample_ids`` are what make paired statistical comparison
    possible: two evaluations can only be compared when they refer to the same
    evaluation units in the same order.
    """

    metrics: dict[str, float]
    per_user: dict[str, np.ndarray] | None
    sample_ids: np.ndarray | None
    n_rows: int
    n_scored_rows: int
    required_k: int
    metadata: dict[str, Any] = field(default_factory=dict)
    # ``None`` means debug collection was off; an empty tuple means it was on
    # and produced nothing. The mapping exposes ``"debug"`` in the second case
    # but not the first, so a caller that asked for debug always finds the key.
    debug_rows: tuple[dict[str, Any], ...] | None = None
    # Identifies the targets these metrics were computed against, so paired
    # comparison can refuse two results that scored the same users on different
    # relevant items. ``None`` for results built by hand rather than by an
    # evaluator; comparison warns rather than failing in that case.
    target_fingerprint: str | None = None

    def __post_init__(self) -> None:
        for key, value in self.metrics.items():
            if not np.isfinite(value):
                raise ValueError(f"aggregate metric {key!r} is not finite: {value!r}")
        self.n_rows = int(self.n_rows)
        self.n_scored_rows = int(self.n_scored_rows)
        self.required_k = int(self.required_k)
        if self.n_rows < 0:
            raise ValueError("n_rows must be >= 0")
        if not 0 <= self.n_scored_rows <= self.n_rows:
            raise ValueError(
                f"n_scored_rows ({self.n_scored_rows}) must be in [0, n_rows={self.n_rows}]"
            )
        if self.required_k < 1:
            raise ValueError("required_k must be >= 1")

        if self.per_user is None:
            if self.sample_ids is not None:
                raise ValueError("sample_ids requires per_user values")
            return

        if set(self.per_user) != set(self.metrics):
            missing = sorted(set(self.metrics) - set(self.per_user))
            extra = sorted(set(self.per_user) - set(self.metrics))
            raise ValueError(
                "per_user keys must match metric keys; "
                f"missing={missing}, unexpected={extra}"
            )
        cleaned: dict[str, np.ndarray] = {}
        for key, values in self.per_user.items():
            array = _owned(np.ascontiguousarray(values, dtype=np.float32), values)
            if array.ndim != 1:
                raise ValueError(f"per_user[{key!r}] must be one-dimensional")
            if array.shape[0] != self.n_scored_rows:
                raise ValueError(
                    f"per_user[{key!r}] has {array.shape[0]} values, "
                    f"expected n_scored_rows={self.n_scored_rows}"
                )
            if not np.isfinite(array).all():
                raise ValueError(f"per_user[{key!r}] contains non-finite values")
            array.setflags(write=False)
            cleaned[key] = array
        self.per_user = cleaned

        if self.sample_ids is None:
            raise ValueError("per_user values require sample_ids")
        ids = _owned(np.asarray(self.sample_ids), self.sample_ids)
        if ids.ndim != 1:
            raise ValueError("sample_ids must be one-dimensional")
        if ids.shape[0] != self.n_scored_rows:
            raise ValueError(
                f"sample_ids has {ids.shape[0]} values, "
                f"expected n_scored_rows={self.n_scored_rows}"
            )
        # Repeated identifiers are legitimate here: the stacked-fold protocol
        # in :func:`compresso_recsys.retrieval.build_eval_holdout` evaluates
        # each user in several folds, so one user owns several rows. Evaluating
        # them is fine; resampling them as independent units is not, which is a
        # question for paired comparison rather than for this constructor.
        #
        # As with the per-user values above: frozen so a later mutation cannot
        # silently invalidate the pairing this result was matched on.
        ids.setflags(write=False)
        self.sample_ids = ids

    def _mapping_view(self) -> dict[str, Any]:
        view: dict[str, Any] = dict(self.metrics)
        view["n_scored_rows"] = self.n_scored_rows
        view["n_units"] = self.n_units
        if self.debug_rows is not None:
            view["debug"] = list(self.debug_rows)
        return view

    def __getitem__(self, key: str) -> Any:
        return self._mapping_view()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._mapping_view())

    def __len__(self) -> int:
        return len(self._mapping_view())

    def __repr__(self) -> str:
        collected = "none" if self.per_user is None else f"{len(self.per_user)} keys"
        return (
            f"EvaluationResult(metrics={self.metrics!r}, "
            f"n_scored_rows={self.n_scored_rows}, n_rows={self.n_rows}, "
            f"required_k={self.required_k}, per_user={collected})"
        )

    @property
    def n_units(self) -> int:
        """Independent evaluation units behind the scored rows.

        Distinct ``sample_ids``, which is smaller than ``n_scored_rows`` when a
        protocol gives one user several rows -- ``eval_draws`` above 1, say. It
        matches :attr:`compresso_recsys.stats.PairwiseComparison.n_units`, the
        count paired comparison actually resamples.

        Without identifiers there is nothing to group by, and every row is its
        own unit, which is also what comparison assumes when it numbers rows
        positionally.
        """
        if self.sample_ids is None:
            return self.n_scored_rows
        return int(np.unique(self.sample_ids).shape[0])

    @property
    def has_per_user(self) -> bool:
        """Whether per-user observations were collected."""
        return self.per_user is not None

    def to_dict(self, *, include_debug: bool = True) -> dict[str, Any]:
        """Return the mapping view as a plain ``dict``.

        Use this where an actual ``dict`` is required, such as JSON
        serialization. Per-user values are deliberately excluded.
        """
        view = self._mapping_view()
        if not include_debug:
            view.pop("debug", None)
        return view
