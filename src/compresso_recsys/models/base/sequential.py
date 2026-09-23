"""The base for models reading chronological histories.

Beyond the prediction contract it carries the plumbing every sequential trainer
shares: the checkpoint entries, the seen-item mask, and the validation each
``fit`` performs before it touches a model.
"""

from __future__ import annotations


from abc import abstractmethod
import time
from typing import (
    Any,
    Hashable,
    Mapping,
    Sequence,
)

import numpy as np
import torch
from scipy.sparse import csr_matrix

from compresso import SRPTensor
from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
)
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models.core.identifiers import ItemVocabulary
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.models.base.identified import _accepts_reporting_keywords
from compresso_recsys.models.base.persistable import BasePersistableRecommender


class BaseSequentialRecommender(BasePersistableRecommender):
    """Reusable base for recommenders that read chronological histories.

    Parallel to :class:`BaseCollaborativeRecommender` rather than derived from
    it. The two differ only in how a user's history arrives — a CSR row of items
    interacted with, or an ordered history that keeps repeats — and crossing that
    with cold-start capability in the type hierarchy would give four classes for
    two ideas. Candidate capability is composed instead: a model that scores
    unseen items owns a catalog rather than inheriting one.

    Implementors provide :attr:`is_fitted`, :attr:`n_items`, and
    :meth:`predict_on_batch`. ``fit`` is deliberately absent from the contract:
    trainers follow the package's existing shape, where
    ``SomeTrainer(config).fit(data)`` returns a fitted model and the model owes
    only the prediction contract.

    Two properties this base is careful not to assume.

    **The source vocabulary need not equal the candidate catalog.**
    :attr:`n_items` describes what can be *scored*. A history may be expressed
    over a different, usually smaller, vocabulary — a truncated context, a
    hashed one — and a cold-capable model scores candidates that never appear in
    any history at all. Nothing here compares the two.

    **Truncation is not exclusion.** ``exclude_seen=True`` must mask every item
    in the *full* history handed to it, even where the encoder reads only a
    suffix. A model that attends to the last 200 interactions must still refuse
    to recommend the 201st.
    """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether the model is ready for prediction."""

    @property
    @abstractmethod
    def n_items(self) -> int | None:
        """Number of scoreable candidates, or ``None`` before fitting."""

    @abstractmethod
    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Return ranked predictions for one batch of histories."""

    def _recommendation_source(
        self,
        rows: list[np.ndarray],
        *,
        vocabulary: ItemVocabulary,
    ) -> ItemSequences:
        return ItemSequences.from_rows(rows, n_items=vocabulary.n_items)

    def _predict_identified(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
    ) -> SRPTensor:
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "sequential recommendations require an ItemSequences source"
            )
        return self.predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )

    def _predict_identified_with_reporting(
        self,
        source: csr_matrix | ItemSequences,
        *,
        k: int,
        exclude_seen: bool,
        candidate_ids: np.ndarray,
        reporter: _Reporter,
    ) -> SRPTensor:
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "sequential recommendations require an ItemSequences source"
            )
        predict = self.predict
        if not _accepts_reporting_keywords(predict):
            predict = BaseSequentialRecommender.predict.__get__(self)
        return predict(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            logger=reporter,
            show_progress=_INHERIT,
        )

    def _prepare_source(self, source: ItemSequences) -> ItemSequences:
        """Check a batch of histories against the fitted model."""
        if not self.is_fitted or self.n_items is None:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before prediction"
            )
        if not isinstance(source, ItemSequences):
            raise TypeError(
                f"{type(self).__name__} predicts from ItemSequences, got "
                f"{type(source).__name__}"
            )
        return source

    @staticmethod
    def _check_unseen_capacity(
        source: ItemSequences,
        *,
        n_items: int,
        k: int,
        candidate_rows: np.ndarray | None = None,
    ) -> None:
        """Require every row to contain at least ``k`` scoreable unseen items."""
        selected = (
            np.ones(n_items, dtype=bool)
            if candidate_rows is None
            else np.zeros(n_items, dtype=bool)
        )
        if candidate_rows is not None:
            selected[candidate_rows] = True
        candidate_count = int(selected.sum())
        for row in range(source.n_rows):
            history = source.row(row)
            scoreable = history[history < n_items]
            seen = np.unique(scoreable)
            available = candidate_count - int(selected[seen].sum())
            if available < k:
                raise ValueError(
                    f"source row {row} has only {available} unseen items, "
                    f"fewer than k={k}"
                )

    def predict(
        self,
        source: ItemSequences,
        *,
        k: int = 100,
        batch_size: int = 1024,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SRPTensor:
        """Predict all histories by repeatedly calling ``predict_on_batch``."""
        reporter = self._prediction_reporter(logger, show_progress)
        source = self._prepare_source(source)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        candidate_count = int(self._candidate_rows(candidate_ids).size)
        if not 1 <= int(k) <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

        columns: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        starts = range(0, source.n_rows, batch_size)
        steps = len(starts)
        started = time.monotonic()
        reporter.log(
            f"predict@{k} started: {source.n_rows} rows | "
            f"{steps} batches of {batch_size}"
        )
        for step, start in enumerate(
            reporter.wrap(
                starts,
                total=steps,
                desc=f"{type(self).__name__} predict@{k}",
            ),
            start=1,
        ):
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            result = self.predict_on_batch(
                source.take_rows(start, start + batch_size),
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
            if result.cols_total != self.n_items:
                raise ValueError(
                    "predict_on_batch() item count must match the candidate catalog"
                )
            columns.append(result.cols)
            values.append(result.vals)
            log_steps = reporter.log_every_n_steps
            if log_steps and step % log_steps == 0:
                reporter.step(
                    f"predict@{k} step {step}/{steps}",
                    step,
                    steps,
                    started,
                )

        if not columns:
            kwargs = (
                {}
                if candidate_ids is None
                else {"candidate_ids": candidate_ids}
            )
            prediction = self.predict_on_batch(
                source,
                k=k,
                exclude_seen=exclude_seen,
                **kwargs,
            )
        else:
            prediction = SRPTensor(
                cols=torch.vstack(columns),
                vals=torch.vstack(values),
                shape=(source.n_rows, self.n_items),
            )
        reporter.log(
            f"predict@{k} finished: "
            f"{_format_duration(time.monotonic() - started)} total | "
            f"{source.n_rows} rows"
        )
        return prediction

    # -- shared trainer plumbing --------------------------------------------
    #
    # Every sequential trainer here keeps its vocabulary in an ItemTokenizer
    # behind a SequenceBatcher and persists the same three entries. Those
    # implementations were identical but for the class named in their error
    # messages, so a subclass now extends them instead of restating them.
    #
    # They are deliberately tolerant of a subclass with no batcher: this is a
    # public extension point, and a model that persists no tokenizer should
    # fail when it tries to save rather than when it is defined.

    def _check_training_sequences(self, sequences: Any) -> None:
        """Reject a training source that is not usable chronological history."""
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                f"{type(self).__name__} trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")

    def _require_batcher(self) -> Any:
        """Return the trainer's batcher, or say it went missing.

        Separate from :meth:`_adopt_batcher_vocabulary` so a trainer can run
        its own batcher check between the two, as SimpleRNN does.
        """
        batcher = getattr(self, "batcher", None)
        if batcher is None:  # pragma: no cover - defensive against mutation
            raise RuntimeError("trainer batcher is unavailable")
        return batcher

    def _adopt_batcher_vocabulary(
        self,
        sequences: ItemSequences,
        item_ids: Sequence[Hashable] | np.ndarray | None,
    ) -> None:
        """Check the batcher against the training catalog and record the IDs.

        Supplied IDs never override the tokenizer's: a batcher handed over for
        its vocabulary is the vocabulary, so disagreeing is an error rather
        than a silent win for either side.
        """
        batcher = self._require_batcher()
        if batcher.tokenizer.n_items != sequences.n_items:
            raise ValueError(
                "batcher tokenizer has "
                f"{batcher.tokenizer.n_items} items, but training sequences "
                f"have {sequences.n_items}"
            )
        tokenizer_ids = getattr(batcher.tokenizer, "item_ids", None)
        if item_ids is not None and tokenizer_ids is not None:
            supplied = ItemVocabulary.from_ids(item_ids).item_ids
            if not np.array_equal(supplied, tokenizer_ids):
                raise ValueError(
                    "item_ids must match the batcher tokenizer item IDs"
                )
        self._set_item_ids(
            tokenizer_ids if item_ids is None else item_ids,
            n_items=sequences.n_items,
        )

    def _checkpoint_trainer_state(self) -> dict[str, Any]:
        """Non-module state for ``state/trainer.json``.

        A subclass that persists more than the window and the history extends
        this rather than rewriting the entry, so every sequential checkpoint
        keeps one shape underneath whatever a model adds to it.
        """
        max_length = self.batcher.max_length
        return {
            # SimpleRNN may leave the window unset, where the transformers
            # reconcile it against the config during fit.
            "max_length": None if max_length is None else int(max_length),
            "history": self.history,
        }

    def _restore_checkpoint_trainer_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        """Install what :meth:`_checkpoint_trainer_state` wrote."""
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError(
                f"{type(self).__name__} training history must be a list"
            )
        self.history = list(history)

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        """Write the tokenizer alongside :meth:`_checkpoint_trainer_state`.

        Item IDs go to their own entry when the tokenizer carries them, since
        they may be arbitrary hashables rather than JSON scalars.
        """
        batcher = getattr(self, "batcher", None)
        if batcher is None or not isinstance(batcher.tokenizer, ItemTokenizer):
            raise TypeError(
                f"{type(self).__name__} checkpoints support ItemTokenizer only"
            )
        writer.write_json(
            "state/trainer.json", self._checkpoint_trainer_state()
        )
        writer.write_json(
            "state/tokenizer.json",
            batcher.tokenizer.to_dict(include_item_ids=False),
        )
        item_ids = batcher.tokenizer.item_ids
        if item_ids is not None:
            writer.write_item_ids("state/tokenizer_item_ids.json", item_ids)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        """Restore the non-module state :meth:`_save_checkpoint_state` wrote."""
        self._restore_checkpoint_trainer_state(
            reader.read_json("state/trainer.json")
        )

    def _mask_seen(self, scores: torch.Tensor, source: ItemSequences) -> None:
        """Forbid every item in the *full* history, truncated part included.

        Scores are indexed by catalog position, and a history may span a wider
        catalog than this model was fitted on -- a later split stage does exactly
        that. Items beyond the fitted catalog are dropped from the mask rather
        than clipped: they were never scoreable, so there is nothing to forbid.
        """
        if source.values.size == 0:
            return
        n_items = int(scores.shape[1])
        # The flat values are already the concatenation of every history, so
        # one scatter covers the batch. np.array copies, both because the
        # buffers are read-only and because torch.from_numpy would share them.
        rows = np.repeat(np.arange(source.n_rows), source.row_lengths)
        cols = np.array(source.values, dtype=np.int64)
        scoreable = cols < n_items
        if not scoreable.all():
            rows, cols = rows[scoreable], cols[scoreable]
        if cols.size == 0:
            return
        scores[
            torch.as_tensor(rows, dtype=torch.long, device=scores.device),
            torch.as_tensor(cols, dtype=torch.long, device=scores.device),
        ] = -torch.inf
