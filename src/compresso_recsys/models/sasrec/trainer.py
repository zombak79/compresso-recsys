"""The training procedure, and the fitted model's prediction path."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from compresso import SRPTensor
from torch import nn

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
)
from compresso_recsys.models.base import BaseSequentialRecommender
from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.persistence import (
    ModelCheckpointReader,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.models.sasrec.config import (
    SASRecConfig,
)
from compresso_recsys.models.sasrec.model import (
    SASRec,
)


class SASRecTrainer(BaseSequentialRecommender):
    """Trains and serves :class:`SASRec`.

    Follows the package's existing shape, where ``fit`` returns the trainer and
    the trainer answers the prediction contract::

        model = SASRecTrainer(SASRecConfig()).fit(split["x_train_sequences"])
        result = evaluate_recommender(
            model, source=split["test_source_sequences"],
            targets=split["test_target_matrix"], metrics=[NDCG(20)],
        )

    The encoder is a *parameter*, not something ``fit`` invents. Passing one is
    how you change the vocabulary -- including giving it an ``unk`` slot so a
    later split stage's unseen items become a reserved id rather than an error.
    Leave its ``max_length`` unset and it inherits the config's window, so the
    number stays in one place::

        batcher = SequenceBatcher(
            ItemTokenizer(n_items, item_ids=split["train_item_ids"]),
        )
        model = SASRecTrainer(SASRecConfig(), batcher).fit(sequences)

    The context window is ``SASRecConfig.max_history_length``, so a shorter one
    is ``SASRecConfig(max_history_length=50)`` rather than a number written on
    the batcher. A batcher that does state its own ``max_length`` must agree
    with the config, and ``fit`` refuses the pair when they differ: the window
    sizes a positional embedding that cannot be extended at prediction time.
    """

    #: Context window used when ``fit`` has to build its own batcher.
    checkpoint_type = "sasrec_trainer"

    def __init__(
        self,
        config: SASRecConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        """Hold the config and encoder; build nothing until ``fit``.

        Sets ``self.cfg``, ``self.device``, ``self.history``, ``self.model``,
        ``self.optimizer``, ``self.batcher``, ``self._owns_batcher`` and
        ``self._n_items``, matching the two sibling trainers so the inherited
        persistence and ``to()`` paths find what they expect.

        ``self._rng`` is one addition. Negative sampling draws from NumPy and
        ``_train_step``'s signature is fixed by the loop that calls it, so the
        generator ``fit`` seeds reaches it as state rather than as an argument.
        It is deliberately not checkpointed: a reloaded model predicts, and a
        further ``fit`` reseeds from ``cfg.seed``.

        ``self._train_batcher`` is the other, and it exists because training
        reads one interaction more than the model has positions for -- see
        :meth:`_train_step`. ``fit`` derives it from ``self.batcher``, so it is
        not checkpointed either: the window that a checkpoint records is the
        model's, and a further ``fit`` derives this from it again.
        """
        self.cfg = config or SASRecConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SASRec | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None
        self._rng: np.random.Generator | None = None
        self._train_batcher: SequenceBatcher | None = None

    # -- contract -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been built and trained."""
        return self.model is not None

    @property
    def n_items(self) -> int | None:
        """Number of scoreable candidates, or ``None`` before fitting."""
        return self._n_items

    # -- training -----------------------------------------------------------

    def fit(
        self,
        sequences: ItemSequences,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SASRecTrainer:
        """Train on chronological histories, one example per position.

        Validates the input, builds a default batcher when none was supplied,
        checks it against the training catalog, records the item IDs, seeds
        Torch and NumPy from ``cfg.seed``, builds the model, optimizer and
        scheduler, then runs ``cfg.epochs`` passes over shuffled rows and
        appends one entry per epoch to :attr:`history`.

        Two of those validations are SASRec's own. A history needs two
        retained interactions to yield even one shifted example, as
        :class:`SimpleRNNTrainer` does and unlike :class:`SimpleGPTTrainer`,
        whose ``CLS`` prefix makes a one-item history trainable. And the
        catalog needs two items, because a negative is drawn from the
        catalog minus the position's own positive.

        Rebuilds the model on every call: early stopping and incremental
        training are not part of this contract.
        """
        reporter = self._reporter(logger, show_progress)
        self._check_training_sequences(sequences)
        if sequences.n_items < 2:
            raise ValueError(
                "SASRec's sampled objective needs at least two items: every "
                "negative is drawn from the catalog minus the position's own "
                f"positive, and a catalog of {sequences.n_items} leaves "
                "nothing to draw"
            )


        if self._owns_batcher:
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items),
                max_length=self.cfg.max_history_length,
            )
        self._adopt_batcher_vocabulary(sequences, item_ids)
        # Resolved before anything reads the window -- truncated_lengths just
        # below is the first thing that would. A batcher stating no window
        # inherits the config's, and the batcher is frozen, so this is a new
        # one rather than a mutation of what the caller handed over.
        if self.batcher.max_length is None:
            self.batcher = replace(
                self.batcher, max_length=self.cfg.max_history_length
            )
        # Left padding is the architecture rather than a preference -- see the
        # SASRec docstring -- so it is set here rather than asked of the caller.
        if self.batcher.padding != "left":
            self.batcher = replace(self.batcher, padding="left")
        self._check_batcher(self.batcher)
        # One interaction wider than the model's window. The next-item shift in
        # _train_step spends a step, so encoding at max_length would leave the
        # last position with no input ever standing on it; encoding at
        # max_length + 1 makes the inputs exactly as long as the positional
        # table. Prediction keeps using self.batcher, which does not shift.
        self._train_batcher = replace(
            self.batcher, max_length=int(self.batcher.max_length) + 1
        )
        # Counted after truncation, because the window is what the model
        # will actually read: a long history whose retained tail is one item
        # is no more trainable than a one-item history. Against the training
        # batcher, since that is the truncation training performs.
        usable = int(
            (self._train_batcher.truncated_lengths(sequences) >= 2).sum()
        )
        if usable == 0:
            raise ValueError(
                "no history retains two or more interactions after "
                "truncation, so there is no next-item example to learn from"
            )

        torch.manual_seed(int(self.cfg.seed))
        # One generator for the row order and the negatives both, so a run
        # is reproducible from cfg.seed alone.
        rng = np.random.default_rng(int(self.cfg.seed))
        self._rng = rng

        self._n_items = self.batcher.tokenizer.n_items
        self.model = self._build_model()

        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            **self.cfg.optimizer_kwargs(),
        )
        optimizer = self.optimizer
        # Binary, not cross entropy: scoring a positive and its negatives
        # independently is what avoids normalising over the catalog.
        objective = nn.BCEWithLogitsLoss()
        self.history = []

        n_rows = sequences.n_rows
        batch_size = self.cfg.batch_size
        starts = range(0, n_rows, batch_size)
        with TrainingProgress(
            reporter,
            label="SASRec",
            epochs=self.cfg.epochs,
            batches=len(starts),
        ) as progress:
            progress.start(
                f"{n_rows} sequences | {self._n_items} items | "
                f"{len(starts)} batches of {batch_size} | "
                f"{self.cfg.epochs} epochs | device {self.device}"
            )
            for epoch in progress.epochs():
                self.model.train()
                order = rng.permutation(n_rows)
                loss_sum, positions = 0.0, 0
                for step_index, start in enumerate(starts, start=1):
                    batch = sequences.select_rows(
                        order[start : start + batch_size]
                    )
                    step = self._train_step(batch, optimizer, objective)
                    if step is not None:
                        batch_loss, batch_positions = step
                        loss_sum += batch_loss * batch_positions
                        positions += batch_positions
                    progress.batch(step_index, loss_sum, positions)
                record = {
                    "epoch": float(epoch),
                    "loss": (
                        loss_sum / positions if positions else float("nan")
                    ),
                    "positions": float(positions),
                }
                self.history.append(record)
                progress.epoch_done(record)
        return self

    def _check_batcher(self, batcher: SequenceBatcher) -> None:
        """Reject a batcher SASRec cannot use.

        The window has one owner, ``cfg.max_history_length``, because it sizes
        the positional embedding and a checkpoint cannot grow one after the
        fact. A batcher naming no window has already inherited it by the time
        this runs; one naming a different window is refused rather than
        silently overruling the config or being silently overruled by it.
        """
        if batcher.max_length is None:  # pragma: no cover - fit resolves it
            raise RuntimeError(
                "batcher window was not resolved before _check_batcher"
            )
        if batcher.max_length != self.cfg.max_history_length:
            raise ValueError(
                f"batcher max_length is {batcher.max_length} but "
                f"cfg.max_history_length is {self.cfg.max_history_length}. "
                "The window sizes the positional table, so it has a single "
                "owner: set it on the config, or leave the batcher's "
                "max_length as None to inherit it"
            )

    def _build_model(self) -> SASRec:
        """Construct :class:`SASRec` from the config and the batcher's tokenizer.

        The tokenizer supplies ``n_items``, ``n_reserved`` and ``pad_id``. The
        window is read off the batcher, which ``fit`` has already reconciled
        with ``cfg.max_history_length``, so the two say the same thing by the
        time the positional table is sized. The config supplies the
        architecture -- ``d_model``, ``n_blocks``, ``n_heads``, ``dropout`` --
        and the result is moved to ``self.device``.
        """
        assert self.batcher is not None
        tokenizer = self.batcher.tokenizer
        return SASRec(
            n_items=tokenizer.n_items,
            n_reserved=tokenizer.n_reserved,
            # No +1: SASRec adds the extra row itself, because it numbers
            # positions from one and keeps index 0 for padding steps.
            max_history_length=int(self.batcher.max_length),
            pad_id=tokenizer.pad_id,
            d_model=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            n_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
        ).to(self.device)

    def _train_step(
        self,
        batch: ItemSequences,
        optimizer: torch.optim.Optimizer,
        objective: nn.Module,
    ) -> tuple[float, int] | None:
        """One optimizer step, or ``None`` when the batch carries no target.

        Encodes the batch, applies :meth:`_with_unk_dropout` to the inputs,
        shifts one step left for the positives, draws ``cfg.n_negatives``
        negatives per position, and scores both through
        :meth:`SASRec.score_items`. The binary objective is applied only where
        the shifted mask is true and the positive is a real item, then the
        losses over positives and negatives are summed.

        The encode runs through ``self._train_batcher``, whose window is one
        wider than the model's. The shift below turns ``n + 1`` interactions
        into ``n`` inputs and ``n`` targets, so a history that fills the window
        puts an input on every position the model owns. Encoding at the model's
        own window instead would yield one input too few, and the highest
        position would never receive a gradient while prediction -- which does
        not shift -- reads it for exactly those full-length histories.

        Returns the mean loss and the number of positions it covers, so ``fit``
        can weight epochs by position count rather than by batch.
        """
        assert self.model is not None and self.batcher is not None
        assert self._train_batcher is not None
        assert self._rng is not None
        tokens, mask = self._train_batcher.encode(batch, device=self.device)
        if tokens.shape[1] < 2:
            # Every row in this batch holds at most one item.
            return None

        # Next-item shift, as SimpleRNNTrainer's, but paid for by the extra
        # interaction the training batcher retained rather than by the last
        # position. Nothing is decoded back to catalog positions here --
        # score_items reads embedding rows, so the reserved offset appears
        # only inside _sample_negatives.
        offset = self.batcher.tokenizer.n_reserved
        inputs = self._with_unk_dropout(tokens[:, :-1], mask[:, :-1])
        positives = tokens[:, 1:]
        # A real item, and one this vocabulary can name. Padding is excluded
        # by the mask; unk by the offset test, because "predict the item I
        # cannot identify" is not a question with an answer.
        # Both ends real. Under right padding the target's mask implied the
        # input's, because real tokens were a prefix; under left padding the
        # step before the first real one has a real target and a pad input, and
        # "given padding, predict this" is not a lesson.
        valid = mask[:, :-1] & mask[:, 1:] & (positives >= offset)
        n_positions = int(valid.sum())
        if n_positions == 0:
            return None

        negatives = self._sample_negatives(batch, positives, self._rng)
        states = self.model(inputs)
        # Scored on the full grid and masked after, rather than gathered first
        # the way the siblings must: what a state is scored against here is
        # n_negatives wide, so there is no rows x length x catalog tensor.
        positive_scores = self.model.score_items(states, positives)[valid]
        negative_scores = self.model.score_items(states, negatives)[valid]
        # Summed rather than averaged: each term is already a mean over its
        # own positions, and the reference weights the two equally.
        loss = objective(positive_scores, torch.ones_like(positive_scores))
        loss = loss + objective(
            negative_scores, torch.zeros_like(negative_scores)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), n_positions

    def _sample_negatives(
        self,
        batch: ItemSequences,
        positives: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Draw ``cfg.n_negatives`` item rows for each position.

        The result is shaped ``(rows, length, n)``, the second form
        :meth:`SASRec.score_items` accepts.

        Negatives are drawn from the catalog rows of the embedding table, never
        from the reserved ids -- padding and ``unk`` are not items and scoring
        them as negatives would train the model to reject its own filler.

        The draw is uniform over the paper's ``I \\ S_u``: every item in the
        user's history is excluded, not merely the position's own positive. An
        item they interacted with earlier -- or later, which the next-item shift
        makes just as reachable -- is one they did engage with, so training the
        model to rank it below the target teaches the opposite of what the data
        says. Excluding the whole set subsumes excluding the positive, which is
        why no separate collision test remains.

        ``S_u`` is read from ``batch`` rather than from the encoded tokens,
        because the paper's exclusion is over the user's sequence and not over
        the window that happens to be retained.

        The mapping avoids a rejection loop whose length would depend on the
        data. Each draw is uniform over ``n_items - |S_u|`` slots and then
        stepped onto the complement: with ``S_u`` sorted, ``seen[j] - j`` is how
        many allowed items fall below ``seen[j]``, so where a draw lands in that
        sequence is exactly how many exclusions it has to step over.
        """
        assert self.model is not None
        n_items = self.model.n_items
        n_reserved = self.model.n_reserved
        device = positives.device

        excluded, n_excluded = self._excluded_items(batch, n_items)
        available = n_items - n_excluded
        if int(available.min(initial=n_items)) < 1:
            raise ValueError(
                "a history covers the entire catalog, so there is no item "
                "outside it left to draw a negative from"
            )

        draws = torch.as_tensor(
            rng.integers(
                0,
                available.reshape(-1, 1, 1),
                size=tuple(positives.shape) + (self.cfg.n_negatives,),
            ),
            dtype=torch.long,
            device=device,
        )
        # Padding sits above every possible draw, so it is never stepped over.
        offsets = excluded - np.arange(excluded.shape[1])
        offsets[excluded >= n_items] = n_items + 1
        steps = torch.searchsorted(
            torch.as_tensor(offsets, dtype=torch.long, device=device),
            draws.reshape(draws.shape[0], -1),
            right=True,
        ).reshape(draws.shape)
        return draws + steps + n_reserved

    @staticmethod
    def _excluded_items(
        batch: ItemSequences, n_items: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Each row's item set, sorted and padded, with the count of real entries.

        Unused slots hold ``n_items``, which is above every catalog position and
        so sorts to the end and compares out of range wherever it is tested.

        Duplicates collapse: an item interacted with twice is one exclusion, and
        counting it twice would shrink the range the draw is uniform over and
        push the mapping past items that were never excluded.
        """
        lengths = batch.row_lengths
        width = int(lengths.max()) if batch.n_rows else 0
        excluded = np.full((batch.n_rows, width), n_items, dtype=np.int64)
        if width:
            filled = np.arange(width)[None, :] < lengths[:, None]
            excluded[filled] = np.asarray(batch.values, dtype=np.int64)
            excluded.sort(axis=1)
            duplicate = np.zeros_like(excluded, dtype=bool)
            duplicate[:, 1:] = excluded[:, 1:] == excluded[:, :-1]
            # Real items only: the pad value repeats by construction.
            duplicate &= excluded < n_items
            excluded[duplicate] = n_items
            excluded.sort(axis=1)
        return excluded, (excluded < n_items).sum(axis=1)

    def _with_unk_dropout(
        self, item_history: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace a fraction of real input positions with ``unk``.

        Applied to the inputs *after* the shift and never to the positives, so a
        corrupted position teaches "an item was here that you cannot identify,
        predict the next one anyway" rather than costing a training example.

        Padding is left alone: only real positions are eligible, or the model
        would learn that ``unk`` and ``pad`` mean the same thing. A no-op when
        the tokenizer names no ``unk`` or ``cfg.unk_dropout`` is zero.
        """
        assert self.batcher is not None
        unk_id = getattr(self.batcher.tokenizer, "unk_id", None)
        if unk_id is None or self.cfg.unk_dropout <= 0.0:
            return item_history
        chosen = (
            torch.rand(item_history.shape, device=item_history.device)
            < self.cfg.unk_dropout
        ) & mask
        return torch.where(
            chosen, torch.full_like(item_history, unk_id), item_history
        )

    # -- prediction ---------------------------------------------------------

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Rank the catalog for each history from its final state.

        Resolves candidates, validates ``k``, and returns an empty
        :class:`~compresso.SRPTensor` for an empty batch. Otherwise encodes the
        source, takes each row's own last real state through the batcher's
        ``gather_final`` -- never ``states[:, -1]``, which is padding for every
        row shorter than the batch maximum -- scores the catalog, masks seen
        items when asked, and takes the top ``k`` over the candidate columns.
        """
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError("SASRecTrainer must be fitted before predicting")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "SASRecTrainer predicts from ItemSequences, got "
                f"{type(source).__name__}"
            )
        n_items = self._n_items
        candidate_rows = self._candidate_rows(candidate_ids)
        candidate_count = int(candidate_rows.size)
        if not 1 <= k <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")
        if exclude_seen:
            self._check_unseen_capacity(
                source,
                n_items=n_items,
                k=k,
                candidate_rows=candidate_rows,
            )

        rows = source.n_rows
        if rows == 0:
            return SRPTensor(
                cols=torch.empty((0, k), dtype=torch.long, device=self.device),
                vals=torch.empty(
                    (0, k), dtype=torch.float32, device=self.device
                ),
                shape=(0, n_items),
            )

        self.model.eval()
        with torch.no_grad():
            item_history, mask = self.batcher.encode(
                source, device=self.device
            )
            final = self.batcher.gather_final(self.model(item_history), mask)
            scores = self.model.score(final)
            if exclude_seen:
                self._mask_seen(scores, source)
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            vals, local_cols = torch.topk(scores[:, candidates], k, dim=1)
            cols = candidates[local_cols]

        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))

    # -- persistence --------------------------------------------------------

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> SASRecTrainer:
        """Rebuild the trainer's shape before learned state is installed.

        Reads the trainer and tokenizer state written by
        :meth:`_save_checkpoint_state`, reconstructs the tokenizer and batcher,
        constructs the trainer with the stored config on ``device``, and builds
        an untrained model for the caller to load a ``state_dict`` into.
        """
        config = dict(config)
        # The stored device is where the model was trained, which says nothing
        # about where it is being loaded. The caller's choice wins, and writing
        # it into the config keeps cfg.device and self.device agreeing.
        config["device"] = str(device)
        trainer_state = reader.read_json("state/trainer.json")
        tokenizer_state = reader.read_json("state/tokenizer.json")
        if reader.exists("state/tokenizer_item_ids.json"):
            tokenizer_state["item_ids"] = reader.read_item_ids(
                "state/tokenizer_item_ids.json"
            )
        tokenizer = ItemTokenizer.from_dict(tokenizer_state)
        # Never None, unlike SimpleRNN's: fit reconciled the window with the
        # config before training, so every checkpoint carries a real one.
        batcher = SequenceBatcher(
            tokenizer,
            max_length=int(trainer_state["max_length"]),
            padding="left",
        )
        trainer = cls(SASRecConfig(**config), batcher)
        trainer._n_items = tokenizer.n_items
        # Built by _build_model rather than inline, so a reloaded model is
        # constructed by exactly the path that trained it.
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _build_checkpoint_optimizer(self) -> None:
        """Construct the optimizer before optimizer state is loaded into it."""
        if self.model is None:
            raise RuntimeError("SASRec model must be built before its optimizer")
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            **self.cfg.optimizer_kwargs(),
        )
