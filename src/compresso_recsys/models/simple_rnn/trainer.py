"""The training procedure, and the fitted model's prediction path."""

from __future__ import annotations

from __future__ import annotations

from typing import Any, Hashable, Literal, Sequence

import numpy as np
import torch
from torch import nn

from compresso import SRPTensor

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
    _validate_log_every_n_steps,
)
from compresso_recsys.persistence import ModelCheckpointReader
from compresso_recsys.sequences import ItemSequences

from ..core.schedule import LRSchedule, build_scheduler, check_schedule
from ..base import BaseSequentialRecommender
from ..sequence_batching import SequenceBatcher
from ..tokenizer import ItemTokenizer
from compresso_recsys.models.simple_rnn.config import (
    SimpleRNNConfig,
)
from compresso_recsys.models.simple_rnn.model import (
    SimpleRNN,
)


class SimpleRNNTrainer(BaseSequentialRecommender):
    """Trains and serves :class:`SimpleRNN`.

    Follows the package's existing shape, where ``fit`` returns the trainer and
    the trainer answers the prediction contract::

        model = SimpleRNNTrainer(SimpleRNNConfig(rnn_type="gru")).fit(
            split["x_train_sequences"]
        )
        result = evaluate_recommender(
            model, source=split["test_source_sequences"],
            targets=split["test_target_matrix"], metrics=[NDCG(20)],
        )

    The encoder is a *parameter*, not something ``fit`` invents. Passing one is
    how you change the context window or the vocabulary -- including giving it
    an ``unk`` slot so a later split stage's unseen items become a token rather
    than an error::

        batcher = SequenceBatcher(
            ItemTokenizer(n_items, item_ids=split["train_item_ids"]),
            max_length=50,
        )
        model = SimpleRNNTrainer(SimpleRNNConfig(), batcher).fit(sequences)

    Without one, ``fit`` builds a default over the training catalog with
    :attr:`DEFAULT_MAX_LENGTH` and right padding.
    A supplied batcher must also use right padding: leading padding would advance
    the recurrent state and turn the first real item into a target of padding.

    Users retaining fewer than two interactions after truncation contribute no
    training example, since a next-item target needs a preceding item. ``fit``
    refuses a dataset where that leaves no usable history. Short histories are
    still predictable: a history the model can read yields its state, and an
    empty history yields the state after a single pad, which is the same for
    every empty row and therefore a learned popularity-like prior.

    :attr:`history` records one entry per epoch, numbered from one as ELSA's is,
    carrying the mean loss and the number of positions it was averaged over. That
    count is worth reading rather than assuming: it is
    ``sum(max(min(length, batcher.max_length) - 1, 0))``, so it shows what
    truncation costs. On MovieLens-1M at the default window of 200, 697 of 6,033
    users exceed it and 80k of 543k training positions are dropped.
    """

    #: Context window used when ``fit`` has to build its own batcher.
    DEFAULT_MAX_LENGTH = 200
    checkpoint_type = "simple_rnn_trainer"

    def __init__(
        self,
        config: SimpleRNNConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config or SimpleRNNConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SimpleRNN | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None

    # -- contract -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def n_items(self) -> int | None:
        return self._n_items

    # -- training -----------------------------------------------------------

    def fit(
        self,
        sequences: ItemSequences,
        *,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SimpleRNNTrainer:
        """Train on chronological histories, one example per row."""
        reporter = self._reporter(logger, show_progress)
        self._check_training_sequences(sequences)

        if self._owns_batcher:
            # Right padding: an RNN reads to each row's own final position, so
            # trailing padding costs nothing and the shift below stays simple.
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items),
                max_length=self.DEFAULT_MAX_LENGTH,
            )
        # Checked before the vocabulary, as it always has been: a malformed
        # batcher is a clearer complaint than a catalog width mismatch.
        self._check_batcher(self._require_batcher())
        self._adopt_batcher_vocabulary(sequences, item_ids)
        usable = int((self.batcher.truncated_lengths(sequences) >= 2).sum())
        if usable == 0:
            raise ValueError(
                "no history retains two or more interactions after truncation, "
                "so there is no next-item example to learn from"
            )

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))

        tokenizer = self.batcher.tokenizer
        self._n_items = tokenizer.n_items
        self.model = self._build_model()

        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        optimizer = self.optimizer
        objective = nn.CrossEntropyLoss()
        self.history = []

        n_rows = sequences.n_rows
        batch_size = self.cfg.batch_size
        starts = range(0, n_rows, batch_size)
        scheduler = build_scheduler(
            optimizer,
            schedule=self.cfg.lr_schedule,
            total_steps=len(starts) * self.cfg.epochs,
            warmup_fraction=self.cfg.warmup_fraction,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )
        # Two bars, as ELSA draws them: epochs outside, batches inside. The
        # inner bar is created once and rewound per epoch rather than a finished
        # one being left behind for each.
        with TrainingProgress(
            reporter,
            label="SimpleRNN",
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
                last_training_lr = float(optimizer.param_groups[0]["lr"])
                for step_index, start in enumerate(starts, start=1):
                    batch = sequences.select_rows(order[start : start + batch_size])
                    batch_lr = float(optimizer.param_groups[0]["lr"])
                    step = self._train_step(batch, optimizer, objective)
                    if step is not None:
                        last_training_lr = batch_lr
                    if scheduler is not None:
                        # Advanced even on a batch the objective declined, so the
                        # curve is the configured shape over the run rather than
                        # one truncated by how many batches carried targets.
                        scheduler.step()
                    if step is not None:
                        batch_loss, batch_positions = step
                        loss_sum += batch_loss * batch_positions
                        positions += batch_positions
                    progress.batch(step_index, loss_sum, positions)
                # Loss only on the live bar: positions is fixed by the data and
                # the context window, so it belongs in history instead.
                record = {
                    "epoch": float(epoch),
                    "loss": (
                        loss_sum / positions if positions else float("nan")
                    ),
                    "positions": float(positions),
                    "lr": last_training_lr,
                }
                self.history.append(record)
                progress.epoch_done(record)
        return self

    @staticmethod
    def _check_batcher(batcher: SequenceBatcher) -> None:
        """Refuse a batcher whose settings this architecture cannot honour."""
        if batcher.padding != "right":
            raise ValueError(
                "SimpleRNN requires right padding: leading padding changes the "
                "recurrent state and next-item target alignment"
            )

    def _build_model(self) -> SimpleRNN:
        if self.batcher is None:
            raise RuntimeError("SimpleRNN batcher is unavailable")
        tokenizer = self.batcher.tokenizer
        return SimpleRNN(
            vocab_size=tokenizer.vocab_size,
            n_items=tokenizer.n_items,
            embedding_dim=self.cfg.embedding_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            rnn_type=self.cfg.rnn_type,
            pad_id=tokenizer.pad_id,
        ).to(self.device)

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> SimpleRNNTrainer:
        config = dict(config)
        config["device"] = str(device)
        trainer_state = reader.read_json("state/trainer.json")
        tokenizer_state = reader.read_json("state/tokenizer.json")
        if reader.exists("state/tokenizer_item_ids.json"):
            tokenizer_state["item_ids"] = reader.read_item_ids(
                "state/tokenizer_item_ids.json"
            )
        tokenizer = ItemTokenizer.from_dict(tokenizer_state)
        max_length = trainer_state.get("max_length")
        batcher = SequenceBatcher(
            tokenizer,
            max_length=None if max_length is None else int(max_length),
        )
        trainer = cls(SimpleRNNConfig(**config), batcher)
        trainer._n_items = tokenizer.n_items
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _build_checkpoint_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("SimpleRNN model must be built before its optimizer")
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _train_step(
        self,
        batch: ItemSequences,
        optimizer: torch.optim.Optimizer,
        objective: nn.Module,
    ) -> tuple[float, int] | None:
        """One optimizer step, or ``None`` when the batch carries no target."""
        assert self.model is not None and self.batcher is not None
        tokens, mask = self.batcher.encode(batch, device=self.device)
        if tokens.shape[1] < 2:
            # Every row in this batch holds at most one item.
            return None

        # Next-item shift. The head is indexed by catalog position while the
        # tokens carry the vocabulary offset, so targets are decoded back --
        # the one place besides encode() where the offset appears at all.
        offset = self.batcher.tokenizer.n_reserved
        inputs = self._with_unk_dropout(tokens[:, :-1], mask[:, :-1])
        target_tokens = tokens[:, 1:]
        targets = target_tokens - offset
        # A real item, and one this vocabulary can name. Padding is excluded by
        # the mask; UNK is excluded by the offset test, because "predict an item
        # I cannot identify" is not a question with an answer.
        valid = mask[:, 1:] & (target_tokens >= offset)
        n_positions = int(valid.sum())
        if n_positions == 0:
            return None

        # Gather before scoring, as predict_on_batch does. Scoring first
        # materialises rows x length x n_items and then throws most of it away:
        # 3.46 GB at batch 128 on a 34k-item catalog against 0.16 GB this way.
        states = self.model(inputs)
        loss = objective(self.model.score(states[valid]), targets[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), n_positions

    def _with_unk_dropout(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace a fraction of real input positions with ``unk``.

        Applied to the inputs *after* the shift and never to the targets, so a
        corrupted position teaches "an item was here that you cannot identify,
        predict the next one anyway" rather than costing a training example.
        Corrupting before the shift would make those positions ``unk`` targets,
        which the objective excludes, so the signal would be lost instead of used.

        Padding is left alone: only real positions are eligible, or the model
        would learn that ``unk`` and ``pad`` mean the same thing.
        """
        assert self.batcher is not None
        unk_id = getattr(self.batcher.tokenizer, "unk_id", None)
        if unk_id is None or self.cfg.unk_dropout <= 0.0:
            return inputs
        chosen = (
            torch.rand(inputs.shape, device=inputs.device) < self.cfg.unk_dropout
        ) & mask
        return torch.where(chosen, torch.full_like(inputs, unk_id), inputs)

    # -- prediction ---------------------------------------------------------

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Rank the catalog for each history from its final recurrent state."""
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError("SimpleRNNTrainer must be fitted before predicting")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "SimpleRNNTrainer predicts from ItemSequences, got "
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
                vals=torch.empty((0, k), dtype=torch.float32, device=self.device),
                shape=(0, n_items),
            )

        self.model.eval()
        with torch.no_grad():
            tokens, mask = self.batcher.encode(source, device=self.device)
            final = self.batcher.gather_final(self.model(tokens), mask)
            logits = self.model.score(final)
            if exclude_seen:
                self._mask_seen(logits, source)
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            vals, local_cols = torch.topk(logits[:, candidates], k, dim=1)
            cols = candidates[local_cols]

        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))
