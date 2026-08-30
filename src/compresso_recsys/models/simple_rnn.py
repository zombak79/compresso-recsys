"""A recurrent next-item recommender — the smallest honest sequential baseline.

One training example per user: read the history left to right and predict the
next item at every position. That is the GRU4Rec objective, and it is the
cheapest thing that actually uses order, which makes it the baseline a
transformer has to beat before its extra machinery has earned anything.

The architecture is deliberately unremarkable::

    ItemSequences
      -> SequenceBatcher.encode        tokens (rows, length), mask
      -> Embedding(vocab, dim, padding_idx=pad_id)
      -> GRU or LSTM                   states (rows, length, hidden)
      -> Linear(hidden, n_items)       one score per catalog item
      -> cross entropy against the history shifted one step left

Two details carry all the risk, and both are pushed into
:class:`~compresso_recsys.models.sequence_batching.SequenceBatcher`.

**Reading the final state.** With right padding, the last *column* is padding
for every row shorter than the batch maximum, so scoring from ``states[:, -1]``
would score most users from a pad embedding. Prediction goes through
:meth:`~compresso_recsys.models.sequence_batching.SequenceBatcher.gather_final`,
which reads each row's own last real position.

**Truncation is not exclusion.** The batcher's ``max_length`` bounds what the
encoder reads, not what the model may recommend: ``exclude_seen`` masks the whole
history, including the part truncation dropped -- and, since a history may span a
wider catalog than the model was fitted on, including nothing it could not have
scored anyway.

Training uses a fixed epoch budget and rebuilds the model on every ``fit`` call;
early stopping and incremental training are not implemented. Tied embeddings
and sampled softmax are also absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from compresso import SRPTensor

from compresso_recsys.sequences import ItemSequences

from ._schedule import LRSchedule, build_scheduler, check_schedule
from .base import BaseSequentialRecommender
from .sequence_batching import SequenceBatcher
from .tokenizer import ItemTokenizer

__all__ = ["SimpleRNN", "SimpleRNNConfig", "SimpleRNNTrainer"]

RNNType = Literal["gru", "lstm"]
OptimizerName = Literal["NAdam", "AdamW"]


@dataclass
class SimpleRNNConfig:
    """Configuration for :class:`SimpleRNNTrainer`.

    ``dropout`` is applied to the states before scoring, and additionally
    between recurrent layers when ``num_layers > 1``. A single-layer RNN has no
    between-layer position to apply it, which is PyTorch's own behaviour rather
    than a choice made here.

    ``unk_dropout`` replaces that fraction of *input* positions with the
    tokenizer's ``unk`` token, teaching the model to read a history containing an
    item it cannot identify. It defaults to a non-zero rate because otherwise
    ``unk`` is never trained at all: the training vocabulary *is* the training
    window, so an out-of-catalog item cannot occur until evaluation, and its
    embedding would still sit at its initialisation when a quarter of a temporal
    test history turns out to need it.

    The right rate tracks the out-of-catalog share the model will actually face,
    which is a property of the split rather than of the model: near zero under
    ``leave_last_out``, and far higher on a late ``temporal`` stage. It is
    ignored when the tokenizer has no ``unk`` to substitute.
    """

    rnn_type: RNNType = "gru"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    unk_dropout: float = 0.05
    lr_schedule: LRSchedule = "constant"
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "NAdam"
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        check_schedule(self.lr_schedule, self.warmup_fraction, self.min_lr_ratio)
        if self.rnn_type not in ("gru", "lstm"):
            raise ValueError(
                f"rnn_type must be 'gru' or 'lstm', got {self.rnn_type!r}"
            )
        for name in ("embedding_dim", "hidden_dim", "num_layers", "batch_size"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.unk_dropout < 1.0:
            raise ValueError(
                f"unk_dropout must be in [0, 1), got {self.unk_dropout}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")


class SimpleRNN(nn.Module):
    """Embedding, recurrence, and a linear head over the catalog.

    The head outputs ``n_items`` scores rather than ``vocab_size``: special
    tokens are never prediction targets, so giving them output columns would
    train weights that can only ever be wrong.

    :meth:`forward` returns states and :meth:`score` turns states into logits,
    kept separate because prediction needs logits at one position per row.
    Scoring first and gathering after would materialise
    ``rows x length x n_items``, which on a real catalog is where the memory
    goes.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        n_items: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        rnn_type: RNNType,
        pad_id: int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=pad_id
        )
        recurrent = nn.GRU if rnn_type == "gru" else nn.LSTM
        self.rnn = recurrent(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch applies this between layers only, so a single-layer RNN
            # would silently ignore it. self.dropout below covers both cases.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, n_items)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Hidden states for every position, shape ``(rows, length, hidden)``."""
        states, _ = self.rnn(self.embedding(tokens))
        return states

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Catalog logits for the given states, one score per item."""
        return self.head(self.dropout(states))


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

    def __init__(
        self,
        config: SimpleRNNConfig | None = None,
        batcher: SequenceBatcher | None = None,
    ) -> None:
        self.cfg = config or SimpleRNNConfig()
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SimpleRNN | None = None
        self.batcher = batcher
        self._n_items: int | None = None

    # -- contract -----------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def n_items(self) -> int | None:
        return self._n_items

    # -- training -----------------------------------------------------------

    def fit(self, sequences: ItemSequences) -> SimpleRNNTrainer:
        """Train on chronological histories, one example per row."""
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "SimpleRNNTrainer trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")

        if self.batcher is None:
            # Right padding: an RNN reads to each row's own final position, so
            # trailing padding costs nothing and the shift below stays simple.
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items),
                max_length=self.DEFAULT_MAX_LENGTH,
            )
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
        self.model = SimpleRNN(
            vocab_size=tokenizer.vocab_size,
            n_items=tokenizer.n_items,
            embedding_dim=self.cfg.embedding_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            rnn_type=self.cfg.rnn_type,
            pad_id=tokenizer.pad_id,
        ).to(self.device)

        optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
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
        epoch_iter = _progress(
            self.cfg.show_progress,
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SimpleRNN fit",
        )
        batch_bar = _progress_bar(
            self.cfg.show_progress, total=len(starts), desc="SimpleRNN epoch 1"
        )
        try:
            for epoch in epoch_iter:
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"SimpleRNN epoch {epoch}")
                loss_sum, positions = 0.0, 0
                last_training_lr = float(optimizer.param_groups[0]["lr"])
                for start in starts:
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
                    if batch_bar is not None:
                        batch_bar.update(1)
                mean_loss = loss_sum / positions if positions else float("nan")
                self.history.append(
                    {
                        "epoch": float(epoch),
                        "loss": mean_loss,
                        "positions": float(positions),
                        "lr": last_training_lr,
                    }
                )
                if hasattr(epoch_iter, "set_postfix"):
                    # Loss only: positions is fixed by the data and the context
                    # window, so it belongs in history rather than on a live bar.
                    epoch_iter.set_postfix({"loss": f"{mean_loss:.4f}"})
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epoch_iter, "close"):
                epoch_iter.close()

        return self

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
        if not 1 <= k <= n_items:
            raise ValueError(f"k must be in [1, {n_items}], got {k}")
        if exclude_seen:
            self._check_unseen_capacity(source, n_items=n_items, k=k)

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
            vals, cols = torch.topk(logits, k, dim=1)

        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))

    def _mask_seen(self, logits: torch.Tensor, source: ItemSequences) -> None:
        """Forbid every item in the *full* history, truncated part included.

        Logits are indexed by catalog position, and a history may span a wider
        catalog than this model was fitted on -- a later split stage does exactly
        that. Items beyond the fitted catalog are dropped from the mask rather
        than clipped: they were never scoreable, so there is nothing to forbid.
        """
        if source.values.size == 0:
            return
        n_items = int(logits.shape[1])
        # The flat values are already the concatenation of every history, so one
        # scatter covers the batch. np.array copies, both because the buffers are
        # read-only and because torch.from_numpy would otherwise share them.
        rows = np.repeat(np.arange(source.n_rows), source.row_lengths)
        cols = np.array(source.values, dtype=np.int64)
        scoreable = cols < n_items
        if not scoreable.all():
            rows, cols = rows[scoreable], cols[scoreable]
        if cols.size == 0:
            return
        logits[
            torch.as_tensor(rows, dtype=torch.long, device=logits.device),
            torch.as_tensor(cols, dtype=torch.long, device=logits.device),
        ] = -torch.inf


def _tqdm():
    """The tqdm class, or ``None`` when it is not installed."""
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return tqdm


def _progress(enabled: bool, iterable, *, total: int, desc: str):
    """Wrap an iterable in a bar, or hand it back untouched."""
    tqdm = _tqdm()
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def _progress_bar(enabled: bool, *, total: int, desc: str):
    """A bar to drive by hand, or ``None`` when progress is unavailable."""
    tqdm = _tqdm()
    if not enabled or tqdm is None:
        return None
    return tqdm(total=total, desc=desc)
