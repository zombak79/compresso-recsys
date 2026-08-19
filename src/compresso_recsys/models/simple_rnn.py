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

**Truncation is not exclusion.** ``max_length`` bounds what the encoder reads,
not what the model may recommend: ``exclude_seen`` masks the whole history,
including the part truncation dropped.

Tied embeddings, learning-rate schedules, early stopping and sampled softmax are
all deliberately absent. This is a baseline, and every one of those is a
separate claim that deserves to be measured on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from torch import nn

from compresso import SRPTensor

from compresso_recsys.sequences import ItemSequences

from .base import BaseSequentialRecommender
from .sequence_batching import SequenceBatcher

__all__ = ["SimpleRNN", "SimpleRNNConfig", "SimpleRNNTrainer"]

RNNType = Literal["gru", "lstm"]
OptimizerName = Literal["NAdam", "AdamW"]


@dataclass
class SimpleRNNConfig:
    """Configuration for :class:`SimpleRNNTrainer`.

    ``max_length`` bounds how far back the encoder reads, keeping cost per user
    constant on datasets where a handful of users have thousands of
    interactions. It is a claim about the model's memory only — the default of
    200 follows the sequential-recommendation literature, and ``None`` reads
    every history in full.

    ``dropout`` is applied to the states before scoring, and additionally
    between recurrent layers when ``num_layers > 1``. A single-layer RNN has no
    between-layer position to apply it, which is PyTorch's own behaviour rather
    than a choice made here.
    """

    rnn_type: RNNType = "gru"
    embedding_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.0
    max_length: int | None = 200
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "NAdam"
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
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
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.max_length is not None and self.max_length < 2:
            raise ValueError(
                "max_length must be >= 2 when set, since a next-item objective "
                f"needs two positions to form one example, got {self.max_length}"
            )


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

    Users with fewer than two interactions contribute no training example, since
    a next-item target needs a preceding item. They are still predictable: a
    history the model can read yields its state, and an empty history yields the
    state after a single pad, which is the same for every empty row and therefore
    a learned popularity-like prior.

    :attr:`history` records one entry per epoch carrying the mean loss and the
    number of positions it was averaged over. That count is worth reading rather
    than assuming: it is ``sum(min(length, max_length) - 1)``, so it shows what
    truncation costs. On MovieLens-1M with the default ``max_length=200``, 697 of
    6,033 users exceed the window and 80k of 543k training positions are dropped.
    """

    def __init__(self, config: SimpleRNNConfig | None = None) -> None:
        self.cfg = config or SimpleRNNConfig()
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SimpleRNN | None = None
        self.batcher: SequenceBatcher | None = None
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
        usable = int((sequences.row_lengths >= 2).sum())
        if usable == 0:
            raise ValueError(
                "no history has two or more interactions, so there is no "
                "next-item example to learn from"
            )

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))

        self._n_items = sequences.n_items
        self.batcher = SequenceBatcher(
            n_items=sequences.n_items,
            max_length=self.cfg.max_length,
            # Right padding: an RNN reads to each row's own final position, so
            # trailing padding costs nothing and the shift below stays simple.
            pad_side="right",
        )
        self.model = SimpleRNN(
            vocab_size=self.batcher.vocab_size,
            n_items=sequences.n_items,
            embedding_dim=self.cfg.embedding_dim,
            hidden_dim=self.cfg.hidden_dim,
            num_layers=self.cfg.num_layers,
            dropout=self.cfg.dropout,
            rnn_type=self.cfg.rnn_type,
            pad_id=self.batcher.pad_id,
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
        bar = _progress_bar(
            self.cfg.show_progress, self.cfg.epochs * len(starts), "training"
        )
        try:
            for epoch in range(self.cfg.epochs):
                self.model.train()
                order = rng.permutation(n_rows)
                loss_sum, positions = 0.0, 0
                for start in starts:
                    batch = sequences.select_rows(order[start : start + batch_size])
                    step = self._train_step(batch, optimizer, objective)
                    if step is not None:
                        batch_loss, batch_positions = step
                        loss_sum += batch_loss * batch_positions
                        positions += batch_positions
                    if bar is not None:
                        bar.update(1)
                mean_loss = loss_sum / positions if positions else float("nan")
                self.history.append(
                    {
                        "epoch": float(epoch),
                        "loss": mean_loss,
                        "positions": float(positions),
                    }
                )
                if bar is not None:
                    bar.set_postfix(loss=f"{mean_loss:.4f}")
        finally:
            if bar is not None:
                bar.close()

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

        # Next-item shift. Under right padding, mask[:, 1:] is exactly the set
        # of positions whose target is a real item, so no separate arithmetic
        # over lengths is needed and padding can never become a target.
        inputs, targets, valid = tokens[:, :-1], tokens[:, 1:], mask[:, 1:]
        n_positions = int(valid.sum())
        if n_positions == 0:
            return None

        logits = self.model.score(self.model(inputs))
        loss = objective(logits[valid], targets[valid])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), n_positions

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
        """Forbid every item in the *full* history, truncated part included."""
        if source.values.size == 0:
            return
        # The flat values are already the concatenation of every history, so one
        # scatter covers the batch. np.array copies, both because the buffers are
        # read-only and because torch.from_numpy would otherwise share them.
        rows = np.repeat(np.arange(source.n_rows), source.row_lengths)
        logits[
            torch.as_tensor(rows, dtype=torch.long, device=logits.device),
            torch.as_tensor(
                np.array(source.values, dtype=np.int64),
                dtype=torch.long,
                device=logits.device,
            ),
        ] = -torch.inf


def _progress_bar(enabled: bool, total: int, desc: str):
    """A tqdm bar when asked for and available, otherwise nothing."""
    if not enabled:
        return None
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - optional dependency
        return None
    return tqdm(total=total, desc=desc)
