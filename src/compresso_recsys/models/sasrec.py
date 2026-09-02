from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from compresso import SRPTensor
from torch import nn

from compresso_recsys.models._schedule import (
    LRSchedule,
    build_scheduler,
    check_schedule,
)
from compresso_recsys.models.base import BaseSequentialRecommender
from compresso_recsys.models.identifiers import ItemVocabulary
from compresso_recsys.models.sequence_batching import SequenceBatcher
from compresso_recsys.models.tokenizer import ItemTokenizer
from compresso_recsys.persistence import (
    ModelCheckpointReader,
    ModelCheckpointWriter,
)
from compresso_recsys.sequences import ItemSequences

__all__ = ["SASRec", "SASRecConfig", "SASRecTrainer"]

# The published architecture's epsilon, against PyTorch's 1e-5 default.
# Nothing in the paper or its follow-ups tunes it.
LAYER_NORM_EPS = 1e-8

OptimizerName = Literal["NAdam", "AdamW"]


@dataclass(frozen=True)
class SASRecConfig:
    """Configuration for :class:`SASRec`.

    The context window is deliberately *not* a field. It belongs to the
    :class:`~compresso_recsys.models.sequence_batching.SequenceBatcher`, which
    already owns how far back a history is read, and it reaches the model as the
    ``max_history_length`` constructor argument that sizes the positional
    embedding. Holding it here as well is how the batcher's window and the
    embedding table drift apart, and the drift is only visible as an index error
    on some later batch.

    ``d_model`` is one width for the whole residual stream: the item embedding,
    the positional embedding, attention and the feed-forward output all share
    it, and ``n_heads`` must divide it. Unlike ``TransformerConfig``, there is no
    ``bias`` switch -- SASRec's projections and norms carry their biases, and the
    feed-forward is ``d_model -> d_model`` with a ReLU rather than the 4x GELU
    block ``SimpleGPT`` uses. Those are the architecture, not options.

    There is likewise no ``tie_embeddings``. SASRec scores a candidate by the dot
    product of the final state with that item's *input* embedding, so the tie is
    structural: an untied SASRec is a different model.

    ``dropout`` is the paper's single rate, applied to the embedding sum, inside
    attention, and between the feed-forward layers -- one knob because the
    reference implementation exposes one, and three independently tuned rates
    would be three numbers nobody has evidence for.

    ``n_negatives`` is how many sampled items each position scores against its
    true next item under the binary objective. One is the paper's setting and is
    enough on MovieLens-scale catalogs; raising it sharpens the gradient on a
    large catalog at a proportional cost per step.

    ``unk_dropout`` replaces that fraction of *input* positions with the
    tokenizer's ``unk`` token, teaching the model to read a history containing an
    item it cannot identify. It defaults to a non-zero rate because otherwise
    ``unk`` is never trained at all: the training vocabulary *is* the training
    window, so an out-of-catalog item cannot occur until evaluation, and its
    embedding would still sit at its initialisation when a quarter of a temporal
    test history turns out to need it. The right rate tracks the out-of-catalog
    share the split will actually produce -- near zero under ``leave_last_out``,
    far higher on a late ``temporal`` stage. It is ignored when the tokenizer has
    no ``unk`` to substitute.

    ``lr_schedule`` defaults to ``cosine`` as ``SimpleGPTConfig``'s does, for the
    reason attention needs it: the earliest steps have learned nothing, so their
    gradients are large and badly aimed, and warmup is what keeps them from
    setting the run's direction.
    """

    d_model: int = 64
    n_blocks: int = 2
    n_heads: int = 1
    dropout: float = 0.2
    n_negatives: int = 1
    unk_dropout: float = 0.05
    lr_schedule: LRSchedule = "cosine"
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    batch_size: int = 128
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "NAdam"
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        check_schedule(self.lr_schedule, self.warmup_fraction, self.min_lr_ratio)
        for name in ("d_model", "n_blocks", "n_heads", "batch_size", "n_negatives"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={self.d_model} "
                f"and n_heads={self.n_heads}"
            )
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
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.optimizer not in ("NAdam", "AdamW"):
            raise ValueError(
                f"optimizer must be 'NAdam' or 'AdamW', got {self.optimizer!r}"
            )


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_rate: float) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return  self.dropout2(
                    self.linear2(
                        self.relu(
                            self.dropout1(
                                self.linear1(
                                    inputs)))))


class SASRec(nn.Module):
    """Item and position embeddings, causal blocks, and a tied dot-product score.

    There is no output head. A candidate is scored by the dot product of a state
    with that candidate's *input* embedding, which is what makes the tie
    structural rather than an option -- see :class:`SASRecConfig`.

    :meth:`forward` returns states; :meth:`score` and :meth:`score_items` turn
    states into scores, kept separate because the two callers want different
    widths. Training scores a handful of sampled items per position, while
    prediction scores the whole catalog at one position per row. Fusing them
    would materialise ``rows x length x n_items``, which on a real catalog is
    where the memory goes.

    The embedding table holds ``n_reserved + n_items`` rows: the reserved ids
    first, the catalog after them, so catalog item ``i`` lives at row
    ``i + n_reserved``. That is
    :class:`~compresso_recsys.models.ItemTokenizer`'s layout, and taking
    ``n_reserved`` rather than a total keeps this module from having to work the
    split out for itself.

    **Padding is on the right**, as
    :class:`~compresso_recsys.models.sequence_batching.SequenceBatcher` produces
    it and unlike the reference implementation. Two consequences are worth
    naming. Causal attention already prevents a real position from reading the
    padding that follows it, so no padding mask is needed. But the last *column*
    is padding for every row shorter than the batch maximum, so a caller must
    take each row's own final state through
    :meth:`~compresso_recsys.models.sequence_batching.SequenceBatcher.gather_final`
    rather than slicing ``states[:, -1]``.

    Right padding also anchors position 1 to the oldest *retained* item, rather
    than anchoring the newest item to a fixed index the way left padding does.
    Since the batcher truncates to the most recent ``max_length``, a history and
    its own truncation agree on the numbering, which is the property that has to
    hold between training and prediction.
    """

    def __init__(
        self,
        *,
        n_items: int,
        n_reserved: int,
        max_history_length: int,
        pad_id: int,
        d_model: int,
        n_blocks: int,
        n_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if n_items < 1:
            raise ValueError(f"n_items must be >= 1, got {n_items}")
        if n_reserved < 1:
            raise ValueError(
                f"n_reserved must be >= 1, got {n_reserved}: padding alone needs "
                "an id below the catalog"
            )
        if max_history_length < 1:
            raise ValueError(
                f"max_history_length must be >= 1, got {max_history_length}"
            )
        if not 0 <= pad_id < n_reserved:
            raise ValueError(
                f"pad_id must be one of the {n_reserved} reserved ids, got {pad_id}"
            )
        self.n_items = int(n_items)
        self.n_reserved = int(n_reserved)
        self.max_history_length = int(max_history_length)
        self.pad_id = int(pad_id)

        self.item_embedding = nn.Embedding(
            self.n_reserved + self.n_items, d_model, padding_idx=pad_id
        )
        # Positions are numbered from one so index 0 stays reserved for padding
        # steps, hence the extra row.
        self.position_embedding = nn.Embedding(
            max_history_length + 1, d_model, padding_idx=0
        )
        self.embedding_dropout = nn.Dropout(p=dropout)

        self.attention_norms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_norms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()

        for _ in range(n_blocks):
            self.attention_norms.append(nn.LayerNorm(d_model, eps=LAYER_NORM_EPS))
            self.attention_layers.append(
                nn.MultiheadAttention(
                    d_model,
                    n_heads,
                    dropout=dropout,
                    batch_first=True,  # keeps (batch, steps, d_model) throughout
                )
            )
            self.forward_norms.append(nn.LayerNorm(d_model, eps=LAYER_NORM_EPS))
            self.forward_layers.append(PointWiseFeedForward(d_model, dropout))

        self.last_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier normal on every matrix, as the reference implementation does."""
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_normal_(parameter)

        # nn.Embedding zeroes padding_idx at construction and the loop above
        # overwrote it. Re-zero explicitly: padding_idx keeps the gradient zero,
        # so whatever sits there at the start stays there for good.
        with torch.no_grad():
            self.item_embedding.weight[self.pad_id].fill_(0)
            self.position_embedding.weight[0].fill_(0)

    def forward(self, item_history: torch.Tensor) -> torch.Tensor:
        """States for every step, shape ``(rows, length, d_model)``.

        ``item_history`` is ``(rows, length)`` of embedding-row ids, right
        padded -- what the batcher's ``encode`` returns. ``states[:, i]`` has
        read ``item_history[:, :i + 1]``, so it is the state from which
        ``item_history[:, i + 1]`` should be predicted.
        """
        if item_history.ndim != 2:
            raise ValueError(
                "item_history must be (rows, length), got "
                f"{tuple(item_history.shape)}"
            )
        n_steps = item_history.shape[1]
        if n_steps > self.max_history_length:
            raise ValueError(
                f"a history of {n_steps} items needs {n_steps} positions, but "
                f"this model was built for {self.max_history_length}"
            )

        real = item_history != self.pad_id
        hidden = self.item_embedding(item_history)
        # Xavier gives the embedding a fan-based scale rather than the unit-ish
        # one the norms downstream expect, and the reference rescales here to
        # compensate. Paired with _init_weights.
        hidden = hidden * (self.item_embedding.embedding_dim**0.5)

        # Padding steps take position 0, whose row is pinned to zero. Causal
        # attention already keeps them out of every real state, so this only
        # stops a pad row's own state from drifting into something readable.
        positions = torch.arange(1, n_steps + 1, device=item_history.device) * real
        hidden = self.embedding_dropout(hidden + self.position_embedding(positions))

        # True marks a pair that may not attend: step i reads 0..i, nothing later.
        causal_mask = torch.triu(
            torch.ones(
                (n_steps, n_steps),
                dtype=torch.bool,
                device=item_history.device,
            ),
            diagonal=1,
        )

        for attention_norm, attention, forward_norm, feed_forward in zip(
            self.attention_norms,
            self.attention_layers,
            self.forward_norms,
            self.forward_layers,
        ):
            normed = attention_norm(hidden)
            # need_weights=False keeps the fused attention kernel. The averaged
            # weights it would otherwise build are discarded here.
            attended, _ = attention(
                normed, normed, normed, attn_mask=causal_mask, need_weights=False
            )
            hidden = hidden + attended
            hidden = hidden + feed_forward(forward_norm(hidden))

        return self.last_norm(hidden)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Catalog scores for the given states, one per item.

        The weight is a *slice* of the embedding rather than its own parameter.
        The reserved rows -- padding, and an unknown item if the tokenizer names
        one -- sit below ``n_reserved`` and so stay out of the scores, which is
        what we want anyway: neither is ever a recommendation.
        """
        return F.linear(states, self.item_embedding.weight[self.n_reserved :])

    def score_items(
        self, states: torch.Tensor, items: torch.Tensor
    ) -> torch.Tensor:
        """Score each state against specific items, for sampled negatives.

        ``items`` holds embedding-row ids: ``(rows, length)`` to score one item
        per step, or ``(rows, length, n)`` for ``n`` of them, and the result
        carries the shape of ``items``. Scoring a handful this way is the point
        of the binary objective -- the full-catalog pass :meth:`score` would give
        is the cost SASRec is avoiding.
        """
        embeddings = self.item_embedding(items)
        if embeddings.ndim == states.ndim + 1:
            states = states.unsqueeze(-2)
        return (embeddings * states).sum(dim=-1)


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
    how you change the context window or the vocabulary -- including giving it an
    ``unk`` slot so a later split stage's unseen items become a reserved id
    rather than an error::

        batcher = SequenceBatcher(
            ItemTokenizer(n_items, item_ids=split["train_item_ids"]),
            max_length=50,
        )
        model = SASRecTrainer(SASRecConfig(), batcher).fit(sequences)

    Unlike :class:`SimpleRNNTrainer`, the batcher's ``max_length`` may not be
    ``None``: it sizes the positional embedding, which cannot be extended at
    prediction time. ``fit`` refuses a batcher without it.
    """

    #: Context window used when ``fit`` has to build its own batcher.
    DEFAULT_MAX_LENGTH = 200
    checkpoint_type = "sasrec_trainer"

    def __init__(
        self,
        config: SASRecConfig | None = None,
        batcher: SequenceBatcher | None = None,
    ) -> None:
        """Hold the config and encoder; build nothing until ``fit``.

        Sets ``self.cfg``, ``self.device``, ``self.history``, ``self.model``,
        ``self.optimizer``, ``self.batcher``, ``self._owns_batcher`` and
        ``self._n_items``, matching the two sibling trainers so the inherited
        persistence and ``to()`` paths find what they expect.

        ``self._rng`` is the one addition. Negative sampling draws from
        NumPy and ``_train_step``'s signature is fixed by the loop that
        calls it, so the generator ``fit`` seeds reaches it as state rather
        than as an argument. It is deliberately not checkpointed: a reloaded
        model predicts, and a further ``fit`` reseeds from ``cfg.seed``.
        """
        self.cfg = config or SASRecConfig()
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SASRec | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None
        self._rng: np.random.Generator | None = None

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
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "SASRecTrainer trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")
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
                max_length=self.DEFAULT_MAX_LENGTH,
            )
        if self.batcher is None:  # pragma: no cover - defensive against mutation
            raise RuntimeError("trainer batcher is unavailable")
        if self.batcher.tokenizer.n_items != sequences.n_items:
            raise ValueError(
                "batcher tokenizer has "
                f"{self.batcher.tokenizer.n_items} items, but training "
                f"sequences have {sequences.n_items}"
            )
        tokenizer_ids = getattr(self.batcher.tokenizer, "item_ids", None)
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
        self._check_batcher(self.batcher)
        # Counted after truncation, because the window is what the model
        # will actually read: a long history whose retained tail is one item
        # is no more trainable than a one-item history.
        usable = int((self.batcher.truncated_lengths(sequences) >= 2).sum())
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
            weight_decay=self.cfg.weight_decay,
        )
        optimizer = self.optimizer
        # Binary, not cross entropy: scoring a positive and its negatives
        # independently is what avoids normalising over the catalog.
        objective = nn.BCEWithLogitsLoss()
        self.history = []

        n_rows = sequences.n_rows
        batch_size = self.cfg.batch_size
        starts = range(0, n_rows, batch_size)
        # The schedule is defined over the whole run, so it needs the step
        # count up front -- which is why this lives here, not in the config.
        scheduler = build_scheduler(
            optimizer,
            schedule=self.cfg.lr_schedule,
            total_steps=len(starts) * self.cfg.epochs,
            warmup_fraction=self.cfg.warmup_fraction,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )
        # Two bars, as ELSA draws them: epochs outside, batches inside. The
        # inner bar is created once and rewound per epoch rather than a
        # finished one being left behind for each.
        epoch_iter = _progress(
            self.cfg.show_progress,
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SASRec fit",
        )
        batch_bar = _progress_bar(
            self.cfg.show_progress, total=len(starts), desc="SASRec epoch 1"
        )
        try:
            for epoch in epoch_iter:
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"SASRec epoch {epoch}")
                loss_sum, positions = 0.0, 0
                last_training_lr = float(optimizer.param_groups[0]["lr"])
                for start in starts:
                    batch = sequences.select_rows(
                        order[start : start + batch_size]
                    )
                    batch_lr = float(optimizer.param_groups[0]["lr"])
                    step = self._train_step(batch, optimizer, objective)
                    if step is not None:
                        last_training_lr = batch_lr
                    if scheduler is not None:
                        # Advanced even when _train_step declined the batch,
                        # so the curve is exactly the configured shape over the
                        # run rather than a slightly truncated one whose floor
                        # depends on how many batches carried targets.
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
                    epoch_iter.set_postfix({"loss": f"{mean_loss:.4f}"})
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epoch_iter, "close"):
                epoch_iter.close()

        return self

    @staticmethod
    def _check_batcher(batcher: SequenceBatcher) -> None:
        """Reject a batcher SASRec cannot use.

        ``max_length`` must be set, because it sizes the positional embedding
        and a checkpoint cannot grow one after the fact.
        """
        if batcher.max_length is None:
            raise ValueError(
                "SASRec needs a bounded context: max_length sizes the "
                "positional table, and learned absolute positions cannot be "
                "extended at prediction time. Set max_length on the batcher"
            )

    def _build_model(self) -> SASRec:
        """Construct :class:`SASRec` from the config and the batcher's tokenizer.

        The tokenizer supplies ``n_items``, ``n_reserved`` and ``pad_id``; the
        batcher supplies ``max_history_length``. The config supplies only the
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

        Returns the mean loss and the number of positions it covers, so ``fit``
        can weight epochs by position count rather than by batch.
        """
        assert self.model is not None and self.batcher is not None
        assert self._rng is not None
        tokens, mask = self.batcher.encode(batch, device=self.device)
        if tokens.shape[1] < 2:
            # Every row in this batch holds at most one item.
            return None

        # Next-item shift, as SimpleRNNTrainer's. Nothing is decoded back to
        # catalog positions here -- score_items reads embedding rows, so the
        # reserved offset appears only inside _sample_negatives.
        offset = self.batcher.tokenizer.n_reserved
        inputs = self._with_unk_dropout(tokens[:, :-1], mask[:, :-1])
        positives = tokens[:, 1:]
        # A real item, and one this vocabulary can name. Padding is excluded
        # by the mask; unk by the offset test, because "predict the item I
        # cannot identify" is not a question with an answer.
        valid = mask[:, 1:] & (positives >= offset)
        n_positions = int(valid.sum())
        if n_positions == 0:
            return None

        negatives = self._sample_negatives(positives, valid, self._rng)
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
        positives: torch.Tensor,
        valid: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Draw ``cfg.n_negatives`` item rows for each position.

        The result is shaped ``(rows, length, n)``, the second form
        :meth:`SASRec.score_items` accepts.

        Negatives are drawn from the catalog rows of the embedding table, never
        from the reserved ids -- padding and ``unk`` are not items and scoring
        them as negatives would train the model to reject its own filler.

        A draw never collides with the position's own positive: it is uniform
        over ``n_items - 1`` slots and then shifted past that positive. That is
        the distribution a rejection loop converges to, reached in one
        comparison rather than in a retry whose length depends on the data.

        What is deliberately *not* reproduced is the reference implementation's
        stronger exclusion, which rejects every item anywhere in the user's
        history. Under this objective an item seen much earlier is a legitimate
        hard negative, and excluding it would mean carrying each row's item set
        into the sampler.
        """
        assert self.model is not None
        n_items = self.model.n_items
        n_reserved = self.model.n_reserved
        draws = torch.as_tensor(
            rng.integers(
                0,
                n_items - 1,
                size=tuple(positives.shape) + (self.cfg.n_negatives,),
            ),
            dtype=torch.long,
            device=positives.device,
        )
        # Compared as catalog positions, not embedding rows, so both sides of
        # the shift mean the same thing. Where valid is false the positive is
        # padding or unk and skips nothing; the objective never reads it.
        positive_rows = (positives - n_reserved).unsqueeze(-1)
        skip = valid.unsqueeze(-1) & (draws >= positive_rows)
        return draws + skip.long() + n_reserved

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
        # Never None, unlike SimpleRNN's: _check_batcher refused an unbounded
        # window at fit time, so every checkpoint carries a real one.
        batcher = SequenceBatcher(
            tokenizer,
            max_length=int(trainer_state["max_length"]),
        )
        trainer = cls(SASRecConfig(**config), batcher)
        trainer._n_items = tokenizer.n_items
        # Built by _build_model rather than inline, so a reloaded model is
        # constructed by exactly the path that trained it.
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        """Write the non-module state: ``max_length``, history, and tokenizer.

        Item IDs go to their own entry when the tokenizer carries them, since
        they may be arbitrary hashables rather than JSON scalars.
        """
        if self.batcher is None or not isinstance(
            self.batcher.tokenizer, ItemTokenizer
        ):
            raise TypeError(
                "SASRecTrainer checkpoints support ItemTokenizer only"
            )
        assert self.batcher.max_length is not None
        writer.write_json(
            "state/trainer.json",
            {
                "max_length": int(self.batcher.max_length),
                "history": self.history,
            },
        )
        writer.write_json(
            "state/tokenizer.json",
            self.batcher.tokenizer.to_dict(include_item_ids=False),
        )
        item_ids = self.batcher.tokenizer.item_ids
        if item_ids is not None:
            writer.write_item_ids("state/tokenizer_item_ids.json", item_ids)

    def _load_checkpoint_state(self, reader: ModelCheckpointReader) -> None:
        """Restore :attr:`history` from the archive."""
        state = reader.read_json("state/trainer.json")
        history = state.get("history")
        if not isinstance(history, list):
            raise TypeError("SASRec training history must be a list") 
        self.history = list(history)

    def _build_checkpoint_optimizer(self) -> None:
        """Construct the optimizer before optimizer state is loaded into it."""
        if self.model is None:
            raise RuntimeError("SASRec model must be built before its optimizer")
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )


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
