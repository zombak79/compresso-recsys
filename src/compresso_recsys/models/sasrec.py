from __future__ import annotations

import time
from collections.abc import Hashable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from compresso import SRPTensor
from torch import nn

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
    _validate_log_every_n_steps,
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

OptimizerName = Literal["Adam"]


@dataclass(frozen=True)
class SASRecConfig:
    """Configuration for :class:`SASRec`.

    ``max_history_length`` is the context window, and this field owns it. It
    sizes the batcher ``fit`` builds when none was passed, and a batcher that
    was passed inherits it whenever that batcher's own ``max_length`` is
    ``None`` -- the usual case, because the reason to hand ``fit`` a batcher is
    the vocabulary it carries rather than the window. Stating the window in both
    places and disagreeing is an error rather than a silent win for either: it
    sizes the positional table, and a table that outlives the run cannot be
    built from a number the config does not know about.

    It belongs here rather than on the trainer because the paper tunes it per
    dataset alongside ``dropout`` -- 200 and 0.2 on MovieLens-1M, 50 and 0.5 on
    the sparse ones -- so a dataset's settings stay one object that a checkpoint
    records whole.

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

    ``betas`` belongs to ``Adam`` and to no other optimizer, which is why it is
    applied through :meth:`optimizer_kwargs` rather than passed unconditionally.
    The reference sets the second moment to 0.98 against PyTorch's 0.999,
    shortening the window the variance estimate averages over -- one sampled
    negative per position makes the gradient noisy between steps but not biased,
    and a longer window spends that noise on a stale scale instead of adapting
    through it.

    The learning rate is deliberately constant: there is no schedule field,
    because the published results are a flat 0.001 for the whole run.
    """

    d_model: int = 50
    n_blocks: int = 2
    n_heads: int = 1
    dropout: float = 0.2
    max_history_length: int = 200
    n_negatives: int = 1
    unk_dropout: float = 0.0
    batch_size: int = 128
    epochs: int = 201
    lr: float = 0.001
    optimizer: OptimizerName = "Adam"
    betas: tuple[float, float] = (0.9, 0.98)

    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "SASRec"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
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
        if self.optimizer != "Adam":
            raise ValueError(
                f"optimizer must be 'Adam', got {self.optimizer!r}"
            )
        # asdict writes a JSON array and reading it back gives a list, so a
        # reloaded config would otherwise carry a different type than a fresh
        # one and compare unequal to it. Frozen, hence object.__setattr__.
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2:
            raise ValueError(f"betas must be two values, got {self.betas!r}")
        if not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError(
                f"betas must each be in [0, 1), got {self.betas!r}"
            )

    def optimizer_kwargs(self) -> dict[str, object]:
        """Optimizer arguments beyond the parameters and ``lr``.

        ``betas`` is Adam's own hyperparameter rather than a universal one, so
        it is selected by :attr:`optimizer` here instead of being handed to
        whatever ``torch.optim`` class the name resolves to. Today that name can
        only be ``Adam``; the indirection is what keeps adding a second one from
        silently passing it an argument it does not take.
        """
        if self.optimizer == "Adam":
            return {"betas": self.betas}
        return {}


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

    **Padding is on the left**, as the reference implementation has it, and
    ``fit`` configures the batcher for it. The reason is the positional table:
    every row is filled to ``max_length``, so the newest interaction always lands
    in the final column and position *n* means "n from the end" for a user with
    twenty interactions and a user with two hundred alike. Under right padding
    position 1 would instead mean "oldest item still retained", which is a
    different anchor for every history length and leaves the highest rows trained
    only by the longest histories.

    It costs two things. Batches are ``max_length`` wide however short their
    histories, and causal masking no longer excludes padding on its own -- the
    pad steps now *precede* the real ones and sit inside every causal window, so
    :meth:`forward` masks them out of attention explicitly.
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
        self.n_heads = int(n_heads)

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

        real_mask = item_history != self.pad_id
        hidden = self.item_embedding(item_history)
        # Xavier gives the embedding a fan-based scale rather than the unit-ish
        # one the norms downstream expect, and the reference rescales here to
        # compensate. Paired with _init_weights.
        hidden = hidden * (self.item_embedding.embedding_dim**0.5)

        # Padding steps take position 0, whose row is pinned to zero. Causal
        # attention already keeps them out of every real state, so this only
        # stops a pad row's own state from drifting into something readable.
        positions = torch.arange(1, n_steps + 1, device=item_history.device) * real_mask
        hidden = self.embedding_dropout(hidden + self.position_embedding(positions))

        # True marks a pair that may not attend: step i reads 0..i, nothing
        # later, and never a padding step. The padding half is what left padding
        # makes necessary -- pad steps precede the real ones, so causal masking
        # alone would let every real step read them.
        causal = torch.triu(
            torch.ones(
                (n_steps, n_steps),
                dtype=torch.bool,
                device=item_history.device,
            ),
            diagonal=1,
        )
        blocked = causal.unsqueeze(0) | ~real_mask.unsqueeze(1)
        # A pad step's own causal window is all padding, and a row masked
        # everywhere softmaxes over nothing and returns NaN, which the residual
        # would then spread to the whole row. Letting every step read itself
        # costs nothing: a pad step's output is discarded either way.
        blocked = blocked & ~torch.eye(
            n_steps, dtype=torch.bool, device=item_history.device
        )
        causal_mask = blocked.repeat_interleave(self.n_heads, dim=0)

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

    def _reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        return _resolve_reporter(
            default_logger=self.logger,
            logger=logger,
            default_show_progress=self.cfg.show_progress,
            show_progress=show_progress,
            prefix=self.cfg.log_prefix,
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

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
                max_length=self.cfg.max_history_length,
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
        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{n_rows} sequences | {self._n_items} items | {len(starts)} batches of "
            f"{batch_size} | {self.cfg.epochs} epochs | device {self.device}"
        )
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SASRec fit",
        )
        batch_bar = reporter.bar(total=len(starts), desc="SASRec epoch 1")
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"SASRec epoch {epoch}")
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
                    if batch_bar is not None:
                        batch_bar.update(1)
                    log_steps = reporter.log_every_n_steps
                    if log_steps and step_index % log_steps == 0:
                        reporter.step(
                            f"epoch {epoch}/{self.cfg.epochs} step "
                            f"{step_index}/{len(starts)}",
                            step_index,
                            len(starts),
                            epoch_started,
                            {
                                "loss": (
                                    loss_sum / positions
                                    if positions
                                    else float("nan")
                                )
                            },
                        )
                mean_loss = loss_sum / positions if positions else float("nan")
                record = {
                    "epoch": float(epoch),
                    "loss": mean_loss,
                    "positions": float(positions),
                }
                self.history.append(record)
                reporter.epoch(
                    f"epoch {epoch}/{self.cfg.epochs}",
                    record,
                    epoch_started,
                )
                if hasattr(epoch_iter, "set_postfix"):
                    epoch_iter.set_postfix({"loss": f"{mean_loss:.4f}"})
        finally:
            if batch_bar is not None:
                batch_bar.close()
            if hasattr(epoch_iter, "close"):
                epoch_iter.close()

        reporter.log(
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} total | "
            f"{len(self.history)} epochs recorded"
        )
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
            **self.cfg.optimizer_kwargs(),
        )


if __name__ == "__main__":
    # A MovieLens-1M reproduction of Kang & McAuley (2018), run through this
    # package's public API.
    #
    # It reports the same run under two protocols, because the obvious question
    # -- "is this number right?" -- has two different right answers and they
    # differ by an order of magnitude. The paper ranks the held-out item against
    # 100 sampled negatives and reports HR@10 and nDCG@10; this package's
    # benchmark table ranks the entire catalog and reports nDCG@20. Neither is
    # wrong, but a sampled number quoted into the full-catalog table would be,
    # so both are printed and each is labelled with what it may be compared to.
    import os
    import time
    from pathlib import Path

    import compresso_recsys as cr
    from compresso_recsys.evaluation import (
        evaluate_ranked_predictions,
        evaluate_recommender,
    )
    from compresso_recsys.metrics import MRR, NDCG, HitRate, Recall

    # -- what the paper fixed ------------------------------------------------

    # Table 3, the MovieLens-1M row. What section 4 is measured against.
    PAPER_HR10 = 0.8245
    PAPER_NDCG10 = 0.5905
    # Section 4.1: rank the true next item against 100 items the user never
    # interacted with, cut at 10.
    N_NEGATIVES = 100
    SAMPLED_CUTOFF = 10
    NEGATIVE_SEED = 0
    # docs/source/sequential-benchmarks.rst ranks the whole catalog at 20.
    FULL_CUTOFF = 20

    CHECKPOINT = Path("artifacts/ml1m/sasrec_paper_ml1m.zip")

    if torch.cuda.is_available():
        DEVICE = "cuda"
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        DEVICE = "mps"
    else:
        DEVICE = "cpu"

    # Every model hyperparameter stays at SASRecConfig's default, because those
    # defaults *are* the paper's MovieLens-1M settings: d=50, two blocks, one
    # head, dropout 0.2, a 200-step window, one negative per position, Adam at
    # 1e-3 with betas (0.9, 0.98), batch 128. A device is not a hyperparameter,
    # so it is the only thing overridden.
    cfg = replace(SASRecConfig(), device=DEVICE)

    # A short run is for checking that the script works, never for reporting.
    # Section 4 withholds the comparison under one rather than leaving a number
    # that looks quotable sitting in a column next to the paper's.
    shortened = os.environ.get("SASREC_EPOCHS")
    if shortened:
        cfg = replace(cfg, epochs=int(shortened))

    started = time.time()

    def elapsed() -> str:
        return f"[{time.time() - started:7.1f}s]"

    print(
        f"device: {DEVICE}   epochs: {cfg.epochs}   "
        f"window: {cfg.max_history_length}"
    )
    if shortened:
        print("  SASREC_EPOCHS is set: this is a smoke run, not a reproduction")

    # -- 1. the split ---------------------------------------------------------

    print(f"\n{elapsed()} checkpoint")
    if CHECKPOINT.exists():
        print(f"  reusing {CHECKPOINT}")
    else:
        print(f"  building {CHECKPOINT} from data/movielens1m")
        cr.build_recsys_checkpoint(
            dataset="ml1m",
            data_dir="data",
            checkpoint_path=str(CHECKPOINT),
            # The one split mode that emits sequences, and the one that matches
            # the paper: it holds out each user's last interaction for test and
            # the second-to-last for validation.
            split_mode="leave_last_out",
            # Section 4.1 discards users and items with fewer than five
            # interactions.
            min_user_support=5,
            item_min_support=5,
            # Every rating, because the paper treats "the presence of a review
            # or rating as implicit feedback" -- a one-star rating is an
            # interaction. This has to be said explicitly: omitting it does not
            # mean "no threshold", it means the ml1m spec's default of 4.0,
            # which silently drops about 45% of the events and would make this
            # a different experiment than the one the numbers below claim.
            # Ratings are integers in 1-5, so 1.0 keeps all of them.
            min_value_to_keep=1.0,
            # Same trap, second door. The builder defaults this to 30, which
            # drops every movie whose title, genres and description together
            # run under thirty words -- 276 of them, and 22k ratings with
            # them. That is a sensible default for the text-conditioned models
            # this package also serves and wrong for a SASRec reproduction,
            # which reads IDs and no text at all.
            min_entity_text_words=0,
            seed=0,
            show_progress=False,
        )

    with cr.read_checkpoint(CHECKPOINT) as root:
        split = cr.load_recsys_split(root)

    train_sequences = split["x_train_sequences"]
    test_source = split["test_source_sequences"]
    test_targets = split["test_target_matrix"]
    item_ids = split["train_item_ids"]

    lengths = np.diff(train_sequences.indptr)
    users = train_sequences.n_rows
    items = train_sequences.n_items
    # Table 2 counts every action. Training holds out two per user, so those go
    # back before the comparison, or this understates the dataset by 1.2%.
    actions = int(train_sequences.values.size) + 2 * users
    print(f"  {users} users, {items} items, {actions} actions")
    print(
        f"  {actions / users:.1f} actions/user, {actions / items:.1f} "
        f"actions/item, median history {int(np.median(lengths))}"
    )
    print("  paper table 2:  6040 users, 3416 items, 163.5 and 289.1")

    # -- 2. train -------------------------------------------------------------

    print(f"\n{elapsed()} train")
    # The batcher carries the real MovieLens IDs, which is what lets recommend()
    # take and return them. It names no window, so the window is the config's
    # and this script cannot drift from the default it exists to run.
    batcher = SequenceBatcher(
        ItemTokenizer(train_sequences.n_items, item_ids=item_ids),
    )
    trainer = SASRecTrainer(cfg, batcher).fit(train_sequences, item_ids=item_ids)

    every = max(1, len(trainer.history) // 10)
    print(f"  {'epoch':>6}  {'loss':>9}")
    for index, entry in enumerate(trainer.history):
        if index % every == 0 or index == len(trainer.history) - 1:
            print(f"  {int(entry['epoch']):>6}  {entry['loss']:>9.4f}")

    # -- 3. scores ------------------------------------------------------------

    def catalog_scores(
        model: SASRecTrainer, source: ItemSequences, *, rows: int = 512
    ) -> np.ndarray:
        """Raw catalog scores per row, batched.

        Built here rather than read out of ``predict`` because the sampled
        protocol needs the score of specific items rather than the best ones,
        and ``predict`` has already discarded everything outside its top k.
        """
        out = np.empty((source.n_rows, source.n_items), dtype=np.float32)
        model.model.eval()
        for start in range(0, source.n_rows, rows):
            stop = min(start + rows, source.n_rows)
            chunk = ItemSequences(
                values=source.values[source.indptr[start] : source.indptr[stop]],
                indptr=source.indptr[start : stop + 1] - source.indptr[start],
                n_items=source.n_items,
            )
            with torch.no_grad():
                tokens, mask = model.batcher.encode(chunk, device=model.device)
                final = model.batcher.gather_final(model.model(tokens), mask)
                out[start:stop] = model.model.score(final).float().cpu().numpy()
        return out

    print(f"\n{elapsed()} scoring {test_source.n_rows} test users")
    scores = catalog_scores(trainer, test_source)

    # -- 4. the paper's protocol: 100 sampled negatives, cut at 10 ------------

    def sampled_protocol(
        scores, interacted, targets, *, n_negatives, cutoff, seed
    ) -> tuple[float, float, int]:
        """Rank each held-out item against negatives the user never touched.

        Ties go to the negative. With ``>`` instead of ``>=`` an untrained model
        that gives every item the same score would report a perfect hit rate.
        """
        rng = np.random.default_rng(seed)
        hits: list[float] = []
        gains: list[float] = []
        for row in range(scores.shape[0]):
            positives = targets[row].indices
            if positives.size == 0:
                continue
            pool = np.flatnonzero(~interacted[row])
            if pool.size < n_negatives:
                continue
            negatives = rng.choice(pool, size=n_negatives, replace=False)
            # Leave-last-out gives one held-out item per user.
            target = int(positives[0])
            above = int((scores[row, negatives] >= scores[row, target]).sum())
            hits.append(float(above < cutoff))
            gains.append(1.0 / np.log2(above + 2) if above < cutoff else 0.0)
        return float(np.mean(hits)), float(np.mean(gains)), len(hits)

    # Everything the user ever touched, so a negative is genuinely unseen: the
    # test source is the whole history bar the last item, and the target is it.
    interacted = (split["test_source_matrix"] + test_targets).astype(bool).toarray()
    hr, ndcg, scored = sampled_protocol(
        scores,
        interacted,
        test_targets,
        n_negatives=N_NEGATIVES,
        cutoff=SAMPLED_CUTOFF,
        seed=NEGATIVE_SEED,
    )

    print(
        f"\n{elapsed()} paper protocol -- "
        f"{N_NEGATIVES} sampled negatives, @{SAMPLED_CUTOFF}"
    )
    print(f"  {'metric':<12}{'this run':>10}{'paper':>10}{'delta':>10}")
    if shortened:
        print(f"  {'HR@10':<12}{hr:>10.4f}{'--':>10}{'--':>10}")
        print(f"  {'NDCG@10':<12}{ndcg:>10.4f}{'--':>10}{'--':>10}")
        print("  paper column withheld: SASREC_EPOCHS shortened this run")
    else:
        print(
            f"  {'HR@10':<12}{hr:>10.4f}{PAPER_HR10:>10.4f}"
            f"{hr - PAPER_HR10:>+10.4f}"
        )
        print(
            f"  {'NDCG@10':<12}{ndcg:>10.4f}{PAPER_NDCG10:>10.4f}"
            f"{ndcg - PAPER_NDCG10:>+10.4f}"
        )
    print(f"  scored {scored} of {test_source.n_rows} users")

    # -- 5. this package's protocol: the whole catalog, cut at 20 -------------

    print(f"\n{elapsed()} benchmark protocol -- full catalog, @{FULL_CUTOFF}")
    metrics = [
        NDCG(FULL_CUTOFF),
        Recall(FULL_CUTOFF),
        HitRate(FULL_CUTOFF),
        MRR(FULL_CUTOFF),
    ]
    result = evaluate_recommender(
        trainer,
        source=test_source,
        targets=test_targets,
        metrics=metrics,
        sample_ids=split["test_eval_user_ids"],
    )

    # Most-popular, scored identically. A sequential model that has silently
    # stopped reading order lands here rather than crashing, so the gap is what
    # says the run is sound before its absolute numbers say anything.
    popularity = np.asarray(split["x_train"].sum(axis=0)).ravel()
    by_popularity = np.argsort(-popularity, kind="stable")
    seen = split["test_source_matrix"].astype(bool).toarray()[:, by_popularity]
    # A stable argsort over the seen flags floats each row's unseen items to the
    # front while leaving them in popularity order.
    unseen_first = np.argsort(seen, axis=1, kind="stable")[:, :FULL_CUTOFF]
    baseline = evaluate_ranked_predictions(
        predictions=SRPTensor(
            cols=torch.from_numpy(by_popularity[unseen_first]).long(),
            # Rank order rather than the counts: tied counts are not a ranking,
            # and the evaluator is right to reject them.
            vals=torch.arange(FULL_CUTOFF, 0, -1, dtype=torch.float32).expand(
                test_source.n_rows, FULL_CUTOFF
            ),
            shape=(test_source.n_rows, train_sequences.n_items),
        ),
        targets=test_targets,
        metrics=metrics,
        sample_ids=split["test_eval_user_ids"],
    )

    print(f"  {'metric':<16}{'SASRec':>10}{'popular':>10}{'lift':>9}")
    for name in result.metrics:
        ours, theirs = result[name], baseline[name]
        lift = f"{ours / theirs:.1f}x" if theirs else "--"
        print(f"  {name:<16}{ours:>10.4f}{theirs:>10.4f}{lift:>9}")
    print(f"  scored {result.n_scored_rows} of {result.n_rows} users")
    print("  this is the column that belongs in sequential-benchmarks.rst")

    # -- 6. validation, to say whether the epoch budget was right -------------

    # The paper trains a fixed budget and so does this. Whether that budget was
    # over or under for this split is a question the test column cannot answer
    # and the validation target can.
    print(f"\n{elapsed()} validation -- full catalog, @{FULL_CUTOFF}")
    validation = evaluate_recommender(
        trainer,
        source=split["val_source_sequences"],
        targets=split["val_target_matrix"],
        metrics=[NDCG(FULL_CUTOFF)],
        sample_ids=split["val_eval_user_ids"],
    )
    val_ndcg = validation[f"ndcg@{FULL_CUTOFF}"]
    test_ndcg = result[f"ndcg@{FULL_CUTOFF}"]
    print(
        f"  val ndcg@{FULL_CUTOFF} {val_ndcg:.4f}   "
        f"test ndcg@{FULL_CUTOFF} {test_ndcg:.4f}"
    )
    print(
        "  the two tracking each other means the budget was about right; "
        "validation far above test means it overfit"
    )

    # -- 7. keep the run ------------------------------------------------------

    trainer.save_to_checkpoint(CHECKPOINT, "sasrec")
    print(f"\n{elapsed()} saved into {CHECKPOINT.name} as 'sasrec'")
    print(f"{elapsed()} done")
