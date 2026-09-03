"""A causal transformer recommender, and the example a sequential model should be.

`SimpleRNN` reads a history one step at a time and carries what it has seen in a
state vector. This reads the whole history at once and lets each position attend
to every earlier one, which is the only architectural difference that matters:
the objective, the vocabulary, the padding and the evaluation are identical.

The architecture is nanoGPT with two recommendation-shaped adjustments::

    ItemSequences
      -> SequenceBatcher.encode          tokens (rows, W), mask
      -> [CLS] + Embedding(vocab, d)     (rows, W + 1, d)
      -> + learned absolute positions
      -> N x pre-LN causal blocks
      -> LayerNorm
      -> Linear(d, n_items)              one score per catalog item
      -> cross entropy against the *unshifted* tokens

**Why the targets are unshifted.** A `CLS` prefix occupies position 0, so
``states[:, i]`` has read `CLS` plus ``tokens[:, :i]`` and therefore predicts
``tokens[:, i]``. The next-item shift stops being arithmetic in the trainer and
becomes a property of the input, which also means every position is a training
example rather than every position but the first — `CLS` buys back one example
per user compared with `SimpleRNN`.

**Why `CLS` is a parameter and not a token.** It could have been a vocabulary
entry, and that would be simpler. It is an `nn.Parameter` so it can be
*conditioned*: a user embedding or a global feature can be added into position 0
per row, which a vocabulary lookup cannot express. Nothing in this library has
user features yet, so today it is a bare learned prefix doing the job `BOS` would
do — including giving an empty history a defined input instead of the state after
reading one pad.

**Why there is no attention mask.** The batcher always pads on the right, so a
causal mask already excludes it: a real token at position ``i`` attends only to
``<= i``, all of which are real. Pad positions do compute garbage and nothing
reads it — the loss is masked and prediction reads each row's last real
position.

The output head is tied to the input embedding by default (``tie_embeddings``),
which halves the parameters.

Training uses a fixed epoch budget and rebuilds the model on every ``fit`` call;
early stopping and incremental training are not implemented. Sampled softmax, a
logit temperature, and pooling other than "read the last real position" are also
absent.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from compresso import SRPTensor

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    _Reporter,
    _format_duration,
    _resolve_reporter,
    _validate_log_every_n_steps,
)
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter

from ._schedule import LRSchedule, build_scheduler, check_schedule
from .base import BaseSequentialRecommender
from .identifiers import ItemVocabulary
from .sequence_batching import SequenceBatcher
from .tokenizer import ItemTokenizer

__all__ = [
    "SimpleGPT",
    "SimpleGPTConfig",
    "SimpleGPTTrainer",
    "TransformerConfig",
]

OptimizerName = Literal["NAdam", "AdamW"]


@dataclass(frozen=True)
class TransformerConfig:
    """The backbone, separated from the recommendation concerns around it.

    A transformer has one width. Unlike :class:`SimpleRNNConfig`, which lets
    ``embedding_dim`` and ``hidden_dim`` differ, the residual stream forces the
    embedding, the attention and the output of every block to share ``d_model``
    — and ``n_heads`` must divide it, since each head takes an equal slice.

    ``bias`` turns off the additive terms in the linear projections and the layer
    norms together. Off by default, following nanoGPT: it is slightly faster and
    marginally better, and having one flag rather than three keeps the
    combinations that were never tested from being expressible.
    """

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self) -> None:
        for name in ("d_model", "n_heads", "n_layers"):
            value = getattr(self, name)
            if value < 1:
                raise ValueError(f"{name} must be >= 1, got {value}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model must be divisible by n_heads, got d_model={self.d_model} "
                f"and n_heads={self.n_heads}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")

    @property
    def head_dim(self) -> int:
        """Width of each attention head."""
        return self.d_model // self.n_heads


@dataclass
class SimpleGPTConfig:
    """Configuration for :class:`SimpleGPTTrainer`.

    ``transformer`` carries the backbone; everything else is about training it.
    The context window is deliberately *not* a field — it belongs to the batcher,
    because it describes what the encoder reads rather than the shape of the
    network, and duplicating it is how the two drift apart. ``rstar`` carries it
    in both places and needs a runtime check to keep them equal.

    ``tie_embeddings`` scores with the input embedding's item rows instead of a
    separate head, halving the parameters. It is on by default; set it ``False``
    to use an independent output projection.

    Tying can change convergence as well as parameter count. ``nn.Linear`` initialises around
    ``+/-1/sqrt(d_model)`` while the embedding starts at ``std=0.02``, so a tied
    head begins with a flatter softmax. Compare variants at independently
    validated budgets rather than assuming their training curves match.

    ``unk_dropout`` replaces that fraction of input positions with the
    tokenizer's ``unk`` token. Non-zero by default because otherwise ``unk`` is
    never trained at all: the training vocabulary *is* the training window, so an
    out-of-catalog item cannot occur until evaluation, and its embedding would
    still sit at initialisation when a quarter of a temporal test history needs
    it. Match the rate to the out-of-catalog share you expect — near zero under
    ``leave_last_out``, far higher on a late ``temporal`` stage.
    """

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    tie_embeddings: bool = True
    lr_schedule: LRSchedule = "cosine"
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    unk_dropout: float = 0.05
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: OptimizerName = "NAdam"
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "SimpleGPT"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if not 0.0 <= self.unk_dropout < 1.0:
            raise ValueError(
                f"unk_dropout must be in [0, 1), got {self.unk_dropout}"
            )
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        check_schedule(self.lr_schedule, self.warmup_fraction, self.min_lr_ratio)


class LayerNorm(nn.Module):
    """Layer norm with an optional bias, which :class:`torch.nn.LayerNorm` lacks."""

    def __init__(self, ndim: int, *, bias: bool) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head self attention where a position may only read earlier ones.

    No mask is built or accepted. ``is_causal=True`` is the whole story, and it
    is sufficient *because* the batcher pads on the right — see the module
    docstring. Passing an additive mask as well would be legal but would give up
    the fused attention kernels for nothing.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rows, length, _ = x.shape
        query, key, value = self.attn(x).split(self.d_model, dim=2)
        shape = (rows, length, self.n_heads, self.d_model // self.n_heads)
        query = query.view(shape).transpose(1, 2)
        key = key.view(shape).transpose(1, 2)
        value = value.view(shape).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(rows, length, self.d_model)
        )
        return self.resid_dropout(self.proj(attended))


class MLP(nn.Module):
    """The position-wise feed-forward half of a block."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.up = nn.Linear(config.d_model, 4 * config.d_model, bias=config.bias)
        self.activation = nn.GELU()
        self.down = nn.Linear(4 * config.d_model, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(self.activation(self.up(x))))


class Block(nn.Module):
    """Pre-norm transformer block: norm before each sublayer, residual around it."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleGPT(nn.Module):
    """Embeddings, a `CLS` prefix, causal blocks, and a linear head.

    The head scores ``n_items`` rather than ``vocab_size``: a special token is
    never a prediction target, so an output column for one could only ever learn
    to be wrong — and it would let a misaligned objective score plausibly instead
    of raising.

    :meth:`forward` returns states and :meth:`score` turns states into logits,
    kept separate because prediction needs logits at one position per row.
    Scoring first would materialise ``rows x length x n_items``, which on a real
    catalog is where the memory goes.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        n_items: int,
        max_positions: int,
        pad_id: int,
        config: TransformerConfig,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        if max_positions < 2:
            raise ValueError(
                "max_positions must be >= 2: one slot for CLS and at least one "
                f"for an item, got {max_positions}"
            )
        if n_items > vocab_size:
            raise ValueError(
                f"n_items ({n_items}) cannot exceed vocab_size ({vocab_size}): "
                "the head scores a subset of the vocabulary"
            )
        self.config = config
        self.tie_embeddings = bool(tie_embeddings)
        # Items occupy the LAST n_items rows of the vocabulary, so this is the
        # tokenizer's n_reserved -- the front-loaded convention the trainer's
        # objective already relies on when it decodes targets. Deriving it keeps
        # the module from carrying a second copy that could disagree.
        self.item_offset = int(vocab_size) - int(n_items)
        self.max_positions = int(max_positions)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_positions, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layers))
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)
        self.head_dropout = nn.Dropout(config.dropout)
        # A tied head keeps its bias: tying is a claim about the weight, and
        # dropping the bias with it would confound two changes in one flag.
        if self.tie_embeddings:
            self.head = None
            self.head_bias = nn.Parameter(torch.zeros(n_items))
        else:
            self.head = nn.Linear(config.d_model, n_items)
            self.head_bias = None

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        # GPT-2's scaled init for the projections that write into the residual
        # stream. Without it the stream's variance grows with depth, since each
        # of the 2 * n_layers residual adds contributes at full scale. nanoGPT
        # matches c_proj by name; matching the modules directly cannot rot.
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
        # nn.Embedding zeroes padding_idx at construction and the initialisation
        # above overwrote it. Re-zero explicitly: padding_idx keeps the gradient
        # zero, so whatever sits there at the start stays there for good.
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 initialisation: every weight normal(0, 0.02), biases zero.

        PyTorch's default for ``nn.Linear`` is uniform over
        ``+/-1/sqrt(fan_in)``, which for ``d_model=128`` is roughly 2.5x wider
        than this. Leaving it there is a silent departure from the architecture
        this model claims to be, and it interacts with a tied head: the output
        weight would start at one scale and the input embedding at another.
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """States for `CLS` and every token, shape ``(rows, length + 1, d_model)``.

        ``states[:, i]`` has read `CLS` and ``tokens[:, :i]``, so it is the state
        from which ``tokens[:, i]`` should be predicted.
        """
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (rows, length), got {tuple(tokens.shape)}")
        rows, length = tokens.shape
        if length + 1 > self.max_positions:
            raise ValueError(
                f"a history of {length} items needs {length + 1} positions "
                f"including CLS, but this model was built for {self.max_positions}"
            )

        prefix = self.cls_token.expand(rows, 1, -1)
        hidden = torch.cat([prefix, self.embedding(tokens)], dim=1)
        positions = torch.arange(length + 1, device=tokens.device)
        hidden = self.embed_dropout(hidden + self.position(positions))
        for block in self.blocks:
            hidden = block(hidden)
        return self.ln_f(hidden)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Catalog logits for the given states, one score per item.

        When tied, the weight is a *slice* of the embedding rather than its own
        parameter: ``pad`` and ``unk`` sit below ``item_offset`` and so stay out
        of the head, which is what we want anyway — neither is ever a target.
        Autograd carries the output-side gradient back into the item rows, so a
        tied embedding is trained from both directions.
        """
        hidden = self.head_dropout(states)
        if self.head is not None:
            return self.head(hidden)
        return F.linear(hidden, self.embedding.weight[self.item_offset :], self.head_bias)


class SimpleGPTTrainer(BaseSequentialRecommender):
    """Trains and serves :class:`SimpleGPT`.

    Follows the package's shape, where ``fit`` returns the trainer and the
    trainer answers the prediction contract::

        model = SimpleGPTTrainer(
            SimpleGPTConfig(transformer=TransformerConfig(d_model=128, n_heads=4)),
            SequenceBatcher(ItemTokenizer(n_items), max_length=200),
        ).fit(split["x_train_sequences"])

    The encoder is a parameter, not something ``fit`` invents, which is how the
    context window and vocabulary are replaceable.
    Without one, ``fit`` builds a default over the training catalog with
    :attr:`DEFAULT_MAX_LENGTH` and right padding.

    One property of that batcher is load-bearing rather than advisory, so
    ``fit`` refuses a batcher without it. ``max_length`` must be set because it
    sizes the positional table and learned absolute positions need a bound.
    Right padding is an invariant of :class:`SequenceBatcher`, which lets the
    causal mask stand in for a padding mask.

    A history of a single interaction is a usable training example here, unlike
    for :class:`SimpleRNNTrainer` — the `CLS` prefix supplies the context, so
    every position is a target rather than every position but the first.

    :attr:`history` records one entry per epoch, numbered from one as ELSA's is,
    carrying the mean loss and the number of positions it was averaged over.
    """

    #: Context window used when ``fit`` has to build its own batcher.
    DEFAULT_MAX_LENGTH = 200
    checkpoint_type = "simple_gpt_trainer"

    def __init__(
        self,
        config: SimpleGPTConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config or SimpleGPTConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SimpleGPT | None = None
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
    ) -> SimpleGPTTrainer:
        """Train on chronological histories, one example per position."""
        reporter = self._reporter(logger, show_progress)
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "SimpleGPTTrainer trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")
        if int((sequences.row_lengths >= 1).sum()) == 0:
            raise ValueError(
                "every history is empty, so there is no next-item example to "
                "learn from"
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
                f"{self.batcher.tokenizer.n_items} items, but training sequences "
                f"have {sequences.n_items}"
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

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))

        self._n_items = self.batcher.tokenizer.n_items
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
        # The schedule is defined over the whole run, so it needs the step count
        # up front -- which is why this lives here rather than in the config.
        scheduler = self._build_scheduler(optimizer, len(starts) * self.cfg.epochs)
        # Two bars, as ELSA draws them: epochs outside, batches inside. The inner
        # bar is created once and rewound per epoch rather than a finished one
        # being left behind for each.
        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{n_rows} sequences | {self._n_items} items | {len(starts)} batches of "
            f"{batch_size} | {self.cfg.epochs} epochs | device {self.device}"
        )
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SimpleGPT fit",
        )
        batch_bar = reporter.bar(total=len(starts), desc="SimpleGPT epoch 1")
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"SimpleGPT epoch {epoch}")
                loss_sum, positions = 0.0, 0
                last_training_lr = float(optimizer.param_groups[0]["lr"])
                for step_index, start in enumerate(starts, start=1):
                    batch = sequences.select_rows(order[start : start + batch_size])
                    batch_lr = float(optimizer.param_groups[0]["lr"])
                    step = self._train_step(batch, optimizer, objective)
                    if step is not None:
                        last_training_lr = batch_lr
                    if scheduler is not None:
                        # Advanced even when _train_step declined the batch, so
                        # the curve is exactly the configured shape over the run
                        # rather than a slightly truncated one whose floor
                        # depends on how many batches happened to carry targets.
                        scheduler.step()
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
                    "lr": last_training_lr,
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

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """The schedule this trainer's config describes, or ``None`` if flat."""
        return build_scheduler(
            optimizer,
            schedule=self.cfg.lr_schedule,
            total_steps=total_steps,
            warmup_fraction=self.cfg.warmup_fraction,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )

    def _build_model(self) -> SimpleGPT:
        """The module this trainer's config and batcher describe.

        Shared by fitting and checkpoint loading so a reloaded model is built
        by exactly the path that trained it.
        """
        assert self.batcher is not None
        tokenizer = self.batcher.tokenizer
        return SimpleGPT(
            vocab_size=tokenizer.vocab_size,
            n_items=tokenizer.n_items,
            # One slot for CLS on top of the longest history the batcher emits.
            max_positions=int(self.batcher.max_length) + 1,
            pad_id=tokenizer.pad_id,
            config=self.cfg.transformer,
            tie_embeddings=self.cfg.tie_embeddings,
        ).to(self.device)

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> SimpleGPTTrainer:
        config = dict(config)
        transformer = TransformerConfig(**dict(config.pop("transformer")))
        config["device"] = str(device)
        trainer_state = reader.read_json("state/trainer.json")
        tokenizer_state = reader.read_json("state/tokenizer.json")
        if reader.exists("state/tokenizer_item_ids.json"):
            tokenizer_state["item_ids"] = reader.read_item_ids(
                "state/tokenizer_item_ids.json"
            )
        tokenizer = ItemTokenizer.from_dict(tokenizer_state)
        batcher = SequenceBatcher(
            tokenizer,
            max_length=int(trainer_state["max_length"]),
        )
        trainer = cls(
            SimpleGPTConfig(transformer=transformer, **config),
            batcher,
        )
        trainer._n_items = tokenizer.n_items
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        if self.batcher is None or not isinstance(
            self.batcher.tokenizer, ItemTokenizer
        ):
            raise TypeError(
                "SimpleGPTTrainer checkpoints support ItemTokenizer only"
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
        state = reader.read_json("state/trainer.json")
        history = state.get("history")
        if not isinstance(history, list):
            raise ValueError("SimpleGPT training history must be a list")
        self.history = list(history)

    def _build_checkpoint_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError("SimpleGPT model must be built before its optimizer")
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    @staticmethod
    def _check_batcher(batcher: SequenceBatcher) -> None:
        """Refuse a batcher whose settings this architecture cannot honour."""
        if batcher.max_length is None:
            raise ValueError(
                "SimpleGPT needs a bounded context: max_length sizes the "
                "positional table, and learned absolute positions cannot be "
                "extended at prediction time. Set max_length on the batcher"
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

        # No shift. CLS occupies position 0, so states[:, i] has read CLS and
        # tokens[:, :i] and therefore predicts tokens[:, i] -- the alignment is a
        # property of the input rather than arithmetic here. Dropping the last
        # state is all that is left of it: nothing follows the final token.
        offset = self.batcher.tokenizer.n_reserved
        targets = tokens - offset
        # A real item, and one this vocabulary can name. Padding is excluded by
        # the mask; unk and any unnamed reserved id by the offset test, because
        # "predict the item I cannot identify" is not a question with an answer.
        valid = mask & (tokens >= offset)
        n_positions = int(valid.sum())
        if n_positions == 0:
            return None

        # Corrupt the inputs only. The targets come from the clean tokens, so a
        # corrupted position teaches "an item was here you cannot identify,
        # predict the following one anyway" rather than costing an example.
        inputs = self._with_unk_dropout(tokens, mask)
        states = self.model(inputs)
        # Gather the scored positions before applying the head, never after. The
        # head is n_items wide, so scoring every position would materialise
        # rows x length x n_items -- 3.5 GB at batch 128 on a 34k catalog, of
        # which the padding is most of it. Indexing first costs 0.16 GB for the
        # same gradient. Prediction has always done this; training now agrees.
        loss = objective(
            self.model.score(states[:, :-1][valid]), targets[valid]
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        return float(loss.detach()), n_positions

    def _with_unk_dropout(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Replace a fraction of real input positions with ``unk``.

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
        """Rank the catalog for each history from its last real state."""
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError("SimpleGPTTrainer must be fitted before predicting")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "SimpleGPTTrainer predicts from ItemSequences, got "
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
            states = self.model(tokens)
            # States are one wider than the mask because of CLS, and
            # gather_final requires them to agree. Extending the mask rather
            # than adjusting indices by hand keeps the empty-history case right:
            # CLS is always real, so a row with no items reads position 0 and
            # scores from the learned prefix instead of from padding.
            prefix = torch.ones(
                (rows, 1), dtype=torch.bool, device=mask.device
            )
            final = self.batcher.gather_final(
                states, torch.cat([prefix, mask], dim=1)
            )
            logits = self.model.score(final)
            if exclude_seen:
                self._mask_seen(logits, source)
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            vals, local_cols = torch.topk(logits[:, candidates], k, dim=1)
            cols = candidates[local_cols]

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
