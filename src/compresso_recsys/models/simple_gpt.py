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

**Why there is no attention mask.** Padding is on the right, so a causal mask
already excludes it: a real token at position ``i`` attends only to ``<= i``, all
of which are real. Pad positions do compute garbage and nothing reads it — the
loss is masked and prediction reads each row's last real position. That is why
:class:`SimpleGPTTrainer` refuses a left-padding batcher rather than quietly
building a key-padding mask.

The output head is tied to the input embedding by default (``tie_embeddings``),
which halves the parameters and measured better on every split tried.

Deliberately absent: learning-rate schedules, early stopping, sampled softmax, a
logit temperature, and any pooling other than "read the last real position".
Each is a separate claim that deserves measuring on its own.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from compresso import SRPTensor

from compresso_recsys.sequences import ItemSequences

from .base import BaseSequentialRecommender
from .sequence_batching import SequenceBatcher
from .tokenizer import ItemTokenizer

__all__ = [
    "SimpleGPT",
    "SimpleGPTConfig",
    "SimpleGPTTrainer",
    "TransformerConfig",
    "load_simple_gpt",
    "save_simple_gpt",
]

OptimizerName = Literal["NAdam", "AdamW"]
LRSchedule = Literal["constant", "cosine"]


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
    separate head, halving the parameters. On by default: it won on every split
    measured -- ML-1M ``leave_last_out`` by ``+0.0127`` (four seed deviations),
    Office ``leave_last_out`` by ``+0.0074`` (six), Office ``temporal`` by
    ``+0.0009`` (under two) -- and in a capacity sweep every tied configuration
    beat every untied one, including a tied model with a third of the parameters
    beating an untied model with all of them. Set it ``False`` to reproduce
    figures recorded before this was the default.

    Tying converges *later*, not faster. ``nn.Linear`` initialises around
    ``+/-1/sqrt(d_model)`` while the embedding starts at ``std=0.02``, so a tied
    head begins with a flatter softmax: on ML-1M it trails untied at ten epochs
    and passes it by twenty. Compare the two at a validated budget rather than a
    fixed one, or the slower start reads as a worse model.

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
    lr_schedule: LRSchedule = "constant"
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

    def __post_init__(self) -> None:
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
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError(
                f"lr_schedule must be 'constant' or 'cosine', got {self.lr_schedule!r}"
            )
        if not 0.0 <= self.warmup_fraction < 1.0:
            raise ValueError(
                f"warmup_fraction must be in [0, 1), got {self.warmup_fraction}"
            )
        if not 0.0 < self.min_lr_ratio <= 1.0:
            raise ValueError(
                f"min_lr_ratio must be in (0, 1], got {self.min_lr_ratio}"
            )


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
        tie_embeddings: bool = False,
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
    context window, the padding side and the vocabulary are all replaceable.
    Without one, ``fit`` builds a default over the training catalog with
    :attr:`DEFAULT_MAX_LENGTH` and right padding.

    Two properties of that batcher are load-bearing rather than advisory, so
    ``fit`` refuses a batcher that lacks them. ``max_length`` must be set,
    because it sizes the positional table and learned absolute positions need a
    bound. And ``pad_side`` must be ``"right"``, because that is what lets a
    causal mask stand in for a padding mask.

    A history of a single interaction is a usable training example here, unlike
    for :class:`SimpleRNNTrainer` — the `CLS` prefix supplies the context, so
    every position is a target rather than every position but the first.

    :attr:`history` records one entry per epoch, numbered from one as ELSA's is,
    carrying the mean loss and the number of positions it was averaged over.
    """

    #: Context window used when ``fit`` has to build its own batcher.
    DEFAULT_MAX_LENGTH = 200

    def __init__(
        self,
        config: SimpleGPTConfig | None = None,
        batcher: SequenceBatcher | None = None,
    ) -> None:
        self.cfg = config or SimpleGPTConfig()
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SimpleGPT | None = None
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

    def fit(self, sequences: ItemSequences) -> SimpleGPTTrainer:
        """Train on chronological histories, one example per position."""
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

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))

        if self.batcher is None:
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items),
                max_length=self.DEFAULT_MAX_LENGTH,
                pad_side="right",
            )
        self._check_batcher(self.batcher)
        self._n_items = self.batcher.tokenizer.n_items
        self.model = self._build_model()

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
        # The schedule is defined over the whole run, so it needs the step count
        # up front -- which is why this lives here rather than in the config.
        scheduler = self._build_scheduler(optimizer, len(starts) * self.cfg.epochs)
        # Two bars, as ELSA draws them: epochs outside, batches inside. The inner
        # bar is created once and rewound per epoch rather than a finished one
        # being left behind for each.
        epoch_iter = _progress(
            self.cfg.show_progress,
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SimpleGPT fit",
        )
        batch_bar = _progress_bar(
            self.cfg.show_progress, total=len(starts), desc="SimpleGPT epoch 1"
        )
        try:
            for epoch in epoch_iter:
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"SimpleGPT epoch {epoch}")
                loss_sum, positions = 0.0, 0
                for start in starts:
                    batch = sequences.select_rows(order[start : start + batch_size])
                    step = self._train_step(batch, optimizer, objective)
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
                mean_loss = loss_sum / positions if positions else float("nan")
                self.history.append(
                    {
                        "epoch": float(epoch),
                        "loss": mean_loss,
                        "positions": float(positions),
                        "lr": float(optimizer.param_groups[0]["lr"]),
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

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Linear warmup then cosine decay, or ``None`` for a flat rate.

        Warmup exists because the first steps of a transformer are the ones most
        able to wreck it: attention has learned nothing, so early gradients are
        large and poorly aimed. Cosine decay then spends the end of the run
        refining rather than bouncing. nanoGPT does both in its training script;
        neither is expressible through the optimizer alone, which is why they
        arrive together as one option rather than two.

        The schedule is measured in optimizer steps, not epochs, so the shape is
        the same whatever the batch size.
        """
        if self.cfg.lr_schedule == "constant":
            return None
        warmup = int(self.cfg.warmup_fraction * total_steps)
        floor = self.cfg.min_lr_ratio

        def factor(step: int) -> float:
            if step < warmup:
                # Step 0 would otherwise train at exactly zero and waste a step.
                return (step + 1) / (warmup + 1)
            if total_steps <= warmup:
                return 1.0
            progress = (step - warmup) / (total_steps - warmup)
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
            return floor + (1.0 - floor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)

    def _build_model(self) -> SimpleGPT:
        """The module this trainer's config and batcher describe.

        Shared by :meth:`fit` and :func:`load_simple_gpt` so a reloaded model is
        built by exactly the path that trained it.
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

    @staticmethod
    def _check_batcher(batcher: SequenceBatcher) -> None:
        """Refuse a batcher whose settings this architecture cannot honour."""
        if batcher.max_length is None:
            raise ValueError(
                "SimpleGPT needs a bounded context: max_length sizes the "
                "positional table, and learned absolute positions cannot be "
                "extended at prediction time. Set max_length on the batcher"
            )
        if batcher.pad_side != "right":
            raise ValueError(
                f"SimpleGPT needs pad_side='right', got {batcher.pad_side!r}. "
                "Right padding is what lets a causal mask exclude the padding on "
                "its own; with padding first, attention would read it"
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


def save_simple_gpt(path: str | Path, trainer: SimpleGPTTrainer) -> None:
    """Write weights, configuration and vocabulary to one file.

    The tokenizer travels with the weights because a served model without it
    cannot say what column 41 means. It is the only part that cannot be
    reconstructed from the config: the config describes the network, while the
    vocabulary describes the data it was fitted on.

    The batcher's ``max_length`` and ``pad_side`` are stored rather than the
    batcher, since those two values are all of it that matters and both are
    already enforced by :meth:`SimpleGPTTrainer.fit`.
    """
    if trainer.model is None or trainer.batcher is None:
        raise RuntimeError("SimpleGPTTrainer must be fitted before saving")
    config = asdict(trainer.cfg)
    # torch.device is picklable but a string keeps the file readable by
    # weights_only=True and by anything that is not torch.
    config["device"] = str(trainer.cfg.device)
    torch.save(
        {
            "format": 1,
            "model_state": trainer.model.state_dict(),
            "config": config,
            "tokenizer": trainer.batcher.tokenizer.to_dict(),
            "max_length": int(trainer.batcher.max_length),
            "pad_side": trainer.batcher.pad_side,
            "history": list(trainer.history),
        },
        Path(path),
    )


def load_simple_gpt(
    path: str | Path, *, device: str | torch.device | None = None
) -> SimpleGPTTrainer:
    """Read back a trainer saved by :func:`save_simple_gpt`, ready to predict.

    Self-contained: no config, tokenizer or batcher is needed at the call site,
    which is the point — those are what a checkpoint exists to carry.

    Read with ``weights_only=True``, so the file is parsed as data rather than
    executed as a pickle.
    """
    state = torch.load(
        Path(path), map_location=device or "cpu", weights_only=True
    )
    if state.get("format") != 1:
        raise ValueError(
            f"unsupported SimpleGPT checkpoint format {state.get('format')!r}"
        )

    config_state = dict(state["config"])
    transformer = TransformerConfig(**config_state.pop("transformer"))
    if device is not None:
        config_state["device"] = str(device)
    config = SimpleGPTConfig(transformer=transformer, **config_state)

    batcher = SequenceBatcher(
        ItemTokenizer.from_dict(state["tokenizer"]),
        max_length=int(state["max_length"]),
        pad_side=state["pad_side"],
    )
    trainer = SimpleGPTTrainer(config, batcher)
    trainer._n_items = batcher.tokenizer.n_items
    trainer.model = trainer._build_model()
    trainer.model.load_state_dict(state["model_state"])
    trainer.model.eval()
    trainer.history = list(state["history"])
    return trainer
