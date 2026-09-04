"""A small bidirectional Transformer for sequence-to-set recommendation.

Unlike :mod:`simple_gpt`, which learns one next-token target at every sequence
position, this model reads a complete history into a ``CLS`` representation and
scores one unordered set of catalog items per row.  That makes it the sequential
model that can consume the target matrix produced by temporal and asymmetric
interaction splits::

    ItemSequences
      -> SequenceBatcher.encode                  tokens, padding mask
      -> [CLS] + item and position embeddings
      -> N x bidirectional, padding-aware blocks
      -> final CLS state
      -> Linear(d_model, n_items)
      -> multinomial cross-entropy against a target set

``fit(..., targets=None)`` reconstructs the set of items in each source history.
Passing a CSR target matrix instead trains the mapping from the source history to
that explicit set.  The distinction is persisted because it controls prediction:
source items remain eligible after explicit-target training, where a source item
may legitimately also be a target.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Hashable, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
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
from compresso_recsys.persistence import ModelCheckpointReader, ModelCheckpointWriter
from compresso_recsys.sequences import ItemSequences

from ._schedule import LRSchedule, build_scheduler, check_schedule
from ._validation import canonical_csr
from .base import BaseSequentialRecommender
from .identifiers import ItemVocabulary
from .sequence_batching import SequenceBatcher
from .simple_gpt import LayerNorm, MLP, TransformerConfig
from .tokenizer import ItemTokenizer

__all__ = [
    "SimpleBidirectionalTransformer",
    "SimpleBidirectionalTransformerConfig",
    "SimpleBidirectionalTransformerTrainer",
]

OptimizerName = Literal["NAdam", "AdamW"]


@dataclass
class SimpleBidirectionalTransformerConfig:
    """Architecture and training settings for the bidirectional trainer."""

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
    log_prefix: str = "SimpleBidirectionalTransformer"
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


class BidirectionalSelfAttention(nn.Module):
    """Multi-head self-attention whose real positions may read one another."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.dropout = config.dropout
        self.attn = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Attend to every non-padding key in the row.

        ``mask`` is true for ``CLS`` and real history positions.  Padding
        queries may compute disposable states, but padding is never visible as a
        key to a real position at any layer.
        """
        if x.ndim != 3:
            raise ValueError(
                f"x must be (rows, length, dim), got {tuple(x.shape)}"
            )
        if mask.shape != x.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match x rows and length "
                f"{tuple(x.shape[:2])}"
            )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have boolean dtype")

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
            attn_mask=mask[:, None, None, :],
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        attended = (
            attended.transpose(1, 2).contiguous().view(rows, length, self.d_model)
        )
        return self.resid_dropout(self.proj(attended))


class BidirectionalBlock(nn.Module):
    """Pre-normalized bidirectional attention followed by an MLP."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(config.d_model, bias=config.bias)
        self.attn = BidirectionalSelfAttention(config)
        self.ln_2 = LayerNorm(config.d_model, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class SimpleBidirectionalTransformer(nn.Module):
    """Item embeddings, bidirectional blocks, and a catalog-scoring head."""

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
                f"n_items ({n_items}) cannot exceed vocab_size ({vocab_size})"
            )
        self.config = config
        self.tie_embeddings = bool(tie_embeddings)
        self.item_offset = int(vocab_size) - int(n_items)
        self.max_positions = int(max_positions)
        self.pad_id = int(pad_id)

        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=pad_id)
        self.position = nn.Embedding(max_positions, config.d_model)
        self.cls_token = nn.Parameter(torch.empty(1, 1, config.d_model))
        self.embed_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            BidirectionalBlock(config) for _ in range(config.n_layers)
        )
        self.ln_f = LayerNorm(config.d_model, bias=config.bias)
        self.head_dropout = nn.Dropout(config.dropout)
        if self.tie_embeddings:
            self.head = None
            self.head_bias = nn.Parameter(torch.zeros(n_items))
        else:
            self.head = nn.Linear(config.d_model, n_items)
            self.head_bias = None

        self.apply(self._init_weights)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        residual_std = 0.02 / math.sqrt(2 * config.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=residual_std)
            nn.init.normal_(block.mlp.down.weight, mean=0.0, std=residual_std)
        with torch.no_grad():
            self.embedding.weight[self.pad_id].zero_()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return states for ``CLS`` and each token."""
        if tokens.ndim != 2:
            raise ValueError(
                f"tokens must be (rows, length), got {tuple(tokens.shape)}"
            )
        if mask.shape != tokens.shape:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match tokens shape "
                f"{tuple(tokens.shape)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have boolean dtype")
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
        full_mask = torch.cat(
            [torch.ones((rows, 1), dtype=torch.bool, device=mask.device), mask],
            dim=1,
        )
        for block in self.blocks:
            hidden = block(hidden, full_mask)
        return self.ln_f(hidden)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Turn one or more hidden states into catalog logits."""
        hidden = self.head_dropout(states)
        if self.head is not None:
            return self.head(hidden)
        return F.linear(
            hidden,
            self.embedding.weight[self.item_offset :],
            self.head_bias,
        )


def _source_target_matrix(sequences: ItemSequences) -> csr_matrix:
    """Binary source membership, preserving every input row."""
    counts = sequences.row_lengths
    rows = np.repeat(np.arange(sequences.n_rows, dtype=np.int64), counts)
    matrix = csr_matrix(
        (
            np.ones(sequences.values.size, dtype=np.float32),
            (rows, np.asarray(sequences.values, dtype=np.int64)),
        ),
        shape=(sequences.n_rows, sequences.n_items),
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.data.fill(1.0)
    matrix.sort_indices()
    return matrix


def _binary_targets(targets: csr_matrix) -> csr_matrix:
    """Canonical binary membership without mutating caller-owned storage."""
    targets = canonical_csr(targets, name="targets")
    if targets.data.size and not np.all(targets.data == 1):
        targets = targets.copy()
        targets.data.fill(1)
    return targets


class SimpleBidirectionalTransformerTrainer(BaseSequentialRecommender):
    """Train a bidirectional sequence encoder against unordered item sets."""

    DEFAULT_MAX_LENGTH = 200
    checkpoint_type = "simple_bidirectional_transformer_trainer"

    def __init__(
        self,
        config: SimpleBidirectionalTransformerConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config or SimpleBidirectionalTransformerConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: SimpleBidirectionalTransformer | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None
        self._trained_with_explicit_targets = False

    @property
    def is_fitted(self) -> bool:
        return self.model is not None

    @property
    def n_items(self) -> int | None:
        return self._n_items

    @property
    def trained_with_explicit_targets(self) -> bool:
        """Whether the most recent fit used a separate target matrix."""
        return self._trained_with_explicit_targets

    def _reporter(self, logger: Any, show_progress: Any) -> _Reporter:
        return _resolve_reporter(
            default_logger=self.logger,
            logger=logger,
            default_show_progress=self.cfg.show_progress,
            show_progress=show_progress,
            prefix=self.cfg.log_prefix,
            log_every_n_steps=self.cfg.log_every_n_steps,
        )

    def fit(
        self,
        sequences: ItemSequences,
        *,
        targets: csr_matrix | None = None,
        item_ids: Sequence[Hashable] | np.ndarray | None = None,
        logger: Any | None = _INHERIT,
        show_progress: bool | None | _Inherit = _INHERIT,
    ) -> SimpleBidirectionalTransformerTrainer:
        """Train on source histories and optional explicit target sets."""
        reporter = self._reporter(logger, show_progress)
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "SimpleBidirectionalTransformerTrainer trains on ItemSequences, "
                f"got {type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")

        explicit_targets = targets is not None
        if targets is None:
            training_targets = _source_target_matrix(sequences)
        else:
            training_targets = _binary_targets(targets)
            expected_shape = (sequences.n_rows, sequences.n_items)
            if training_targets.shape != expected_shape:
                raise ValueError(
                    f"targets shape {training_targets.shape} must match sequences "
                    f"shape {expected_shape}"
                )
        if training_targets.nnz == 0:
            raise ValueError("training targets contain no positive items")

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
                raise ValueError("item_ids must match the batcher tokenizer item IDs")
        self._set_item_ids(
            tokenizer_ids if item_ids is None else item_ids,
            n_items=sequences.n_items,
        )
        self._check_batcher(self.batcher)

        torch.manual_seed(int(self.cfg.seed))
        rng = np.random.default_rng(int(self.cfg.seed))
        self._n_items = self.batcher.tokenizer.n_items
        self._trained_with_explicit_targets = explicit_targets
        self.model = self._build_model()
        self._build_checkpoint_optimizer()
        assert self.optimizer is not None
        optimizer = self.optimizer
        self.history = []

        n_rows = sequences.n_rows
        batch_size = self.cfg.batch_size
        starts = range(0, n_rows, batch_size)
        scheduler = self._build_scheduler(optimizer, len(starts) * self.cfg.epochs)
        target_mode = "explicit" if explicit_targets else "source reconstruction"
        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{n_rows} sequences | {self._n_items} items | "
            f"{training_targets.nnz} target memberships ({target_mode}) | "
            f"{len(starts)} batches of {batch_size} | {self.cfg.epochs} epochs | "
            f"device {self.device}"
        )
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="SimpleBidirectionalTransformer fit",
        )
        batch_bar = reporter.bar(
            total=len(starts), desc="SimpleBidirectionalTransformer epoch 1"
        )
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(n_rows)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(
                        f"SimpleBidirectionalTransformer epoch {epoch}"
                    )
                loss_sum, target_rows = 0.0, 0
                last_training_lr = float(optimizer.param_groups[0]["lr"])
                for step_index, start in enumerate(starts, start=1):
                    selected = order[start : start + batch_size]
                    batch_lr = float(optimizer.param_groups[0]["lr"])
                    step = self._train_step(
                        sequences.select_rows(selected),
                        training_targets[selected],
                    )
                    if step is not None:
                        last_training_lr = batch_lr
                    if scheduler is not None:
                        scheduler.step()
                    if step is not None:
                        batch_loss, batch_target_rows = step
                        loss_sum += batch_loss * batch_target_rows
                        target_rows += batch_target_rows
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
                                    loss_sum / target_rows
                                    if target_rows
                                    else float("nan")
                                )
                            },
                        )
                mean_loss = loss_sum / target_rows if target_rows else float("nan")
                record = {
                    "epoch": float(epoch),
                    "loss": mean_loss,
                    "target_rows": float(target_rows),
                    "lr": last_training_lr,
                }
                self.history.append(record)
                reporter.epoch(
                    f"epoch {epoch}/{self.cfg.epochs}", record, epoch_started
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
        return build_scheduler(
            optimizer,
            schedule=self.cfg.lr_schedule,
            total_steps=total_steps,
            warmup_fraction=self.cfg.warmup_fraction,
            min_lr_ratio=self.cfg.min_lr_ratio,
        )

    def _build_model(self) -> SimpleBidirectionalTransformer:
        if self.batcher is None:
            raise RuntimeError("trainer batcher is unavailable")
        tokenizer = self.batcher.tokenizer
        return SimpleBidirectionalTransformer(
            vocab_size=tokenizer.vocab_size,
            n_items=tokenizer.n_items,
            max_positions=int(self.batcher.max_length) + 1,
            pad_id=tokenizer.pad_id,
            config=self.cfg.transformer,
            tie_embeddings=self.cfg.tie_embeddings,
        ).to(self.device)

    @staticmethod
    def _check_batcher(batcher: SequenceBatcher) -> None:
        if batcher.max_length is None:
            raise ValueError(
                "SimpleBidirectionalTransformer needs a bounded context; set "
                "max_length on the batcher"
            )

    def _train_step(
        self,
        batch: ItemSequences,
        targets: csr_matrix,
    ) -> tuple[float, int] | None:
        """Optimize one batch, skipping rows whose target sets are empty."""
        assert self.model is not None
        assert self.batcher is not None
        assert self.optimizer is not None
        tokens, mask = self.batcher.encode(batch, device=self.device)
        inputs = self._with_unk_dropout(tokens, mask)
        states = self.model(inputs, mask)
        logits = self.model.score(states[:, 0])

        dense = targets.astype(np.float32, copy=False).toarray()
        target = torch.from_numpy(dense).to(self.device)
        counts = target.sum(dim=1)
        active = counts > 0
        n_active = int(active.sum())
        if n_active == 0:
            return None
        target_distribution = target[active] / counts[active, None]
        loss = -(
            target_distribution * F.log_softmax(logits[active], dim=1)
        ).sum(dim=1).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach()), n_active

    def _with_unk_dropout(
        self, inputs: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        assert self.batcher is not None
        unk_id = getattr(self.batcher.tokenizer, "unk_id", None)
        if unk_id is None or self.cfg.unk_dropout <= 0.0:
            return inputs
        chosen = (
            torch.rand(inputs.shape, device=inputs.device) < self.cfg.unk_dropout
        ) & mask
        return torch.where(chosen, torch.full_like(inputs, unk_id), inputs)

    def _effective_exclude_seen(self, exclude_seen: bool) -> bool:
        return exclude_seen and not self._trained_with_explicit_targets

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Rank catalog items from the bidirectional ``CLS`` representation."""
        if self.model is None or self.batcher is None or self._n_items is None:
            raise RuntimeError(
                "SimpleBidirectionalTransformerTrainer must be fitted before "
                "predicting"
            )
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "SimpleBidirectionalTransformerTrainer predicts from "
                f"ItemSequences, got {type(source).__name__}"
            )
        n_items = self._n_items
        candidate_rows = self._candidate_rows(candidate_ids)
        candidate_count = int(candidate_rows.size)
        if not 1 <= k <= candidate_count:
            raise ValueError(f"k must be in [1, {candidate_count}], got {k}")

        exclude_seen = self._effective_exclude_seen(exclude_seen)
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
            states = self.model(tokens, mask)
            logits = self.model.score(states[:, 0])
            if exclude_seen:
                self._mask_seen(logits, source)
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            vals, local_cols = torch.topk(logits[:, candidates], k, dim=1)
            cols = candidates[local_cols]
        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))

    @staticmethod
    def _mask_seen(logits: torch.Tensor, source: ItemSequences) -> None:
        if source.values.size == 0:
            return
        n_items = int(logits.shape[1])
        rows = np.repeat(np.arange(source.n_rows), source.row_lengths)
        cols = np.asarray(source.values, dtype=np.int64)
        scoreable = cols < n_items
        rows, cols = rows[scoreable], cols[scoreable]
        if cols.size == 0:
            return
        logits[
            torch.as_tensor(rows, dtype=torch.long, device=logits.device),
            torch.as_tensor(cols, dtype=torch.long, device=logits.device),
        ] = -torch.inf

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> SimpleBidirectionalTransformerTrainer:
        config = dict(config)
        transformer = TransformerConfig(**dict(config.pop("transformer")))
        config["device"] = str(device)
        state = reader.read_json("state/trainer.json")
        max_length = state.get("max_length")
        if (
            isinstance(max_length, bool)
            or not isinstance(max_length, int)
            or max_length < 1
        ):
            raise ValueError(
                "SimpleBidirectionalTransformer max_length must be a positive integer"
            )
        tokenizer_state = reader.read_json("state/tokenizer.json")
        if reader.exists("state/tokenizer_item_ids.json"):
            tokenizer_state["item_ids"] = reader.read_item_ids(
                "state/tokenizer_item_ids.json"
            )
        tokenizer = ItemTokenizer.from_dict(tokenizer_state)
        trainer = cls(
            SimpleBidirectionalTransformerConfig(
                transformer=transformer, **config
            ),
            SequenceBatcher(
                tokenizer,
                max_length=max_length,
                padding=state.get("padding", "right"),
            ),
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
                "SimpleBidirectionalTransformerTrainer checkpoints support "
                "ItemTokenizer only"
            )
        assert self.batcher.max_length is not None
        writer.write_json(
            "state/trainer.json",
            {
                "max_length": int(self.batcher.max_length),
                "padding": self.batcher.padding,
                "history": self.history,
                "trained_with_explicit_targets": self._trained_with_explicit_targets,
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
            raise ValueError(
                "SimpleBidirectionalTransformer training history must be a list"
            )
        explicit = state.get("trained_with_explicit_targets")
        if not isinstance(explicit, bool):
            raise ValueError(
                "SimpleBidirectionalTransformer target mode must be a bool"
            )
        self.history = list(history)
        self._trained_with_explicit_targets = explicit

    def _build_checkpoint_optimizer(self) -> None:
        if self.model is None:
            raise RuntimeError(
                "SimpleBidirectionalTransformer model must be built before its "
                "optimizer"
            )
        self.optimizer = getattr(torch.optim, self.cfg.optimizer)(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
