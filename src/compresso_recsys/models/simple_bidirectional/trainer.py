"""The training procedure, and the fitted model's prediction path."""

from __future__ import annotations

from __future__ import annotations

from typing import Any, Hashable, Literal, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
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
from ..core.validation import canonical_csr
from ..base import BaseSequentialRecommender
from ..sequence_batching import SequenceBatcher
from ..simple_gpt import LayerNorm, MLP, TransformerConfig
from ..tokenizer import ItemTokenizer
from compresso_recsys.models.simple_bidirectional.config import (
    SimpleBidirectionalTransformerConfig,
)
from compresso_recsys.models.simple_bidirectional.model import (
    SimpleBidirectionalTransformer,
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

    def _prepare_targets(self, targets: csr_matrix) -> csr_matrix:
        """Canonicalize an explicit target matrix before training.

        Binary membership by default, which is what this trainer's objective
        assumes: a target is either in the set or it is not. A subclass training
        on *graded* targets - how long an item was watched, how strongly it was
        rated - overrides this to keep the stored values, and pairs it with a
        loss that can consume them.

        This is the only place the grading is discarded. Everything downstream
        already carries values faithfully: ``InteractionBatchSampler`` packs
        ``matrix.data`` as it finds it and ``dense_training_target`` scatters it
        unchanged. Without this hook a subclass would have to reimplement the
        whole of :meth:`fit` to keep them.
        """
        return _binary_targets(targets)

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
        self._check_training_sequences(sequences)

        explicit_targets = targets is not None
        if targets is None:
            training_targets = _source_target_matrix(sequences)
        else:
            training_targets = self._prepare_targets(targets)
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
        self._adopt_batcher_vocabulary(sequences, item_ids)
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
        with TrainingProgress(
            reporter,
            label="SimpleBidirectionalTransformer",
            epochs=self.cfg.epochs,
            batches=len(starts),
        ) as progress:
            progress.start(
                f"{n_rows} sequences | {self._n_items} items | "
                f"{training_targets.nnz} target memberships ({target_mode}) | "
                f"{len(starts)} batches of {batch_size} | "
                f"{self.cfg.epochs} epochs | device {self.device}"
            )
            for epoch in progress.epochs():
                self.model.train()
                order = rng.permutation(n_rows)
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
                    progress.batch(step_index, loss_sum, target_rows)
                record = {
                    "epoch": float(epoch),
                    "loss": (
                        loss_sum / target_rows if target_rows else float("nan")
                    ),
                    "target_rows": float(target_rows),
                    "lr": last_training_lr,
                }
                self.history.append(record)
                progress.epoch_done(record)
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

    def _checkpoint_trainer_state(self) -> dict[str, Any]:
        """Add the padding side and the target mode ``fit`` ran in.

        Padding is a stored choice here rather than an architectural constant:
        unlike SASRec or SimpleGPT this model accepts either side, so a
        checkpoint has to say which one produced its weights.
        """
        state = super()._checkpoint_trainer_state()
        state["padding"] = self.batcher.padding
        state["trained_with_explicit_targets"] = (
            self._trained_with_explicit_targets
        )
        return state

    def _restore_checkpoint_trainer_state(
        self,
        state: Mapping[str, Any],
    ) -> None:
        super()._restore_checkpoint_trainer_state(state)
        explicit = state.get("trained_with_explicit_targets")
        if not isinstance(explicit, bool):
            raise ValueError(
                "SimpleBidirectionalTransformer target mode must be a bool"
            )
        self._trained_with_explicit_targets = explicit

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

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
