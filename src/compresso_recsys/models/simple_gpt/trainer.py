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
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.persistence import ModelCheckpointReader

from ..core.schedule import LRSchedule, build_scheduler, check_schedule
from ..base import BaseSequentialRecommender
from ..sequence_batching import SequenceBatcher
from ..tokenizer import ItemTokenizer
from compresso_recsys.models.simple_gpt.config import (
    SimpleGPTConfig,
    TransformerConfig,
)
from compresso_recsys.models.simple_gpt.model import (
    SimpleGPT,
)


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
    This trainer requires right padding, which lets the causal mask stand in for
    a padding mask. A left-padded batcher is rejected before the model is built.

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
        self._check_training_sequences(sequences)
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
        self._adopt_batcher_vocabulary(sequences, item_ids)
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
        with TrainingProgress(
            reporter,
            label="SimpleGPT",
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
                        # Advanced even when _train_step declined the batch, so
                        # the curve is exactly the configured shape over the run
                        # rather than a slightly truncated one whose floor
                        # depends on how many batches happened to carry targets.
                        scheduler.step()
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
                    "lr": last_training_lr,
                }
                self.history.append(record)
                progress.epoch_done(record)
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
        if batcher.padding != "right":
            raise ValueError(
                "SimpleGPT requires right padding: its causal attention mask "
                "does not mask leading padding"
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
