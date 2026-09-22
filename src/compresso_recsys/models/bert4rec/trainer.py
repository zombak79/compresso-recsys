"""Cloze training and next-item prediction for BERT4Rec.

Sun et al., *BERT4Rec: Sequential Recommendation with Bidirectional Encoder
Representations from Transformer*, CIKM 2019. The model in :mod:`.model` is the
paper's architecture; this module is §3.6, "Model Learning", and the batching
the paper leaves to its implementation.

The shape of a fit::

    ItemSequences
      -> sliding windows over histories longer than N       (reference)
      -> duplication_factor independent maskings per window (§3.6)
      -> + one last-item-masked sample per window           (§3.6)
      -> tokenize, right-pad to the batch width
      -> Bert4Rec -> catalog logits at the masked positions
      -> cross entropy against the true items               (Eq. 8)

Prediction appends ``[mask]`` to the history and ranks the catalog from that
position's output, which is the paper's remedy for the train/test mismatch the
Cloze objective creates.

Where the paper does not specify something, this follows the authors' released
TensorFlow implementation (https://github.com/FeiSun/BERT4Rec) and says so at
the point of the decision.
"""

from __future__ import annotations

import time
from collections.abc import Hashable, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from compresso import SRPTensor
from torch import nn
from torch.nn import functional as F

from compresso_recsys._reporting import (
    _INHERIT,
    _format_duration,
    _Inherit,
    _Reporter,
    _resolve_reporter,
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

from .config import Bert4RecConfig
from .model import Bert4Rec

__all__ = ["Bert4RecTrainer"]

#: ``pad`` fills the batch rectangle and must never be an item; ``mask`` is the
#: Cloze token the model predicts at; ``unk`` represents an item outside the
#: fitted catalog, which a later split stage can produce. The reference reserves
#: id 0 for padding and gives ``[MASK]`` its own entry for the same reasons.
SPECIAL_TOKENS = {"pad": 0, "mask": 1, "unk": 2}


def _sliding_windows(history: np.ndarray, width: int, step: int) -> list[np.ndarray]:
    """Split one history into overlapping windows of at most ``width``.

    A history no longer than the window is one sample. A longer one is cut into
    windows ending at the most recent interaction and walking backwards by
    ``step``, with the oldest window pinned to the start so no interaction is
    dropped. That is ``create_training_instances`` in the reference generator:
    the windows are produced back-to-front and then reversed, so the returned
    list is in chronological order.

    Without this, everything before the last ``N`` interactions of a long
    history would never appear in training at all.
    """
    length = int(history.size)
    if length <= width:
        return [history]
    starts = list(range(length - width, 0, -step))
    starts.append(0)
    return [history[start : start + width] for start in reversed(starts)]


class Bert4RecTrainer(BaseSequentialRecommender):
    """Train :class:`Bert4Rec` with the Cloze objective and rank from ``[mask]``.

    ``fit`` builds the vocabulary, the batcher and the model unless a batcher was
    supplied for its vocabulary, then runs ``cfg.epochs`` passes over the Cloze
    samples. ``predict_on_batch`` appends ``[mask]`` and ranks the catalog from
    the final hidden state at that position.

    The trainer owns tokenization because the checkpointed histories deliberately
    do not: :class:`~compresso_recsys.ItemSequences` holds catalog indices with
    no padding and no special tokens, so ``[mask]``, ``[pad]`` and the right
    padding that makes a dense batch are all decided here.
    """

    checkpoint_type = "bert4rec_trainer"

    def __init__(
        self,
        config: Bert4RecConfig | None = None,
        batcher: SequenceBatcher | None = None,
        logger: Any | None = None,
    ) -> None:
        self.cfg = config or Bert4RecConfig()
        self.logger = logger
        self.device = torch.device(self.cfg.device)
        self.history: list[dict[str, float]] = []
        self.model: Bert4Rec | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.batcher = batcher
        self._owns_batcher = batcher is None
        self._n_items: int | None = None
        # Set by fit and by _from_checkpoint_config, both of which know the
        # window. Declared here so the attribute exists on an unfitted trainer
        # rather than being absent until one of them runs.
        self._predict_batcher: SequenceBatcher | None = None

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
    ) -> Bert4RecTrainer:
        """Train on chronological histories with the Cloze objective.

        Validates the input, builds a default batcher when none was supplied,
        reconciles its window with the config, records the item IDs, seeds Torch
        and NumPy from ``cfg.seed``, builds the model and optimizer, then runs
        ``cfg.epochs`` passes over freshly masked samples.

        Masking is redrawn every epoch rather than fixed once. The reference
        materializes ``dupe_factor`` maskings of every sequence to a file before
        training and then reads that file each epoch; drawing them per epoch
        gives the same distribution over a run without holding the expansion in
        memory, and is strictly more varied for the same number of steps.

        Rebuilds the model on every call: early stopping and incremental
        training are not part of this contract.
        """
        reporter = self._reporter(logger, show_progress)
        if not isinstance(sequences, ItemSequences):
            raise TypeError(
                "Bert4RecTrainer trains on ItemSequences, got "
                f"{type(sequences).__name__}"
            )
        if sequences.n_rows == 0:
            raise ValueError("cannot train on zero sequences")
        if sequences.n_items < 1:
            raise ValueError("cannot train on an empty catalog")

        if self._owns_batcher:
            self.batcher = SequenceBatcher(
                ItemTokenizer(sequences.n_items, special_tokens=SPECIAL_TOKENS),
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
        self._check_tokenizer(self.batcher.tokenizer)
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
        # A batcher handed over for its vocabulary usually states no window; it
        # inherits the config's, which is the number that sized the positional
        # table. Stating both and disagreeing is an error rather than a silent
        # win for either.
        if self.batcher.max_length is None:
            self.batcher = replace(
                self.batcher, max_length=self.cfg.max_history_length
            )
        elif int(self.batcher.max_length) != self.cfg.max_history_length:
            raise ValueError(
                f"batcher max_length ({self.batcher.max_length}) and config "
                f"max_history_length ({self.cfg.max_history_length}) disagree; "
                "the window sizes the positional table, so it cannot be two "
                "numbers"
            )
        # Right padding is the reference's, which zero-fills to the right of
        # each sequence. It also keeps position 0 meaning "oldest retained
        # interaction" for every row, which is what a learned positional table
        # reads under the Cloze objective -- unlike SASRec, nothing here gathers
        # a final state whose column depends on the row's length.
        if self.batcher.padding != "right":
            self.batcher = replace(self.batcher, padding="right")

        # Prediction appends [mask] to the history, so the encoder must have a
        # position free for it. Training encodes at the full window; prediction
        # truncates one shorter and then appends.
        self._predict_batcher = replace(
            self.batcher, max_length=self.cfg.max_history_length - 1
        )

        torch.manual_seed(int(self.cfg.seed))
        # One generator for window choice, masking and row order alike, so a run
        # is reproducible from cfg.seed alone.
        rng = np.random.default_rng(int(self.cfg.seed))
        self._n_items = sequences.n_items
        self.model = self._build_model()
        self._build_checkpoint_optimizer()
        assert self.optimizer is not None
        optimizer = self.optimizer
        self.history = []

        samples = self._training_windows(sequences)
        if not samples:
            raise ValueError(
                "no history has an item to mask, so there is no Cloze example "
                "to learn from"
            )

        n_samples = len(samples)
        batch_size = self.cfg.batch_size
        starts = range(0, n_samples, batch_size)
        scheduler = self._build_scheduler(optimizer, len(starts) * self.cfg.epochs)

        fit_started = time.monotonic()
        reporter.log(
            "fit started: "
            f"{sequences.n_rows} sequences -> {n_samples} Cloze samples | "
            f"{self._n_items} items | rho={self.cfg.mask_proportion} | "
            f"{len(starts)} batches of {batch_size} | {self.cfg.epochs} epochs | "
            f"device {self.device}"
        )
        epoch_iter = reporter.wrap(
            range(1, self.cfg.epochs + 1),
            total=self.cfg.epochs,
            desc="Bert4Rec fit",
        )
        batch_bar = reporter.bar(total=len(starts), desc="Bert4Rec epoch 1")
        try:
            for epoch in epoch_iter:
                epoch_started = time.monotonic()
                self.model.train()
                order = rng.permutation(n_samples)
                if batch_bar is not None:
                    batch_bar.reset(total=len(starts))
                    batch_bar.set_description(f"Bert4Rec epoch {epoch}")
                loss_sum, predicted = 0.0, 0
                last_training_lr = float(optimizer.param_groups[0]["lr"])
                for step_index, start in enumerate(starts, start=1):
                    selected = order[start : start + batch_size]
                    batch_lr = float(optimizer.param_groups[0]["lr"])
                    step = self._train_step(
                        [samples[int(index)] for index in selected], rng
                    )
                    if scheduler is not None:
                        scheduler.step()
                    if step is not None:
                        last_training_lr = batch_lr
                        batch_loss, batch_predicted = step
                        loss_sum += batch_loss * batch_predicted
                        predicted += batch_predicted
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
                                    loss_sum / predicted
                                    if predicted
                                    else float("nan")
                                )
                            },
                        )
                mean_loss = loss_sum / predicted if predicted else float("nan")
                record = {
                    "epoch": float(epoch),
                    "loss": mean_loss,
                    "masked_items": float(predicted),
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
            f"fit finished: {_format_duration(time.monotonic() - fit_started)} "
            f"total | {len(self.history)} epochs recorded"
        )
        return self

    # -- Cloze sample construction ------------------------------------------

    def _training_windows(
        self, sequences: ItemSequences
    ) -> list[tuple[np.ndarray, bool]]:
        """Every window to train on, paired with whether to mask only its last.

        Three things happen here, all from §3.6 and the reference generator.
        A history longer than the window becomes several overlapping windows, so
        its early interactions are trained on. Each window is repeated
        ``duplication_factor`` times, which is how one history yields the many
        samples the Cloze objective is credited with -- the masking itself is
        drawn per step, so the repeats differ. And each history contributes one
        last-item-masked sample, the paper's fine-tuning for the actual
        prediction task -- one per window, as ``mask_last`` does.

        Empty histories are dropped: there is nothing to mask in one.
        """
        width = self.cfg.max_history_length
        step = max(1, int(self.cfg.sliding_window_step * width))
        samples: list[tuple[np.ndarray, bool]] = []
        for row in range(sequences.n_rows):
            history = np.asarray(sequences.row(row), dtype=np.int64)
            if history.size == 0:
                continue
            windows = _sliding_windows(history, width, step)
            for _ in range(self.cfg.duplication_factor):
                samples.extend((window, False) for window in windows)
            if self.cfg.last_item_samples:
                # §3.6's fine-tuning sample: a sequence with only the final
                # position masked, "to better match the sequential
                # recommendation task".
                #
                # One per window, not one per user: create_training_instances
                # calls mask_last on the training branch too, and mask_last
                # iterates the user's whole document, which is the list of
                # windows. A history that fits the window is a single window, so
                # this differs from one-per-user only for the long histories the
                # sliding window exists to serve.
                #
                # The masked position is the last of the window, so the context
                # it leaves is ``width - 1`` interactions -- exactly what
                # predict_on_batch encodes before appending [mask]. A wider
                # context would train "recover the interaction you were shown",
                # one step behind the question prediction asks. SASRec pays for
                # the same shift by encoding one interaction wider than its
                # window.
                #
                # Masking the last item of a window with only one position
                # leaves no context at all, so the model would be asked to
                # predict an item from nothing. The reference asserts the
                # position is non-zero rather than handling it; dropping the
                # sample is the same requirement without the crash.
                samples.extend(
                    (window, True) for window in windows if window.size >= 2
                )
        return samples

    def _cloze_batch(
        self,
        windows: list[tuple[np.ndarray, bool]],
        rng: np.random.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask, tokenize and right-pad one batch of windows.

        Returns ``(tokens, padding_mask, masked, labels)``. ``masked`` marks the
        positions the loss is taken at and ``labels`` holds the token id of the
        true item at each of them, in row-major order over ``masked`` -- the
        order :meth:`Bert4Rec.forward` gathers them in when it is handed the
        same index.

        The index has to be passed rather than recovered from the tokens: below
        a ``mask_token_probability`` of 1.0 a chosen position may keep its item
        or take a random one, and the loss covers those as much as the masked
        ones, so ``tokens == mask`` finds fewer positions than there are labels.

        Positions are chosen per §3.6: ``round(len * rho)`` of them, at least
        one, capped at ``max_predictions``, sampled without replacement. The
        reference is Python 2, where ``round`` goes half away from zero rather
        than to even, but no published rho puts ``len * rho`` on a half-integer,
        so the two agree everywhere it has been run. A
        chosen position becomes ``[mask]`` with probability
        ``mask_token_probability``; the reference's default of 1.0 means it
        always does, and the remainder splits evenly between keeping the item
        and drawing a random one.
        """
        assert self.batcher is not None
        tokenizer = self.batcher.tokenizer
        mask_id = tokenizer.token_id("mask")
        n_reserved = tokenizer.n_reserved
        n_items = tokenizer.n_items

        width = max(window.size for window, _ in windows)
        rows = len(windows)
        tokens = np.full((rows, width), tokenizer.pad_id, dtype=np.int64)
        padding = np.zeros((rows, width), dtype=bool)
        masked = np.zeros((rows, width), dtype=bool)
        labels: list[np.ndarray] = []

        for row, (window, force_last) in enumerate(windows):
            length = int(window.size)
            encoded = tokenizer.encode_indices(window)
            tokens[row, :length] = encoded
            padding[row, :length] = True

            if force_last:
                chosen = np.array([length - 1], dtype=np.int64)
            else:
                count = min(
                    self.cfg.max_predictions,
                    max(1, round(length * self.cfg.mask_proportion)),
                )
                chosen = rng.choice(length, size=count, replace=False)
                chosen.sort()

            masked[row, chosen] = True
            labels.append(encoded[chosen])

            if force_last or self.cfg.mask_token_probability >= 1.0:
                tokens[row, chosen] = mask_id
                continue
            draws = rng.random(chosen.size)
            replacement = np.where(
                draws < self.cfg.mask_token_probability,
                mask_id,
                # Below the threshold the reference splits the remainder in
                # half: keep the original item, or substitute a random one.
                np.where(
                    rng.random(chosen.size) < 0.5,
                    encoded[chosen],
                    rng.integers(n_reserved, n_reserved + n_items, chosen.size),
                ),
            )
            tokens[row, chosen] = replacement

        return (
            torch.from_numpy(tokens).to(self.device),
            torch.from_numpy(padding).to(self.device),
            torch.from_numpy(masked).to(self.device),
            torch.from_numpy(np.concatenate(labels)).to(self.device),
        )

    def _train_step(
        self,
        windows: list[tuple[np.ndarray, bool]],
        rng: np.random.Generator,
    ) -> tuple[float, int] | None:
        """Optimize one batch against Eq. 8.

        Eq. 8 is the negative log-likelihood of the softmax in Eq. 7, which is
        what ``cross_entropy`` computes from logits -- by the log-sum-exp
        identity rather than by forming the probability and taking its log. The
        two are the same function of the same inputs; only the second one
        underflows. Where a probability reaches zero in float32 its log is
        infinite, and clamping it makes the loss locally constant, so the
        gradient at exactly the positions the model is most wrong about is zero.
        """
        assert self.model is not None
        assert self.optimizer is not None
        if not windows:
            return None

        tokens, padding, masked, labels = self._cloze_batch(windows, rng)
        if labels.numel() == 0:
            return None

        logits = self.model(tokens, padding, masked)
        loss = F.cross_entropy(logits, labels)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # "The gradient is clipped when its l2 norm exceeds a threshold of 5."
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.max_grad_norm
        )
        self.optimizer.step()
        return float(loss.detach()), int(labels.numel())

    # -- prediction ---------------------------------------------------------

    def predict_on_batch(
        self,
        source: ItemSequences,
        *,
        k: int,
        exclude_seen: bool = True,
        candidate_ids: Sequence[Hashable] | np.ndarray | None = None,
    ) -> SRPTensor:
        """Rank the catalog from a ``[mask]`` appended to each history.

        §3.6, "Test": the Cloze objective predicts a masked position, while the
        task is to predict the future, so the token is appended to the end of
        the history and the recommendation is read from its hidden state.

        The ranking itself is this package's and not the paper's. §4.2 ranks
        the held-out item against 100 popularity-sampled negatives; this ranks
        the whole catalog and, by default, strikes out what the user has
        already seen. Full-catalog ranking is the harder and more honest
        measurement, but it is a different one, so HR@k and NDCG@k from here
        are not comparable to the paper's tables.

        Every row gets exactly one masked position, and rows are padded to a
        common width, so the model's masked-position gather returns one row of
        scores per source row in order.
        """
        if self.model is None or self._n_items is None:
            raise RuntimeError("Bert4RecTrainer must be fitted before predicting")
        if not isinstance(source, ItemSequences):
            raise TypeError(
                "Bert4RecTrainer predicts from ItemSequences, got "
                f"{type(source).__name__}"
            )
        if self._predict_batcher is None:  # pragma: no cover - defensive
            raise RuntimeError("trainer batcher is unavailable")

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
            tokens, padding = self._encode_with_mask(source)
            logits = self.model(tokens, padding)
            # The model scores the whole vocabulary; the catalog is the part
            # above the reserved ids, and nothing downstream of a model sees a
            # token id. These are logits rather than probabilities, which ranks
            # identically -- the softmax is monotonic -- and matches what every
            # other model in the package returns as a score.
            scores = logits[:, self._item_offset :]
            if exclude_seen:
                self._mask_seen(scores, source)
            candidates = torch.from_numpy(candidate_rows).long().to(self.device)
            vals, local_cols = torch.topk(scores[:, candidates], k, dim=1)
            cols = candidates[local_cols]

        return SRPTensor(cols=cols, vals=vals, shape=(rows, n_items))

    def _encode_with_mask(
        self, source: ItemSequences
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode histories with ``[mask]`` appended to each.

        The batcher truncates to one short of the window, so appending keeps
        every row inside the positional table. The appended token is a real
        position rather than padding -- it is the one the model reads -- so it
        is marked in the padding mask.
        """
        assert self.model is not None
        batcher = self._predict_batcher
        tokens, padding = batcher.encode(source, device=self.device)
        lengths = padding.sum(dim=1)
        rows = tokens.shape[0]
        column = torch.arange(rows, device=tokens.device)

        pad_id = batcher.tokenizer.pad_id
        widened = torch.full(
            (rows, int(tokens.shape[1]) + 1),
            pad_id,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        widened[:, : tokens.shape[1]] = tokens
        widened[column, lengths] = self.model.mask

        widened_padding = torch.zeros(
            widened.shape, dtype=torch.bool, device=tokens.device
        )
        widened_padding[:, : padding.shape[1]] = padding
        widened_padding[column, lengths] = True
        return widened, widened_padding

    @property
    def _item_offset(self) -> int:
        assert self.batcher is not None
        return int(self.batcher.tokenizer.n_reserved)

    def _mask_seen(self, scores: torch.Tensor, source: ItemSequences) -> None:
        """Forbid every item in the *full* history, truncated part included.

        Scores are indexed by catalog position, and a history may span a wider
        catalog than this model was fitted on -- a later split stage does
        exactly that. Items beyond the fitted catalog are dropped from the mask
        rather than clipped: they were never scoreable, so there is nothing to
        forbid.
        """
        if source.values.size == 0:
            return
        n_items = int(scores.shape[1])
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

    # -- construction -------------------------------------------------------

    @staticmethod
    def _check_tokenizer(tokenizer: Any) -> None:
        """Require the vocabulary to carry a ``mask`` token.

        The Cloze objective is not expressible without one: masking with an item
        id would teach the model that that item means "predict here".
        """
        token_id = getattr(tokenizer, "token_id", None)
        if token_id is None:
            raise TypeError(
                "Bert4Rec needs a tokenizer with named special tokens, got "
                f"{type(tokenizer).__name__}"
            )
        try:
            token_id("mask")
        except KeyError:
            raise ValueError(
                "Bert4Rec's Cloze objective needs a 'mask' special token; "
                "build the tokenizer with "
                "ItemTokenizer(n_items, special_tokens="
                f"{SPECIAL_TOKENS})"
            ) from None

    def _build_model(self) -> Bert4Rec:
        if self.batcher is None:
            raise RuntimeError("trainer batcher is unavailable")
        tokenizer = self.batcher.tokenizer
        model = Bert4Rec(
            n_items=tokenizer.n_items,
            max_history_length=self.cfg.max_history_length,
            d_model=self.cfg.d_model,
            n_blocks=self.cfg.n_blocks,
            n_heads=self.cfg.n_heads,
            dropout=self.cfg.dropout,
            att_dropout=self.cfg.att_dropout,
            pad=tokenizer.pad_id,
            mask=tokenizer.token_id("mask"),
            n_reserved=tokenizer.n_reserved,
        ).to(self.device)
        self._init_weights(model)
        return model

    def _init_weights(self, model: Bert4Rec) -> None:
        """Truncated normal with standard deviation ``initializer_range``.

        §4.3 reads "initialized using truncated normal distribution in the range
        [-0.02, 0.02]", but the reference's ``create_initializer`` passes 0.02 as
        ``stddev`` to ``tf.truncated_normal_initializer``, which truncates at two
        standard deviations:
        https://github.com/FeiSun/BERT4Rec/blob/b5a1c2eddbe5c2cb4ae6c7a7845b3e6f251f90ba/modeling.py#L371-L373

        So 0.02 is the deviation and 0.04 the bound, not the other way round;
        reading it as the bound shrinks every weight by half and leaves the tied
        output head dotting a normalized state against embeddings too small to
        separate any two items.

        The padding row is zeroed afterwards so a padded position embeds to
        nothing.
        """
        std = float(self.cfg.initializer_range)
        bound = 2.0 * std
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(
                    module.weight, std=std, a=-bound, b=bound
                )
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.MultiheadAttention):
                for weight in (module.in_proj_weight, module.out_proj.weight):
                    if weight is not None:
                        nn.init.trunc_normal_(weight, std=std, a=-bound, b=bound)
                for bias in (module.in_proj_bias, module.out_proj.bias):
                    if bias is not None:
                        nn.init.zeros_(bias)
        with torch.no_grad():
            model.item_embedding.weight[model.pad].zero_()

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer, total_steps: int
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Linear warmup then linear decay toward zero.

        §4.3 specifies "linear decay of the learning rate", which the shared
        ``_schedule`` helper does not offer -- it has constant and cosine -- so
        it is written here rather than by widening a helper two other models
        depend on.

        The shape is ``create_optimizer``'s: a ``polynomial_decay`` of power 1
        from ``init_lr`` to zero across *all* ``num_train_steps``, which a
        linear warmup overrides for the first stretch rather than displacing.
        So the decay is measured from step 0 and not from the end of warmup,
        and the rate re-enters a little below ``init_lr`` once the warmup lets
        go, instead of restarting at it.

        Neither end of the schedule spends a step at exactly zero: the last
        step lands at ``1 / total_steps`` of the peak, and the warmup starts at
        ``1 / (warmup + 1)`` where ``create_optimizer`` starts at 0. A step
        whose learning rate is zero computes a gradient, clips it, and applies
        nothing.
        """
        if total_steps <= 1:
            return None
        # Not clamped to total_steps: create_optimizer ramps toward a flat
        # num_warmup_steps and simply ends early if the run is shorter, which
        # leaves a short fit training below the peak rate throughout rather
        # than compressing the ramp to fit.
        warmup = int(self.cfg.warmup_steps)

        def factor(step: int) -> float:
            if step < warmup:
                # Step 0 would otherwise train at exactly zero and waste a step.
                return (step + 1) / (warmup + 1)
            return max(0.0, 1.0 - step / total_steps)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)

    def _build_checkpoint_optimizer(self) -> None:
        """Construct the optimizer before optimizer state is loaded into it.

        Weight decay skips biases and LayerNorm parameters, matching
        ``exclude_from_weight_decay`` in the reference. Decaying a normalization
        scale pulls it toward zero, which shrinks the activations it exists to
        standardize.

        The decoupling itself is equivalent: the reference applies
        ``param -= lr * (update + weight_decay * param)`` where AdamW scales the
        parameter by ``1 - lr * weight_decay`` and then takes the Adam step,
        which is the same update rearranged.

        One thing is not equivalent. ``AdamWeightDecayOptimizer`` computes
        ``update = next_m / (sqrt(next_v) + epsilon)`` with no bias correction,
        which the comment there calls a deliberate departure from Adam;
        ``torch.optim.AdamW`` always corrects. The ratio of the two step sizes
        is ``sqrt(1 - beta2**t) / (1 - beta1**t)``, so this takes roughly a
        third of the reference's step at t = 1 and agrees with it to within a
        percent by t ~ 1000. Warmup covers most of that stretch. Matching it
        exactly would mean carrying an optimizer, which is a poor trade for a
        difference that has vanished before the schedule reaches its peak.
        """
        if self.model is None:
            raise RuntimeError("Bert4Rec model must be built before its optimizer")
        decayed, spared = [], []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.ndim == 1 or name.endswith(".bias"):
                spared.append(parameter)
            else:
                decayed.append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decayed, "weight_decay": self.cfg.weight_decay},
                {"params": spared, "weight_decay": 0.0},
            ],
            lr=self.cfg.lr,
            betas=tuple(self.cfg.betas),
            eps=self.cfg.epsilon,
        )

    # -- persistence --------------------------------------------------------

    @classmethod
    def _from_checkpoint_config(
        cls,
        config: dict,
        reader: ModelCheckpointReader,
        *,
        device: torch.device,
    ) -> Bert4RecTrainer:
        """Rebuild the trainer's shape before learned state is installed."""
        config = dict(config)
        # The stored device is where the model was trained, which says nothing
        # about where it is being loaded. The caller's choice wins.
        config["device"] = str(device)
        # betas arrives from JSON as a list; Bert4RecConfig.__post_init__
        # normalizes it back to a tuple, so nothing is needed here.
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
            padding="right",
        )
        trainer = cls(Bert4RecConfig(**config), batcher)
        trainer._n_items = tokenizer.n_items
        trainer._predict_batcher = replace(
            batcher, max_length=int(trainer_state["max_length"]) - 1
        )
        # Built by _build_model rather than inline, so a reloaded model is
        # constructed by exactly the path that trained it.
        trainer.model = trainer._build_model()
        return trainer

    def _checkpoint_module(self) -> nn.Module | None:
        return self.model

    def _save_checkpoint_state(self, writer: ModelCheckpointWriter) -> None:
        """Write the non-module state: ``max_length``, history, and tokenizer."""
        if self.batcher is None or not isinstance(
            self.batcher.tokenizer, ItemTokenizer
        ):
            raise TypeError(
                "Bert4RecTrainer checkpoints support ItemTokenizer only"
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
            raise TypeError("Bert4Rec training history must be a list")
        self.history = list(history)
