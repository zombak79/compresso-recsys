"""Configuration for :class:`~compresso_recsys.models.Bert4Rec`.

Every default here is the published setting. Where the paper is silent the
value comes from the authors' TensorFlow implementation at
https://github.com/FeiSun/BERT4Rec, which is noted per field, because a number
with two possible provenances is the one most likely to drift.

"The reference" there means the four ``run_*.sh`` scripts, which are what
produced the paper's tables. The flag defaults in ``run.py`` and
``gen_data_fin.py`` are a third thing again, inherited from the BERT
pretraining code the repository was forked from, and several of them are
values no published run ever used -- ``batch_size`` 32, ``learning_rate``
5e-5, ``num_train_steps`` 100000, ``num_warmup_steps`` 10000. A citation
here naming a flag default rather than a script is a citation to the wrong
tier, so the fields below say which one they mean.

The scripts disagree with each other per dataset, so a single default set
cannot be all four at once. These follow ml-20m, the long-sequence run the
default ``max_history_length`` of 200 already commits to, except for
``dropout``, which is ml-1m's 0.2 against ml-20m's 0.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from compresso_recsys._reporting import _validate_log_every_n_steps

__all__ = ["Bert4RecConfig"]

OptimizerName = Literal["AdamW"]


@dataclass(frozen=True)
class Bert4RecConfig:
    """Architecture, Cloze, and optimization settings for BERT4Rec.

    **Architecture.** ``n_blocks`` and ``n_heads`` are the paper's ``L = 2`` and
    ``h = 2`` (§4.3). ``d_model`` is 64, the smallest width at which §4.5 reports
    the model saturating, and the width the released configs use. The
    feed-forward inner width is ``4 * d_model`` and is not a field: Eq. 3 fixes
    it, and the released config agrees (``intermediate_size`` 256 at
    ``hidden_size`` 64).

    ``max_history_length`` is ``N``, the truncation point *and* the size of the
    positional table (§3.4). The paper uses 200 for the MovieLens datasets and
    50 for the sparse ones; 200 is the default because the released configs set
    ``max_position_embeddings`` to it. A history longer than ``N`` keeps its most
    recent ``N`` interactions, matching ``item_seq[-max_num_tokens:]`` in the
    reference generator -- a context window is a claim about recency.

    ``dropout`` and ``att_dropout`` are the reference's ``hidden_dropout_prob``
    and ``attention_probs_dropout_prob``. They are two fields rather than one
    because the reference exposes two; the paper names only the tuning range
    (§4.3). The released ``bert_config`` files disagree -- 0.2/0.2 for ml-1m,
    0.1/0.1 for ml-20m -- and 0.2 is ml-1m's, the run the paper reports in
    most detail. ``dropout`` covers all three of the reference's
    hidden-dropout sites -- the postprocessed input embedding, the attention
    output, and the feed-forward output -- while ``att_dropout`` applies to the
    attention probabilities alone.

    **Cloze.** ``mask_proportion`` is the paper's rho: the fraction of positions
    masked per training sequence (§3.6). The paper tunes it per dataset -- 0.2
    for MovieLens, 0.4 for Steam, 0.6 for Beauty (§4.3) -- and 0.2 is the
    default for the long-sequence case the default ``max_history_length``
    assumes. ``max_predictions`` caps the count per sequence at
    ``max_predictions_per_seq``, which the scripts set per dataset: 40 for
    ml-1m, 20 for ml-20m and steam, 30 for beauty. 20 is ml-20m's, and at
    ``N`` 200 with rho 0.2 it is that run's exact pairing -- so the cap binds
    at 20 rather than the 40 rho alone would ask for on a full-length window,
    which is a published choice and not an accident of the two numbers
    meeting.

    ``mask_token_probability`` is the reference's ``mask_prob``, and its default
    of 1.0 is the interesting one: it *disables* BERT's 80/10/10 corruption, so
    a chosen position always becomes ``[MASK]``. The paper describes exactly
    that -- "replace them with a special token ``[mask]``" -- with no mention of
    random or kept tokens. Below 1.0 the remainder splits evenly between keeping
    the original item and drawing a random one, as in ``create_masked_lm_
    predictions``.

    ``duplication_factor`` and ``sliding_window_step`` reproduce ``dupe_factor``
    (10 in all four scripts) and ``prop_sliding_window`` (0.5 in ml-1m, ml-20m
    and steam; 0.1 in beauty, whose ``N`` is 50). The first re-masks every
    sequence that many times per fit, which is how the Cloze objective turns
    one history into the many samples §3.6 claims for it. It is not a slower
    spelling of ``epochs``: it also fixes the ratio of randomly masked samples
    to ``last_item_samples`` ones at 10:1, which is what weights the Cloze
    objective against the fine-tuning objective.

    The second slides a window over any history longer than
    ``max_history_length`` in steps of
    ``sliding_window_step * max_history_length``, so the early part of a long
    history is trained on rather than discarded. At the default ``N`` of 200
    that is a window every 100 interactions. Beauty's 0.1 would place one
    every 20 and cost five times as much per epoch; it is paired with an ``N``
    of 50, where it means a window every 5.

    ``last_item_samples`` adds the paper's fine-tuning samples: an extra
    sequence with only the final item masked, "to better match the sequential
    recommendation task" (§3.6). ``create_training_instances`` appends these
    unconditionally via ``mask_last``, which runs over a user's whole document
    -- so one per sliding window, not one per user -- and the default is
    ``True``.

    Every Cloze sample keeps at least one position unmasked, so the count is
    also capped at ``len - 1`` and a ``mask_proportion`` of 1.0 means "all but
    one". A history of a single interaction therefore yields no sample at all:
    masking its only position would leave the input identical for every such
    history, teaching nothing but a popularity prior ``duplication_factor``
    times over. The reference's ``create_masked_lm_predictions`` has no such
    floor and would mask the lone position; this one is the package's.

    ``unk_dropout`` replaces that fraction of the *context* positions -- real
    positions not chosen for prediction -- with the tokenizer's ``unk`` token,
    teaching the model to read a history containing an item it cannot
    identify. It defaults to zero for paper parity. Set it above zero when
    otherwise ``unk`` would never be trained: the training vocabulary *is* the
    training window, so an out-of-catalog item cannot occur until evaluation.
    Until then the ``unk`` row is trained only as a wrong answer in the tied
    softmax, so it reaches a temporal test history as a vector pushed away from
    every masked state rather than as a learned "unknown item". The right rate
    tracks the out-of-catalog share the split will actually produce -- near zero
    under ``leave_last_out``, far higher on a late ``temporal`` stage. Chosen
    positions are never replaced, so no label is lost and ``unk`` never stands
    in for ``[mask]``. It is ignored when the tokenizer has no ``unk`` to
    substitute.

    **Optimization.** The paper's §4.3 settings, and the scripts agree with it
    throughout: ``lr`` 1e-4 and ``batch_size`` 256 are both §4.3's *and* what
    all four scripts pass, so there is nothing to adjudicate. (``run.py``
    defaults those flags to 5e-5 and 32, which is the fork's inheritance and
    not a published setting.) ``optimization.py`` supplies the rest: decoupled
    weight decay 0.01, ``betas`` of (0.9, 0.999), ``epsilon`` 1e-6 against
    PyTorch's 1e-8, gradients clipped at an l2 norm of 5, and a
    ``polynomial_decay`` of power 1, which is the paper's linear decay. Weight
    decay is excluded from LayerNorm and bias parameters, as
    ``exclude_from_weight_decay`` does.

    ``warmup_steps`` is the scripts' ``num_warmup_steps``, 100 in all four,
    and it is a count rather than a fraction of the run because that is what
    the reference means by it. Warmup buys a fixed number of steps for Adam's
    zero-initialized second moment to become a usable estimate, and that cost
    does not grow because the run is longer: 100 steps is 100 steps at 400000
    total, as the scripts are, and at 400. As a fraction it would be 2.5e-4,
    which is the right ratio only at the scripts' length and rounds to no
    warmup at all below about 4000 steps -- so a short fit would take its very
    first step, the one against that zero-initialized estimate, at the full
    learning rate. The other sequential models here do take a
    ``warmup_fraction``; none of them is reproducing a published step count.

    ``epochs`` has no counterpart at all. The reference trains a fixed
    ``num_train_steps`` -- 400000, or 800000 for steam -- over a corpus
    generated once; here the step count is
    ``epochs * samples / batch_size`` and so depends on the dataset. 10 is a
    default for this package's shape, not a published figure.

    ``initializer_range`` is the truncated-normal width the paper gives as
    [-0.02, 0.02] and the configs give as ``initializer_range`` 0.02.
    """

    # -- architecture -------------------------------------------------------
    d_model: int = 64
    n_blocks: int = 2
    n_heads: int = 2
    dropout: float = 0.2
    att_dropout: float = 0.2
    max_history_length: int = 200
    initializer_range: float = 0.02

    # -- Cloze objective ----------------------------------------------------
    mask_proportion: float = 0.2
    max_predictions: int = 20
    mask_token_probability: float = 1.0
    duplication_factor: int = 10
    sliding_window_step: float = 0.5
    last_item_samples: bool = True
    unk_dropout: float = 0.0

    # -- optimization -------------------------------------------------------
    batch_size: int = 256
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)
    epsilon: float = 1e-6
    max_grad_norm: float = 5.0
    warmup_steps: int = 100
    optimizer: OptimizerName = "AdamW"

    # -- runtime ------------------------------------------------------------
    device: str | torch.device = "cpu"
    show_progress: bool = True
    seed: int = 0
    log_prefix: str = "Bert4Rec"
    log_every_n_steps: int = 1000

    def __post_init__(self) -> None:
        _validate_log_every_n_steps(self.log_every_n_steps)
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.d_model % self.n_heads:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads}): each head reads d_model / h of the "
                "residual stream"
            )
        if self.n_blocks < 1:
            raise ValueError(f"n_blocks must be >= 1, got {self.n_blocks}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.att_dropout < 1.0:
            raise ValueError(
                f"att_dropout must be in [0, 1), got {self.att_dropout}"
            )
        if self.max_history_length < 2:
            raise ValueError(
                "max_history_length must be >= 2: a Cloze example needs a "
                "masked position and at least one position of context, got "
                f"{self.max_history_length}"
            )
        if self.initializer_range <= 0.0:
            raise ValueError(
                f"initializer_range must be > 0, got {self.initializer_range}"
            )
        if not 0.0 < self.mask_proportion <= 1.0:
            raise ValueError(
                "mask_proportion is the fraction of positions masked per "
                f"sequence and must be in (0, 1], got {self.mask_proportion}"
            )
        if self.max_predictions < 1:
            raise ValueError(
                f"max_predictions must be >= 1, got {self.max_predictions}"
            )
        if not 0.0 <= self.mask_token_probability <= 1.0:
            raise ValueError(
                "mask_token_probability must be in [0, 1], got "
                f"{self.mask_token_probability}"
            )
        if self.duplication_factor < 1:
            raise ValueError(
                f"duplication_factor must be >= 1, got {self.duplication_factor}"
            )
        if not 0.0 < self.sliding_window_step <= 1.0:
            raise ValueError(
                "sliding_window_step is a fraction of max_history_length and "
                f"must be in (0, 1], got {self.sliding_window_step}"
            )
        if not 0.0 <= self.unk_dropout < 1.0:
            raise ValueError(
                f"unk_dropout must be in [0, 1), got {self.unk_dropout}"
            )
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.lr <= 0.0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.weight_decay < 0.0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        # asdict writes a JSON array and reading it back gives a list, so a
        # reloaded config would otherwise carry a different type than a fresh
        # one and compare unequal to it. Frozen, hence object.__setattr__.
        object.__setattr__(self, "betas", tuple(self.betas))
        if len(self.betas) != 2:
            raise ValueError(f"betas must be two values, got {self.betas!r}")
        if not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError(f"betas must each be in [0, 1), got {self.betas!r}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if self.max_grad_norm <= 0.0:
            raise ValueError(
                f"max_grad_norm must be > 0, got {self.max_grad_norm}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        if self.optimizer != "AdamW":
            raise ValueError(
                "BERT4Rec's published optimizer is Adam with decoupled weight "
                f"decay, which PyTorch spells AdamW; got {self.optimizer!r}"
            )
