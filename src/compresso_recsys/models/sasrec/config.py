"""Published settings for the model, as validated dataclasses."""

from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from compresso_recsys._reporting import (
    _INHERIT,
    _Inherit,
    TrainingProgress,
    _validate_log_every_n_steps,
)


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
    item it cannot identify. It defaults to zero for paper parity. Set it above
    zero when otherwise ``unk`` would never be trained: the training vocabulary
    *is* the training window, so an out-of-catalog item cannot occur until
    evaluation, and its embedding would still sit at its initialisation when a
    quarter of a temporal test history turns out to need it. The right rate
    tracks the out-of-catalog share the split will actually produce -- near zero
    under ``leave_last_out``, far higher on a late ``temporal`` stage. It is
    ignored when the tokenizer has no ``unk`` to substitute.

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
        for name in (
            "d_model",
            "n_blocks",
            "n_heads",
            "max_history_length",
            "batch_size",
            "n_negatives",
        ):
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
