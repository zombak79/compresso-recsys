"""Validation during training, and early stopping on it.

Trainers that drive their epochs through the shared progress loop accept two
optional arguments in ``fit``::

    trainer.fit(train, validation=Validation(val_source, val_targets, [NDCG(20)],
                                              sample_ids=val_ids),
                early_stopping=EarlyStopping("ndcg@20", patience=5))

``validation`` alone records validation metrics in the training history
(``val_<metric>`` keys) without stopping. ``early_stopping`` also stops once the
monitored metric has not improved for ``patience`` validations and, by
default, restores the weights of the best validation.

Validation goes through the public
:func:`~compresso_recsys.evaluation.evaluate_recommender`, so the numbers are
the ones a post-hoc evaluation of the same weights would give. It draws no
random numbers, so a run with validation trains exactly like one without.

``epochs`` in the trainer's config becomes the maximum. Trainers whose
learning-rate schedule spans the configured epochs (cosine, warmup) keep that
schedule, so a run stopped early ends partway through it.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

__all__ = ["EarlyStopping", "Validation"]


@dataclass(frozen=True)
class Validation:
    """What to evaluate on between epochs.

    ``every`` is the number of epochs between validations; the final epoch is
    always validated. ``exclude_seen`` is passed to the evaluator unchanged.
    """

    source: Any
    targets: Any
    metrics: Sequence[Any] | None = None
    sample_ids: Sequence[Any] | np.ndarray | None = None
    every: int = 1
    batch_size: int = 1024
    exclude_seen: bool | None = None

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError(f"every must be a positive integer, got {self.every!r}")
        if isinstance(self.batch_size, bool) or not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size!r}")


@dataclass(frozen=True)
class EarlyStopping:
    """When to stop, and what to keep.

    ``metric`` names a key the validation produces, e.g. ``"ndcg@20"``.
    ``patience`` counts validations without an improvement of more than
    ``min_delta``; it only starts counting after epoch ``min_epochs``. With
    ``restore_best`` the weights of the best validation are reloaded before
    ``fit`` returns.
    """

    metric: str
    patience: int
    mode: Literal["max", "min"] = "max"
    min_delta: float = 0.0
    min_epochs: int = 0
    restore_best: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or not isinstance(self.patience, int) or self.patience < 1:
            raise ValueError(f"patience must be a positive integer, got {self.patience!r}")
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")
        if not np.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError(f"min_delta must be a finite value >= 0, got {self.min_delta!r}")
        if isinstance(self.min_epochs, bool) or not isinstance(self.min_epochs, int) or self.min_epochs < 0:
            raise ValueError(f"min_epochs must be an integer >= 0, got {self.min_epochs!r}")


def _module_of(owner: Any) -> Any:
    module = owner._checkpoint_module() if hasattr(owner, "_checkpoint_module") else None
    if module is None:
        module = getattr(owner, "model", None)
    return module


class _ValidationMonitor:
    """Validates a trainer between epochs; decides when to stop; keeps the best state."""

    def __init__(self, owner: Any, validation: Validation, early_stopping: EarlyStopping | None) -> None:
        if not isinstance(validation, Validation):
            raise TypeError(f"validation must be a Validation, got {type(validation).__name__}")
        if early_stopping is not None and not isinstance(early_stopping, EarlyStopping):
            raise TypeError(f"early_stopping must be an EarlyStopping, got {type(early_stopping).__name__}")
        self.owner = owner
        self.validation = validation
        self.early_stopping = early_stopping
        self.stop = False
        self.best: dict[str, Any] | None = None
        self._best_state: dict[str, Any] | None = None
        self._bad = 0

    def _evaluate(self) -> dict[str, float]:
        from compresso_recsys.evaluation import evaluate_recommender

        module = _module_of(self.owner)
        was_training = bool(getattr(module, "training", False))
        # Some trainers mark themselves fitted only after the last epoch; the
        # weights are scoreable already, so validation lifts the flag for the
        # duration of the call and puts it back.
        lifted = getattr(self.owner, "_is_fitted", None) is False
        if lifted:
            self.owner._is_fitted = True
        try:
            if module is not None and hasattr(module, "eval"):
                module.eval()
            result = evaluate_recommender(
                self.owner,
                source=self.validation.source,
                targets=self.validation.targets,
                metrics=self.validation.metrics,
                sample_ids=self.validation.sample_ids,
                batch_size=self.validation.batch_size,
                exclude_seen=self.validation.exclude_seen,
            )
        finally:
            if lifted:
                self.owner._is_fitted = False
            if module is not None and hasattr(module, "train") and was_training:
                module.train()
        return dict(result.metrics)

    def after_epoch(self, epoch: int, record: dict[str, Any], *, is_last: bool) -> None:
        """Validate when due, merge ``val_*`` into ``record``, and decide whether to stop."""
        if epoch % self.validation.every != 0 and not is_last:
            return
        metrics = self._evaluate()
        record.update({f"val_{key}": value for key, value in metrics.items()})
        rule = self.early_stopping
        if rule is None:
            return
        if rule.metric not in metrics:
            raise KeyError(
                f"early_stopping.metric {rule.metric!r} is not among the validation "
                f"metrics {sorted(metrics)}"
            )
        score = float(metrics[rule.metric])
        if self.best is None:
            improved = True
        else:
            previous = float(self.best[f"val_{rule.metric}"])
            improved = (score > previous + rule.min_delta if rule.mode == "max"
                        else score < previous - rule.min_delta)
        if improved:
            self.best = {"epoch": int(epoch), **{f"val_{k}": float(v) for k, v in metrics.items()}}
            self.owner.best_ = dict(self.best)
            self._bad = 0
            if rule.restore_best:
                module = _module_of(self.owner)
                if module is not None and hasattr(module, "state_dict"):
                    self._best_state = {
                        key: value.detach().to("cpu", copy=True) if hasattr(value, "detach") else copy.deepcopy(value)
                        for key, value in module.state_dict().items()
                    }
        elif epoch > rule.min_epochs:
            self._bad += 1
        if self._bad >= rule.patience:
            self.stop = True
        if self.stop or is_last:
            self._restore()

    def _restore(self) -> None:
        if self._best_state is None or self.early_stopping is None or not self.early_stopping.restore_best:
            return
        module = _module_of(self.owner)
        if module is None:
            return
        device = next((p.device for p in module.parameters()), None) if hasattr(module, "parameters") else None
        state = {key: value.to(device) if device is not None and hasattr(value, "to") else value
                 for key, value in self._best_state.items()}
        module.load_state_dict(state)


def _make_monitor(owner: Any, validation: Validation | None,
                  early_stopping: EarlyStopping | None) -> _ValidationMonitor | None:
    """The monitor for one fit, or ``None`` when no validation was asked for.

    Resets ``owner.best_`` so it always describes the latest fit.
    """
    owner.best_ = None
    if validation is None:
        if early_stopping is not None:
            raise ValueError("early_stopping needs validation=: there is nothing to stop on")
        return None
    return _ValidationMonitor(owner, validation, early_stopping)
