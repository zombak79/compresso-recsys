from __future__ import annotations

import time
import warnings
from numbers import Integral
from typing import Any, Callable, Mapping, Sequence


class _Inherit:
    """Sentinel for "use the object's default value"."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic, for signatures
        return "<inherit>"


_INHERIT = _Inherit()


def _validate_show_progress(
    value: Any,
    *,
    allow_inherit: bool,
) -> bool | None | _Inherit:
    """Validate a resolved value or a public per-call override."""
    if isinstance(value, bool):
        return value
    if allow_inherit and (value is None or value is _INHERIT):
        return value
    expected = "a bool or None" if allow_inherit else "a bool"
    raise ValueError(f"show_progress must be {expected}")


def _format_duration(seconds: float, unit_name: str | None = None) -> str:
    """Render a duration in whichever of s / ms / us keeps it readable."""
    suffix = f"/{unit_name}" if unit_name is not None else ""
    if seconds >= 1 or seconds == 0:
        return f"{seconds:.0f}s{suffix}"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.0f}ms{suffix}"
    return f"{seconds * 1e6:.0f}us{suffix}"


def _format_metric(value: Any) -> str:
    """Render a metric without rounding small nonzero values to zero."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}" if abs(number) > 1e-3 else f"{number:.4e}"


def _format_metrics(
    metrics: Mapping[str, Any],
    *,
    skip: frozenset[str] = frozenset(),
) -> str:
    return " | ".join(
        f"{key}: {_format_metric(value)}"
        for key, value in metrics.items()
        if key not in skip
    )


def _validate_log_every_n_steps(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("log_every_n_steps must be an integer")
    if int(value) < 0:
        raise ValueError("log_every_n_steps must be >= 0")
    return int(value)


class _Reporter:
    """One call's resolved progress sink.

    The failure latch and progress-bar decision live here rather than on a
    trainer. A broken logger therefore disables only this call, and concurrent
    calls cannot alter each other's reporting state.
    """

    def __init__(
        self,
        logger: Any | None,
        show_progress: bool,
        prefix: str,
        log_every_n_steps: int,
        *,
        allow_stdout_fallback: bool = False,
    ) -> None:
        self.logger = logger
        # A logger and tqdm carry the same progress. The logger always wins.
        resolved_progress = _validate_show_progress(
            show_progress,
            allow_inherit=False,
        )
        assert isinstance(resolved_progress, bool)
        self.show_progress = resolved_progress and logger is None
        self.prefix = str(prefix)
        validated_interval = _validate_log_every_n_steps(log_every_n_steps)
        self.log_every_n_steps = validated_interval if logger is not None else 0
        self.allow_stdout_fallback = bool(allow_stdout_fallback)
        self.disabled = False

    @property
    def active(self) -> bool:
        """Whether formatting and emitting a line can still be useful."""
        return self.logger is not None and not self.disabled

    def log(self, message: str) -> None:
        """Emit one line, latching off if the duck-typed sink raises."""
        if not self.active:
            return
        try:
            self.logger.info(f"[{self.prefix}] {message}")
        except Exception as exc:
            self.disabled = True
            try:
                warnings.warn(
                    "logger.info raised "
                    f"{exc!r}; the call continues with logging disabled",
                    RuntimeWarning,
                    stacklevel=2,
                )
            except Exception:
                # A warnings-as-errors policy must not turn a courtesy notice
                # into a failed multi-hour training run.
                pass

    def wrap(self, iterable, *, total: int | None = None, desc: str | None = None):
        """Return an iterable behind tqdm only when no logger supersedes it."""
        if not self.show_progress:
            return iterable
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional display helper
            return iterable
        return tqdm(iterable, total=total, desc=desc)

    def bar(self, *, total: int, desc: str):
        """Create a manually driven tqdm bar, or return ``None``."""
        if not self.show_progress:
            return None
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional display helper
            return None
        return tqdm(total=total, desc=desc)

    def step(
        self,
        label: str,
        step: int,
        steps: int,
        started: float,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        """Log timing, ETA, and optional running metrics inside a long pass."""
        if not self.active:
            return
        elapsed = time.monotonic() - started
        per_step = elapsed / max(1, step)
        segments = [
            _format_duration(per_step, "step"),
            f"{_format_duration(elapsed)} elapsed",
            f"{_format_duration(per_step * max(0, steps - step))} remaining",
        ]
        if metrics:
            segments.append(_format_metrics(metrics))
        self.log(f"{label}: " + " | ".join(segments))

    def epoch(
        self,
        label: str,
        record: Mapping[str, Any],
        started: float,
    ) -> None:
        """Log one completed epoch from the record also stored in history."""
        if not self.active:
            return
        segments = [_format_duration(time.monotonic() - started, "epoch")]
        metrics = _format_metrics(record, skip=frozenset({"epoch"}))
        if metrics:
            segments.append(metrics)
        self.log(f"{label}: " + " | ".join(segments))


def _resolve_reporter(
    *,
    default_logger: Any | None,
    logger: Any,
    default_show_progress: bool,
    show_progress: Any,
    prefix: str,
    log_every_n_steps: int,
) -> _Reporter:
    """Resolve constructor/config defaults against one call's overrides."""
    if isinstance(logger, _Reporter):
        # Internal callers may carry one already-resolved reporter through
        # nested prediction paths. Preserve its call-scoped failure latch.
        return logger
    inherited_logger = logger is _INHERIT
    resolved_logger = default_logger if inherited_logger else logger
    # ``None`` historically meant "inherit" on ELSA/TEASERGD progress
    # overrides. Keep accepting it while the sentinel makes new signatures
    # unambiguous.
    validated_progress = _validate_show_progress(
        show_progress,
        allow_inherit=True,
    )
    inherited_progress = (
        validated_progress is _INHERIT or validated_progress is None
    )
    if not inherited_progress:
        assert isinstance(validated_progress, bool)
        bar = validated_progress
    elif not inherited_logger and resolved_logger is None:
        # Explicit logger=None means a deliberately quiet call unless a bar was
        # requested explicitly alongside it.
        bar = False
    else:
        bar = bool(default_show_progress)
    return _Reporter(
        resolved_logger,
        bar,
        prefix,
        log_every_n_steps,
        allow_stdout_fallback=inherited_logger and resolved_logger is None,
    )


class TrainingProgress:
    """The reporting scaffolding around a fit loop, not the loop itself.

    A trainer keeps its own ``for epoch`` statement and its own loop body. What
    it hands over is only the part every trainer was repeating around them: the
    epoch and batch bars, the per-epoch timer, the periodic step line, the epoch
    line, and the started/finished pair. That scaffolding was written out
    fifteen times across eleven modules and differed between them only in a
    progress-bar label and the name of the loss denominator.

    Deliberately not a runner. Nothing here takes a training step as a callback,
    so early stopping, an extra metric, or a changed schedule stays an edit to
    code that is still visible inside ``fit``.

    A trainer whose fit has two logged phases -- ELSA and TEASERGD both do --
    builds one of these per phase rather than reusing one across both.
    """

    def __init__(
        self,
        reporter: _Reporter,
        *,
        label: str,
        epochs: int,
        batches: int,
        metric: str = "loss",
    ) -> None:
        self.reporter = reporter
        self.label = str(label)
        # Autoencoders report a reconstruction loss rather than a plain one,
        # and the name has to match what the trainer stores in its history for
        # the epoch line and the history record to agree.
        self.metric = str(metric)
        self.n_epochs = int(epochs)
        self.n_batches = int(batches)
        self._epoch_iter: Any = None
        self._batch_bar: Any = None
        self._epoch = 0
        self._epoch_started = 0.0
        self._started = 0.0
        self._recorded = 0

    def __enter__(self) -> "TrainingProgress":
        self._started = time.monotonic()
        self._epoch_iter = self.reporter.wrap(
            range(1, self.n_epochs + 1),
            total=self.n_epochs,
            desc=f"{self.label} fit",
        )
        self._batch_bar = self.reporter.bar(
            total=self.n_batches, desc=f"{self.label} epoch 1"
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        if self._batch_bar is not None:
            self._batch_bar.close()
        if hasattr(self._epoch_iter, "close"):
            self._epoch_iter.close()
        # The finished line follows the bars being torn down, as it did when
        # each trainer wrote it after its own finally block.
        if exc_type is None:
            self.reporter.log(
                "fit finished: "
                f"{_format_duration(time.monotonic() - self._started)} total | "
                f"{self._recorded} epochs recorded"
            )
        return False

    def start(self, detail: str) -> None:
        """Log the opening line, whose content is the model's own."""
        self.reporter.log(f"fit started: {detail}")

    def epochs(self):
        """Yield ``1..epochs``, timing each and relabelling the batch bar."""
        for epoch in self._epoch_iter:
            self._epoch = epoch
            self._epoch_started = time.monotonic()
            if self._batch_bar is not None:
                self._batch_bar.reset(total=self.n_batches)
                self._batch_bar.set_description(f"{self.label} epoch {epoch}")
            yield epoch

    def batch(
        self,
        step_index: int,
        loss_sum: Any = 0.0,
        count: int = 0,
        *,
        metrics: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        """Advance the bar and, on the configured interval, log the step.

        ``loss_sum`` and ``count`` are the running totals whose ratio is the
        mean loss so far; trainers name the denominator differently -- masked
        items, positions, target rows -- but divide it the same way. A sum
        accumulated on device arrives as a tensor, so the ratio is coerced
        rather than formatted as one.

        A trainer reporting several metrics passes ``metrics`` instead. Pass a
        callable when producing them costs something: MultVAE stacks three
        device tensors, and a callable runs only on the steps that log, which
        is what kept that sync off the batch path before.
        """
        if self._batch_bar is not None:
            self._batch_bar.update(1)
        interval = self.reporter.log_every_n_steps
        if not interval or step_index % interval != 0:
            return
        if metrics is None:
            resolved: Mapping[str, Any] = {
                self.metric: float(loss_sum / count) if count else float("nan")
            }
        elif callable(metrics):
            resolved = metrics()
        else:
            resolved = metrics
        self.reporter.step(
            f"epoch {self._epoch}/{self.n_epochs} "
            f"step {step_index}/{self.n_batches}",
            step_index,
            self.n_batches,
            self._epoch_started,
            resolved,
        )

    def epoch_done(
        self,
        record: Mapping[str, Any],
        *,
        postfix: Sequence[str] | None = None,
    ) -> None:
        """Log the completed epoch from the record also stored in history.

        ``postfix`` names the record keys worth showing on the live bar, and
        defaults to the trainer's metric alone. MultVAE shows its KL weight
        beside the loss because the weight moves on a schedule during the run.
        """
        self._recorded += 1
        self.reporter.epoch(
            f"epoch {self._epoch}/{self.n_epochs}", record, self._epoch_started
        )
        if not hasattr(self._epoch_iter, "set_postfix"):
            return
        keys = (self.metric,) if postfix is None else tuple(postfix)
        shown = {
            key: f"{float(record[key]):.4f}"
            for key in keys
            if record.get(key) is not None
        }
        if shown:
            self._epoch_iter.set_postfix(shown)
