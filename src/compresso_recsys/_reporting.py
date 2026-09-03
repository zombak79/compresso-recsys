from __future__ import annotations

import time
import warnings
from numbers import Integral
from typing import Any, Mapping


class _Inherit:
    """Sentinel for "use the object's default value"."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic, for signatures
        return "<inherit>"


_INHERIT = _Inherit()


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
        self.show_progress = bool(show_progress) and logger is None
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
    inherited_logger = logger is _INHERIT
    resolved_logger = default_logger if inherited_logger else logger
    # ``None`` historically meant "inherit" on ELSA/TEASERGD progress
    # overrides. Keep accepting it while the sentinel makes new signatures
    # unambiguous.
    inherited_progress = show_progress is _INHERIT or show_progress is None
    if not inherited_progress:
        bar = bool(show_progress)
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
