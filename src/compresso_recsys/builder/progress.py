"""The build's own progress sink, shared by the stages reporting through it."""

from __future__ import annotations

from typing import Any




class _CheckpointProgress:
    def __init__(self, *, enabled: bool, total: int) -> None:
        self.enabled = enabled
        self.current = False
        self.bar: Any = None
        if not enabled:
            return
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency
            return
        self.bar = tqdm(total=total, unit="step", desc="Building checkpoint")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.bar is not None:
            if self.current and exc_type is None:
                self.bar.update(1)
            self.bar.close()

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        if self.bar is None:
            print(f"[compresso-recsys] {message}", flush=True)
            return
        if self.current:
            self.bar.update(1)
        self.current = True
        self.bar.set_description_str(message)

    def detail(self, message: str) -> None:
        """Update the active step label without advancing the progress bar."""
        if not self.enabled:
            return
        if self.bar is None:
            print(f"[compresso-recsys] {message}", flush=True)
            return
        self.bar.set_description_str(message)
