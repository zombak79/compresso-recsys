from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from compresso_recsys.models import (
    ELSACompressionConfig,
    ELSAConfig,
    ELSATrainer,
    TEASERGDConfig,
    TEASERGDTrainer,
)

EPOCHS = 3


class _StubBar:
    """tqdm-compatible recorder for the calls a trainer makes on a bar."""

    created: list["_StubBar"] = []

    def __init__(self, iterable=None, *, total=None, desc=None, **kwargs):
        self.iterable = iterable
        self.total = total
        self.descriptions = [desc]
        self.resets = 0
        self.updates = 0
        self.closes = 0
        self.postfixes: list[object] = []
        _StubBar.created.append(self)

    def __iter__(self):
        return iter(self.iterable)

    def reset(self, total=None):
        self.resets += 1
        self.total = total

    def set_description(self, desc):
        self.descriptions.append(desc)

    def set_postfix(self, *args, **kwargs):
        self.postfixes.append(kwargs or args)

    def update(self, n=1):
        self.updates += n

    def close(self):
        self.closes += 1


@pytest.fixture
def bars(monkeypatch):
    """Make ``from tqdm.auto import tqdm`` resolve to the recorder."""
    _StubBar.created = []
    auto = types.ModuleType("tqdm.auto")
    auto.tqdm = _StubBar
    root = types.ModuleType("tqdm")
    root.auto = auto
    # Stub modules need a spec: torch's dynamo walks sys.modules and calls
    # find_spec(), which rejects entries whose __spec__ is None.
    root.__spec__ = ModuleSpec("tqdm", loader=None)
    auto.__spec__ = ModuleSpec("tqdm.auto", loader=None)
    monkeypatch.setitem(sys.modules, "tqdm", root)
    monkeypatch.setitem(sys.modules, "tqdm.auto", auto)
    return _StubBar.created


def _interactions() -> csr_matrix:
    return csr_matrix(
        np.array(
            [
                [1, 1, 1, 0, 0, 0],
                [0, 1, 1, 1, 0, 0],
                [1, 0, 0, 1, 1, 0],
                [0, 0, 1, 1, 1, 1],
                [1, 1, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
    )


def _item_features() -> np.ndarray:
    return np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 1],
        ],
        dtype=np.float32,
    )


def _fit_elsa(*, show_progress: bool = True):
    trainer = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            batch_size=2,
            max_output=6,
            epochs=EPOCHS,
            lr=1e-2,
            show_progress=show_progress,
            seed=7,
        )
    )
    return trainer.fit(_interactions())


def _fit_teaser_gd(*, show_progress: bool = True):
    trainer = TEASERGDTrainer(
        TEASERGDConfig(
            epochs=EPOCHS,
            batch_size=2,
            max_output=4,
            lr=0.01,
            show_progress=show_progress,
            include_popularity=False,
            coefficient_regularization_samples=16,
            seed=3,
        )
    )
    return trainer.fit(_interactions(), _item_features())


TRAINERS = pytest.mark.parametrize(
    ("fit", "label", "trainer_class", "step"),
    [
        (_fit_elsa, "ELSA", ELSATrainer, "train_step"),
        (_fit_teaser_gd, "TEASERGD", TEASERGDTrainer, "_train_step"),
    ],
)


@TRAINERS
def test_fit_reuses_one_batch_bar_across_epochs(bars, fit, label, trainer_class, step):
    """Regression: every epoch used to leave its own finished bar behind."""
    fit()

    # One bar for the epoch counter, one reused for batches. Nothing per epoch.
    assert len(bars) == 2
    epoch_bar, batch_bar = bars

    assert batch_bar.resets == EPOCHS
    assert batch_bar.updates >= EPOCHS  # at least one batch per epoch
    assert batch_bar.updates % EPOCHS == 0
    assert batch_bar.closes == 1
    assert batch_bar.descriptions[1:] == [
        f"{label} epoch {epoch}" for epoch in range(1, EPOCHS + 1)
    ]
    assert len(epoch_bar.postfixes) == EPOCHS


@TRAINERS
def test_fit_closes_bars_when_a_step_fails(
    bars, monkeypatch, fit, label, trainer_class, step
):
    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(trainer_class, step, explode)
    with pytest.raises(RuntimeError, match="boom"):
        fit()

    assert len(bars) == 2
    assert bars[1].closes == 1


@TRAINERS
def test_show_progress_false_creates_no_bars(bars, fit, label, trainer_class, step):
    trainer = fit(show_progress=False)

    assert bars == []
    assert len(trainer.history) == EPOCHS


@TRAINERS
def test_missing_tqdm_leaves_training_working(
    monkeypatch, fit, label, trainer_class, step
):
    """tqdm is optional, so an import failure must not break fit()."""
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)

    trainer = fit()
    assert len(trainer.history) == EPOCHS


def test_elsa_mask_search_reuses_one_bar(bars):
    """The unbounded mask-search loop must not accumulate bars either."""
    trainer = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            batch_size=2,
            max_output=6,
            epochs=1,
            lr=1e-2,
            show_progress=True,
            seed=7,
            compression=ELSACompressionConfig(
                k_target=2,
                k_schedule=(4, 3, 2),
                stability_window=1,
                change_threshold=100.0,
                mask_update_interval=1,
            ),
        )
    )
    trainer.fit(_interactions())

    mask_epochs = [
        record for record in trainer.history if record.get("phase") == "mask_search"
    ]
    assert len(mask_epochs) > 1, "mask search should run several epochs"

    mask_bars = [
        bar
        for bar in bars
        if any("mask stage" in str(desc) for desc in bar.descriptions)
    ]
    assert len(mask_bars) == 1
    bar = mask_bars[0]
    assert bar.resets == len(mask_epochs)
    assert bar.closes == 1


def test_real_tqdm_bar_is_reused():
    """The stubs above cannot catch API drift in tqdm itself."""
    tqdm_module = pytest.importorskip("tqdm.auto")
    created: list[object] = []
    original = tqdm_module.tqdm

    class _Tracked(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    tqdm_module.tqdm = _Tracked
    try:
        trainer = _fit_teaser_gd()
    finally:
        tqdm_module.tqdm = original

    assert len(created) == 2
    assert created[1].desc.startswith(f"TEASERGD epoch {EPOCHS}")
    assert len(trainer.history) == EPOCHS
