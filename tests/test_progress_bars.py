from __future__ import annotations

import sys
import types
from importlib.machinery import ModuleSpec

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall
from compresso_recsys.models import (
    EASE,
    ELSACompressionConfig,
    ELSAConfig,
    ELSATrainer,
    ItemTokenizer,
    MultDAEConfig,
    MultDAETrainer,
    MultVAEConfig,
    MultVAETrainer,
    SequenceBatcher,
    SimpleGPTConfig,
    SimpleGPTTrainer,
    SimpleRNNConfig,
    SimpleRNNTrainer,
    TEASERGDConfig,
    TEASERGDTrainer,
    TransformerConfig,
    WarmCatalogAdapter,
)
from compresso_recsys.sequences import ItemSequences

EPOCHS = 3


class _RecordingLogger:
    """Minimal sink: the reporting contract is only ``info(str)``."""

    def __init__(self):
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(message)


class _RaisingLogger:
    def __init__(self):
        self.calls = 0

    def info(self, message: str) -> None:
        self.calls += 1
        raise RuntimeError("collector unavailable")


class _LegacyPredictionOverrides(EASE):
    """A v0.3.0-style extension with the released prediction signatures."""

    def __init__(self):
        super().__init__()
        self.hook_calls = 0
        self.predict_calls = 0

    def _predict_identified(
        self,
        source,
        *,
        k,
        exclude_seen,
        candidate_ids,
    ):
        self.hook_calls += 1
        return super()._predict_identified(
            source,
            k=k,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
        )

    def predict(
        self,
        source,
        *,
        k=100,
        batch_size=1024,
        exclude_seen=True,
        candidate_ids=None,
        show_progress=False,
    ):
        self.predict_calls += 1
        return super().predict(
            source,
            k=k,
            batch_size=batch_size,
            exclude_seen=exclude_seen,
            candidate_ids=candidate_ids,
            show_progress=show_progress,
        )


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


def _fit_elsa(*, show_progress: bool = True, logger=None, log_every_n_steps=1000):
    trainer = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            batch_size=2,
            max_output=6,
            epochs=EPOCHS,
            lr=1e-2,
            show_progress=show_progress,
            seed=7,
            log_every_n_steps=log_every_n_steps,
        ),
        logger=logger,
    )
    return trainer.fit(_interactions())


def _fit_teaser_gd(
    *, show_progress: bool = True, logger=None, log_every_n_steps=1000
):
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
            log_every_n_steps=log_every_n_steps,
        ),
        logger=logger,
    )
    return trainer.fit(_interactions(), _item_features())


def _fit_mult_dae(
    *, show_progress: bool = True, logger=None, log_every_n_steps=1000
):
    trainer = MultDAETrainer(
        MultDAEConfig(
            latent_dim=4,
            epochs=EPOCHS,
            batch_size=2,
            lr=1e-2,
            dropout=0.0,
            show_progress=show_progress,
            seed=4,
            log_every_n_steps=log_every_n_steps,
        ),
        logger=logger,
    )
    return trainer.fit(_interactions())


def _fit_mult_vae(
    *, show_progress: bool = True, logger=None, log_every_n_steps=1000
):
    trainer = MultVAETrainer(
        MultVAEConfig(
            latent_dim=3,
            hidden_dim=5,
            epochs=EPOCHS,
            batch_size=2,
            lr=1e-2,
            dropout=0.0,
            show_progress=show_progress,
            seed=4,
            log_every_n_steps=log_every_n_steps,
        ),
        logger=logger,
    )
    return trainer.fit(_interactions())


def _histories(n_items: int = 3):
    """Chronological histories over the same catalog the matrices use."""
    return ItemSequences.from_rows(
        [[0, 1, 2], [1, 2], [2, 0, 1], [0, 2]], n_items=n_items
    )


def _fit_simple_rnn(
    *, show_progress: bool = True, logger=None, log_every_n_steps=1000
):
    trainer = SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=4,
            hidden_dim=8,
            epochs=EPOCHS,
            batch_size=2,
            lr=1e-2,
            unk_dropout=0.0,
            show_progress=show_progress,
            seed=5,
            log_every_n_steps=log_every_n_steps,
        ),
        SequenceBatcher(ItemTokenizer(3), max_length=8),
        logger=logger,
    )
    return trainer.fit(_histories())


def _fit_simple_gpt(
    *, show_progress: bool = True, logger=None, log_every_n_steps=1000
):
    trainer = SimpleGPTTrainer(
        SimpleGPTConfig(
            transformer=TransformerConfig(d_model=8, n_heads=2, n_layers=1, dropout=0.0),
            epochs=EPOCHS,
            batch_size=2,
            lr=1e-2,
            unk_dropout=0.0,
            show_progress=show_progress,
            seed=5,
            log_every_n_steps=log_every_n_steps,
        ),
        SequenceBatcher(ItemTokenizer(3), max_length=8),
        logger=logger,
    )
    return trainer.fit(_histories())


TRAINERS = pytest.mark.parametrize(
    ("fit", "label", "trainer_class", "step"),
    [
        (_fit_elsa, "ELSA", ELSATrainer, "train_step"),
        (_fit_teaser_gd, "TEASERGD", TEASERGDTrainer, "_train_step"),
        (_fit_mult_dae, "MultDAE", MultDAETrainer, "_train_step"),
        (_fit_mult_vae, "MultVAE", MultVAETrainer, "_train_step"),
        # The sequential trainers draw bars the same way and were untested.
        (_fit_simple_rnn, "SimpleRNN", SimpleRNNTrainer, "_train_step"),
        (_fit_simple_gpt, "SimpleGPT", SimpleGPTTrainer, "_train_step"),
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


@TRAINERS
def test_logger_replaces_both_bars_and_reports_steps_and_epochs(
    bars,
    capsys,
    fit,
    label,
    trainer_class,
    step,
):
    logger = _RecordingLogger()

    trainer = fit(logger=logger, log_every_n_steps=1)

    assert bars == []
    assert len(trainer.history) == EPOCHS
    assert len([line for line in logger.lines if " step " in line]) >= EPOCHS
    epoch_lines = [
        line
        for line in logger.lines
        if line.startswith(f"[{label}] epoch ") and " step " not in line
    ]
    assert len(epoch_lines) == EPOCHS
    assert logger.lines[0].startswith(f"[{label}] fit started:")
    assert logger.lines[-1].startswith(f"[{label}] fit finished:")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_mult_dae_names_reconstruction_metric_in_logger_and_tqdm(bars):
    logger = _RecordingLogger()

    logged = _fit_mult_dae(
        show_progress=True,
        logger=logger,
        log_every_n_steps=1,
    )

    assert bars == []
    assert set(logged.history[0]) == {"epoch", "reconstruction_loss"}
    assert any("reconstruction_loss:" in line for line in logger.lines)

    displayed = _fit_mult_dae(show_progress=True)

    assert len(bars) == 2
    (postfix,) = bars[0].postfixes[-1]
    assert postfix == {
        "reconstruction_loss": f"{displayed.history[-1]['reconstruction_loss']:.4f}"
    }


def test_per_call_logger_overrides_constructor_logger_and_explicit_bar(bars):
    constructor_logger = _RecordingLogger()
    call_logger = _RecordingLogger()
    trainer = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            batch_size=2,
            max_output=6,
            epochs=1,
            show_progress=True,
        ),
        logger=constructor_logger,
    )

    trainer.fit(
        _interactions(),
        logger=call_logger,
        show_progress=True,
    )

    assert bars == []
    assert call_logger.lines
    assert constructor_logger.lines == []


def test_explicit_none_makes_one_call_quiet(bars, capsys):
    logger = _RecordingLogger()
    trainer = ELSATrainer(
        ELSAConfig(
            latent_dim=4,
            batch_size=2,
            max_output=6,
            epochs=1,
            show_progress=True,
        ),
        logger=logger,
    )

    trainer.fit(_interactions(), logger=None)

    assert bars == []
    assert logger.lines == []
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_raising_logger_warns_once_and_does_not_abort_fit():
    logger = _RaisingLogger()

    with pytest.warns(RuntimeWarning, match="logging disabled"):
        trainer = _fit_elsa(logger=logger, log_every_n_steps=1)

    assert len(trainer.history) == EPOCHS
    assert logger.calls == 1


def test_warning_promoted_to_error_still_does_not_abort_fit():
    import warnings

    logger = _RaisingLogger()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        trainer = _fit_elsa(logger=logger)

    assert len(trainer.history) == EPOCHS
    assert logger.calls == 1


@pytest.mark.parametrize(
    ("fit", "source"),
    [
        (_fit_elsa, _interactions),
        (_fit_mult_dae, _interactions),
        (_fit_simple_gpt, _histories),
        (_fit_teaser_gd, _interactions),
    ],
)
def test_logger_replaces_prediction_bar(bars, fit, source):
    trainer = fit(show_progress=False, log_every_n_steps=1)
    logger = _RecordingLogger()
    prediction_source = source()
    rows = (
        prediction_source.shape[0]
        if isinstance(prediction_source, csr_matrix)
        else prediction_source.n_rows
    )
    batches = (rows + 1) // 2

    predictions = trainer.predict(
        prediction_source,
        k=1,
        batch_size=2,
        exclude_seen=False,
        logger=logger,
        show_progress=True,
    )

    assert predictions.rows == rows
    assert bars == []
    assert (
        f"predict@1 started: {rows} rows | {batches} batches of 2"
        in logger.lines[0]
    )
    assert any(f"predict@1 step 1/{batches}" in line for line in logger.lines)
    assert logger.lines[-1].endswith(f"{rows} rows")


@pytest.mark.parametrize(
    ("fit", "label"),
    [
        (_fit_mult_dae, "MultDAE"),
        (_fit_simple_gpt, "SimpleGPT"),
        (_fit_teaser_gd, "TEASERGD"),
    ],
)
def test_recommend_can_override_constructor_reporting(bars, fit, label):
    constructor_logger = _RecordingLogger()
    trainer = fit(
        show_progress=True,
        logger=constructor_logger,
        log_every_n_steps=1,
    )
    constructor_logger.lines.clear()

    inherited = trainer.recommend([[0]], k=1, exclude_seen=False)

    assert inherited.valid_counts.tolist() == [1]
    assert constructor_logger.lines[0].startswith(f"[{label}] predict@1 started:")
    assert constructor_logger.lines[-1].startswith(f"[{label}] predict@1 finished:")
    assert bars == []

    constructor_logger.lines.clear()
    quiet = trainer.recommend(
        [[0]],
        k=1,
        exclude_seen=False,
        logger=None,
    )

    assert quiet.valid_counts.tolist() == [1]
    assert constructor_logger.lines == []
    assert bars == []

    call_logger = _RecordingLogger()
    redirected = trainer.recommend(
        [[0]],
        k=1,
        exclude_seen=False,
        logger=call_logger,
        show_progress=True,
    )

    assert redirected.valid_counts.tolist() == [1]
    assert call_logger.lines[0].startswith(f"[{label}] predict@1 started:")
    assert call_logger.lines[-1].startswith(f"[{label}] predict@1 finished:")
    assert constructor_logger.lines == []
    assert bars == []


def test_recommend_override_reaches_optimized_predict_and_catalog_adapter(bars):
    logger = _RecordingLogger()
    ease = EASE().fit(_interactions())

    ease_result = ease.recommend(
        [[0]],
        k=1,
        exclude_seen=False,
        logger=logger,
        show_progress=True,
    )

    assert ease_result.valid_counts.tolist() == [1]
    assert logger.lines[0].startswith("[EASE] predict@1 started:")
    assert logger.lines[-1].startswith("[EASE] predict@1 finished:")
    assert bars == []

    constructor_logger = _RecordingLogger()
    wrapped = _fit_mult_dae(
        show_progress=True,
        logger=constructor_logger,
        log_every_n_steps=1,
    )
    adapter = WarmCatalogAdapter(
        wrapped,
        train_item_ids=np.arange(6),
        catalog_item_ids=np.arange(7),
    )
    constructor_logger.lines.clear()

    adapter_result = adapter.recommend(
        [[0, 6]],
        k=1,
        exclude_seen=False,
        logger=None,
    )

    assert adapter_result.valid_counts.tolist() == [1]
    assert constructor_logger.lines == []
    assert bars == []


def test_recommend_preserves_released_prediction_override_signatures(bars):
    model = _LegacyPredictionOverrides().fit(_interactions())

    result = model.recommend([[0]], k=1, exclude_seen=False)

    assert result.valid_counts.tolist() == [1]
    assert model.hook_calls == 1
    assert model.predict_calls == 1

    logger = _RecordingLogger()
    reported = model.recommend(
        [[0]],
        k=1,
        exclude_seen=False,
        logger=logger,
    )

    assert reported.valid_counts.tolist() == [1]
    assert model.hook_calls == 1
    assert model.predict_calls == 1
    assert logger.lines[0].startswith("[_LegacyPredictionOverrides] predict@1 started:")
    assert logger.lines[-1].startswith(
        "[_LegacyPredictionOverrides] predict@1 finished:"
    )
    assert bars == []


def test_recommend_latches_a_raising_logger_across_prediction_groups():
    trainer = _fit_mult_dae(show_progress=False)
    logger = _RaisingLogger()

    with pytest.warns(RuntimeWarning, match="logging disabled") as caught:
        result = trainer.recommend(
            [[], [0]],
            k=6,
            exclude_seen=True,
            logger=logger,
        )

    assert result.valid_counts.tolist() == [6, 5]
    assert logger.calls == 1
    assert len(caught) == 1


@pytest.mark.parametrize("show_progress", ["false", 0, 1, [], object()])
def test_empty_recommend_validates_show_progress(show_progress):
    trainer = _fit_mult_dae(show_progress=False)

    with pytest.raises(ValueError, match="show_progress must be a bool or None"):
        trainer.recommend([], k=1, show_progress=show_progress)


@pytest.mark.parametrize("show_progress", ["false", 0, 1, [], object()])
def test_evaluation_validates_show_progress(show_progress):
    trainer = _fit_elsa(show_progress=False)
    source = csr_matrix((0, trainer.n_items), dtype=np.float32)

    with pytest.raises(ValueError, match="show_progress must be a bool"):
        evaluate_recommender(
            trainer,
            source=source,
            targets=source,
            metrics=[CalibratedRecall(1)],
            show_progress=show_progress,
        )


def test_logger_replaces_evaluation_bar(bars, capsys):
    trainer = _fit_elsa(show_progress=False)
    logger = _RecordingLogger()
    source = _interactions()

    result = evaluate_recommender(
        trainer,
        source=source,
        targets=source,
        metrics=[CalibratedRecall(1)],
        batch_size=2,
        show_progress=True,
        logger=logger,
        log_every_n_steps=1,
    )

    assert result.n_rows == source.shape[0]
    assert bars == []
    assert logger.lines[0].startswith(
        "[evaluation] evaluate recommender@1 started:"
    )
    assert any("evaluate recommender@1 step 1/3" in line for line in logger.lines)
    assert logger.lines[-1].startswith(
        "[evaluation] evaluate recommender@1 finished:"
    )
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


@pytest.mark.parametrize(
    "config",
    [
        ELSAConfig,
        MultDAEConfig,
        MultVAEConfig,
        SimpleGPTConfig,
        SimpleRNNConfig,
        TEASERGDConfig,
    ],
)
def test_negative_log_interval_is_rejected(config):
    with pytest.raises(ValueError, match="log_every_n_steps must be >= 0"):
        config(log_every_n_steps=-1)


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
