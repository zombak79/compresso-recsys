"""The on-disk shape of a model checkpoint, pinned as an explicit contract.

``tests/test_model_persistence.py`` covers round-trips: save with this code,
load with this code, get the same predictions back. A round-trip cannot catch a
change that rewrites the writer and the reader together -- the trip still
closes, but an archive written by an earlier release no longer loads. And
:class:`ModelCheckpointReader` refuses a mismatched
:data:`~compresso_recsys.persistence.MODEL_CHECKPOINT_VERSION` outright, so
there is no migration path to soften that.

These tests pin what is actually written: which entries a checkpoint contains,
and which keys each JSON entry carries. They exist because
``_save_checkpoint_state`` is implemented once per trainer, several of those
implementations are near-identical, and consolidating them must not quietly
change the format. Note that the sequential trainers are not interchangeable
here: ``simple_bidirectional`` writes two keys the other three do not.

A deliberate format change should update these expectations *and* bump
``MODEL_CHECKPOINT_VERSION``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable
import zipfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from compresso_recsys import ItemSequences
from compresso_recsys.models import (
    ELSAConfig,
    ELSATrainer,
    ItemTokenizer,
    MultDAEConfig,
    MultDAETrainer,
    MultVAEConfig,
    MultVAETrainer,
    SASRecConfig,
    SASRecTrainer,
    SequenceBatcher,
    SimpleBidirectionalTransformerConfig,
    SimpleBidirectionalTransformerTrainer,
    SimpleGPTConfig,
    SimpleGPTTrainer,
    SimpleRNNConfig,
    SimpleRNNTrainer,
    TEASERGDConfig,
    TEASERGDTrainer,
    TransformerConfig,
)
from compresso_recsys.persistence import (
    MODEL_CHECKPOINT_FORMAT,
    MODEL_CHECKPOINT_VERSION,
)

N_ITEMS = 6

INTERACTIONS = csr_matrix(
    np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 1, 0, 0, 1, 1],
        ],
        dtype=np.float32,
    )
)

ITEM_FEATURES = np.array(
    [[1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1], [1, 0, 1]],
    dtype=np.float32,
)

# Entries every checkpoint carries: the manifest and config written by save(),
# the identity block written by _save_checkpoint_common_state, and the module
# state dictionary.
COMMON_ENTRIES = frozenset(
    {
        "manifest.json",
        "config.json",
        "identity/source_item_ids.json",
        "state/model.pt",
        "state/trainer.json",
    }
)

# Sequential trainers additionally persist their tokenizer.
TOKENIZER_ENTRIES = frozenset(
    {"state/tokenizer.json", "state/tokenizer_item_ids.json"}
)

TOKENIZER_KEYS = frozenset({"n_items", "n_reserved", "special_tokens"})


def _sequences() -> ItemSequences:
    return ItemSequences.from_rows(
        [[0, 1, 2], [1, 3], [2, 4, 5], [0, 5]], n_items=N_ITEMS
    )


def _batcher(padding: str) -> SequenceBatcher:
    tokenizer = ItemTokenizer(
        N_ITEMS, item_ids=np.array([f"item-{i}" for i in range(N_ITEMS)])
    )
    return SequenceBatcher(tokenizer, max_length=3, padding=padding)


def _elsa():
    return ELSATrainer(
        ELSAConfig(
            latent_dim=4, batch_size=2, max_output=4, epochs=1,
            show_progress=False, seed=7,
        )
    ).fit(INTERACTIONS)


def _mult_vae():
    return MultVAETrainer(
        MultVAEConfig(
            latent_dim=3, hidden_dim=5, dropout=0.2, epochs=1, batch_size=2,
            lr=1e-2, show_progress=False, seed=7,
        )
    ).fit(INTERACTIONS)


def _mult_dae():
    return MultDAETrainer(
        MultDAEConfig(
            latent_dim=4, dropout=0.2, epochs=1, batch_size=2, lr=1e-2,
            show_progress=False, seed=7,
        )
    ).fit(INTERACTIONS)


def _teaser_gd():
    return TEASERGDTrainer(
        TEASERGDConfig(
            epochs=1, batch_size=2, max_output=4, lr=0.01, show_progress=False,
            include_popularity=False, coefficient_regularization_samples=16,
            seed=3,
        )
    ).fit(INTERACTIONS, ITEM_FEATURES)


def _sasrec():
    # Left padding is SASRec's architecture, not a preference.
    return SASRecTrainer(
        SASRecConfig(
            d_model=8, n_blocks=1, n_heads=2, max_history_length=3, epochs=1,
            batch_size=2, show_progress=False, seed=0,
        ),
        _batcher("left"),
    ).fit(_sequences())


def _simple_gpt():
    return SimpleGPTTrainer(
        SimpleGPTConfig(
            transformer=TransformerConfig(
                d_model=8, n_heads=2, n_layers=1, dropout=0.0
            ),
            epochs=1, batch_size=2, show_progress=False, seed=0,
        ),
        _batcher("right"),
    ).fit(_sequences())


def _simple_rnn():
    return SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=8, hidden_dim=12, epochs=1, batch_size=2,
            show_progress=False, seed=0,
        ),
        _batcher("right"),
    ).fit(_sequences())


def _simple_bidirectional():
    return SimpleBidirectionalTransformerTrainer(
        SimpleBidirectionalTransformerConfig(
            transformer=TransformerConfig(
                d_model=8, n_heads=2, n_layers=1, dropout=0.0
            ),
            tie_embeddings=False, epochs=1, batch_size=8,
            lr_schedule="constant", unk_dropout=0.0, show_progress=False,
            seed=0,
        ),
        _batcher("right"),
    ).fit(_sequences())


@dataclass(frozen=True)
class Layout:
    build: Callable[[], object]
    model_type: str
    entries: frozenset[str]
    trainer_keys: frozenset[str]


LAYOUTS: dict[str, Layout] = {
    "elsa": Layout(
        _elsa, "elsa_trainer", COMMON_ENTRIES,
        frozenset({"history", "input_dim"}),
    ),
    "mult_vae": Layout(
        _mult_vae, "mult_vae_trainer", COMMON_ENTRIES,
        frozenset({"history", "n_items", "updates"}),
    ),
    "mult_dae": Layout(
        _mult_dae, "mult_dae_trainer", COMMON_ENTRIES,
        frozenset({"history", "n_items"}),
    ),
    "teaser_gd": Layout(
        _teaser_gd, "teaser_gd_trainer",
        COMMON_ENTRIES | {
            "state/training_features.npy",
            "state/train_item_indices.npy",
            "state/train_item_mask.npy",
            "catalog/state.json",
            "catalog/candidate_features.npy",
            "catalog/candidate_item_ids.json",
            "catalog/source_item_ids.json",
            "catalog/source_popularity.npy",
        },
        frozenset({
            "history", "n_items", "n_features", "n_train_items",
            "n_training_users", "feature_names", "training_feature_storage",
        }),
    ),
    "sasrec": Layout(
        _sasrec, "sasrec_trainer", COMMON_ENTRIES | TOKENIZER_ENTRIES,
        frozenset({"history", "max_length"}),
    ),
    "simple_gpt": Layout(
        _simple_gpt, "simple_gpt_trainer", COMMON_ENTRIES | TOKENIZER_ENTRIES,
        frozenset({"history", "max_length"}),
    ),
    "simple_rnn": Layout(
        _simple_rnn, "simple_rnn_trainer", COMMON_ENTRIES | TOKENIZER_ENTRIES,
        frozenset({"history", "max_length"}),
    ),
    "simple_bidirectional": Layout(
        _simple_bidirectional, "simple_bidirectional_transformer_trainer",
        COMMON_ENTRIES | TOKENIZER_ENTRIES,
        # Two keys no other sequential trainer writes. A consolidated
        # _save_checkpoint_state has to keep making room for them.
        frozenset({
            "history", "max_length", "padding",
            "trained_with_explicit_targets",
        }),
    ),
}


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, zipfile.Path]:
    """Save every trainer once; the layout tests only read the archives."""
    directory = tmp_path_factory.mktemp("layout")
    saved = {}
    for name, layout in LAYOUTS.items():
        path = directory / f"{name}.zip"
        layout.build().save(path)
        saved[name] = path
    return saved


def _read_json(path, entry: str):
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read(entry))


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_checkpoint_contains_exactly_the_expected_entries(name, checkpoints):
    with zipfile.ZipFile(checkpoints[name]) as archive:
        assert set(archive.namelist()) == set(LAYOUTS[name].entries)


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_trainer_state_carries_exactly_the_expected_keys(name, checkpoints):
    state = _read_json(checkpoints[name], "state/trainer.json")
    assert set(state) == set(LAYOUTS[name].trainer_keys)


@pytest.mark.parametrize(
    "name",
    sorted(n for n, l in LAYOUTS.items() if "state/tokenizer.json" in l.entries),
)
def test_tokenizer_state_carries_exactly_the_expected_keys(name, checkpoints):
    state = _read_json(checkpoints[name], "state/tokenizer.json")
    assert set(state) == set(TOKENIZER_KEYS)


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_manifest_declares_the_pinned_format_and_model_type(name, checkpoints):
    manifest = _read_json(checkpoints[name], "manifest.json")
    assert manifest == {
        "format": MODEL_CHECKPOINT_FORMAT,
        "version": MODEL_CHECKPOINT_VERSION,
        "model_type": LAYOUTS[name].model_type,
        "optimizer_included": False,
    }


def test_checkpoint_version_is_pinned():
    """Bumping the version is a deliberate act, so it fails here first."""
    assert MODEL_CHECKPOINT_VERSION == 1
    assert MODEL_CHECKPOINT_FORMAT == "compresso.recsys.model"
