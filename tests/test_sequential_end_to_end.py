"""One builder call to one ``compare_models`` call, across both model families.

Every other sequential test covers one link: the structure, the batcher, the
contract, the model. This one is about the chain holding together — a
chronological split producing both views of the same events, a matrix model and a
sequential model each reading the view it understands, and the statistics layer
comparing them without knowing that either representation exists.

The data is synthetic but the pipeline is not: the real ``leave_last_out``
builder, a real checkpoint round trip, and both real models.

The signal is deliberately one that *only* order carries. Every user reads a
window of consecutive items from a cycle ``0 -> 1 -> ... -> 19 -> 0``, and the
held-out target is the item that comes next. From a source window ``{s..s+6}``
the two plausible candidates are ``s-1`` and ``s+7``, and they are
indistinguishable by co-occurrence: both sit one step outside a window endpoint
and both share the same number of training windows with it. Only the direction of
time separates them, so a set-based model cannot do better than picking an end,
and one that has learned any consistent asymmetry picks the wrong one every time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compresso_recsys.builder import _build_args, _build_leave_last_out_split
from compresso_recsys.checkpoint import load_recsys_split, save_recsys_split
from compresso_recsys.evaluation import evaluate_recommender
from compresso_recsys.metrics import CalibratedRecall, NDCG
from compresso_recsys.models import (
    EASE,
    EASEConfig,
    SimpleRNNConfig,
    SimpleRNNTrainer,
)
from compresso_recsys.stats import compare_models

N_ITEMS = 20
HISTORY = 8
N_USERS = 120

SEQUENCE_KEYS = (
    "x_train_sequences",
    "train_source_sequences",
    "val_source_sequences",
    "test_source_sequences",
)


def _cycle_events() -> pd.DataFrame:
    """User ``u`` reads ``HISTORY`` consecutive items starting at ``u % N_ITEMS``.

    Ids are zero-padded so that sorting them lexicographically -- which is what
    the builder does to fix the column order -- agrees with the cycle. Without
    the padding, ``i10`` sorts before ``i2`` and the catalog indices become an
    unreadable permutation of the sequence.
    """
    return pd.DataFrame(
        [
            {
                "user_id": f"u{user:03d}",
                "item_id": f"i{(user + step) % N_ITEMS:02d}",
                "value": 1.0,
                "timestamp": 1_000_000 + step * 90_000,
            }
            for user in range(N_USERS)
            for step in range(HISTORY)
        ]
    )


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    """A ``leave_last_out`` split, through a real checkpoint round trip."""
    payload = _build_leave_last_out_split(
        _build_args(dataset="goodbooks", split_mode="leave_last_out"),
        _cycle_events(),
    )
    root = tmp_path_factory.mktemp("checkpoint")
    save_recsys_split(
        root,
        item_ids=payload["item_ids"],
        x_train=payload["x_train"],
        train_source_matrix=payload["train_source_matrix"],
        train_target_matrix=payload["train_target_matrix"],
        val_source_indices=payload["val_holdout"]["source_indices"],
        val_target_indices=payload["val_holdout"]["target_indices"],
        test_source_indices=payload["test_holdout"]["source_indices"],
        test_target_indices=payload["test_holdout"]["target_indices"],
        **{key: payload[key] for key in SEQUENCE_KEYS},
    )
    return load_recsys_split(root)


@pytest.fixture(scope="module")
def results(split):
    """Both families fitted and evaluated against one target matrix."""
    sequential = SimpleRNNTrainer(
        SimpleRNNConfig(
            embedding_dim=16,
            hidden_dim=32,
            epochs=40,
            batch_size=32,
            lr=0.01,
            max_length=None,
            show_progress=False,
            seed=0,
        )
    ).fit(split["x_train_sequences"])
    matrix = EASE(EASEConfig(l2=1.0)).fit(split["x_train"])

    metrics = [CalibratedRecall(1), CalibratedRecall(5), NDCG(5)]
    targets = split["test_target_matrix"]
    ids = split["test_user_ids"]
    return {
        "ease": evaluate_recommender(
            matrix,
            source=split["test_source_matrix"],
            targets=targets,
            metrics=metrics,
            sample_ids=ids,
        ),
        "simple_rnn": evaluate_recommender(
            sequential,
            source=split["test_source_sequences"],
            targets=targets,
            metrics=metrics,
            sample_ids=ids,
        ),
    }


# --------------------------------------------------------------------------
# the two views
# --------------------------------------------------------------------------


def test_the_builder_gives_both_views_of_the_same_events(split):
    sequences = split["test_source_sequences"]
    matrix = split["test_source_matrix"]

    assert sequences.n_rows == matrix.shape[0] == N_USERS
    assert sequences.n_items == matrix.shape[1] == N_ITEMS
    for row in range(N_USERS):
        assert set(sequences.row(row).tolist()) == set(matrix[row].indices.tolist())


def test_only_the_sequence_view_knows_which_item_comes_next(split):
    """The wrap-around rows, where sorting is visibly destructive.

    User 15 reads 15, 16, 17, 18, 19, 0, 1 and the answer is 2. A CSR row renders
    that history as 0, 1, 15, 16, 17, 18, 19, whose *last* entry is 19 -- so the
    matrix view does not merely lose the ordering, it points at the wrong item.
    """
    sequences = split["test_source_sequences"]
    matrix = split["test_source_matrix"]
    targets = split["test_target_matrix"]

    for user in (15, 18, 19):
        history = sequences.row(user).tolist()
        expected = [(user + step) % N_ITEMS for step in range(HISTORY - 1)]
        assert history == expected
        assert targets[user].indices.tolist() == [(user + HISTORY - 1) % N_ITEMS]
        # The same events, sorted, end somewhere else entirely.
        assert matrix[user].indices.tolist() == sorted(history)
        assert matrix[user].indices[-1] != history[-1]


# --------------------------------------------------------------------------
# what each family can see
# --------------------------------------------------------------------------


def test_the_sequential_model_reads_the_cycle_off_the_last_item(results):
    assert results["simple_rnn"]["calibrated_recall@1"] == pytest.approx(1.0)


def test_the_matrix_model_finds_the_right_pair_and_the_wrong_end(results):
    """Why the comparison below measures order rather than model quality.

    EASE has the target in its top 5 for every user, so it identifies the two
    candidates outside the window perfectly. It then puts the *past* neighbour
    first almost every time, because co-occurrence is symmetric and carries no
    direction. That is a systematic failure, not noise, and it is the only thing
    the sequential model is being credited with fixing here.
    """
    ease = results["ease"]

    assert ease["calibrated_recall@5"] == pytest.approx(1.0)
    assert ease["calibrated_recall@1"] < 0.2
    # Rank two for nearly everyone: NDCG is then 1 / log2(3) per user.
    assert ease["ndcg@5"] == pytest.approx(1.0 / np.log2(3.0), abs=0.05)


def test_both_families_are_scored_against_the_same_targets(results):
    """What lets comparison accept the pair, and what identifiers cannot prove."""
    assert (
        results["ease"].target_fingerprint
        == results["simple_rnn"].target_fingerprint
        is not None
    )


# --------------------------------------------------------------------------
# one comparison over two source representations
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def report(results):
    return compare_models(
        results,
        metrics=["ndcg@5", "calibrated_recall@1"],
        reference="ease",
        alternative="greater",
        n_resamples=999,
    )


def test_the_comparison_finds_the_order_signal(report):
    for comparison in report.comparisons:
        assert comparison.baseline == "ease"
        assert comparison.candidate == "simple_rnn"
        assert comparison.difference > 0.3, comparison.metric
        assert comparison.ci_low > 0.0, comparison.metric
        assert comparison.significant, comparison.metric
        assert comparison.adjusted_p_value < 0.01, comparison.metric


def test_users_are_the_unit_of_analysis(report):
    """One row per user here, so units and rows agree -- and both are pinned."""
    for comparison in report.comparisons:
        assert comparison.n_samples == comparison.n_units == N_USERS
        assert comparison.n_nonzero <= N_USERS


def test_the_report_is_a_single_family(report):
    assert len(report.comparisons) == 2
    assert report.reference == "ease"
    assert set(report.model_names) == {"ease", "simple_rnn"}
    assert report.correction == "holm"
    # Two hypotheses, so correction must have moved at least one p-value.
    assert any(
        comparison.adjusted_p_value > comparison.p_value
        for comparison in report.comparisons
    )


def test_a_metric_that_cannot_separate_them_is_reported_as_a_tie(results):
    """Both models have the target in the top 5 for every user.

    Every paired difference is exactly zero, which is a finding rather than a
    failure: it says the two models are level at that cutoff. The warning exists
    so a p-value of 1.0 is not mistaken for a weak result.
    """
    with pytest.warns(RuntimeWarning, match="mean paired difference of exactly zero"):
        tied = compare_models(
            results,
            metrics=["calibrated_recall@5"],
            reference="ease",
            n_resamples=999,
        )

    (comparison,) = tied.comparisons
    assert comparison.difference == 0.0
    assert comparison.n_nonzero == 0
    assert comparison.tie_rate == 1.0
    assert comparison.p_value == 1.0
    assert not comparison.significant
