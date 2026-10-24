"""Run the code of docs/source/adding-a-metric.rst, so the tutorial cannot go stale."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from compresso import SRPTensor
from scipy.sparse import csr_matrix

from compresso_recsys.evaluation import evaluate_ranked_predictions

PAGE = Path(__file__).resolve().parents[1] / "docs" / "source" / "adding-a-metric.rst"
_BLOCK = re.compile(r"\.\. code-block:: python\n\n((?:(?:[ \t]+.*)?\n)+)")


def _blocks() -> list[str]:
    import textwrap

    return [textwrap.dedent(block) for block in _BLOCK.findall(PAGE.read_text())]


@pytest.fixture(scope="module")
def namespace():
    blocks = _blocks()
    assert len(blocks) == 3  # the class, its use, its test
    scope: dict = {}
    exec(compile(blocks[0], str(PAGE), "exec"), scope)
    return scope


def test_the_documented_test_passes(namespace):
    exec(compile(_blocks()[2], str(PAGE), "exec"), dict(namespace))


def test_the_documented_metric_works_end_to_end(namespace):
    """The usage block needs a checkpoint; run the same calls on a small split instead."""
    import compresso_recsys as cr

    rng = np.random.default_rng(0)
    x_train = csr_matrix((rng.random((60, 80)) < 0.2).astype(np.float32))
    source = csr_matrix((rng.random((20, 80)) < 0.2).astype(np.float32))
    targets = csr_matrix(((rng.random((20, 80)) < 0.1) & (source.toarray() == 0)).astype(np.float32))
    split = {"x_train": x_train, "test_source_matrix": source, "test_target_matrix": targets,
             "test_eval_user_ids": np.arange(20)}
    usage = _blocks()[1].replace("import compresso_recsys as cr\n", "")
    scope = dict(namespace, cr=cr, split=split,
                 ease=cr.EASE(cr.EASEConfig(l2=10.0)).fit(x_train),
                 popularity=cr.PopularityBaseline().fit(x_train))
    exec(compile(usage, str(PAGE), "exec"), scope)
    popular = scope["results"]["Pop"].metrics["popularity_percentile@10"]
    assert popular > scope["results"]["EASE"].metrics["popularity_percentile@10"] - 1e-9
    assert set(scope["report"].to_frame()["metric"]) == {"ndcg@20", "popularity_percentile@20"}


def test_documented_edge_cases(namespace):
    metric_class = namespace["PopularityPercentile"]
    predictions = SRPTensor(cols=torch.tensor([[3, 2, 1], [0, 1, 2]]),
                            vals=torch.tensor([[3.0, 2.0, 1.0]] * 2), shape=(2, 4))
    empty = csr_matrix((2, 4), dtype=np.float32)
    targets = csr_matrix(np.array([[1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.float32))
    metric = metric_class([1, 3], item_counts=[1, 2, 3, 4])
    result = evaluate_ranked_predictions(predictions=predictions, targets=targets, metrics=[metric], batch_size=1)
    assert np.isfinite(list(result.metrics.values())).all()
    whole = evaluate_ranked_predictions(predictions=predictions, targets=targets,
                                        metrics=[metric_class([1, 3], item_counts=[1, 2, 3, 4])])
    assert result.metrics == pytest.approx(whole.metrics)
    with pytest.raises(ValueError):
        evaluate_ranked_predictions(predictions=predictions, targets=targets,
                                    metrics=[metric_class(5, item_counts=[1, 2, 3, 4])])
    assert evaluate_ranked_predictions(predictions=predictions, targets=empty,
                                       metrics=[metric_class(1, item_counts=[1, 2, 3, 4])]).n_scored_rows == 0
