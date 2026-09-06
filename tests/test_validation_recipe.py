"""Offline integration checks for the documented benchmark runner."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.sparse import csr_matrix


def recipe():
    path = Path(__file__).parents[1] / "examples/validation/benchmark.py"
    spec = importlib.util.spec_from_file_location("benchmark_recipe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("model", ["ease", "multdae", "multvae"])
def test_training_recipe(model):
    runner = recipe()
    rng = np.random.default_rng(7)
    train = csr_matrix((rng.random((12, 120)) < 0.15).astype(np.float32))
    source = csr_matrix(([1., 1.], ([0, 1], [0, 1])), shape=(2, 120))
    target = csr_matrix(([1., 1.], ([0, 1], [2, 3])), shape=(2, 120))
    ids = np.arange(120).astype(str)
    split = {"x_train": train, "train_item_ids": ids}
    for phase in ("val", "test"):
        split.update({f"{phase}_item_ids": ids,
                      f"{phase}_source_matrix": source,
                      f"{phase}_target_matrix": target})
    result = runner.train(split, model, epochs=[1], l2_values=[10.],
                          device="cpu", seed=7)
    assert len(result["trials"]) == 1
    for key in ("calibrated_recall@20", "calibrated_recall@50", "ndcg@100"):
        assert 0 <= result["test"][key] <= 1


def test_build_protocol(monkeypatch, tmp_path):
    runner = recipe()
    calls = []
    monkeypatch.setattr(runner, "build_recsys_checkpoint", lambda **kw: calls.append(kw))
    runner.build("ml20m", tmp_path / "split.zip", tmp_path)
    assert calls[0]["val_users"] == calls[0]["test_users"] == 10_000
    assert calls[0]["eval_draws"] == 1
    assert calls[0]["min_entity_text_words"] == 0
    assert calls[0]["min_value_to_keep"] == 4
