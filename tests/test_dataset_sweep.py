"""Small offline checks for the dataset statistics and baseline sweep."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix


@pytest.fixture
def sweep():
    path = Path(__file__).parents[1] / "examples/validation/dataset_sweep.py"
    spec = importlib.util.spec_from_file_location("dataset_sweep", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repeated_events_do_not_inflate_density(sweep):
    data = pd.DataFrame({"user_id": ["u", "u", "u", "v"], "item_id": ["a", "a", "b", "b"],
                         "timestamp": [1, 2, 3, None]})
    stats = sweep.frame_stats(data)
    assert stats["n_events"] == 4
    assert stats["n_unique_pairs"] == 3
    assert stats["sparsity"] == .25
    assert stats["repeat_event_fraction"] == .25
    assert stats["timestamp_coverage"] == .75


def test_counts_distinguish_rows_from_users_and_inactive_items(sweep):
    matrix = csr_matrix([[1, 0, 0], [0, 1, 0]])
    result = sweep.matrix_stats(matrix, ["same-user", "same-user"])
    assert result["n_users"] == 1 and result["n_rows"] == 2
    assert result["active_items"] == 2 and result["n_items"] == 3
    assert result["sparsity"] == pytest.approx(2 / 3)


def test_phase_alignment_and_cold_targets(sweep):
    # Global ID order and phase order intentionally differ.
    split = {"item_ids": np.array(["b", "future", "a", "cold"]),
             "train_item_ids": np.array(["a", "b"]),
             "x_train": csr_matrix([[1, 0], [1, 1]])}
    for phase in ("val", "test"):
        split[f"{phase}_item_ids"] = np.array(["cold", "a", "b"])
        split[f"{phase}_source_matrix"] = csr_matrix([[0, 1, 0]])
        split[f"{phase}_target_matrix"] = csr_matrix([[0, 0, 1]])
        split[f"{phase}_eval_user_ids"] = np.array(["u"])
    args = sweep.parse_args(["--cutoffs", "1", "--neighbors", "1", "--baselines", "popularity", "itemknn"])
    results = sweep.evaluate_baselines(split, args)
    for name in ("popularity", "itemknn"):
        assert results[name]["status"] == "ok"
        assert results[name]["test"]["metrics"]["recall@1"] == 1.
    assert "future" not in split["val_item_ids"]
    aligned = sweep.align_columns(split["x_train"], split["train_item_ids"], split["item_ids"])
    np.testing.assert_array_equal(aligned.toarray(), [[0, 0, 1, 0], [1, 0, 1, 0]])


def test_cutoff_exclusions_and_knn_resource_skip(sweep):
    ids = np.array(["a", "b"])
    split = {"item_ids": ids, "train_item_ids": ids, "x_train": csr_matrix([[1, 0]])}
    for phase in ("val", "test"):
        split.update({f"{phase}_item_ids": ids, f"{phase}_source_matrix": csr_matrix([[1, 0]]),
                      f"{phase}_target_matrix": csr_matrix([[0, 1]]),
                      f"{phase}_eval_user_ids": np.array(["u"])})
    args = sweep.parse_args(["--cutoffs", "2", "--knn-max-items", "1", "--baselines", "popularity", "itemknn"])
    result = sweep.evaluate_baselines(split, args)
    assert result["popularity"]["test"]["status"] == "skipped"
    assert result["popularity"]["test"]["excluded_rows"] == 1
    assert result["itemknn"]["status"] == "skipped"


def test_sweep_builds_reports_and_resumes_offline(sweep, tmp_path):
    data = tmp_path / "data" / "dbbook"
    data.mkdir(parents=True)
    with zipfile.ZipFile(data / "dbbook_interaction_data.zip", "w") as z:
        z.writestr("interaction_data/train.tsv", "".join(
            f"{user}\t{item}\t1\n" for user in range(8) for item in range(8) if item != user))
        z.writestr("interaction_data/test.tsv", "".join(f"{user}\t{user}\t1\n" for user in range(8)))
        z.writestr("interaction_data/DBbook_Items_DBpedia_mapping.tsv",
                   "DBbook_ItemID\tname\tDBpedia_uri\n" + "".join(f"{i}\tBook {i}\turi:{i}\n" for i in range(8)))
    overrides = tmp_path / "overrides.json"
    overrides.write_text(json.dumps({"dbbook": {"val_users": 1, "test_users": 1, "min_user_support": 1,
                                               "val_items": 1, "test_items": 1}}))
    argv = ["--datasets", "dbbook", "--data-dir", str(data.parent), "--output", str(tmp_path / "out"),
            "--builder-overrides", str(overrides), "--cutoffs", "1", "--neighbors", "2"]
    command = [sys.executable, sweep.__file__, "--threads-per-worker", "1"]
    first = subprocess.run(command + argv, capture_output=True, text=True)
    assert first.returncode == 0, first.stdout + first.stderr
    paths = list((tmp_path / "out").glob("*/result.json"))
    assert len(paths) == 5
    results = [json.loads(path.read_text()) for path in paths]
    assert sum(result["status"] == "complete" for result in results) == 3
    assert sum(result["status"] == "unsupported" for result in results) == 2
    assert all("stats" in result for result in results if result["status"] == "complete")
    assert (tmp_path / "out" / "summary.md").exists()
    old_times = [path.stat().st_mtime_ns for path in paths]
    resumed = subprocess.run(command + argv + ["--resume"], capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert [path.stat().st_mtime_ns for path in paths] == old_times
    repeated = subprocess.run(command + argv, capture_output=True, text=True)
    assert repeated.returncode != 0
    assert "FileExistsError" in repeated.stderr


def test_unsupported_modes_are_explicit(sweep):
    assert sweep.unsupported("goodbooks", "temporal")
    assert sweep.unsupported("ml1m", "official")
    assert sweep.unsupported("ml1m", "leave_last_out") is None


def test_two_parallel_dataset_workers_and_default_first_pass(sweep, tmp_path):
    assert sweep.parse_args([]).baselines == ["popularity"]
    result = subprocess.run([
        sys.executable, sweep.__file__, "--workers", "2", "--threads-per-worker", "1",
        "--datasets", "goodbooks", "lfm2k", "--splits", "temporal", "--output", str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert {record["dataset"] for record in records} == {"goodbooks", "lfm2k"}
    assert all(record["status"] == "unsupported" for record in records)
