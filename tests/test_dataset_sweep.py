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


def test_copied_script_uses_imported_package_provenance(sweep, tmp_path, monkeypatch):
    expected = sweep.code_version()
    copied = tmp_path / "dataset_sweep.py"
    copied.write_bytes(Path(sweep.__file__).read_bytes())
    monkeypatch.setattr(sweep, "__file__", str(copied))
    assert sweep.code_version() == expected


def test_package_without_git_provenance_is_quiet(sweep, tmp_path, monkeypatch, capfd):
    installed = tmp_path / "__init__.py"
    installed.touch()
    monkeypatch.setattr(sweep.cr, "__file__", str(installed))
    result = sweep.code_version()
    assert result["commit"] is None
    assert result["script_sha256"]
    assert not capfd.readouterr().err


def test_untracked_package_does_not_claim_enclosing_repo_commit(sweep, monkeypatch):
    # Wheel installed in this checkout's ignored .venv: Git can find HEAD,
    # but it belongs to the enclosing project, not the installed package.
    repo = Path(sweep.__file__).resolve().parents[2]
    monkeypatch.setattr(sweep.cr, "__file__", str(repo / ".venv" / "__init__.py"))
    assert sweep.code_version()["commit"] is None


def test_missing_git_is_optional(sweep, monkeypatch):
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(sweep.subprocess, "check_output", unavailable)
    assert sweep.code_version()["commit"] is None


def test_amazon_category_job_selection(sweep):
    assert sweep.parse_args([]).amazon_categories == ["Toys_and_Games"]
    args = sweep.parse_args(["--datasets", "amazon2023", "dbbook",
                             "--amazon-categories", "toys", "Office_Products"])
    assert sweep.dataset_jobs(args) == [("amazon2023", "Toys_and_Games"),
                                        ("amazon2023", "Office_Products"), ("dbbook", None)]
    assert sweep.parse_args(["--amazon-category", "toys"]).amazon_categories == ["Toys_and_Games"]


@pytest.mark.parametrize("categories", [["toys", "Toys_and_Games"], [""], ["../foo"], ["all"]])
def test_amazon_rejects_duplicate_or_unsafe_categories(sweep, categories):
    with pytest.raises(SystemExit):
        sweep.parse_args(["--amazon-categories", *categories])


def test_category_cannot_be_changed_through_overrides(sweep, tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"amazon2023": {"amazon_category": "Electronics"}}))
    with pytest.raises(ValueError, match="--amazon-categories"):
        sweep.main(["--builder-overrides", str(settings), "--output", str(tmp_path / "out")])


def test_parallel_amazon_category_jobs_and_resume(sweep, tmp_path):
    # Unsupported mode exercises process scheduling/identity without downloads.
    command = [sys.executable, sweep.__file__, "--workers", "20", "--threads-per-worker", "1",
               "--datasets", "amazon2023", "--amazon-categories", "toys", "Office_Products",
               "--splits", "official", "--output", str(tmp_path)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Starting 2 dataset/category workers for 2 jobs" in result.stdout
    records = [json.loads(line) for line in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert [record["amazon_category"] for record in records] == ["Toys_and_Games", "Office_Products"]
    assert all(record["status"] == "unsupported" for record in records)
    for record in records:
        assert record["build_parameters"]["amazon_category"] == record["amazon_category"]
        assert sweep.dataset_label(record) in (tmp_path / "summary.md").read_text()
    paths = sorted(tmp_path.glob("*/result.json"))
    old_times = [path.stat().st_mtime_ns for path in paths]
    resumed = subprocess.run(command + ["--resume"], capture_output=True, text=True)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert [path.stat().st_mtime_ns for path in paths] == old_times


def test_amazon_subsets_build_independent_checkpoints_and_stats(sweep, tmp_path, monkeypatch, capsys):
    def load_subset(ds, config, *, split="full"):
        assert ds.metadata_text_fields == ("title", "features", "description", "categories")
        size = 8 if ds.category == "Toys_and_Games" else 10
        if config == ds.metadata_config:
            return pd.DataFrame({"parent_asin": [f"i{i}" for i in range(size)],
                                 "title": [f"{ds.category} product {i}" for i in range(size)]})
        assert config == ds.interactions_config
        return pd.DataFrame([
            {"user_id": f"u{u}", "parent_asin": f"i{i}", "rating": 5, "timestamp": u * size + i}
            for u in range(size) for i in range(size) if u != i
        ])

    monkeypatch.setattr(sweep.cr.AmazonReviews2023, "_load_hf_dataframe", load_subset)
    args = sweep.parse_args([
        "--datasets", "amazon2023", "--amazon-categories", "toys", "Office_Products",
        "--splits", "user_split", "item_split", "--threads-per-worker", "1", "--cutoffs", "1",
        "--data-dir", str(tmp_path / "data"), "--output", str(tmp_path / "out"),
    ])
    overrides = {"amazon2023": {"val_users": 1, "test_users": 1, "val_items": 1, "test_items": 1,
                                "min_user_support": 1, "item_min_support": 1}}
    records = []
    for dataset, category in sweep.dataset_jobs(args):
        records.extend(sweep.run_dataset(args, dataset, overrides, {}, category))
    assert len(records) == 4
    assert all(record["status"] == "complete" for record in records), records
    assert [record["loaded_data"]["n_items"] for record in records] == [8, 8, 10, 10]
    assert len({record["signature"] for record in records}) == 4
    for category in args.amazon_categories:
        assert len(list(args.output.glob(f"amazon2023-{category}-*/checkpoint.zip"))) == 2
        assert (args.data_dir / "amazon2023" / category).is_dir()
    log = capsys.readouterr().out
    for category in args.amazon_categories:
        assert f"Running amazon2023[{category}]/user_split" in log
    sweep.write_summary(args.output, records)
    report = (args.output / "summary.md").read_text()
    loaded = report.split("## Training splits")[0]
    for category in args.amazon_categories:
        assert loaded.count(f"amazon2023[{category}]") == 1
        assert report.count(f"amazon2023[{category}]") == 11
