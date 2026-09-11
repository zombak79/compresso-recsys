"""Offline parity and constraint tests for Amazon support profiling."""
import importlib.util
import json
import argparse
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from compresso_recsys import builder
from compresso_recsys.datasets.base import RecSysDataset


@pytest.fixture
def profile():
    path = Path(__file__).parents[1] / "examples/validation/amazon_profile.py"
    spec = importlib.util.spec_from_file_location("amazon_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def interactions():
    rng = np.random.default_rng(13)
    rows = []
    for user in range(45):
        for item in rng.choice(24, rng.integers(5, 20), replace=False):
            rows.append((f"u{user}", f"i{item}", 1., 1600000000000 + rng.integers(0, 100) * 3600000))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "value", "timestamp"])


@pytest.mark.parametrize("us,it", [(2, 2), (3, 2), (4, 1), (5, 1), (5, 5), (8, 10), (15, 5), (100, 100)])
def test_core_matches_real_preprocessing(profile, interactions, us, it):
    graph = profile.CoreGraph(interactions)
    actual = interactions.iloc[graph.filter(us, it)].reset_index(drop=True)
    expected = RecSysDataset.preprocess_interactions_for_recsys(
        interactions, min_value_to_keep=None, user_min_support=us, item_min_support=it)
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize("us,it", [(2, 2), (3, 2), (4, 1), (5, 1), (5, 3), (8, 3), (10, 5)])
def test_temporal_counter_matches_real_builder(profile, interactions, us, it):
    args, _ = builder._resolve_args(builder._build_args(dataset="amazon2023", split_mode="temporal",
        min_user_support=us, item_min_support=it, temporal_period_hours=20))
    graph = profile.TemporalGraph(interactions, 20)
    try:
        expected = graph.counts(us, it)
    except ValueError as exc:
        with pytest.raises(ValueError, match="no users after support filtering"):
            builder._build_temporal_split(args, interactions)
        assert "no users after support filtering" in str(exc)
        return
    payload = builder._build_temporal_split(args, interactions)
    actual = profile.payload_stats(payload)
    assert {key: actual[key] for key in ("users", "items", "pairs")} == {
        key: expected[key] for key in ("users", "items", "pairs")}


def test_temporal_keeps_inherited_inactive_catalog_columns(profile):
    # Old items cannot disappear from the cap merely by having no test events.
    rows = [(f"u{u}", f"i{i}", 1., float(t) * 3600) for u in range(10)
            for i, t in enumerate([0, 1, 2, 20, 40, 60, 80, 90])]
    frame = pd.DataFrame(rows, columns=["user_id", "item_id", "value", "timestamp"])
    graph = profile.TemporalGraph(frame, 20)
    stats = graph.counts(5, 5)
    assert stats["items"] == 8
    assert stats["users"] == 10


def test_boundary_search_returns_feasible_nonempty_settings(profile):
    def measure(us, it):
        return {"users": max(0, 140 - us * it), "items": max(0, 60 - it * 3), "pairs": 1000 // it}
    candidates, tested = profile.boundary_candidates(measure, 40, 30, user_supports=(5, 10))
    assert all(c["counts"]["users"] <= 40 and c["counts"]["items"] <= 30 for c in candidates)
    by_user = {us: min(c["item_min_support"] for c in candidates if c["min_user_support"] == us)
               for us in (5, 10)}
    assert by_user == {5: 20, 10: 10}
    assert len(tested) < 40


def test_caps_and_coverage_objective(profile):
    assert profile.limits("Books") == (499999, 100000)
    assert profile.limits("Electronics") == (499999, 100000)
    assert profile.limits("Toys_and_Games") == (100000, 20000)
    assert profile.size_score({"users": 60000, "items": 15000, "pairs": 500000}) > profile.size_score(
        {"users": 95000, "items": 3000, "pairs": 1000000})
    assert len(profile.CATEGORIES) == len(set(profile.CATEGORIES)) == 33


def test_user_rich_selection_retains_a_useful_catalog_and_eval_users(profile):
    def candidate(users, items, pairs, support=2, evaluated=2000):
        return {"parameters": {"min_user_support": support}, "stats": {
            "users": users, "items": items, "pairs": pairs,
            "val_users": evaluated, "test_users": evaluated}}

    fashion = candidate(35714, 10217, 68051, evaluated=3551)
    item_heavy = candidate(744, 15198, 15739, support=15, evaluated=12)
    tiny_catalog = candidate(3595, 362, 6749, evaluated=359)
    assert profile.candidate_quality(fashion) > profile.candidate_quality(item_heavy)
    assert profile.candidate_quality(fashion) > profile.candidate_quality(tiny_catalog)
    # Keep longer histories if they already meet the size and evaluation aims.
    healthy = candidate(97533, 18043, 728532, support=5)
    short = candidate(99999, 19999, 900000, support=2)
    assert profile.candidate_quality(healthy) > profile.candidate_quality(short)
    # Requesting a large partition is not enough: use actual eligible users.
    inadequate = candidate(95000, 19000, 2000000, support=5, evaluated=999)
    assert profile.candidate_quality(fashion) > profile.candidate_quality(inadequate)
    assert profile.size_score({"users": 20000, "items": 10000, "pairs": 50000}) > profile.size_score(
        {"users": 1000, "items": 10000, "pairs": 50000})


def test_support_floors_are_protocol_specific(profile):
    assert profile.user_supports("user_split")[:3] == (2, 3, 4)
    assert min(profile.user_supports("leave_last_out")) == 4
    assert min(profile.user_supports("item_split")) == 2
    assert min(profile.user_supports("temporal")) == 2


def test_metadata_and_all_ratings_with_global_temporal_endpoints(profile, tmp_path):
    ds = profile.AmazonReviews2023(data_dir=tmp_path / "data", category="All_Beauty", show_progress=False)
    root = ds.root / "mcauley"
    root.mkdir(parents=True)
    frame = pd.DataFrame([(f"u{u}", f"i{i}", (i % 5) + 1, 3600 * (i + 1))
                          for u in range(10) for i in range(6)],
                         columns=["user_id", "parent_asin", "rating", "timestamp"])
    # Ineligible users still set the real temporal boundaries; missing metadata does not.
    extras = pd.DataFrame([("early", "i0", 1, 0), ("late", "i0", 1, 360000),
                           ("absent", "no-meta", 5, 720000)], columns=frame.columns)
    frame = pd.concat([frame, extras], ignore_index=True)
    frame.to_csv(root / "All_Beauty.csv.gz", index=False)
    pd.DataFrame({"parent_asin": [f"i{i}" for i in range(6)]}).to_json(
        root / "meta_All_Beauty.jsonl.gz", orient="records", lines=True)
    directory = tmp_path / "prepared"
    prepared, manifest = profile.prepare(ds, directory, cached_only=True)
    assert set(prepared.value) == {1.}
    assert len(prepared) == 62
    assert prepared.timestamp.min() == 0
    assert prepared.timestamp.max() == 360000
    assert "no-meta" not in set(prepared.item_id)
    assert manifest["metadata_eligible_events"] == 62
    assert manifest["raw_valid_rating_events"] == 63
    assert all(len(s["sha256"]) == 64 for s in manifest["sources"])
    again, reloaded = profile.prepare(ds, directory, cached_only=True)
    pd.testing.assert_frame_equal(prepared, again)
    assert manifest == reloaded
    # Endpoint-only users are eventually removed by support, not sampled.
    assert profile.CoreGraph(prepared).counts(5, 1)["users"] == 10


def test_streaming_prepare_keeps_short_histories_and_rebuilds_old_policy(profile, tmp_path, monkeypatch):
    ds = profile.AmazonReviews2023(data_dir=tmp_path / "data", category="All_Beauty", show_progress=False)
    root = ds.root / "mcauley"
    root.mkdir(parents=True)
    rows = [(f"u{n}", f"i{i}", 1., 100 + i) for n in (2, 3, 4) for i in range(n)]
    rows.extend([("early", "i0", 2., 0), ("late", "i0", 5., 1000), ("missing", "absent", 1., 2000)])
    raw = pd.DataFrame(rows, columns=["user_id", "parent_asin", "rating", "timestamp"])
    raw.to_csv(root / "All_Beauty.csv.gz", index=False)
    pd.DataFrame({"parent_asin": ["i0", "i1", "i2", "i3"]}).to_json(
        root / "meta_All_Beauty.jsonl.gz", orient="records", lines=True)
    original_chunks = profile.rating_chunks
    def small_chunks(path):
        for frame in original_chunks(path):
            for start in range(0, len(frame), 2):
                yield frame.iloc[start:start + 2]
    monkeypatch.setattr(profile, "rating_chunks", small_chunks)
    directory = tmp_path / "prepared"
    # Hybrid preparation must never load the full parquet back into RAM.
    with monkeypatch.context() as context:
        context.setattr(profile.pd, "read_parquet", lambda *a, **k: pytest.fail("Full frame loaded locally"))
        frame, manifest = profile.prepare(ds, directory, cached_only=True, load_frame=False)
        assert frame is None
        assert manifest["prepared"]["rows"] == 11
        assert not list(directory.glob("prepare-*"))
    frame, _ = profile.load_prepared(directory, "All_Beauty")
    assert profile.CoreGraph(frame).counts(2, 1) == {"users": 3, "items": 4, "pairs": 9}
    assert (frame.timestamp.min(), frame.timestamp.max()) == (0, 1000)
    assert set(frame.value) == {1.}
    # A same-schema cache from a stricter policy must be REBUILT, not silently
    # reused or rejected in a way that leaves the coordinator stuck forever.
    manifest["min_user_support_upper_bound"] = 5
    profile.write_json(directory / "input.json", manifest)
    again, new_manifest = profile.prepare(ds, directory, cached_only=True)
    assert new_manifest["min_user_support_upper_bound"] == 2
    pd.testing.assert_frame_equal(frame, again)


def test_temporal_screening_rejects_empty_stages(profile, interactions):
    graph = profile.TemporalGraph(interactions, 20)
    with pytest.raises(ValueError, match="no users"):
        graph.counts(1000, 1000)
    candidates, measured = profile.boundary_candidates(graph.counts, 100, 100, user_supports=(1000,))
    assert not candidates and measured


def test_verify_all_four_splits_and_report(profile, interactions, tmp_path):
    graph = profile.CoreGraph(interactions)
    counts = graph.counts(5, 1)
    entries = {}
    for split in profile.SPLITS:
        measured = profile.TemporalGraph(interactions, 20).counts(5, 1) if split == "temporal" else counts
        candidate = {"min_user_support": 5, "item_min_support": 1, "counts": measured}
        entry = profile.verify_candidate(interactions, graph, candidate, category="All_Beauty",
                                          split=split, period_hours=20, seed=42)
        assert entry["stats"]["val_users"] > 0
        assert entry["stats"]["test_users"] > 0
        assert entry["parameters"]["min_user_support"] == 5
        entries[split] = {"status": "verified", **entry}
    profile.write_json(tmp_path / "All_Beauty/result.json", {"category": "All_Beauty", "splits": entries})
    profile.write_report(tmp_path)
    assert "| All_Beauty | temporal | verified | 5/1 |" in (tmp_path / "summary.md").read_text()
    json.dumps(entries, allow_nan=False)


def write_prepared(profile, frame, root, category="All_Beauty"):
    directory = root / category
    directory.mkdir(parents=True)
    frame.to_parquet(directory / "input.parquet", index=False)
    manifest = {"schema": profile.SCHEMA, "all_ratings": True, "min_entity_text_words": 0,
                "min_user_support_upper_bound": min(profile.USER_SUPPORTS), "sources": []}
    manifest = profile.seal_prepared(directory, category, manifest, len(frame))
    return directory, manifest


def test_prepared_input_validates_checksum_category_and_values(profile, interactions, tmp_path):
    directory, manifest = write_prepared(profile, interactions, tmp_path)
    actual, loaded = profile.load_prepared(directory, "All_Beauty")
    pd.testing.assert_frame_equal(actual, interactions)
    assert loaded == manifest
    with pytest.raises(ValueError, match="category"):
        profile.load_prepared(directory, "Books")
    (directory / "input.parquet").write_bytes((directory / "input.parquet").read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        profile.load_prepared(directory, "All_Beauty")
    interactions["value"] = 4.
    interactions.to_parquet(directory / "input.parquet", index=False)
    profile.seal_prepared(directory, "All_Beauty", manifest, len(interactions))
    with pytest.raises(ValueError, match="binary"):
        profile.load_prepared(directory, "All_Beauty")


def test_temporal_float_timestamp_conversion_does_not_mutate_read_only_input(profile):
    values = np.array([1600000000000., 1600003600000.])
    values.flags.writeable = False
    series = pd.Series(values, copy=False)
    result = builder._timestamps_in_seconds(series)
    np.testing.assert_array_equal(result, [1600000000., 1600003600.])
    np.testing.assert_array_equal(series, values)


def test_prepared_profiling_never_accesses_raw_sources(profile, interactions, tmp_path, monkeypatch):
    write_prepared(profile, interactions, tmp_path / "inputs")
    monkeypatch.setattr(profile, "prepare", lambda *a, **k: pytest.fail("Remote work must not access raw sources"))
    args = argparse.Namespace(output=tmp_path / "results", prepared_root=tmp_path / "inputs",
                              seed=42, temporal_period_hours=20, splits=profile.SPLITS)
    result = profile.profile_category("All_Beauty", args)
    assert all(entry["status"] == "verified" for entry in result["splits"].values())
    assert result["context"]["package_sha256"] == profile.RUN_CODE["package_sha256"]
    monkeypatch.setattr(profile, "boundary_candidates", lambda *a, **k: pytest.fail("Verified results must resume"))
    assert profile.profile_category("All_Beauty", args) == result


def test_remote_worker_processes_ready_inputs_offline(profile, interactions, tmp_path):
    # Stretch the fixture to the default 339-day windows without changing its stages.
    interactions.timestamp = 1600000000000 + (interactions.timestamp - 1600000000000) * (8136 / 20)
    directory, manifest = write_prepared(profile, interactions, tmp_path / "inputs")
    profile.write_json(directory / "READY.json", {"sha256": manifest["prepared"]["sha256"]})
    worker = Path(profile.__file__).with_name("amazon_remote_worker.py")
    run = subprocess.run([sys.executable, str(worker), "--input-root", str(tmp_path / "inputs"),
                          "--output", str(tmp_path / "results"), "--categories", "All_Beauty", "--workers", "1"],
                         text=True, capture_output=True, timeout=90)
    assert run.returncode == 0, run.stdout + run.stderr
    state = json.loads((tmp_path / "results/remote-state.json").read_text())
    assert state["status"] == "complete"
    assert state["completed"] == ["All_Beauty"]
    assert state["categories"] == ["All_Beauty"]
    assert (tmp_path / "results/All_Beauty/profile.log").exists()


def test_remote_worker_flags_inputs_over_memory_budget_without_loading(profile, interactions, tmp_path):
    directory, manifest = write_prepared(profile, interactions, tmp_path / "inputs")
    profile.write_json(directory / "READY.json", {"sha256": manifest["prepared"]["sha256"]})
    worker = Path(profile.__file__).with_name("amazon_remote_worker.py")
    run = subprocess.run([sys.executable, str(worker), "--input-root", str(tmp_path / "inputs"),
                          "--output", str(tmp_path / "results"), "--categories", "All_Beauty",
                          "--memory-budget-gib", "0.000001"], text=True, capture_output=True, timeout=30)
    assert run.returncode == 1, run.stdout + run.stderr
    state = json.loads((tmp_path / "results/remote-state.json").read_text())
    assert state["status"] == "needs_attention"
    assert "Estimated working memory" in state["failures"]["All_Beauty"]
    assert not (tmp_path / "results/All_Beauty/profile.log").exists()
