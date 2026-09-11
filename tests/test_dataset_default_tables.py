"""Per-dataset documentation tables use measured defaults, not raw source totals."""
import copy
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import pytest

import compresso_recsys as cr
from compresso_recsys import builder
from compresso_recsys.datasets import Steam, DBbook

ROOT = Path(__file__).parents[1]
ARCHIVE_PATH = ROOT / "docs/source/_static/dataset-default-measurements.json"


@pytest.fixture
def tables(monkeypatch):
    directory = ROOT / "examples/validation"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("dataset_default_tables", directory / "dataset_default_tables.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_non_amazon_dataset_has_a_current_default_table(tables):
    archive = json.loads(ARCHIVE_PATH.read_text())
    records = archive["datasets"]
    assert {r["dataset"] for r in records} == set(builder.DATASETS) - {"amazon2023"} == set(tables.NAMES)
    guide = (ROOT / "docs/source/datasets.rst").read_text()
    for record in records:
        assert tables.render_table(record).rstrip() in guide
        assert set(record["splits"]) == set(tables.SPLITS)
        for mode, entry in record["splits"].items():
            reason = tables.unsupported(record["dataset"], mode)
            if reason:
                assert entry == {"status": "unsupported", "reason": reason}
                continue
            assert entry["parameters"] == tables.parameters(record["dataset"], mode)[1]
            if entry["status"] == "measured":
                assert record["sources"]
                s = entry["stats"]
                for phase in ("val", "test"):
                    n, cold = s[f"{phase}_target_items"], s[f"{phase}_cold_target_items"]
                    assert 0 <= cold <= n <= s[f"{phase}_catalog_items"]
                    assert s[f"{phase}_cold_target_item_fraction"] == pytest.approx(cold / n)
                assert mode in record["metadata"]["splits"]
            else:
                pytest.fail(f"Unmeasured supported default: {record['dataset']}/{mode}")
        assert record["metadata"]["status"] == "measured"
        if record["metadata"]["status"] == "measured":
            m = record["metadata"]
            assert 0 <= m["text_ge10_items"] <= m["union_items"]
            assert m["image_url_items"] is None or 0 <= m["image_url_items"] <= m["union_items"]
        for source in record["sources"]:
            assert source["bytes"] > 0 and len(source["sha256"]) == 64
    assert "Full source cache not available" not in guide


def test_three_feature_tables_publish_verified_encoder_coverage(tables):
    archive = json.loads(ARCHIVE_PATH.read_text())
    records = {r["dataset"]: r for r in archive["datasets"]}
    for dataset, count in (("ml1m", 10), ("dbbook", 6), ("lfm2k", 8)):
        metadata = records[dataset]["metadata"]
        features = metadata["precomputed_features"]
        assert features["status"] == "measured" and features["verified_release"]
        assert features["union_items"] == metadata["union_items"]
        assert len(features["source"]["sha256"]) == 64
        assert len(features["spaces"]) == count
        for name, space in features["spaces"].items():
            assert 0 <= space["items"] <= min(space["source_items"], features["union_items"])
            assert space["dimension"] > 0
            assert space["interaction_derived"] == (dataset == "lfm2k" and name.startswith("text/"))
            assert name + ":" in tables.render_table(records[dataset])


def test_metadata_counts_align_ids_skip_missing_and_use_inclusive_ten_words(tables):
    frame = pd.DataFrame({"item_id": ["a", "b", "excluded"],
                          "entity_text": ["word " * 10, "word " * 9, "word " * 20],
                          "image_url": [" https://example.org/a ", "not a URL", "https://example.org/x"]})
    result = tables.metadata_counts(frame, {"b", "missing", "a"})
    assert result == {"union_items": 3, "image_url_items": 1, "text_ge10_items": 1}
    assert tables.metadata_counts(frame.drop(columns="image_url"), {"a"})["image_url_items"] is None
    with pytest.raises(ValueError, match="Duplicate metadata IDs"):
        tables.metadata_counts(pd.concat([frame, frame]), {"a"})


def test_missing_caches_never_construct_or_download_datasets(tables, tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "_make_dataset", lambda *args: pytest.fail("Must not load an uncached dataset"))
    for dataset in tables.NAMES:
        record = tables.measure_dataset(dataset, [tmp_path])
        rendered = tables.render_table(record)
        assert "Not measured" in rendered
        assert "Union: 0" not in rendered
        assert all(entry["status"] in {"not_measured", "unsupported"} for entry in record["splits"].values())


@pytest.mark.parametrize("mode", ["user_split", "item_split", "leave_last_out", "temporal", "official"])
def test_collector_agrees_with_checkpoint_builder(tables, tmp_path, monkeypatch, mode):
    cls, dataset = (DBbook, "dbbook") if mode == "official" else (Steam, "steam")
    rows = [{"user_id": f"u{u}", "item_id": f"i{i}", "value": 1.,
             "timestamp": 1600000000 + (i * 24 + u) * 3600,
             "source_split": "test" if i >= 10 else "train"}
            for u in range(8) for i in range(12)]
    # Official test items must be in the training vocabulary, with no pair overlap.
    if mode == "official":
        for row in rows:
            row["source_split"] = "test" if (int(row["item_id"][1:]) + int(row["user_id"][1:])) % 5 == 0 else "train"
    frame = pd.DataFrame(rows)
    if mode != "official":
        frame = pd.concat([frame, frame.iloc[[0]].assign(timestamp=1600000001)], ignore_index=True)
    metadata = pd.DataFrame({"item_id": [f"i{i}" for i in range(12)], "title": ["useful title"] * 12})

    def prepare(ds):
        ds.finish(frame, metadata)

    monkeypatch.setattr(cls, "prepare", prepare)
    options = dict(dataset=dataset, split_mode=mode, data_dir=str(tmp_path),
                   val_users=2, test_users=2, val_items=2, test_items=2,
                   min_user_support=1, item_min_support=1, temporal_period_hours=48,
                   annotation_source="none", show_progress=False)
    args, spec = builder._resolve_args(builder._build_args(**options))
    ds = builder._make_dataset(args, spec)
    stats, catalog = tables.split_measurement(args, ds, ds.get_interactions())
    path = cr.build_recsys_checkpoint(**options, checkpoint_path=str(tmp_path / "check.zip"))
    with cr.read_checkpoint(path) as root:
        payload = cr.load_recsys_split(root)
        assert stats["pairs"] == payload["x_train"].nnz
        assert stats["train_users"] == len(payload["train_user_ids"])
        assert stats["items"] == len(payload["item_ids"])
        assert set(payload["item_ids"]).issubset(catalog)
        warm = set(payload["train_item_ids"][np.unique(payload["x_train"].indices)])
        for phase in ("val", "test"):
            target = payload[f"{phase}_target_matrix"]
            source = payload[f"{phase}_source_matrix"]
            eligible = (np.diff(source.indptr) > 0) & (np.diff(target.indptr) > 0)
            ids = payload[f"{phase}_item_ids"][np.unique(target[eligible].indices)]
            assert stats[f"{phase}_users"] == len(np.unique(payload[f"{phase}_eval_user_ids"][eligible]))
            assert stats[f"{phase}_target_items"] == len(ids)
            assert stats[f"{phase}_cold_target_items"] == len(set(ids) - warm)


def test_unknown_counts_and_no_image_exposure_are_not_reported_as_zero(tables):
    records = {r["dataset"]: r for r in json.loads(ARCHIVE_PATH.read_text())["datasets"]}
    assert "Image URLs: not exposed" in tables.render_table(records["ml1m"])
    assert "≥1 image: 0" not in tables.render_table(records["ml1m"])
    assert "Window: 720 h" in tables.render_table(records["gowalla"])
    assert records["gowalla"]["splits"]["temporal"]["status"] == "measured"
    assert "Not supported" in tables.render_table(records["goodbooks"])
    failed = copy.deepcopy(records["ml1m"])
    failed["splits"]["temporal"] = {**failed["splits"]["temporal"], "status": "failed", "reason": "Empty stage"}
    assert "Build failed" in tables.render_table(failed)


def test_precomputed_coverage_matches_imported_availability(tables, tmp_path, monkeypatch):
    from compresso_recsys.checkpoint import save_recsys_split
    from compresso_recsys.embeddings import load_item_embeddings
    from compresso_recsys.multimodal import import_multimodal_embeddings
    from scipy.sparse import csr_matrix

    path = tmp_path / "features.zip"
    # Two media per artist are one available item; a zero vector remains valid.
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("resnet152.json", json.dumps({"1_1": [0., 0.], "1_2": [0., 0.],
                                                       "02_1": [1., 2.], "999_1": [3., 4.]}))
    monkeypatch.setattr(tables, "ENCODERS", {"image/resnet152": "resnet152.json"})
    monkeypatch.setitem(tables.ARCHIVE_MD5, "lfm2k", tables.hashlib.md5(path.read_bytes()).hexdigest())
    result = tables.feature_counts("lfm2k", path, {"1", "2", "missing"})
    space = result["spaces"]["image/resnet152"]
    assert space == {"items": 2, "source_items": 3, "dimension": 2, "interaction_derived": False}
    catalog = np.array(["1", "2", "missing"])
    root = tmp_path / "checkpoint"
    save_recsys_split(root, x_train=csr_matrix([[1., 1., 0.]]),
                     val_source_indices=[np.array([0])], val_target_indices=[np.array([1])],
                     test_source_indices=[np.array([0])], test_target_indices=[np.array([1])],
                     val_source_matrix=csr_matrix([[1., 0., 0.]]),
                     val_target_matrix=csr_matrix([[0., 1., 0.]]),
                     test_source_matrix=csr_matrix([[1., 0., 0.]]),
                     test_target_matrix=csr_matrix([[0., 1., 0.]]), item_ids=catalog,
                     train_user_ids=np.array(["u"]), val_eval_user_ids=np.array(["v"]),
                     test_eval_user_ids=np.array(["t"]))
    import_multimodal_embeddings(root, dataset="lfm2k", features="image/resnet152", archive_path=path)
    loaded = load_item_embeddings(root, "image/resnet152")
    assert int(loaded["available"].sum()) == space["items"]


def test_precomputed_coverage_requires_verified_release(tables, tmp_path):
    path = tmp_path / "wrong.zip"
    path.write_bytes(b"not the release")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        tables.feature_counts("ml1m", path, {"1"})


def test_requested_but_missing_features_remain_visible(tables):
    record = copy.deepcopy(json.loads(ARCHIVE_PATH.read_text())["datasets"][0])
    record["metadata"]["precomputed_features"] = {
        "status": "not_measured", "reason": "Feature archive missing; use --download-root"}
    rendered = tables.render_table(record)
    assert "Precomputed features: not measured" in rendered
    assert "use --download-root" in rendered


def test_merge_preserves_runs_and_rejects_duplicates_partial_or_changed_defaults(tables, tmp_path):
    records = json.loads(ARCHIVE_PATH.read_text())["datasets"]
    paths = [tmp_path / "first.json", tmp_path / "second.json"]
    for path, record in zip(paths, records[:2]):
        path.write_text(json.dumps({"datasets": [record], "code_sha256": {"builder.py": "fingerprint"}}))
    combined = tables.merge_archives(paths)
    assert combined["datasets"] == records[:2]
    assert len(combined["runs"]) == 2
    assert combined["runs"][0]["code_sha256"] == {"builder.py": "fingerprint"}
    with pytest.raises(ValueError, match="Duplicate datasets"):
        tables.merge_archives([paths[0], paths[0]])
    partial = json.loads(paths[0].read_text())
    partial["in_progress"] = {}
    paths[0].write_text(json.dumps(partial))
    with pytest.raises(ValueError, match="Incomplete measurement"):
        tables.merge_archives(paths)
    partial.pop("in_progress")
    partial["datasets"][0]["splits"]["user_split"]["parameters"]["seed"] += 1
    paths[0].write_text(json.dumps(partial))
    with pytest.raises(ValueError, match="Not installed defaults"):
        tables.merge_archives(paths)
