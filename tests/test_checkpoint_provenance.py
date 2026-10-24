"""Data-checkpoint manifest v2: provenance, version checks, and safe extraction."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compresso_recsys as cr
from compresso_recsys import _provenance, builder
from compresso_recsys.checkpoint import (
    CheckpointFormatError,
    load_manifest,
    load_provenance,
    read_checkpoint,
    update_checkpoint,
)
from compresso_recsys.datasets import RecSysDataset


class _ToyDataset(RecSysDataset):
    """Reads ``ratings.csv`` from its root; writes it there on first use."""

    name = "toyprov"
    url = "https://example.org/data/ratings.csv"
    default_text_fields = ("entity_text",)

    def __init__(self, data_dir="data", *, metadata_text_fields=None, min_entity_text_words=0) -> None:
        super().__init__(data_dir)

    def prepare(self) -> None:
        path = self.root / "ratings.csv"
        if not path.exists():
            rng = np.random.default_rng(0)
            rows = [(f"u{u}", f"i{i}", 5.0, 1_000 + k)
                    for u in range(80) for k, i in enumerate(rng.choice(40, size=12, replace=False))]
            pd.DataFrame(rows, columns=["user_id", "item_id", "value", "timestamp"]).to_csv(path, index=False)
        frame = pd.read_csv(path, dtype={"user_id": str, "item_id": str})
        self._interactions = frame
        items = sorted(frame["item_id"].unique())
        self._item_metadata = pd.DataFrame({"item_id": items, "entity_text": [f"item {i}" for i in items]})


@pytest.fixture
def toy(monkeypatch, tmp_path):
    monkeypatch.setitem(builder.DATASETS, "toyprov", builder.DatasetSpec(
        _ToyDataset, str(tmp_path / "default.zip"), seed=0, val_users=10, test_users=10,
        min_value_to_keep=None, min_entity_text_words=0,
    ))
    data_dir = tmp_path / "data"

    def build(name: str = "toy.zip", **kwargs) -> Path:
        path = tmp_path / name
        cr.build_recsys_checkpoint(dataset="toyprov", data_dir=str(data_dir), checkpoint_path=str(path),
                                   split_mode="user_split", annotation_source="none", show_progress=False,
                                   **kwargs)
        return path

    build.data_dir = data_dir
    return build


def _manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("manifest.json"))


def _rewrite(path: Path, edit) -> None:
    """Rewrite a checkpoint's manifest without going through update_checkpoint."""
    with zipfile.ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    manifest = json.loads(entries["manifest.json"])
    edit(manifest)
    entries["manifest.json"] = json.dumps(manifest).encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)


def test_build_writes_manifest_v2_with_provenance(toy):
    manifest = _manifest(toy())
    assert manifest["version"] == 2 and "upgraded_from" not in manifest
    env = manifest["provenance"]["data"]
    assert env["compresso_recsys"] == _provenance._version("compresso-recsys")
    assert env["created_at"].endswith("+00:00") and env["python"]
    assert set(env["packages"]) == {"torch", "numpy", "scipy", "pandas", "pyarrow", "compresso-pytorch"}
    assert "host" not in json.dumps(env).lower()


def test_build_records_resolved_build_args(toy):
    build = _manifest(toy(seed=7))["stages"]["data"]["build"]
    assert build["seed"] == 7 and build["split_mode"] == "user_split" and build["dataset"] == "toyprov"
    assert build["val_users"] == 10
    assert not {"data_dir", "checkpoint_path", "show_progress"} & set(build)  # nothing machine-local


def test_sources_hashed_with_size_and_sha256(toy):
    path = toy()
    sources = _manifest(path)["stages"]["data"]["sources"]
    source = toy.data_dir / "toyprov" / "ratings.csv"
    assert sources == [{"name": "ratings.csv", "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "url": "https://example.org/data/ratings.csv"}]


def test_source_hash_sidecar_reused_when_unchanged_and_refreshed_when_changed(toy, monkeypatch):
    toy()
    calls = []
    original = _provenance.file_sha256
    monkeypatch.setattr(_provenance, "file_sha256", lambda path: calls.append(path) or original(path))
    toy("again.zip")
    assert not [p for p in calls if p.name == "ratings.csv"]
    source = toy.data_dir / "toyprov" / "ratings.csv"
    source.write_text(source.read_text() + "u0,i39,5.0,99999\n")
    toy("changed.zip")
    assert [p for p in calls if p.name == "ratings.csv"]


def test_content_sha256_identical_across_rebuilds(toy):
    first, second = toy("a.zip"), toy("b.zip")
    digest = _manifest(first)["stages"]["data"]["content_sha256"]
    assert digest and digest == _manifest(second)["stages"]["data"]["content_sha256"]


def test_content_sha256_ignores_npz_write_times(tmp_path):
    """The raw .npz bytes carry ZIP timestamps; the digest covers what they hold."""
    first, second = tmp_path / "a" / "data", tmp_path / "b" / "data"
    for directory, stamp in ((first, (2020, 1, 1, 0, 0, 0)), (second, (2024, 6, 6, 6, 6, 6))):
        directory.mkdir(parents=True)
        with zipfile.ZipFile(directory / "x.npz", "w") as archive:
            archive.writestr(zipfile.ZipInfo("x.npy", date_time=stamp), b"same bytes")
    assert (first / "x.npz").read_bytes() != (second / "x.npz").read_bytes()
    assert _provenance.content_sha256(first.parent) == _provenance.content_sha256(second.parent)


def test_content_sha256_ignores_models_and_follows_embeddings(toy):
    path = toy()
    before = _manifest(path)["stages"]["data"]["content_sha256"]
    with read_checkpoint(path) as root:
        split = cr.load_recsys_split(root)
    cr.EASE().fit(split["x_train"]).save_to_checkpoint(path, "ease")
    assert _manifest(path)["stages"]["data"]["content_sha256"] == before
    with update_checkpoint(path) as root:
        cr.save_item_embeddings(root, "text/toy", item_ids=split["item_ids"],
                                embeddings=np.ones((split["item_ids"].size, 2), dtype=np.float32))
    assert _manifest(path)["stages"]["data"]["content_sha256"] != before


def test_git_block_is_null_outside_a_repository_and_never_fails(toy, monkeypatch, tmp_path):
    monkeypatch.setattr(_provenance, "_repository_root", lambda: tmp_path / "nowhere")
    assert _manifest(toy("a.zip"))["provenance"]["data"]["git"] is None
    monkeypatch.setattr(_provenance, "_repository_root", lambda: Path(__file__).resolve().parents[1])
    (tmp_path / "nowhere").mkdir()

    def broken(*args, **kwargs):
        raise OSError("no git")

    monkeypatch.setattr(_provenance.subprocess, "run", broken)
    assert _manifest(toy("b.zip"))["provenance"]["data"]["git"] is None


def _as_v1(manifest: dict) -> None:
    manifest["version"] = 1
    manifest.pop("provenance", None)
    for key in ("build", "sources", "content_sha256"):
        manifest["stages"]["data"].pop(key, None)


def test_read_v1_checkpoint_without_provenance(toy):
    path = toy()
    _rewrite(path, _as_v1)
    with read_checkpoint(path) as root:
        assert cr.load_recsys_split(root)["x_train"].nnz > 0
        record = load_provenance(root)
        assert record["version"] == 1
        assert record["provenance"] is None and record["content_sha256"] is None
        with pytest.raises(CheckpointFormatError, match="records no provenance"):
            load_provenance(root, require=True)


def test_writing_into_v1_upgrades_without_touching_existing_stage_metadata(toy):
    path = toy()
    _rewrite(path, _as_v1)
    stage_before = _manifest(path)["stages"]["data"]
    with update_checkpoint(path):
        pass
    manifest = _manifest(path)
    assert manifest["version"] == 2 and manifest["upgraded_from"] == 1
    assert "provenance" not in manifest  # the data stage was not rewritten, so it has none
    stage_after = dict(manifest["stages"]["data"])
    assert stage_after.pop("content_sha256")
    assert stage_after == stage_before


def test_unknown_future_version_raises_clear_error(toy):
    path = toy()
    _rewrite(path, lambda manifest: manifest.update(version=3))
    before = path.read_bytes()
    with pytest.raises(CheckpointFormatError, match=r"reads versions \(1, 2\)"):
        with read_checkpoint(path):
            pass
    with pytest.raises(CheckpointFormatError, match="version 3"):
        with update_checkpoint(path):
            pass
    assert path.read_bytes() == before


@pytest.mark.parametrize("edit", [
    lambda manifest: manifest.update(version="2"),
    lambda manifest: manifest.update(format="compresso.recsys.model"),
])
def test_non_integer_version_and_wrong_format_raise(toy, edit):
    path = toy()
    _rewrite(path, edit)
    with pytest.raises(CheckpointFormatError):
        with read_checkpoint(path):
            pass


def test_a_model_archive_is_not_a_data_checkpoint(tmp_path):
    model = tmp_path / "ease.ckpt"
    from scipy.sparse import csr_matrix

    cr.EASE().fit(csr_matrix(np.eye(5, dtype=np.float32))).save(model)
    with pytest.raises(CheckpointFormatError, match="ModelClass.load"):
        with read_checkpoint(model):
            pass


@pytest.mark.parametrize("name", ["../evil", "/abs", "C:\\evil", "a/../../evil", "\\evil"])
def test_read_and_update_reject_zip_slip(tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("manifest.json", json.dumps({"format": "compresso.recsys.zip", "version": 1}))
        archive.writestr(name, b"x")
    for opener in (read_checkpoint, update_checkpoint):
        with pytest.raises(ValueError, match="data checkpoint contains an unsafe path"):
            with opener(path):
                pass
    assert not (tmp_path / "evil").exists()


def test_persistence_still_rejects_zip_slip(tmp_path):
    path = tmp_path / "model.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../evil", b"x")
    with pytest.raises(ValueError, match="model checkpoint contains an unsafe path"):
        cr.EASE.load(path)


def test_load_manifest_of_a_fresh_build_matches_load_provenance(toy):
    with read_checkpoint(toy()) as root:
        record = load_provenance(root, require=True)
        assert record["version"] == 2
        assert record["content_sha256"] == _provenance.content_sha256(root)
        assert record["build"] == load_manifest(root)["stages"]["data"]["build"]
