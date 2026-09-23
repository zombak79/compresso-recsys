"""The manifest: what a checkpoint says about itself, and how it is updated."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import tempfile
import zipfile


from compresso_recsys.checkpoint.io import _zip_dir


MANIFEST_NAME = "manifest.json"


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return {"format": "compresso.recsys.zip", "version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def update_stage_manifest(root: str | Path, stage: str, metadata: dict[str, Any]) -> None:
    manifest = load_manifest(root)
    manifest.setdefault("format", "compresso.recsys.zip")
    manifest.setdefault("version", 1)
    manifest.setdefault("stages", {})[stage] = metadata
    save_manifest(root, manifest)


def save_json(root: str | Path, relpath: str, data: dict[str, Any]) -> Path:
    path = Path(root) / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_json(root: str | Path, relpath: str) -> dict[str, Any]:
    return json.loads((Path(root) / relpath).read_text(encoding="utf-8"))


@contextmanager
def update_checkpoint(path: str | Path) -> Iterator[Path]:
    """Extract a zip checkpoint to a temp dir, let caller edit it, then rewrite it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        if path.exists():
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(root)
        yield root
        _zip_dir(root, path)


@contextmanager
def read_checkpoint(path: str | Path) -> Iterator[Path]:
    """Extract a zip checkpoint to a read-only temp workspace."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(root)
        yield root
