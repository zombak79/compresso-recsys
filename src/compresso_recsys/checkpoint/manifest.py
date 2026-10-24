"""The manifest: what a checkpoint says about itself, and how it is updated.

Version 2 adds provenance without changing the data layout:

- ``provenance.<stage>``: the environment that wrote each stage (library and
  package versions, git commit, Python, platform family; never a hostname);
- ``stages.data.build``: the resolved build arguments;
- ``stages.data.sources``: name, size, SHA-256 and URL of each raw input file;
- ``stages.data.content_sha256``: a digest of ``data/`` and ``embeddings/``
  that is identical across rebuilds, unlike the ZIP bytes.

Readers check ``format`` and ``version`` and refuse what they do not know, so
a future checkpoint fails with a clear message instead of a ``KeyError``, and
an update never rewrites (and silently downgrades) one. Writing into a version
1 checkpoint upgrades it and records ``upgraded_from``.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import tempfile
import zipfile


from compresso_recsys._zipsafe import safe_extract
from compresso_recsys.checkpoint.io import _zip_dir


MANIFEST_NAME = "manifest.json"
DATA_CHECKPOINT_FORMAT = "compresso.recsys.zip"
SUPPORTED_DATA_CHECKPOINT_VERSIONS = (1, 2)
CURRENT_DATA_CHECKPOINT_VERSION = 2


class CheckpointFormatError(ValueError):
    """A file is not a data checkpoint this version of compresso-recsys can read."""


def _validate_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    fmt = manifest.get("format", DATA_CHECKPOINT_FORMAT)
    if fmt != DATA_CHECKPOINT_FORMAT:
        raise CheckpointFormatError(
            f"{path} is not a compresso-recsys data checkpoint (format={fmt!r}). "
            "Model archives written by model.save() load with ModelClass.load()."
        )
    version = manifest.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version not in SUPPORTED_DATA_CHECKPOINT_VERSIONS:
        from compresso_recsys._provenance import _version

        raise CheckpointFormatError(
            f"{path} uses data-checkpoint format version {version!r}; this is compresso-recsys "
            f"{_version('compresso-recsys')}, which reads versions {SUPPORTED_DATA_CHECKPOINT_VERSIONS}. "
            "Upgrade compresso-recsys (pip install -U compresso-recsys)."
        )


def load_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST_NAME
    if not path.exists():
        return {"format": DATA_CHECKPOINT_FORMAT, "version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    path = Path(root) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _upgrade(manifest: dict[str, Any], *, existed: bool) -> None:
    """Raise the version to the current one, never lower it, and note an upgrade."""
    manifest.setdefault("format", DATA_CHECKPOINT_FORMAT)
    version = manifest.get("version", 1)
    if existed and version < CURRENT_DATA_CHECKPOINT_VERSION and "upgraded_from" not in manifest:
        manifest["upgraded_from"] = version
    manifest["version"] = max(version, CURRENT_DATA_CHECKPOINT_VERSION)


def update_stage_manifest(root: str | Path, stage: str, metadata: dict[str, Any]) -> None:
    """Record a stage's metadata and the environment that wrote it."""
    from compresso_recsys._provenance import environment

    existed = (Path(root) / MANIFEST_NAME).exists()
    manifest = load_manifest(root)
    _upgrade(manifest, existed=existed)
    manifest.setdefault("stages", {})[stage] = metadata
    manifest.setdefault("provenance", {})[stage] = environment()
    save_manifest(root, manifest)


def _refresh_content_digest(root: Path) -> None:
    """Keep ``stages.data.content_sha256`` in step with ``data/`` and ``embeddings/``."""
    from compresso_recsys._provenance import content_sha256

    path = root / MANIFEST_NAME
    if not path.exists():
        return
    manifest = load_manifest(root)
    data = manifest.get("stages", {}).get("data")
    if data is None:
        return
    digest = content_sha256(root)
    if data.get("content_sha256") == digest and manifest.get("version", 1) >= CURRENT_DATA_CHECKPOINT_VERSION:
        return
    _upgrade(manifest, existed=True)
    data["content_sha256"] = digest
    save_manifest(root, manifest)


def load_provenance(root: str | Path, *, require: bool = False) -> dict[str, Any]:
    """What a checkpoint records about how it was built; ``None`` for anything absent.

    Version 1 checkpoints record none of it. With ``require=True`` that raises
    instead of returning ``None`` values.
    """
    manifest = load_manifest(root)
    data = manifest.get("stages", {}).get("data", {})
    record = {
        "version": manifest.get("version", 1),
        "upgraded_from": manifest.get("upgraded_from"),
        "provenance": manifest.get("provenance"),
        "build": data.get("build"),
        "sources": data.get("sources"),
        "content_sha256": data.get("content_sha256"),
    }
    if require and (record["provenance"] is None or record["content_sha256"] is None):
        raise CheckpointFormatError(
            f"{root} records no provenance (data-checkpoint version {record['version']}); rebuild it with "
            "compresso-recsys >= 0.4 to record it"
        )
    return record


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
                safe_extract(zf, root, kind="data checkpoint")
            _validate_manifest(load_manifest(root), path)
        yield root
        _refresh_content_digest(root)
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
            safe_extract(zf, root, kind="data checkpoint")
        _validate_manifest(load_manifest(root), path)
        yield root
