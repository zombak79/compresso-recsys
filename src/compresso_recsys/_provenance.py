"""Provenance of data checkpoints: the build environment, raw-source hashes, a content digest.

Nothing here names the machine: checkpoints are shared, so the environment
block records versions and the platform family, never a hostname or a path.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

_PACKAGES = ("torch", "numpy", "scipy", "pandas", "pyarrow", "compresso-pytorch")
_CHUNK = 8 << 20
# Directories whose bytes define the data of a checkpoint. Models and the
# clustering stage live elsewhere and do not change the digest.
CONTENT_DIRS = ("data", "embeddings")


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _repository_root() -> Path:
    import compresso_recsys

    return Path(compresso_recsys.__file__).resolve().parents[2]


def _git_block() -> dict[str, Any] | None:
    """HEAD and dirtiness of the source checkout, or None for a wheel install or any failure."""
    try:
        root = _repository_root()
        if not (root / ".git").exists():
            return None
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=2, check=False)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True, timeout=2, check=False)
        if head.returncode != 0 or status.returncode != 0:
            return None
        return {"sha": head.stdout.strip(), "dirty": bool(status.stdout.strip())}
    except Exception:  # noqa: BLE001 - provenance must never fail a build
        return None


def environment() -> dict[str, Any]:
    """The block written to ``provenance.<stage>`` when a stage is written."""
    return {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "compresso_recsys": _version("compresso-recsys"),
        "git": _git_block(),
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": {name: _version(name) for name in _PACKAGES},
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_file_sha256(path: Path) -> str:
    """SHA-256 of a raw source file, reusing ``<file>.sha256.json`` while size and mtime match."""
    path = Path(path)
    stat = path.stat()
    sidecar = path.with_name(path.name + ".sha256.json")
    try:
        cached = json.loads(sidecar.read_text())
        if cached.get("size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            return str(cached["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    digest = file_sha256(path)
    try:
        sidecar.write_text(json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}))
    except OSError:  # a read-only data directory still gets the hash, just not the cache
        pass
    return digest


def source_records(files: list[tuple[Path, str | None]]) -> list[dict[str, Any]]:
    """``[{name, bytes, sha256, url}]`` for the raw inputs a dataset adapter read."""
    records = []
    for path, url in files:
        path = Path(path)
        records.append({"name": path.name, "bytes": path.stat().st_size,
                        "sha256": cached_file_sha256(path), "url": url})
    return records


def _content_file_digest(path: Path) -> str:
    """A digest of what a file holds rather than how it was packed.

    ``.npz`` archives carry write timestamps in their ZIP headers, so their raw
    bytes differ between two identical builds; for them the digest covers each
    member's name and uncompressed bytes instead.
    """
    if path.suffix == ".npz" and zipfile.is_zipfile(path):
        digest = hashlib.sha256()
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                digest.update(name.encode())
                digest.update(b"\0")
                digest.update(hashlib.sha256(archive.read(name)).digest())
        return "npz:" + digest.hexdigest()
    return file_sha256(path)


def content_sha256(root: str | Path) -> str | None:
    """Digest of a checkpoint's data: sorted (relative path, file digest) pairs under data/ and embeddings/.

    ``None`` when the checkpoint has no data stage. The manifest, models and the
    clustering stage are excluded, so adding a model leaves it unchanged.
    """
    root = Path(root)
    files = sorted(
        path for directory in CONTENT_DIRS if (root / directory).is_dir()
        for path in (root / directory).rglob("*") if path.is_file()
    )
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_content_file_digest(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()
