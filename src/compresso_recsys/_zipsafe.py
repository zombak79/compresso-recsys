"""Extracting ZIP archives without letting a member write outside the target."""

from __future__ import annotations

import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath


def check_members(archive: zipfile.ZipFile, root: Path, *, kind: str) -> None:
    """Refuse members that are absolute, climb with ``..``, or resolve outside ``root``."""
    resolved_root = root.resolve()
    for member in archive.infolist():
        name = member.filename
        posix = PurePosixPath(name)
        windows = PureWindowsPath(name)
        unsafe = (
            posix.is_absolute()
            or ".." in posix.parts
            or ".." in windows.parts
            or bool(windows.drive)
            or name.startswith("\\")
        )
        if not unsafe:
            target = (resolved_root / name).resolve()
            unsafe = target != resolved_root and resolved_root not in target.parents
        if unsafe:
            raise ValueError(f"{kind} contains an unsafe path: {name!r}")


def safe_extract(archive: zipfile.ZipFile, root: Path, *, kind: str) -> None:
    """Check every member, then extract them all."""
    check_members(archive, root, kind=kind)
    archive.extractall(root)
