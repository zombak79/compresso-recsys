"""Reading and writing the array shapes a checkpoint stores.

Optional entries are the theme: a checkpoint may or may not carry sequences,
string ids or integer columns, and every reader here answers None rather than
raising when an entry is simply absent.
"""

from __future__ import annotations

from pathlib import Path
import zipfile

import numpy as np
from scipy.sparse import csr_matrix

from compresso_recsys.sequences import (
    ItemSequences,
    load_item_sequences,
)


def _as_obj_array(xs: list[np.ndarray]) -> np.ndarray:
    return np.array([np.asarray(x, dtype=np.int64) for x in xs], dtype=object)


def _read_obj_array(x: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.int64) for v in x.tolist()]


def _load_optional_sequences(path: Path) -> ItemSequences | None:
    """Read sequences if the checkpoint has them, else ``None``."""
    return load_item_sequences(path) if path.exists() else None


def _indices_to_csr(rows: list[np.ndarray], *, n_cols: int) -> csr_matrix:
    indptr = [0]
    indices: list[np.ndarray] = []
    for row in rows:
        row = np.asarray(row, dtype=np.int64)
        indices.append(row)
        indptr.append(indptr[-1] + int(row.size))
    flat_indices = np.concatenate(indices).astype(np.int64, copy=False) if indices else np.array([], dtype=np.int64)
    data = np.ones(flat_indices.size, dtype=np.float32)
    return csr_matrix(
        (data, flat_indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(rows), int(n_cols)),
        dtype=np.float32,
    )


def _save_optional_str_array(path: Path, values: np.ndarray | list[str] | None) -> None:
    if values is not None:
        np.save(path, np.asarray(values).astype(str))


def _load_optional_str_array(path: Path) -> np.ndarray | None:
    return np.load(path, allow_pickle=False).astype(str) if path.exists() else None


def _first_existing(*paths: Path) -> Path:
    """The first path that exists, or the first given so the default applies."""
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def _load_optional_int_array(path: Path, default: np.ndarray | None = None) -> np.ndarray:
    if path.exists():
        return np.load(path, allow_pickle=False)
    if default is None:
        return np.array([], dtype=np.int64)
    return default


def _zip_dir(root: Path, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(p for p in root.rglob("*") if p.is_file()):
            zf.write(file, file.relative_to(root).as_posix())
    tmp.replace(path)
