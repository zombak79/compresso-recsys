from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
import json
import shutil
import tempfile
import zipfile

import numpy as np
import pandas as pd
from compresso.clustering import load_cluster_graph, save_cluster_graph
from compresso.clustering.types import SparseClusterSet
from scipy.sparse import csr_matrix, load_npz, save_npz


MANIFEST_NAME = "manifest.json"
SPLIT_DIR = "data"
CLUSTERING_DIR = "clustering"
CLUSTER_GRAPH_NAME = "graph.json"

__all__ = [
    "update_checkpoint",
    "read_checkpoint",
    "load_manifest",
    "save_manifest",
    "update_stage_manifest",
    "save_json",
    "load_json",
    "save_recsys_split",
    "load_recsys_split",
    "save_cluster_graph_stage",
    "load_cluster_graph_stage",
]


def _as_obj_array(xs: list[np.ndarray]) -> np.ndarray:
    return np.array([np.asarray(x, dtype=np.int64) for x in xs], dtype=object)


def _read_obj_array(x: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.int64) for v in x.tolist()]


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


def save_recsys_split(
    root: str | Path,
    *,
    item_ids: np.ndarray,
    x_train: csr_matrix,
    train_item_ids: np.ndarray | list[str] | None = None,
    val_item_ids: np.ndarray | list[str] | None = None,
    test_item_ids: np.ndarray | list[str] | None = None,
    val_source_indices: list[np.ndarray],
    val_target_indices: list[np.ndarray],
    test_source_indices: list[np.ndarray],
    test_target_indices: list[np.ndarray],
    train_source_matrix: csr_matrix | None = None,
    train_target_matrix: csr_matrix | None = None,
    val_source_matrix: csr_matrix | None = None,
    val_target_matrix: csr_matrix | None = None,
    test_source_matrix: csr_matrix | None = None,
    test_target_matrix: csr_matrix | None = None,
    train_user_ids: np.ndarray | list[str] | None = None,
    val_user_ids: np.ndarray | list[str] | None = None,
    test_user_ids: np.ndarray | list[str] | None = None,
    val_eval_user_ids: np.ndarray | list[str] | None = None,
    test_eval_user_ids: np.ndarray | list[str] | None = None,
    train_item_indices: np.ndarray | None = None,
    val_item_indices: np.ndarray | None = None,
    test_item_indices: np.ndarray | None = None,
    entity_tag_matrix: csr_matrix | None = None,
    tag_names: np.ndarray | list[str] | None = None,
    entity_metadata: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Write the split stage of a checkpoint.

    Item partitions
    ---------------
    ``train_item_indices``, ``val_item_indices`` and ``test_item_indices`` are
    positions into ``item_ids`` naming the items **that phase introduces**, not
    the items it may score. They partition the catalog by first appearance:

    - ``user_split``: training spans every item and the later phases introduce
      none, so the train partition is the full range and val/test are empty.
    - ``item_split``: three disjoint partitions, the val/test ones being the
      cold items held out of training.
    - ``leave_last_out``: nothing is held out of the catalog. An item lands in
      the val or test partition only when every one of its occurrences falls in
      a held-out tail, so on dense data both partitions are empty and on sparse
      data they hold the genuinely new items.
    - ``temporal``: each phase introduces the items first seen in its window,
      so the partitions are consecutive ranges of the growing catalog.

    An empty partition therefore means "this phase introduces no new items",
    which is not the same as "this phase has no candidates". The candidate space
    of a phase is ``{phase}_item_ids``, which defaults to ``item_ids`` when not
    given. Callers that select feature or metadata rows for a phase should index
    with that phase's ``*_item_ids`` (or the union of partitions up to it),
    because mirroring ``train_item_indices`` into a later phase silently yields
    an empty selection for splits that hold no items out.

    Passing ``None`` for a partition omits its file, and
    :func:`load_recsys_split` then falls back to the whole catalog for train and
    to an empty array for val/test. Prefer writing all three explicitly.
    """
    root = Path(root)
    data_dir = root / SPLIT_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    n_items = len(item_ids)
    train_item_ids = np.asarray(
        item_ids if train_item_ids is None else train_item_ids
    ).astype(str)
    val_item_ids = np.asarray(
        item_ids if val_item_ids is None else val_item_ids
    ).astype(str)
    test_item_ids = np.asarray(
        item_ids if test_item_ids is None else test_item_ids
    ).astype(str)
    train_source_matrix = x_train if train_source_matrix is None else train_source_matrix
    train_target_matrix = train_source_matrix if train_target_matrix is None else train_target_matrix
    val_source_matrix = (
        _indices_to_csr(val_source_indices, n_cols=len(val_item_ids))
        if val_source_matrix is None
        else val_source_matrix
    )
    val_target_matrix = (
        _indices_to_csr(val_target_indices, n_cols=len(val_item_ids))
        if val_target_matrix is None
        else val_target_matrix
    )
    test_source_matrix = (
        _indices_to_csr(test_source_indices, n_cols=len(test_item_ids))
        if test_source_matrix is None
        else test_source_matrix
    )
    test_target_matrix = (
        _indices_to_csr(test_target_indices, n_cols=len(test_item_ids))
        if test_target_matrix is None
        else test_target_matrix
    )

    pairs = (
        ("train", train_source_matrix, train_target_matrix, train_item_ids),
        ("validation", val_source_matrix, val_target_matrix, val_item_ids),
        ("test", test_source_matrix, test_target_matrix, test_item_ids),
    )
    for name, source, target, ids in pairs:
        if source.shape != target.shape:
            raise ValueError(f"{name} source and target matrix shapes must match")
        if source.shape[1] != len(ids):
            raise ValueError(
                f"{name} matrix columns must match {name} item IDs length"
            )
    if x_train.shape != train_source_matrix.shape:
        raise ValueError("x_train shape must match train source matrix shape")

    save_npz(data_dir / "train_source_matrix.npz", train_source_matrix.tocsr())
    save_npz(data_dir / "train_target_matrix.npz", train_target_matrix.tocsr())
    save_npz(data_dir / "val_source_matrix.npz", val_source_matrix.tocsr())
    save_npz(data_dir / "val_target_matrix.npz", val_target_matrix.tocsr())
    save_npz(data_dir / "test_source_matrix.npz", test_source_matrix.tocsr())
    save_npz(data_dir / "test_target_matrix.npz", test_target_matrix.tocsr())
    # Backward-compatible training matrix; temporal checkpoints store the
    # source/target union here while retaining each side separately above.
    save_npz(data_dir / "train_matrix.npz", x_train.tocsr())
    np.save(data_dir / "train_item_ids.npy", train_item_ids)
    np.save(data_dir / "val_item_ids.npy", val_item_ids)
    np.save(data_dir / "test_item_ids.npy", test_item_ids)
    np.savez_compressed(
        data_dir / "split.npz",
        item_ids=np.asarray(item_ids).astype(str),
        val_source_indices=_as_obj_array(val_source_indices),
        val_target_indices=_as_obj_array(val_target_indices),
        test_source_indices=_as_obj_array(test_source_indices),
        test_target_indices=_as_obj_array(test_target_indices),
    )
    _save_optional_str_array(data_dir / "train_user_ids.npy", train_user_ids)
    _save_optional_str_array(data_dir / "val_user_ids.npy", val_user_ids)
    _save_optional_str_array(data_dir / "test_user_ids.npy", test_user_ids)
    _save_optional_str_array(data_dir / "val_eval_user_ids.npy", val_eval_user_ids)
    _save_optional_str_array(data_dir / "test_eval_user_ids.npy", test_eval_user_ids)
    if train_item_indices is not None:
        np.save(data_dir / "train_item_indices.npy", np.asarray(train_item_indices, dtype=np.int64))
    if val_item_indices is not None:
        np.save(data_dir / "val_item_indices.npy", np.asarray(val_item_indices, dtype=np.int64))
    if test_item_indices is not None:
        np.save(data_dir / "test_item_indices.npy", np.asarray(test_item_indices, dtype=np.int64))
    if entity_tag_matrix is not None:
        if entity_tag_matrix.shape[0] != len(item_ids):
            raise ValueError("entity_tag_matrix rows must match item_ids length")
        if tag_names is None:
            raise ValueError("tag_names must be provided when entity_tag_matrix is provided")
        tag_names_arr = np.asarray(tag_names).astype(str)
        if entity_tag_matrix.shape[1] != len(tag_names_arr):
            raise ValueError("tag_names length must match entity_tag_matrix columns")
        save_npz(data_dir / "entity_tags.npz", entity_tag_matrix.tocsr().astype(np.float32))
        np.save(data_dir / "tag_names.npy", tag_names_arr)
    if entity_metadata is not None:
        meta = entity_metadata.copy()
        if "item_id" not in meta.columns:
            raise ValueError("entity_metadata must contain an item_id column")
        meta["item_id"] = meta["item_id"].astype(str)
        meta = meta.set_index("item_id", drop=False).reindex(np.asarray(item_ids).astype(str)).reset_index(drop=True)
        meta.to_csv(data_dir / "entity_metadata.csv", index=False)
    update_stage_manifest(root, "data", metadata or {})


def load_recsys_split(root: str | Path) -> dict[str, Any]:
    """Read the split stage of a checkpoint.

    See :func:`save_recsys_split` for what ``*_item_indices`` mean: they are the
    items each phase *introduces*, so they are empty for phases that hold no
    items out, while ``*_item_ids`` give the candidate space and default to
    ``item_ids``.

    For checkpoints written before every partition was stored explicitly, a
    missing ``train_item_indices.npy`` loads as the full catalog range and
    missing val/test files load as empty arrays.
    """
    root = Path(root)
    split = np.load(root / SPLIT_DIR / "split.npz", allow_pickle=True)
    tags_path = root / SPLIT_DIR / "entity_tags.npz"
    tag_names_path = root / SPLIT_DIR / "tag_names.npy"
    metadata_path = root / SPLIT_DIR / "entity_metadata.csv"
    train_item_indices_path = root / SPLIT_DIR / "train_item_indices.npy"
    val_item_indices_path = root / SPLIT_DIR / "val_item_indices.npy"
    test_item_indices_path = root / SPLIT_DIR / "test_item_indices.npy"
    train_user_ids_path = root / SPLIT_DIR / "train_user_ids.npy"
    val_user_ids_path = root / SPLIT_DIR / "val_user_ids.npy"
    test_user_ids_path = root / SPLIT_DIR / "test_user_ids.npy"
    val_eval_user_ids_path = root / SPLIT_DIR / "val_eval_user_ids.npy"
    test_eval_user_ids_path = root / SPLIT_DIR / "test_eval_user_ids.npy"
    train_item_ids_path = root / SPLIT_DIR / "train_item_ids.npy"
    val_item_ids_path = root / SPLIT_DIR / "val_item_ids.npy"
    test_item_ids_path = root / SPLIT_DIR / "test_item_ids.npy"
    train_source_matrix_path = root / SPLIT_DIR / "train_source_matrix.npz"
    train_target_matrix_path = root / SPLIT_DIR / "train_target_matrix.npz"
    val_source_matrix_path = root / SPLIT_DIR / "val_source_matrix.npz"
    val_target_matrix_path = root / SPLIT_DIR / "val_target_matrix.npz"
    test_source_matrix_path = root / SPLIT_DIR / "test_source_matrix.npz"
    test_target_matrix_path = root / SPLIT_DIR / "test_target_matrix.npz"
    train_matrix_path = root / SPLIT_DIR / "train_matrix.npz"
    item_ids = split["item_ids"]
    train_item_ids = (
        np.load(train_item_ids_path, allow_pickle=False).astype(str)
        if train_item_ids_path.exists()
        else item_ids
    )
    val_item_ids = (
        np.load(val_item_ids_path, allow_pickle=False).astype(str)
        if val_item_ids_path.exists()
        else item_ids
    )
    test_item_ids = (
        np.load(test_item_ids_path, allow_pickle=False).astype(str)
        if test_item_ids_path.exists()
        else item_ids
    )
    train_source_matrix = (
        load_npz(train_source_matrix_path).tocsr()
        if train_source_matrix_path.exists()
        else load_npz(train_matrix_path).tocsr()
    )
    x_train = (
        load_npz(train_matrix_path).tocsr()
        if train_matrix_path.exists()
        else train_source_matrix
    )
    return {
        "item_ids": item_ids,
        "train_item_ids": train_item_ids,
        "val_item_ids": val_item_ids,
        "test_item_ids": test_item_ids,
        "x_train": x_train,
        "train_source_matrix": train_source_matrix,
        "train_target_matrix": (
            load_npz(train_target_matrix_path).tocsr()
            if train_target_matrix_path.exists()
            else x_train
        ),
        "val_source_matrix": (
            load_npz(val_source_matrix_path).tocsr()
            if val_source_matrix_path.exists()
            else _indices_to_csr(_read_obj_array(split["val_source_indices"]), n_cols=len(val_item_ids))
        ),
        "val_target_matrix": (
            load_npz(val_target_matrix_path).tocsr()
            if val_target_matrix_path.exists()
            else _indices_to_csr(_read_obj_array(split["val_target_indices"]), n_cols=len(val_item_ids))
        ),
        "test_source_matrix": (
            load_npz(test_source_matrix_path).tocsr()
            if test_source_matrix_path.exists()
            else _indices_to_csr(_read_obj_array(split["test_source_indices"]), n_cols=len(test_item_ids))
        ),
        "test_target_matrix": (
            load_npz(test_target_matrix_path).tocsr()
            if test_target_matrix_path.exists()
            else _indices_to_csr(_read_obj_array(split["test_target_indices"]), n_cols=len(test_item_ids))
        ),
        "val_source_indices": _read_obj_array(split["val_source_indices"]),
        "val_target_indices": _read_obj_array(split["val_target_indices"]),
        "test_source_indices": _read_obj_array(split["test_source_indices"]),
        "test_target_indices": _read_obj_array(split["test_target_indices"]),
        "train_user_ids": _load_optional_str_array(train_user_ids_path),
        "val_user_ids": _load_optional_str_array(val_user_ids_path),
        "test_user_ids": _load_optional_str_array(test_user_ids_path),
        "val_eval_user_ids": _load_optional_str_array(val_eval_user_ids_path),
        "test_eval_user_ids": _load_optional_str_array(test_eval_user_ids_path),
        "train_item_indices": _load_optional_int_array(
            train_item_indices_path,
            default=np.arange(len(item_ids), dtype=np.int64),
        ),
        "val_item_indices": _load_optional_int_array(val_item_indices_path),
        "test_item_indices": _load_optional_int_array(test_item_indices_path),
        "entity_tag_matrix": load_npz(tags_path).tocsr() if tags_path.exists() else None,
        "tag_names": np.load(tag_names_path, allow_pickle=False) if tag_names_path.exists() else None,
        "entity_metadata": pd.read_csv(metadata_path, dtype={"item_id": str}) if metadata_path.exists() else None,
    }


def save_cluster_graph_stage(
    root: str | Path,
    graph: SparseClusterSet,
    *,
    stage_dir: str = CLUSTERING_DIR,
    metadata: dict[str, Any] | None = None,
) -> Path:
    root = Path(root)
    path = root / stage_dir / CLUSTER_GRAPH_NAME
    save_cluster_graph(graph, path)
    update_stage_manifest(
        root,
        stage_dir,
        {
            "graph_path": f"{stage_dir}/{CLUSTER_GRAPH_NAME}",
            "n_nodes": len(graph.clusters),
            "n_active_clusters": len(graph.active_clusters),
            **(metadata or {}),
        },
    )
    return path


def load_cluster_graph_stage(
    root: str | Path,
    *,
    stage_dir: str = CLUSTERING_DIR,
) -> SparseClusterSet:
    return load_cluster_graph(Path(root) / stage_dir / CLUSTER_GRAPH_NAME)
