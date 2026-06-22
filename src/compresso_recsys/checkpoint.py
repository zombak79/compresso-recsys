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
ELSA_DIR = "elsa"
SAE_DIR = "sae"
SBERT_DIR = "sbert"
SBERT_SAE_DIR = "sbert_sae"
COMPRESSED_ELSA_DIR = "compressed_elsa"
CLUSTERING_DIR = "clustering"
CLUSTER_GRAPH_NAME = "graph.json"


def _as_obj_array(xs: list[np.ndarray]) -> np.ndarray:
    return np.array([np.asarray(x, dtype=np.int64) for x in xs], dtype=object)


def _read_obj_array(x: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.int64) for v in x.tolist()]


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
    val_source_indices: list[np.ndarray],
    val_target_indices: list[np.ndarray],
    test_source_indices: list[np.ndarray],
    test_target_indices: list[np.ndarray],
    train_item_indices: np.ndarray | None = None,
    val_item_indices: np.ndarray | None = None,
    test_item_indices: np.ndarray | None = None,
    entity_tag_matrix: csr_matrix | None = None,
    tag_names: np.ndarray | list[str] | None = None,
    entity_metadata: pd.DataFrame | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    root = Path(root)
    data_dir = root / SPLIT_DIR
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    save_npz(data_dir / "train_matrix.npz", x_train.tocsr())
    np.savez_compressed(
        data_dir / "split.npz",
        item_ids=np.asarray(item_ids).astype(str),
        val_source_indices=_as_obj_array(val_source_indices),
        val_target_indices=_as_obj_array(val_target_indices),
        test_source_indices=_as_obj_array(test_source_indices),
        test_target_indices=_as_obj_array(test_target_indices),
    )
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
    root = Path(root)
    split = np.load(root / SPLIT_DIR / "split.npz", allow_pickle=True)
    tags_path = root / SPLIT_DIR / "entity_tags.npz"
    tag_names_path = root / SPLIT_DIR / "tag_names.npy"
    metadata_path = root / SPLIT_DIR / "entity_metadata.csv"
    train_item_indices_path = root / SPLIT_DIR / "train_item_indices.npy"
    val_item_indices_path = root / SPLIT_DIR / "val_item_indices.npy"
    test_item_indices_path = root / SPLIT_DIR / "test_item_indices.npy"
    item_ids = split["item_ids"]
    return {
        "item_ids": item_ids,
        "x_train": load_npz(root / SPLIT_DIR / "train_matrix.npz").tocsr(),
        "val_source_indices": _read_obj_array(split["val_source_indices"]),
        "val_target_indices": _read_obj_array(split["val_target_indices"]),
        "test_source_indices": _read_obj_array(split["test_source_indices"]),
        "test_target_indices": _read_obj_array(split["test_target_indices"]),
        "train_item_indices": (
            np.load(train_item_indices_path, allow_pickle=False)
            if train_item_indices_path.exists()
            else np.arange(len(item_ids), dtype=np.int64)
        ),
        "val_item_indices": (
            np.load(val_item_indices_path, allow_pickle=False)
            if val_item_indices_path.exists()
            else np.array([], dtype=np.int64)
        ),
        "test_item_indices": (
            np.load(test_item_indices_path, allow_pickle=False)
            if test_item_indices_path.exists()
            else np.array([], dtype=np.int64)
        ),
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
