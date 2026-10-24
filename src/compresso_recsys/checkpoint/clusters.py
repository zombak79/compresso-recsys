"""The clustering stage: an item-similarity graph stored beside a split."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from compresso.clustering import load_cluster_graph, save_cluster_graph
from compresso.clustering.types import SparseClusterSet

from compresso_recsys.checkpoint.manifest import update_stage_manifest


CLUSTERING_DIR = "clustering"


CLUSTER_GRAPH_NAME = "graph.json"


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
