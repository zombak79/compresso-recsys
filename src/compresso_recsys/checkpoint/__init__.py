"""The on-disk checkpoint: a split, its manifest, and anything stored beside it.

A checkpoint is a directory, not a file, and the pieces are written and read
independently. :mod:`~compresso_recsys.checkpoint.io` handles the array shapes,
where most entries are optional and absence is answered with ``None`` rather
than an exception. :mod:`~compresso_recsys.checkpoint.manifest` is what the
checkpoint says about itself. :mod:`~compresso_recsys.checkpoint.validation`
runs the cross-stage checks -- a later stage nesting inside the catalog of the
one before it -- which are cheap here and expensive to discover after training.
:mod:`~compresso_recsys.checkpoint.split` and
:mod:`~compresso_recsys.checkpoint.clusters` are the two stage kinds.
"""

from compresso_recsys.checkpoint.manifest import (
    MANIFEST_NAME,
    load_json,
    load_manifest,
    read_checkpoint,
    save_json,
    save_manifest,
    update_checkpoint,
    update_stage_manifest,
)
from compresso_recsys.checkpoint.split import (
    SPLIT_DIR,
    load_recsys_split,
    save_recsys_split,
)
from compresso_recsys.checkpoint.clusters import (
    CLUSTER_GRAPH_NAME,
    CLUSTERING_DIR,
    load_cluster_graph_stage,
    save_cluster_graph_stage,
)

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
