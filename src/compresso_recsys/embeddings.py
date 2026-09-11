"""ID-aligned, optional item features stored inside recommender checkpoints."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .checkpoint import load_manifest, save_manifest

__all__ = ["save_item_embeddings", "load_item_embeddings", "list_item_embeddings"]


def _name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[\w-]+(?:/[\w-]+)?", name, flags=re.ASCII):
        raise ValueError("Embedding name must be a safe name or modality/encoder pair")
    return name


def _ids(values):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("item_ids must be one-dimensional")
    values = values.astype(str)
    if len(set(values.tolist())) != len(values):
        raise ValueError("item_ids must be unique")
    return values


def save_item_embeddings(root, name, *, item_ids, embeddings, available=None, metadata=None):
    """Save finite float32 features, explicit IDs, a presence mask, and provenance.

    Call inside ``update_checkpoint``. Names such as ``text/minilm`` are independent
    feature spaces. Missing rows are stored as zeros, never inferred from values.
    """
    name, ids = _name(name), _ids(item_ids)
    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 1:
        raise ValueError("embeddings must have shape (len(item_ids), positive dimension)")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must be finite")
    mask = np.ones(len(ids), dtype=bool) if available is None else np.asarray(available)
    if mask.dtype != bool or mask.shape != (len(ids),):
        raise ValueError("available must be a boolean mask aligned with item_ids")
    values = values.copy()
    values[~mask] = 0
    relative = f"embeddings/{name}"
    destination = Path(root) / relative
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "values.npy", values, allow_pickle=False)
    np.save(destination / "item_ids.npy", ids, allow_pickle=False)
    np.save(destination / "available.npy", mask, allow_pickle=False)
    manifest = load_manifest(root)
    manifest.setdefault("item_embeddings", {})[name] = {
        "format_version": 1, "path": relative, "shape": list(values.shape),
        "dtype": "float32", "available_items": int(mask.sum()),
        "metadata": dict(metadata or {}),
    }
    save_manifest(root, manifest)


def load_item_embeddings(root, name, *, item_ids=None):
    """Load one feature space, optionally reindexing to a requested item catalog.

    Unknown IDs become zero rows with ``available=False``. Old checkpoints have
    no registered feature spaces; an absent name raises ``KeyError``.
    """
    name = _name(name)
    entry = load_manifest(root).get("item_embeddings", {})[name]
    if entry.get("format_version") != 1:
        raise ValueError("Unsupported item embedding format")
    # Derive the path from the validated name, never trust a manifest path.
    directory = Path(root) / "embeddings" / name
    ids = _ids(np.load(directory / "item_ids.npy", allow_pickle=False))
    values = np.load(directory / "values.npy", allow_pickle=False)
    mask = np.load(directory / "available.npy", allow_pickle=False)
    if (values.ndim != 2 or values.shape[0] != len(ids) or values.shape[1] < 1
            or values.dtype != np.float32 or not np.isfinite(values).all()
            or mask.dtype != bool or mask.shape != (len(ids),)):
        raise ValueError("Invalid stored embedding matrix or availability mask")
    if item_ids is not None:
        requested = _ids(item_ids)
        index = {key: row for row, key in enumerate(ids)}
        aligned = np.zeros((len(requested), values.shape[1]), dtype=np.float32)
        present = np.zeros(len(requested), dtype=bool)
        for row, key in enumerate(requested):
            if key in index:
                aligned[row] = values[index[key]]
                present[row] = mask[index[key]]
        ids, values, mask = requested, aligned, present
    return {"item_ids": ids, "embeddings": values, "available": mask,
            "metadata": entry.get("metadata", {})}


def list_item_embeddings(root):
    """Return names and descriptors without loading feature matrices."""
    return load_manifest(root).get("item_embeddings", {})
