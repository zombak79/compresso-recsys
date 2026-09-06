"""Import pretrained SWAP features without executing downloaded pickle files."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import zipfile

import numpy as np

from .checkpoint import load_manifest, load_recsys_split, update_checkpoint
from .datasets._download import download
from .datasets.multimodal import SOURCE_PAGE
from .embeddings import save_item_embeddings

ENCODERS = {
    "text/minilm": "all-MiniLM-L6-v2.json",
    "text/mpnet": "all-mpnet-base-v2.json",
    "image/resnet152": "resnet152.json",
    "image/vgg": "vgg.json",
    "image/vit_cls": "vit_cls.json",
    "image/vit_avg": "vit_avg.json",
    "audio/vggish": "vggish.json",
    "audio/whisper": "whisper.json",
    "video/i3d": "i3d.json",
    "video/r2p1d": "r2p1d.json",
}
ARCHIVE_MD5 = {"ml1m": "8f184920d99edd58d1f2063450f4c0c6",
               "dbbook": "124334de88be80fbd5590f27786b1cea",
               "lfm2k": "7e5ba969073f5886743d301450083c0b"}


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate embedding item ID: {key}")
        result[key] = value
    return result


def _selection(dataset, features):
    if dataset not in ARCHIVE_MD5:
        raise ValueError("SWAP features support only ml1m, dbbook, lfm2k")
    names = features.split(",") if isinstance(features, str) else list(features)
    names = [name.strip() for name in names]
    if not names or len(set(names)) != len(names):
        raise ValueError("Choose at least one feature space, without duplicates")
    for name in names:
        if name not in ENCODERS or (dataset != "ml1m" and name.startswith("video/")) or (
                dataset == "dbbook" and name.startswith("audio/")):
            raise ValueError(f"Unsupported feature {name!r} for {dataset}")
    return names


def import_multimodal_embeddings(root, *, dataset, features, data_dir="data",
                                archive_path=None, show_progress=True):
    """Attach selected feature spaces to an extracted checkpoint.

    Automatic downloads use the fixed Zenodo record and verify its published MD5.
    ``archive_path`` accepts a local ID-keyed JSON ZIP (including custom fixtures),
    whose SHA-256 is recorded but whose contents are not claimed to be official.
    Last.fm media vectors are mean-pooled by artist ID, matching the upstream
    preprocessing. No normalization, imputation, or interaction filtering occurs.
    """
    names = _selection(dataset, features)
    split = load_recsys_split(root)
    recorded_dataset = load_manifest(root).get("stages", {}).get("data", {}).get("dataset")
    if recorded_dataset is not None and recorded_dataset != dataset:
        raise ValueError(f"Checkpoint dataset {recorded_dataset!r} does not match {dataset!r}")
    automatic = archive_path is None
    if automatic:
        archive_path = Path(data_dir) / "multimodal" / f"{dataset}_mm_json.zip"
        download(f"{SOURCE_PAGE}/files/{dataset}_mm_json.zip", archive_path,
                 show_progress=show_progress)
    archive_path = Path(archive_path)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    if automatic and md5.hexdigest() != ARCHIVE_MD5[dataset]:
        raise ValueError(f"Checksum mismatch for {archive_path}; replace the invalid cached file")
    catalog = split["item_ids"].astype(str)
    with zipfile.ZipFile(archive_path) as archive:
        for name in names:
            members = [member for member in archive.namelist()
                       if member.rsplit("/", 1)[-1] == ENCODERS[name]
                       and not member.startswith("__MACOSX/")]
            if len(members) != 1:
                raise ValueError(f"Expected exactly one {ENCODERS[name]} in {archive_path}")
            with archive.open(members[0]) as stream:
                data = json.load(stream, object_pairs_hook=_unique_pairs)
            if not isinstance(data, dict) or not data:
                raise ValueError("Expected a nonempty item-ID-to-vector JSON object")
            dimension = None
            for key, value in data.items():
                vector = np.asarray(value, dtype=np.float32)
                if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
                    raise ValueError(f"Invalid embedding vector for item {key}")
                dimension = len(vector) if dimension is None else dimension
                if len(vector) != dimension:
                    raise ValueError("Embedding dimensions must agree")
                data[key] = vector
            if dataset == "lfm2k":
                # Upstream JSON preserves media IDs like '6347_1', '6347_2'.
                # Its MMRec conversion averages these vectors at artist level.
                groups = {}
                for key, vector in data.items():
                    if not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", key):
                        raise ValueError(f"Invalid Last.fm artist/media ID: {key}")
                    groups.setdefault(str(int(key.split("_")[0])), []).append(vector)
                data = {key: np.mean(vectors, axis=0, dtype=np.float32)
                        for key, vectors in groups.items()}
            values = np.zeros((len(catalog), dimension), dtype=np.float32)
            mask = np.array([key in data for key in catalog], dtype=bool)
            if not mask.any():
                raise ValueError("No feature IDs match the checkpoint catalog")
            for row in np.flatnonzero(mask):
                values[row] = data[catalog[row]]
            save_item_embeddings(root, name, item_ids=catalog, embeddings=values,
                                 available=mask, metadata={
                                     "source": SOURCE_PAGE if automatic else "local JSON archive",
                                     "dataset": dataset, "encoder_file": ENCODERS[name],
                                     "archive_sha256": sha.hexdigest(),
                                     "verified_release": automatic,
                                     "normalization": "none; upstream vectors preserved",
                                     "pooling": "mean by artist ID prefix" if dataset == "lfm2k" else "none",
                                     "interaction_derived": dataset == "lfm2k" and name.startswith("text/"),
                                 })


def enrich_multimodal_checkpoint(checkpoint_path, **kwargs):
    """Atomically enrich an existing ML-1M, DBbook, or Last.fm checkpoint."""
    if not Path(checkpoint_path).is_file():
        raise FileNotFoundError(checkpoint_path)
    with update_checkpoint(checkpoint_path) as root:
        import_multimodal_embeddings(root, **kwargs)
