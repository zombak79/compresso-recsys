"""Interactions from the versioned SWAP multimodal dataset release."""
from __future__ import annotations

import zipfile

import pandas as pd

from ._download import download
from ._public import PublicDataset

SOURCE_PAGE = "https://zenodo.org/records/15403972"


def read_member(archive, basename, **kwargs):
    names = [name for name in archive.namelist()
             if name.rsplit("/", 1)[-1] == basename and not name.startswith("__MACOSX/")]
    if len(names) != 1:
        raise ValueError(f"Archive must contain exactly one {basename}")
    with archive.open(names[0]) as stream:
        return pd.read_csv(stream, sep="\t", **kwargs)


class DBbook(PublicDataset):
    """DBbook binary feedback; retains the upstream train/test label per row."""

    name = "dbbook"
    has_timestamps = False
    timestamp_precision = None
    source_page = SOURCE_PAGE
    default_text_fields = ("title",)

    def download(self):
        download(f"{SOURCE_PAGE}/files/dbbook_interaction_data.zip",
                 self.root / "dbbook_interaction_data.zip", show_progress=self.show_progress)

    def prepare(self):
        self.download()
        frames = []
        with zipfile.ZipFile(self.root / "dbbook_interaction_data.zip") as archive:
            for phase in ("train", "test"):
                frame = read_member(archive, f"{phase}.tsv", header=None,
                                    names=["user_id", "item_id", "value"],
                                    dtype={"user_id": str, "item_id": str, "value": float})
                frame["source_split"] = phase
                frame["timestamp"] = float("nan")
                frames.append(frame)
            metadata = read_member(archive, "DBbook_Items_DBpedia_mapping.tsv", dtype=str)
        metadata = metadata.rename(columns={"DBbook_ItemID": "item_id", "name": "title"})
        self.finish(pd.concat(frames, ignore_index=True), metadata)

    def get_official_split(self):
        """Return the supplied train/test frames without merging or resplitting."""
        frame = self.get_interactions()
        return {phase: frame[frame.source_split == phase].copy() for phase in ("train", "test")}


class LastFM2K(PublicDataset):
    """Artist listening counts; tagging dates are not listening timestamps."""

    name = "lfm2k"
    has_timestamps = False
    timestamp_precision = None
    source_page = SOURCE_PAGE
    default_text_fields = ("name",)

    def download(self):
        download(f"{SOURCE_PAGE}/files/lfm2k_interaction_data.zip",
                 self.root / "lfm2k_interaction_data.zip", show_progress=self.show_progress)

    def prepare(self):
        self.download()
        with zipfile.ZipFile(self.root / "lfm2k_interaction_data.zip") as archive:
            frame = read_member(archive, "user_artists.dat", dtype={"userID": str, "artistID": str})
            metadata = read_member(archive, "artists.dat", dtype=str)
        frame = frame.rename(columns={"userID": "user_id", "artistID": "item_id", "weight": "value"})
        frame["timestamp"] = float("nan")
        self.finish(frame, metadata.rename(columns={"id": "item_id", "pictureURL": "image_url"}))
