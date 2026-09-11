from __future__ import annotations

import zipfile

import pandas as pd

from ._download import cached_interactions, download
from ._public import PublicDataset


class TasteProfile(PublicDataset):
    """MSD Taste Profile triplets: play counts, no timestamps or item text.

    The original source currently serves this archive over HTTP. Additional
    MSD metadata and mismatch lists are not part of this interaction adapter.
    """

    name = "taste-profile"
    has_timestamps = False
    timestamp_precision = None
    source_page = "http://millionsongdataset.com/tasteprofile/"
    url = "http://millionsongdataset.com/sites/default/files/challenge/train_triplets.txt.zip"

    def download(self) -> None:
        download(self.url, self.root / "train_triplets.txt.zip", show_progress=self.show_progress)

    def _frames(self):
        with zipfile.ZipFile(self.root / "train_triplets.txt.zip") as archive:
            names = [name for name in archive.namelist() if name.rsplit("/", 1)[-1] == "train_triplets.txt"]
            if len(names) != 1:
                raise ValueError("Taste Profile archive must contain one train_triplets.txt")
            with archive.open(names[0]) as stream:
                for frame in pd.read_csv(stream, sep="\t", header=None,
                                         names=["user_id", "item_id", "value"],
                                         dtype={"user_id": str, "item_id": str}, chunksize=100_000):
                    frame["timestamp"] = float("nan")
                    yield frame

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(self.root / "train_triplets.txt.zip", self._frames)
        self.finish(interactions, pd.DataFrame(columns=["item_id"]))
