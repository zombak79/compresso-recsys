from __future__ import annotations

import pandas as pd

from ._download import cached_interactions, download, unix_seconds
from ._public import PublicDataset


class Gowalla(PublicDataset):
    """Raw SNAP check-ins with timestamps and location coordinates.

    Repeated check-ins are retained. This is not the remapped LightGCN split.
    Coordinates are taken from the first source occurrence of each location.
    The social graph is not required or downloaded.
    """

    name = "gowalla"
    source_page = "https://snap.stanford.edu/data/loc-gowalla.html"
    url = "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"

    def download(self) -> None:
        download(self.url, self.root / "loc-gowalla_totalCheckins.txt.gz", show_progress=self.show_progress)

    def _raw_frames(self):
        return pd.read_csv(self.root / "loc-gowalla_totalCheckins.txt.gz", sep="\t", header=None,
                           names=["user_id", "timestamp", "latitude", "longitude", "item_id"],
                           dtype={"user_id": str, "item_id": str}, chunksize=100_000)

    def _frames(self):
        for frame in self._raw_frames():
            frame["value"] = 1.0
            frame["timestamp"] = unix_seconds(frame["timestamp"])
            yield frame[["user_id", "item_id", "value", "timestamp"]]

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(self.root / "loc-gowalla_totalCheckins.txt.gz", self._frames)
        metadata = pd.concat([
            frame[["item_id", "latitude", "longitude"]].drop_duplicates("item_id")
            for frame in self._raw_frames()
        ], ignore_index=True).drop_duplicates("item_id")
        self.finish(interactions, metadata)
