from __future__ import annotations

import tarfile

import pandas as pd

from ._download import cached_interactions, download, unix_seconds
from ._public import PublicDataset


class NetflixPrize(PublicDataset):
    """Netflix Prize ratings with dates, movie titles and release years.

    Downloads the original-format archive from Internet Archive. No Kaggle
    dependency. Original Netflix terms apply; Compresso does not redistribute it.
    """

    name = "netflix"
    default_text_fields = ("title", "release_year")
    timestamp_precision = "day"
    source_page = "https://archive.org/details/nf_prize_dataset.tar"
    url = "https://archive.org/download/nf_prize_dataset.tar/nf_prize_dataset.tar.gz"

    def download(self) -> None:
        download(self.url, self.root / "nf_prize_dataset.tar.gz", show_progress=self.show_progress)

    @staticmethod
    def _ratings(archive):
        rows = []
        for member in archive:
            if not member.isfile() or not member.name.rsplit("/", 1)[-1].startswith("mv_"):
                continue
            with archive.extractfile(member) as binary:
                item_id = binary.readline().decode("utf-8").strip()
                if not item_id.endswith(":"):
                    raise ValueError(f"Missing movie header in {member.name}")
                item_id = item_id[:-1]
                for line in binary:
                    user_id, value, date = line.decode("utf-8").strip().split(",")
                    rows.append((user_id, item_id, float(value), date))
                    if len(rows) >= 100_000:
                        yield NetflixPrize._frame(rows)
                        rows = []
        if rows:
            yield NetflixPrize._frame(rows)

    @staticmethod
    def _frame(rows):
        frame = pd.DataFrame(rows, columns=["user_id", "item_id", "value", "timestamp"])
        frame["timestamp"] = unix_seconds(frame["timestamp"])
        return frame

    def _frames(self):
        with tarfile.open(self.root / "nf_prize_dataset.tar.gz", "r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.rsplit("/", 1)[-1] == "training_set.tar":
                    with archive.extractfile(member) as stream, tarfile.open(fileobj=stream, mode="r|*") as inner:
                        yield from self._ratings(inner)
                    return
        # Also accept archives with the per-movie files directly inside.
        with tarfile.open(self.root / "nf_prize_dataset.tar.gz", "r|gz") as archive:
            yield from self._ratings(archive)

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(self.root / "nf_prize_dataset.tar.gz", self._frames)
        if interactions.empty:
            raise ValueError("Netflix archive contains no training ratings")
        rows = []
        with tarfile.open(self.root / "nf_prize_dataset.tar.gz", "r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.rsplit("/", 1)[-1] == "movie_titles.txt":
                    with archive.extractfile(member) as stream:
                        for line in stream:
                            item_id, year, title = line.decode("latin-1").rstrip("\r\n").split(",", 2)
                            rows.append((item_id, year if year != "NULL" else "", title))
                    break
        self.finish(interactions, pd.DataFrame(rows, columns=["item_id", "release_year", "title"]))
