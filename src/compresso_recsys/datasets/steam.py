from __future__ import annotations

import pandas as pd

from ._download import cached_interactions, download, gzip_records, unix_seconds
from ._public import PublicDataset


class Steam(PublicDataset):
    """Raw McAuley Steam reviews with original product IDs and game metadata.

    Every review is an interaction, including negative reviews (BERT4Rec's
    convention). Dates have day precision; equal dates retain source order.
    Metadata does not have to exist for every reviewed game.
    """

    name = "steam"
    default_text_fields = ("title", "genres", "tags", "developer", "publisher")
    source_page = "https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data"
    timestamp_precision = "day"
    reviews_url = "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_reviews.json.gz"
    metadata_url = "https://mcauleylab.ucsd.edu/public_datasets/data/steam/steam_games.json.gz"

    def download(self) -> None:
        download(self.reviews_url, self.root / "steam_reviews.json.gz", show_progress=self.show_progress)
        download(self.metadata_url, self.root / "steam_games.json.gz", show_progress=self.show_progress)

    def _review_frames(self):
        rows = []
        for row in gzip_records(self.root / "steam_reviews.json.gz"):
            rows.append((row["username"], row["product_id"], 1.0, row["date"]))
            if len(rows) >= 100_000:
                yield self._frame(rows)
                rows = []
        if rows:
            yield self._frame(rows)

    @staticmethod
    def _frame(rows):
        frame = pd.DataFrame(rows, columns=["user_id", "item_id", "value", "timestamp"])
        frame["timestamp"] = unix_seconds(frame["timestamp"])
        return frame

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(self.root / "steam_reviews.json.gz", self._review_frames)
        rows = []
        for row in gzip_records(self.root / "steam_games.json.gz"):
            if row.get("id") is None:
                continue
            result = {"item_id": str(row["id"]), "title": row.get("title") or row.get("app_name", "")}
            for field in ("genres", "tags", "specs"):
                value = row.get(field, [])
                result[field] = "|".join(map(str, value)) if isinstance(value, list) else str(value or "")
            for field in ("developer", "publisher", "release_date", "price", "url", "early_access"):
                value = row.get(field)
                result[field] = "" if value is None else str(value)
            rows.append(result)
        metadata = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["item_id"])
        self.finish(interactions, metadata)
