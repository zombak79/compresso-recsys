from __future__ import annotations

import zipfile
from urllib.request import urlretrieve
from typing import Iterable

import pandas as pd

from .base import RecSysDataset


class MovieLens20M(RecSysDataset):
    name = "movielens20m"
    default_text_fields = ("title", "genres", "description")
    url = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"
    text_descriptions_url = (
        "https://raw.githubusercontent.com/recombee/beeformer/main/"
        "_datasets/ml20m/item_text_descriptions.feather"
    )

    def __init__(
        self,
        data_dir: str = "data",
        *,
        metadata_text_fields: Iterable[str] | None = None,
        min_entity_text_words: int = 0,
    ) -> None:
        self.metadata_text_fields = tuple(metadata_text_fields or self.default_text_fields)
        self.min_entity_text_words = int(min_entity_text_words)
        super().__init__(data_dir=data_dir)

    def download(self) -> None:
        zip_path = self.root / "ml-20m.zip"
        if not zip_path.exists():
            urlretrieve(self.url, zip_path)

        extract_dir = self.root / "ml-20m"
        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.root)

        descriptions_path = self.root / "item_text_descriptions.feather"
        if not descriptions_path.exists():
            urlretrieve(self.text_descriptions_url, descriptions_path)

    def prepare(self) -> None:
        self.download()
        ratings_path = self.root / "ml-20m" / "ratings.csv"
        movies_path = self.root / "ml-20m" / "movies.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

        ratings = pd.read_csv(ratings_path)
        ratings = ratings.rename(
            columns={
                "userId": "user_id",
                "movieId": "item_id",
                "rating": "value",
                "timestamp": "timestamp",
            }
        )
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["item_id"] = ratings["item_id"].astype(str)

        self._interactions = ratings[["user_id", "item_id", "value", "timestamp"]].copy()

        if movies_path.exists():
            movies = pd.read_csv(movies_path)
            movies = movies.rename(columns={"movieId": "item_id", "title": "title", "genres": "genres"})
            movies["item_id"] = movies["item_id"].astype(str)
            descriptions_path = self.root / "item_text_descriptions.feather"
            if descriptions_path.exists():
                descriptions = pd.read_feather(descriptions_path)
                descriptions = descriptions.rename(
                    columns={"movieId": "item_id", "llama31_description": "description"}
                )
                descriptions["item_id"] = descriptions["item_id"].astype(str)
                movies = movies.merge(descriptions[["item_id", "description"]], on="item_id", how="left")
            keep = [c for c in ["item_id", "title", "genres", "description"] if c in movies.columns]
            self._item_metadata = self.add_entity_text(
                movies[keep],
                fields=self.metadata_text_fields,
                min_words=self.min_entity_text_words,
            )
        else:
            self._item_metadata = self.add_entity_text(
                pd.DataFrame(columns=["item_id", "title", "genres", "description"]),
                fields=self.metadata_text_fields,
                min_words=self.min_entity_text_words,
            )
        self._interactions = self.restrict_interactions_to_metadata_items(self._interactions, self._item_metadata)
