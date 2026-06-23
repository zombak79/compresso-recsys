from __future__ import annotations

import zipfile
from urllib.request import urlretrieve
from typing import Iterable

import pandas as pd

from .base import RecSysDataset


class Goodbooks(RecSysDataset):
    name = "goodbooks"
    default_text_fields = ("title", "authors", "description")
    url = "https://github.com/zygmuntz/goodbooks-10k/releases/download/v1.0/goodbooks-10k.zip"
    text_descriptions_url = (
        "https://github.com/recombee/beeformer/raw/refs/heads/main/"
        "_datasets/goodbooks/item_text_descriptions.feather"
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
        zip_path = self.root / "goodbooks-10k.zip"
        if not zip_path.exists():
            urlretrieve(self.url, zip_path)

        marker = self.root / "ratings.csv"
        if not marker.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.root)

        descriptions_path = self.root / "item_text_descriptions.feather"
        if not descriptions_path.exists():
            urlretrieve(self.text_descriptions_url, descriptions_path)

    def prepare(self) -> None:
        self.download()
        ratings_path = self.root / "ratings.csv"
        books_path = self.root / "books.csv"

        if not ratings_path.exists():
            raise FileNotFoundError(f"Missing ratings file: {ratings_path}")

        ratings = pd.read_csv(ratings_path)
        ratings = ratings.rename(columns={"user_id": "user_id", "book_id": "item_id", "rating": "value"})
        ratings["user_id"] = ratings["user_id"].astype(str)
        ratings["item_id"] = ratings["item_id"].astype(str)
        ratings["timestamp"] = None

        self._interactions = ratings[["user_id", "item_id", "value", "timestamp"]].copy()

        if books_path.exists():
            books = pd.read_csv(books_path)
            books = books.rename(columns={"book_id": "item_id", "title": "title", "authors": "authors"})
            books["item_id"] = books["item_id"].astype(str)
            descriptions_path = self.root / "item_text_descriptions.feather"
            if descriptions_path.exists():
                descriptions = pd.read_feather(descriptions_path)
                descriptions = descriptions.rename(
                    columns={"book_id": "item_id", "llama31_description": "description"}
                )
                descriptions["item_id"] = descriptions["item_id"].astype(str)
                books = books.merge(descriptions[["item_id", "description"]], on="item_id", how="left")
            keep = [c for c in ["item_id", "title", "authors", "average_rating", "description"] if c in books.columns]
            self._item_metadata = self.add_entity_text(
                books[keep],
                fields=self.metadata_text_fields,
                min_words=self.min_entity_text_words,
            )
        else:
            self._item_metadata = self.add_entity_text(
                pd.DataFrame(columns=["item_id", "title", "authors", "description"]),
                fields=self.metadata_text_fields,
                min_words=self.min_entity_text_words,
            )
        self._interactions = self.restrict_interactions_to_metadata_items(self._interactions, self._item_metadata)
