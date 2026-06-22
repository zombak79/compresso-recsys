from __future__ import annotations

import json
from typing import Any, Iterable

import pandas as pd

from .base import RecSysDataset


DEFAULT_TEXT_FIELDS = ("title", "features", "description", "categories")
CATEGORY_ALIASES = {
    "beauty": "All_Beauty",
    "clothing": "Clothing_Shoes_and_Jewelry",
    "electronics": "Electronics",
    "toys": "Toys_and_Games",
    "toys_and_games": "Toys_and_Games",
}


class AmazonReviews2023(RecSysDataset):
    """Amazon Reviews 2023 category dataset loaded from Hugging Face.

    The recommender pipeline only needs compact rating-only interactions plus
    item metadata. Reviews are intentionally not downloaded.
    """

    name = "amazon2023"
    hf_name = "McAuley-Lab/Amazon-Reviews-2023"

    def __init__(
        self,
        data_dir: str = "data",
        *,
        category: str = "Toys_and_Games",
        metadata_text_fields: Iterable[str] = DEFAULT_TEXT_FIELDS,
        min_entity_text_words: int = 0,
    ) -> None:
        self.category = self.normalize_category(category)
        self.metadata_text_fields = tuple(metadata_text_fields)
        self.min_entity_text_words = int(min_entity_text_words)
        super().__init__(data_dir=data_dir)
        self.root = self.data_dir / self.name / self.category
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def interactions_config(self) -> str:
        return f"0core_rating_only_{self.category}"

    @property
    def metadata_config(self) -> str:
        return f"raw_meta_{self.category}"

    @staticmethod
    def normalize_category(category: str) -> str:
        key = category.strip()
        if not key:
            raise ValueError("Amazon category cannot be empty")
        return CATEGORY_ALIASES.get(key.lower(), key)

    def download(self) -> None:
        # Hugging Face datasets handles local caching. `prepare` triggers the
        # category-specific downloads; this method exists for interface parity.
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _stringify(value: Any, *, separator: str = " ") -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            parts: list[str] = []
            for key, val in value.items():
                val_text = AmazonReviews2023._stringify(val, separator=separator)
                if val_text:
                    parts.append(f"{key}: {val_text}")
            return separator.join(parts).strip()
        if isinstance(value, (list, tuple, set)):
            parts = [AmazonReviews2023._stringify(v, separator=separator) for v in value]
            return separator.join(p for p in parts if p).strip()
        return str(value).strip()

    @staticmethod
    def _parse_details(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if not text or text == "{}":
            return ""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text

    @classmethod
    def build_entity_text(cls, row: pd.Series, fields: Iterable[str]) -> str:
        parts: list[str] = []
        for field in fields:
            if field not in row:
                continue
            value = cls._parse_details(row[field]) if field == "details" else row[field]
            text = cls._stringify(value, separator="\n" if field in {"features", "description"} else " > ")
            if text:
                label = field.replace("_", " ").title()
                parts.append(f"{label}: {text}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _word_count(text: str) -> int:
        return len(str(text).split())

    def _load_hf_dataframe(self, config: str, *, split: str = "full") -> pd.DataFrame:
        try:
            from datasets import load_dataset
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install Hugging Face datasets to load Amazon Reviews 2023, e.g. "
                "`pip install compresso-recsys[datasets]`."
            ) from e

        dataset = load_dataset(self.hf_name, config, split=split, trust_remote_code=True)
        return dataset.to_pandas()

    def load_timestamp_splits_with_history(self) -> dict[str, pd.DataFrame]:
        """Load McAuley's timestamp split with per-row history fields."""
        config = f"0core_timestamp_w_his_{self.category}"
        return {
            "train": self._load_hf_dataframe(config, split="train"),
            "valid": self._load_hf_dataframe(config, split="valid"),
            "test": self._load_hf_dataframe(config, split="test"),
        }

    def prepare(self) -> None:
        self.download()

        meta = self._load_hf_dataframe(self.metadata_config, split="full")
        if "parent_asin" not in meta.columns:
            raise ValueError(f"Amazon metadata config {self.metadata_config!r} has no parent_asin column")
        meta = meta.rename(columns={"parent_asin": "item_id"}).copy()
        meta["item_id"] = meta["item_id"].astype(str)
        meta = meta.drop_duplicates(subset=["item_id"], keep="first")
        meta["entity_text"] = meta.apply(lambda row: self.build_entity_text(row, self.metadata_text_fields), axis=1)
        if self.min_entity_text_words > 0:
            meta = meta[meta["entity_text"].map(self._word_count) >= self.min_entity_text_words].copy()

        interactions = self._load_hf_dataframe(self.interactions_config, split="full")
        expected = {"user_id", "parent_asin", "rating", "timestamp"}
        missing = expected.difference(interactions.columns)
        if missing:
            raise ValueError(f"Amazon interactions config {self.interactions_config!r} is missing columns {sorted(missing)}")
        interactions = interactions.rename(columns={"parent_asin": "item_id", "rating": "value"})
        interactions = interactions[["user_id", "item_id", "value", "timestamp"]].copy()
        interactions["user_id"] = interactions["user_id"].astype(str)
        interactions["item_id"] = interactions["item_id"].astype(str)
        interactions["value"] = pd.to_numeric(interactions["value"], errors="coerce")
        interactions["timestamp"] = pd.to_numeric(interactions["timestamp"], errors="coerce")
        interactions = interactions.dropna(subset=["user_id", "item_id", "value"])

        valid_items = set(meta["item_id"].astype(str))
        interactions = interactions[interactions["item_id"].isin(valid_items)].reset_index(drop=True)

        preferred = [
            "item_id",
            "title",
            "store",
            "main_category",
            "categories",
            "features",
            "description",
            "details",
            "price",
            "average_rating",
            "rating_number",
            "entity_text",
        ]
        keep = [col for col in preferred if col in meta.columns]
        self._item_metadata = meta[keep].reset_index(drop=True)
        self._interactions = interactions
