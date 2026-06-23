from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


@dataclass
class SplitBundle:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


class RecSysDataset:
    """Thin base class for interaction datasets used in example pipelines.

    Canonical interactions schema:
    - user_id: str
    - item_id: str
    - value: float
    - timestamp: int | float | None
    """

    name: str = "base"

    def __init__(self, data_dir: str | Path = "data") -> None:
        self.data_dir = Path(data_dir)
        self.root = self.data_dir / self.name
        self.root.mkdir(parents=True, exist_ok=True)

        self._interactions: Optional[pd.DataFrame] = None
        self._item_metadata: Optional[pd.DataFrame] = None

    def download(self) -> None:
        raise NotImplementedError

    def prepare(self) -> None:
        raise NotImplementedError

    def get_interactions(self) -> pd.DataFrame:
        if self._interactions is None:
            self.prepare()
        assert self._interactions is not None
        return self._interactions.copy()

    def get_item_metadata(self) -> pd.DataFrame:
        if self._item_metadata is None:
            self.prepare()
        assert self._item_metadata is not None
        return self._item_metadata.copy()

    @staticmethod
    def _metadata_value_to_text(value: Any, *, separator: str = " ") -> str:
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
                text = RecSysDataset._metadata_value_to_text(val, separator=separator)
                if text:
                    parts.append(f"{key}: {text}")
            return separator.join(parts).strip()
        if isinstance(value, (list, tuple, set)):
            parts = [RecSysDataset._metadata_value_to_text(v, separator=separator) for v in value]
            return separator.join(p for p in parts if p).strip()
        return str(value).strip()

    @classmethod
    def build_entity_text(cls, row: pd.Series, fields: Iterable[str]) -> str:
        parts: list[str] = []
        for field in fields:
            if field not in row:
                continue
            text = cls._metadata_value_to_text(row[field], separator="\n")
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def entity_text_word_count(text: str) -> int:
        return len(str(text).split())

    def add_entity_text(
        self,
        metadata: pd.DataFrame,
        *,
        fields: Iterable[str],
        min_words: int = 0,
    ) -> pd.DataFrame:
        out = metadata.copy()
        out["entity_text"] = out.apply(lambda row: self.build_entity_text(row, fields), axis=1)
        if min_words > 0:
            out = out[out["entity_text"].map(self.entity_text_word_count) >= int(min_words)].copy()
        return out.reset_index(drop=True)

    def restrict_interactions_to_metadata_items(
        self,
        interactions: pd.DataFrame,
        metadata: pd.DataFrame,
    ) -> pd.DataFrame:
        if "item_id" not in metadata.columns:
            raise ValueError("metadata must contain item_id")
        valid_items = set(metadata["item_id"].astype(str))
        out = interactions.copy()
        out["item_id"] = out["item_id"].astype(str)
        return out[out["item_id"].isin(valid_items)].reset_index(drop=True)

    def split_users_strong_generalization(
        self,
        *,
        val_users: int,
        test_users: int,
        min_user_support: int = 1,
        random_state: int = 42,
        interactions: Optional[pd.DataFrame] = None,
    ) -> SplitBundle:
        df = self.get_interactions() if interactions is None else interactions.copy()
        if min_user_support > 1:
            counts = df.groupby("user_id")["item_id"].nunique()
            keep_users = counts[counts >= min_user_support].index
            df = df[df["user_id"].isin(keep_users)].copy()
        users = np.array(sorted(df["user_id"].unique()))
        rng = np.random.default_rng(random_state)
        rng.shuffle(users)

        if val_users + test_users >= len(users):
            raise ValueError("val_users + test_users must be smaller than number of users")

        val_set = set(users[:val_users])
        test_set = set(users[val_users : val_users + test_users])

        is_val = df["user_id"].isin(val_set)
        is_test = df["user_id"].isin(test_set)

        val = df[is_val].copy()
        test = df[is_test].copy()
        train = df[~(is_val | is_test)].copy()

        return SplitBundle(train=train, val=val, test=test)

    @staticmethod
    def preprocess_interactions_for_recsys(
        df: pd.DataFrame,
        *,
        min_value_to_keep: Optional[float] = 4.0,
        user_min_support: int = 5,
        item_min_support: int = 1,
        set_all_values_to: Optional[float] = 1.0,
        max_steps: int = 0,
    ) -> pd.DataFrame:
        """Paper-style preprocessing: threshold, binarize, iterative pruning, categorical cleanup."""
        out = df.copy()
        out["user_id"] = out["user_id"].astype(str)
        out["item_id"] = out["item_id"].astype(str)
        out["value"] = out["value"].astype(float)

        if min_value_to_keep is not None:
            out = out[out["value"] >= float(min_value_to_keep)].copy()

        if set_all_values_to is not None:
            out["value"] = float(set_all_values_to)

        step = 0
        while True:
            step += 1
            n_before = len(out)

            if item_min_support > 1:
                item_counts = out.groupby("item_id")["user_id"].size()
                keep_items = item_counts[item_counts >= item_min_support].index
                out = out[out["item_id"].isin(keep_items)]

            if user_min_support > 1:
                user_counts = out.groupby("user_id")["item_id"].size()
                keep_users = user_counts[user_counts >= user_min_support].index
                out = out[out["user_id"].isin(keep_users)]

            n_after = len(out)
            if n_after == n_before:
                break
            if max_steps > 0 and step >= max_steps:
                break

        out["user_id"] = out["user_id"].astype("category").cat.remove_unused_categories()
        out["item_id"] = out["item_id"].astype("category").cat.remove_unused_categories()
        out["user_id"] = out["user_id"].astype(str)
        out["item_id"] = out["item_id"].astype(str)
        return out.reset_index(drop=True)

    def to_hf_dataset(self, df: Optional[pd.DataFrame] = None):
        """Convert interactions to HuggingFace Dataset.

        Import is optional so core library does not hard-depend on datasets.
        """
        if df is None:
            df = self.get_interactions()
        try:
            from datasets import Dataset
        except Exception as e:  # pragma: no cover - optional dependency
            raise ImportError("Install `datasets` to use HF conversion.") from e
        return Dataset.from_pandas(df.reset_index(drop=True), preserve_index=False)

    @staticmethod
    def to_sparse_matrix(df: pd.DataFrame):
        """Return (X, user_ids, item_ids) where X is user x item CSR."""
        users = pd.Index(sorted(df["user_id"].astype(str).unique()))
        items = pd.Index(sorted(df["item_id"].astype(str).unique()))

        u_codes = pd.Categorical(df["user_id"].astype(str), categories=users).codes
        i_codes = pd.Categorical(df["item_id"].astype(str), categories=items).codes

        vals = df["value"].astype(float).to_numpy()
        x = csr_matrix((vals, (u_codes, i_codes)), shape=(len(users), len(items)), dtype=np.float32)
        return x, users, items
