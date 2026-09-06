from __future__ import annotations

from typing import Iterable

import pandas as pd

from .base import RecSysDataset


class PublicDataset(RecSysDataset):
    """Shared configuration for the additional public interaction datasets."""

    default_text_fields: tuple[str, ...] = ()
    has_timestamps = True
    source_page = ""
    timestamp_precision = "seconds"

    def __init__(self, data_dir="data", *, metadata_text_fields: Iterable[str] | None = None,
                 min_entity_text_words: int = 0, show_progress: bool = True):
        super().__init__(data_dir)
        self.metadata_text_fields = tuple(
            self.default_text_fields if metadata_text_fields is None else metadata_text_fields
        )
        self.min_entity_text_words = int(min_entity_text_words)
        if self.min_entity_text_words < 0:
            raise ValueError("min_entity_text_words must be >= 0")
        self.show_progress = show_progress

    def finish(self, interactions: pd.DataFrame, metadata: pd.DataFrame) -> None:
        # A missing description must not erase a valid interaction or its item.
        metadata = metadata.copy()
        metadata["item_id"] = metadata["item_id"].astype(str)
        metadata = metadata.drop_duplicates("item_id", keep="first")
        catalog = pd.DataFrame({"item_id": interactions["item_id"].unique()})
        metadata = catalog.merge(metadata, on="item_id", how="outer", sort=False)
        if metadata.empty:
            metadata["entity_text"] = pd.Series(dtype=str)
            self._item_metadata = metadata
            self._interactions = interactions
            return
        self._item_metadata = self.add_entity_text(
            metadata, fields=self.metadata_text_fields, min_words=self.min_entity_text_words,
        )
        self._interactions = (
            self.restrict_interactions_to_metadata_items(interactions, self._item_metadata)
            if self.min_entity_text_words else interactions
        )
