from __future__ import annotations

import pandas as pd

from compresso_recsys.datasets.base import RecSysDataset


def test_add_entity_text_joins_fields_and_filters_by_words(tmp_path):
    ds = RecSysDataset(data_dir=tmp_path)
    metadata = pd.DataFrame(
        {
            "item_id": ["a", "b"],
            "title": ["Long Enough Title", "Tiny"],
            "authors": ["Author Name", ""],
            "description": ["This description has enough words", ""],
        }
    )

    out = ds.add_entity_text(metadata, fields=["title", "authors", "description"], min_words=5)

    assert out["item_id"].tolist() == ["a"]
    assert out["entity_text"].iloc[0] == "Long Enough Title\nAuthor Name\nThis description has enough words"


def test_restrict_interactions_to_metadata_items(tmp_path):
    ds = RecSysDataset(data_dir=tmp_path)
    interactions = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2"],
            "item_id": ["a", "b", "c"],
            "value": [1.0, 1.0, 1.0],
            "timestamp": [1, 2, 3],
        }
    )
    metadata = pd.DataFrame({"item_id": ["a", "c"]})

    out = ds.restrict_interactions_to_metadata_items(interactions, metadata)

    assert out["item_id"].tolist() == ["a", "c"]
