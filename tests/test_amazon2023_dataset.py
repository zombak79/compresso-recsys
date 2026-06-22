from __future__ import annotations

import pandas as pd

from compresso_recsys.datasets.amazon2023 import AmazonReviews2023


class FakeAmazon2023(AmazonReviews2023):
    def _load_hf_dataframe(self, config: str, *, split: str = "full") -> pd.DataFrame:
        if config.startswith("raw_meta_"):
            return pd.DataFrame(
                {
                    "parent_asin": ["A", "B", "C"],
                    "title": ["Robot Toy", "Tiny", "Circuit Kit"],
                    "features": [["Programmable", "STEM learning"], [], ["Build circuits"]],
                    "description": [["A detailed robotics kit for children."], [], ["Hands-on electronics project."]],
                    "categories": [["Toys", "STEM"], ["Toys"], ["Electronics"]],
                    "store": ["ToyCo", "SmallCo", "CircuitCo"],
                }
            )
        if config.startswith("0core_rating_only_"):
            return pd.DataFrame(
                {
                    "user_id": ["u1", "u1", "u2", "u3"],
                    "parent_asin": ["A", "B", "C", "missing"],
                    "rating": ["5.0", "4.0", "3.0", "5.0"],
                    "timestamp": ["1", "2", "3", "4"],
                }
            )
        raise AssertionError(config)


def test_amazon2023_uses_category_specific_configs(tmp_path):
    ds = AmazonReviews2023(data_dir=tmp_path, category="Toys_and_Games")

    assert ds.interactions_config == "0core_rating_only_Toys_and_Games"
    assert ds.metadata_config == "raw_meta_Toys_and_Games"


def test_amazon2023_builds_entity_text_from_configured_fields():
    row = pd.Series(
        {
            "title": "Robot Toy",
            "features": ["Programmable", "STEM learning"],
            "description": ["A detailed robotics kit."],
            "categories": ["Toys", "STEM"],
        }
    )

    text = AmazonReviews2023.build_entity_text(row, ["title", "features", "description", "categories"])

    assert "Title: Robot Toy" in text
    assert "Programmable" in text
    assert "A detailed robotics kit." in text
    assert "Toys" in text


def test_amazon2023_prepare_filters_items_by_entity_text(tmp_path):
    ds = FakeAmazon2023(
        data_dir=tmp_path,
        category="Toys_and_Games",
        metadata_text_fields=["title", "features", "description", "categories"],
        min_entity_text_words=5,
    )

    interactions = ds.get_interactions()
    metadata = ds.get_item_metadata()

    assert interactions["item_id"].tolist() == ["A", "C"]
    assert set(metadata["item_id"]) == {"A", "C"}
    assert "entity_text" in metadata.columns
