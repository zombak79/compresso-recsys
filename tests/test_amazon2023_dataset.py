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
                    "images": [
                        [
                            {
                                "thumb": "https://example.com/a-thumb.jpg",
                                "large": "https://example.com/a-large.jpg",
                                "hi_res": "https://example.com/a-hires.jpg",
                            }
                        ],
                        [],
                        [{"large": "https://example.com/c-large.jpg"}],
                    ],
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


def test_amazon2023_resolves_official_source_files(tmp_path):
    ds = AmazonReviews2023(data_dir=tmp_path, category="Office_Products")
    ds._hf_files_for_path = lambda path: [
        {
            "type": "file",
            "path": "raw_meta_Office_Products/full-00000-of-00002.parquet",
            "size": 123,
        },
        {
            "type": "file",
            "path": "raw_meta_Office_Products/full-00001-of-00002.parquet",
            "size": 456,
        },
    ]

    meta_sources = ds._hf_source_for_config(ds.metadata_config)
    ratings_source = ds._hf_source_for_config(ds.interactions_config)[0]
    temporal_source = ds._hf_source_for_config(
        "0core_timestamp_w_his_Office_Products",
        split="valid",
    )[0]

    assert [source["kind"] for source in meta_sources] == ["parquet", "parquet"]
    assert str(meta_sources[0]["local_path"]) == "huggingface/raw_meta_Office_Products/full-00000-of-00002.parquet"
    assert meta_sources[0]["url"].endswith("/raw_meta_Office_Products/full-00000-of-00002.parquet")
    assert ratings_source["url"].endswith("/benchmark/0core/rating_only/Office_Products.csv")
    assert ratings_source["kind"] == "csv"
    assert temporal_source["url"].endswith("/benchmark/0core/timestamp_w_his/Office_Products.valid.csv")
    assert temporal_source["kind"] == "csv"


def test_amazon2023_loads_cached_huggingface_source_files(tmp_path):
    ds = AmazonReviews2023(data_dir=tmp_path, category="Toys_and_Games")
    meta_path = ds.root / "huggingface" / "raw_meta_Toys_and_Games" / "full-00000-of-00001.parquet"
    ratings_path = ds.root / "huggingface" / "benchmark" / "0core" / "rating_only" / "Toys_and_Games.csv"
    meta_path.parent.mkdir(parents=True)
    ratings_path.parent.mkdir(parents=True)

    pd.DataFrame({"parent_asin": ["A"], "title": ["Robot Toy"]}).to_parquet(meta_path, index=False)
    pd.DataFrame(
        {
            "user_id": ["u1"],
            "parent_asin": ["A"],
            "rating": [5.0],
            "timestamp": [123],
        }
    ).to_csv(ratings_path, index=False)
    ds._source_groups_for_config = lambda config, split="full": [
        [
            {
                "url": "https://example.invalid/source",
                "local_path": meta_path.relative_to(ds.root) if config == ds.metadata_config else ratings_path.relative_to(ds.root),
                "kind": "parquet" if config == ds.metadata_config else "csv",
                "size": None,
            }
        ]
    ]

    meta = ds._load_hf_dataframe(ds.metadata_config)
    ratings = ds._load_hf_dataframe(ds.interactions_config)

    assert meta.to_dict(orient="records") == [{"parent_asin": "A", "title": "Robot Toy"}]
    assert ratings["user_id"].tolist() == ["u1"]
    assert ratings["parent_asin"].tolist() == ["A"]


def test_amazon2023_only_sends_hf_token_to_huggingface(tmp_path, monkeypatch):
    ds = AmazonReviews2023(data_dir=tmp_path, category="Toys_and_Games")
    monkeypatch.setenv("HF_TOKEN", "secret")

    assert ds._headers_for_url("https://huggingface.co/datasets/example") == {"Authorization": "Bearer secret"}
    assert ds._headers_for_url("https://mcauleylab.ucsd.edu/public_datasets/data/file.csv") == {}


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


def test_amazon2023_extracts_image_urls_in_preferred_order():
    urls = AmazonReviews2023.extract_image_urls(
        [
            {
                "thumb": "https://example.com/thumb.jpg",
                "large": "https://example.com/large.jpg",
                "hi_res": "https://example.com/hires.jpg",
            },
            {"large": "https://example.com/second.jpg"},
        ]
    )

    assert urls == [
        "https://example.com/hires.jpg",
        "https://example.com/large.jpg",
        "https://example.com/thumb.jpg",
        "https://example.com/second.jpg",
    ]


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
    assert "image_url" not in metadata.columns
    assert "image_urls" not in metadata.columns


def test_amazon2023_can_include_image_url_metadata(tmp_path):
    ds = FakeAmazon2023(
        data_dir=tmp_path,
        category="Toys_and_Games",
        metadata_text_fields=["title", "features", "description", "categories"],
        min_entity_text_words=5,
        include_image_urls=True,
    )

    metadata = ds.get_item_metadata()
    by_item = metadata.set_index("item_id")

    assert by_item.loc["A", "image_url"] == "https://example.com/a-hires.jpg"
    assert by_item.loc["C", "image_url"] == "https://example.com/c-large.jpg"
    assert "https://example.com/a-large.jpg" in by_item.loc["A", "image_urls"]
