"""Which datasets the builder knows, and the defaults each was measured with.

The registry is data rather than logic: every entry states the split sizes and
support thresholds its dataset was profiled against, so a checkpoint built from
a name alone reproduces the documented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from compresso_recsys.datasets import AmazonReviews2023, Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.datasets import Steam, NetflixPrize, TasteProfile, Gowalla
from compresso_recsys.datasets import DBbook, LastFM2K
from compresso_recsys.datasets._public import PublicDataset


DEFAULT_TEMPORAL_PERIOD_HOURS = 339 * 24


@dataclass(frozen=True)
class DatasetSpec:
    cls: type
    checkpoint_path: str
    seed: int
    val_users: int
    test_users: int
    min_user_support: int = 5
    item_min_support: int = 1
    min_value_to_keep: float | None = 4.0
    set_all_values_to: float = 1.0
    min_entity_text_words: int = 30
    temporal_period_hours: float = DEFAULT_TEMPORAL_PERIOD_HOURS


DATASETS = {
    "dbbook": DatasetSpec(DBbook, "artifacts/dbbook/recsys_checkpoint.zip", seed=42,
                          val_users=500, test_users=1000, min_value_to_keep=1.0,
                          min_entity_text_words=0),
    "lfm2k": DatasetSpec(LastFM2K, "artifacts/lfm2k/recsys_checkpoint.zip", seed=42,
                         val_users=200, test_users=400, min_value_to_keep=None,
                         min_entity_text_words=0),
    "steam": DatasetSpec(Steam, "artifacts/steam/recsys_checkpoint.zip", seed=42,
                         val_users=10000, test_users=10000, min_value_to_keep=None,
                         min_entity_text_words=0),
    "netflix": DatasetSpec(NetflixPrize, "artifacts/netflix/recsys_checkpoint.zip", seed=98765,
                           val_users=40000, test_users=40000, min_entity_text_words=0),
    "taste-profile": DatasetSpec(TasteProfile, "artifacts/taste-profile/recsys_checkpoint.zip", seed=98765,
                                 val_users=50000, test_users=50000, min_user_support=20,
                                 item_min_support=200, min_value_to_keep=None, min_entity_text_words=0),
    "gowalla": DatasetSpec(Gowalla, "artifacts/gowalla/recsys_checkpoint.zip", seed=42,
                           val_users=10000, test_users=10000, min_user_support=10,
                           item_min_support=10, min_value_to_keep=None, min_entity_text_words=0,
                           temporal_period_hours=720),
    "goodbooks": DatasetSpec(Goodbooks, "artifacts/goodbooks/recsys_checkpoint.zip", seed=0, val_users=1000, test_users=2500),
    "ml1m": DatasetSpec(MovieLens1M, "artifacts/ml1m/recsys_checkpoint.zip", seed=42, val_users=500, test_users=1000),
    "ml20m": DatasetSpec(MovieLens20M, "artifacts/ml20m/recsys_checkpoint.zip", seed=42, val_users=2500, test_users=5000),
    "amazon2023": DatasetSpec(
        AmazonReviews2023,
        "artifacts/amazon2023/{amazon_category}/recsys_checkpoint.zip",
        seed=42,
        # Some category graphs are too sparse for a 20-core.
        # Keep useful user histories without recursively deleting rare items.
        val_users=100,
        test_users=200,
        min_user_support=5,
        item_min_support=1,
        min_value_to_keep=None,
        set_all_values_to=1.0,
        min_entity_text_words=0,
    ),
}


def _temporal_period_hours(dataset: str, value: float | None) -> float:
    value = DATASETS[dataset].temporal_period_hours if value is None else value
    if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
        raise ValueError("temporal_period_hours must be finite and > 0")
    return float(value)


def _make_dataset(args, spec: DatasetSpec):
    default_fields = getattr(spec.cls, "default_text_fields", ())
    fields = (
        [field.strip() for field in args.metadata_text_fields.split(",") if field.strip()]
        if args.metadata_text_fields
        else list(default_fields)
    )
    if not fields and not issubclass(spec.cls, PublicDataset):
        raise ValueError("--metadata_text_fields must contain at least one field")
    if issubclass(spec.cls, PublicDataset):
        return spec.cls(data_dir=args.data_dir, metadata_text_fields=fields,
                        min_entity_text_words=args.min_entity_text_words,
                        show_progress=getattr(args, "show_progress", True))
    if spec.cls is AmazonReviews2023:
        return AmazonReviews2023(
            data_dir=args.data_dir,
            category=args.amazon_category,
            metadata_text_fields=fields,
            min_entity_text_words=args.min_entity_text_words,
            include_image_urls=getattr(args, "include_image_urls", False),
            show_progress=getattr(args, "show_progress", True),
        )
    return spec.cls(
        data_dir=args.data_dir,
        metadata_text_fields=fields,
        min_entity_text_words=args.min_entity_text_words,
    )
