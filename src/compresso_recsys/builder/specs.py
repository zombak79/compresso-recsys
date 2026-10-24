"""Which datasets the builder knows, and the defaults each was measured with.

The registry is data rather than logic: every entry states the split sizes and
support thresholds its dataset was profiled against, so a checkpoint built from
a name alone reproduces the documented one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from compresso_recsys.datasets import AmazonReviews2023, Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.datasets import Steam, NetflixPrize, TasteProfile, Gowalla
from compresso_recsys.datasets import RetailRocket, Music4AllOnion, OTTO, Yambda
from compresso_recsys.datasets import DBbook, LastFM2K
from compresso_recsys.datasets._public import PublicDataset
from compresso_recsys.builder.options import _dataset_options_for


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
    #: Adapter options this dataset is registered with, e.g. a sample fraction
    #: for a source too large to hold whole. Anything the caller passes wins, so
    #: a default here narrows what a bare build reads without taking the choice
    #: away. OTTO and Music4All-Onion set one because their full logs do not fit
    #: in memory as a DataFrame, so a bare build of either would fail rather
    #: than produce the numbers the measured tables document.
    dataset_options: dict[str, Any] = field(default_factory=dict)
    #: Width of each temporal target window. The split needs three of them to
    #: fit inside the log, so a dataset whose history is shorter than three
    #: times this cannot build a temporal split at all.
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
    # 4.5 months of history, so the 339-day global window cannot fit three
    # target periods and the temporal split fails outright. Fourteen days
    # spends 42 of those days on evaluation and leaves roughly 95 for training.
    "retailrocket": DatasetSpec(RetailRocket, "artifacts/retailrocket/recsys_checkpoint.zip", seed=42,
                                val_users=2500, test_users=5000, min_user_support=5,
                                item_min_support=5, min_value_to_keep=None, min_entity_text_words=0,
                                temporal_period_hours=14 * 24),
    # A quarter of a billion listens does not fit in memory, and the cost is not
    # the interactions frame -- at 43 bytes a row that is under two gigabytes --
    # but the split payload built from it, which holds a Python list entry per
    # event. One calendar year is 33.7M events. Window by time rather than by
    # sampling rows: dropping random events destroys the adjacency any
    # sequential claim rests on. Three 339-day windows do not fit in one year,
    # so temporal uses 30 days, leaving nine months of training history.
    "music4all-onion": DatasetSpec(Music4AllOnion, "artifacts/music4all-onion/recsys_checkpoint.zip",
                                   seed=42, val_users=2500, test_users=5000, min_user_support=5,
                                   item_min_support=5, min_value_to_keep=None, min_entity_text_words=0,
                                   dataset_options={"start": "2014-01-01", "end": "2015-01-01"},
                                   temporal_period_hours=30 * 24),
    # 194M clicks does not fit either. A tenth of the sessions is 19.5M events,
    # the largest of these three defaults -- kept because that cache already
    # exists and rebuilding it means re-parsing 11 GB of JSON. Lower it to 0.05
    # if the split payload proves too large. Sample whole sessions, never rows:
    # a session cut in half is not a shorter session, it is a different one.
    # Measured at this sample: 543,448 users over 360,552 items, so the 10,000
    # hold-outs are comfortable. The log is only 672 hours long, so temporal
    # uses two days -- six days of evaluation against twenty-two of history.
    "otto": DatasetSpec(OTTO, "artifacts/otto/recsys_checkpoint.zip", seed=42,
                        val_users=10000, test_users=10000, min_user_support=5,
                        item_min_support=5, min_value_to_keep=None, min_entity_text_words=0,
                        dataset_options={"session_sample": 0.1},
                        temporal_period_hours=2 * 24),
    # The smallest variant is still 46.5M listens, and the release ships no
    # smaller one, so a fifth of the users -- 9.3M events -- is the default.
    # Sample users rather than rows for the same reason OTTO samples sessions.
    # Few users, very long histories: this sample measures 1,800 users over
    # 149,971 items, which is why the hold-outs are hundreds rather than the
    # thousands every other dataset can afford. Both fall under the 1,000-user
    # mark the tables flag, and that is the dataset's shape, not a mistake.
    # The log runs 7,222 hours, so temporal uses 30 days.
    "yambda": DatasetSpec(Yambda, "artifacts/yambda/recsys_checkpoint.zip", seed=42,
                          val_users=200, test_users=400, min_user_support=5,
                          item_min_support=5, min_value_to_keep=None, min_entity_text_words=0,
                          dataset_options={"user_sample": 0.2},
                          temporal_period_hours=30 * 24),
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
    # Validated against this adapter's signature by _resolve_args, so an unknown
    # name has already been rejected by the time it would become a TypeError.
    options = _dataset_options_for(spec, args)
    if issubclass(spec.cls, PublicDataset):
        return spec.cls(data_dir=args.data_dir, metadata_text_fields=fields,
                        min_entity_text_words=args.min_entity_text_words,
                        show_progress=getattr(args, "show_progress", True),
                        **options)
    if spec.cls is AmazonReviews2023:
        return AmazonReviews2023(
            data_dir=args.data_dir,
            category=args.amazon_category,
            metadata_text_fields=fields,
            min_entity_text_words=args.min_entity_text_words,
            include_image_urls=getattr(args, "include_image_urls", False),
            show_progress=getattr(args, "show_progress", True),
            **options,
        )
    return spec.cls(
        data_dir=args.data_dir,
        metadata_text_fields=fields,
        min_entity_text_words=args.min_entity_text_words,
        **options,
    )
