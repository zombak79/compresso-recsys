"""Retailrocket ecommerce behaviour log, downloaded by hand from Kaggle."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import pandas as pd

from ._download import cached_interactions
from ._public import PublicDataset

#: Every event type the log records.
EVENT_TYPES = ("view", "addtocart", "transaction")

#: Views are 96.7% of the log and are the click analogue every other adapter
#: yields. The canonical schema has no event-type column, so keeping carts and
#: transactions as well would not make the data richer -- it would make a
#: purchase indistinguishable from a page view. Pass ``events`` to override that
#: default when a purchase and a page view really are the same event to you.
KEPT_EVENTS = ("view",)


def _event_selection(events: Sequence[str]) -> tuple[str, ...]:
    """Normalise and check an event-type selection.

    Rejecting an unknown name matters more here than it looks: the filter is a
    plain ``isin``, so a typo would not raise -- it would yield an empty log and
    surface much later as a dataset with no interactions.
    """
    selection = tuple(dict.fromkeys(str(event) for event in events))
    if not selection:
        raise ValueError("events must name at least one event type")
    unknown = [event for event in selection if event not in EVENT_TYPES]
    if unknown:
        raise ValueError(f"unknown event types {unknown}; choose from {list(EVENT_TYPES)}")
    return selection


class RetailRocket(PublicDataset):
    """Visitor events over 4.5 months: 2.76M rows, 1.41M visitors, 235k items.

    Repeated views of the same item are retained; they are most of what makes
    this log a sequence rather than a set. Item properties are published as
    hashed values, so there is no usable item text and the catalog carries no
    metadata beyond the ids the events reference.

    Sequences are short — under two events per visitor on average — so the
    leave-last-out protocol, which needs four, keeps only a small and heavily
    self-selected minority of visitors. Measure that share before reading
    anything into a metric computed on it.
    """

    name = "retailrocket"
    source_page = "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset"
    timestamp_precision = "milliseconds"

    #: Kaggle serves the archive to signed-in users only, so there is no URL to
    #: fetch. Only this member of the archive is read; the rest may stay zipped.
    events_file = "events.csv"

    def __init__(
        self,
        data_dir="data",
        *,
        events: Sequence[str] = KEPT_EVENTS,
        metadata_text_fields=None,
        min_entity_text_words: int = 0,
        show_progress: bool = True,
    ) -> None:
        self.events = _event_selection(events)
        super().__init__(
            data_dir,
            metadata_text_fields=metadata_text_fields,
            min_entity_text_words=min_entity_text_words,
            show_progress=show_progress,
        )

    def download(self) -> None:
        """Check for the manually placed export rather than fetching anything.

        Raising beats a warning here: ``prepare`` would otherwise continue and
        fail inside the CSV reader, reporting a missing file instead of the
        manual step that is actually required.
        """
        if not (self.root / self.events_file).is_file():
            raise FileNotFoundError(
                f"{self.name} cannot be downloaded automatically. Sign in at "
                f"{self.source_page}, download ecommerce-dataset.zip, and extract "
                f"{self.events_file} to {self.root}."
            )

    def _frames(self):
        for frame in pd.read_csv(
            self.root / self.events_file,
            usecols=["timestamp", "visitorid", "event", "itemid"],
            dtype={"visitorid": str, "itemid": str, "event": str},
            chunksize=500_000,
        ):
            frame = frame[frame["event"].isin(self.events)]
            yield pd.DataFrame(
                {
                    "user_id": frame["visitorid"],
                    "item_id": frame["itemid"],
                    "value": 1.0,
                    # Integer milliseconds. unix_seconds() cannot read these:
                    # pandas parses a bare integer as nanoseconds, which would
                    # place every event in 1970 without raising.
                    "timestamp": frame["timestamp"] / 1000.0,
                }
            )

    def _cache_version(self) -> int:
        """Cache key for the event selection.

        The cache is keyed on the source file alone, so a different selection
        has to change the version or it would read the previous one's parquet.
        The default selection keeps the original key, so adding this option did
        not invalidate caches built before it existed.
        """
        if self.events == KEPT_EVENTS:
            return 1
        return int(hashlib.sha256(f"1|events={self.events}".encode()).hexdigest()[:8], 16)

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(
            self.root / self.events_file, self._frames, version=self._cache_version(),
        )
        self.finish(interactions, pd.DataFrame({"item_id": pd.Series(dtype=str)}))
