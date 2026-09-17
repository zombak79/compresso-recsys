"""Yandex Music listening events, with the flag saying which were recommended."""
from __future__ import annotations

import hashlib

import pandas as pd
import pyarrow.parquet as pq

from ._download import cached_interactions, download
from ._public import PublicDataset

REPOSITORY = "https://huggingface.co/datasets/yandex/yambda"
VARIANTS = ("50m", "500m", "5b")


class Yambda(PublicDataset):
    """Listening events with an ``is_organic`` flag on every row.

    That flag is the reason to reach for this dataset. Every other log here
    records what users did without recording how much of it the platform chose
    for them, so exposure bias can only be named as a caveat. Here it can be
    measured: ``organic_only=True`` keeps the events a user started themselves
    and drops the ones a recommender served.

    **Timestamps are seconds since the start of the log, not since 1970**, and
    they are rounded to multiples of five. Both facts matter more than they
    look:

    * The relative epoch is silent if assumed wrong -- values around 2.6e7 parse
      as January 1970 without raising -- so nothing here converts them, and the
      builder's magnitude normalisation leaves numbers this small untouched.
    * Five-second rounding creates ties *by construction*. A tie rate above zero
      here says nothing about the platform's logging quality, unlike on a
      dataset recorded at second precision.

    ``variant`` selects the release size: ``"50m"`` (the default), ``"500m"`` or
    ``"5b"``, named for roughly that many interactions. Only the default fits in
    memory as a DataFrame on an ordinary machine; the other two are one and two
    orders of magnitude past it. Unlike OTTO and Music4All-Onion this adapter has
    no sampling or windowing argument to bring a larger release back down, so
    reaching for ``"500m"`` or ``"5b"`` means providing the memory to hold it.
    Each variant is cached under its own filename, so switching between them
    re-reads rather than reusing the previous one's rows.

    The published files are sorted by ``(uid, timestamp)``. Order within a tie is
    therefore whatever that sort produced, so the within-tie ascent diagnostic
    measures the sort rather than the data, and cannot be read the way it can on
    a dataset stored in its original write order.

    The release also carries likes, dislikes, unlikes, undislikes and audio
    embeddings. This adapter reads listens only: the canonical interaction schema
    has one event kind, and folding a dislike into the same column as a play
    would record the two as the same thing.
    """

    name = "yambda"
    source_page = REPOSITORY
    timestamp_precision = "5 seconds, relative to the start of the log"

    def __init__(
        self,
        data_dir="data",
        *,
        variant: str = "50m",
        organic_only: bool = False,
        user_sample: float | None = None,
        metadata_text_fields=None,
        min_entity_text_words: int = 0,
        show_progress: bool = True,
    ) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.variant = variant
        self.organic_only = bool(organic_only)
        if user_sample is not None and not 0.0 < user_sample <= 1.0:
            raise ValueError(f"user_sample must lie in (0, 1], got {user_sample!r}")
        self.user_sample = user_sample
        super().__init__(
            data_dir,
            metadata_text_fields=metadata_text_fields,
            min_entity_text_words=min_entity_text_words,
            show_progress=show_progress,
        )

    @property
    def url(self) -> str:
        return f"{REPOSITORY}/resolve/main/flat/{self.variant}/listens.parquet"

    @property
    def events_path(self):
        # The variant is in the filename because one data directory may hold
        # several of them, and they are not interchangeable.
        return self.root / f"listens-{self.variant}.parquet"

    def download(self) -> None:
        download(self.url, self.events_path, show_progress=self.show_progress)

    def _keeps(self, users: pd.Series) -> pd.Series:
        """Deterministic membership for a batch of user ids.

        ``hash()`` is salted per process, so a sampled dataset would differ
        between runs and the parquet cache would be wrong the moment it was
        reused. Hashing the distinct ids rather than every row keeps this a
        per-user cost on a log with thousands of events per user.
        """
        if self.user_sample is None:
            return pd.Series(True, index=users.index)
        threshold = self.user_sample * 2**32
        unique = users.unique()
        keep = {
            user for user in unique
            if int.from_bytes(hashlib.sha256(user.encode()).digest()[:4], "big") < threshold
        }
        return users.isin(keep)

    def _frames(self):
        columns = ["uid", "item_id", "timestamp", "is_organic"]
        reader = pq.ParquetFile(self.events_path)
        for batch in reader.iter_batches(batch_size=1_000_000, columns=columns):
            frame = batch.to_pandas()
            if self.organic_only:
                frame = frame[frame["is_organic"] == 1]
            frame = frame[self._keeps(frame["uid"].astype(str))]
            yield pd.DataFrame(
                {
                    "user_id": frame["uid"].astype(str),
                    "item_id": frame["item_id"].astype(str),
                    "value": 1.0,
                    # Already seconds. Converting would move every event to 1970.
                    "timestamp": frame["timestamp"].astype("float64"),
                }
            )

    def _cache_version(self) -> int:
        """Cache key for the options that select rows.

        Only options differing from their default are folded in, so adding
        ``user_sample`` did not invalidate caches built before it existed. The
        variant is absent because each one already has its own filename.
        """
        signature = f"1|{self.organic_only}"
        if self.user_sample is not None:
            signature += f"|user_sample={self.user_sample}"
        return int(hashlib.sha256(signature.encode()).hexdigest()[:8], 16)

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(
            self.events_path,
            self._frames,
            version=self._cache_version(),
        )
        self.finish(interactions, pd.DataFrame({"item_id": pd.Series(dtype=str)}))
