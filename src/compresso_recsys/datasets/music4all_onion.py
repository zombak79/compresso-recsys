"""Last.fm listening events from the Music4All-Onion release."""
from __future__ import annotations

import hashlib

import pandas as pd

from ._download import cached_interactions, download, unix_seconds
from ._public import PublicDataset

RECORD = "https://zenodo.org/records/6609677"


def window_version(start: str | None, end: str | None) -> int:
    """Cache signature for a time window.

    :func:`cached_interactions` keys its parquet on the source file alone, so
    two windows would otherwise share one cache and the second would silently
    read the first one's rows. Folding the window into ``version`` costs a
    rebuild when the window changes and never returns the wrong events.
    """
    return int(hashlib.sha256(f"1|{start}|{end}".encode()).hexdigest()[:8], 16)


class Music4AllOnion(PublicDataset):
    """252,984,396 listening events by 119,140 users over 56,512 tracks.

    Heavy repeat consumption: a user replaying one track several times in a row
    is ordinary here, and those replays are retained as separate events. That
    makes this the one registered dataset where "predict the next listen" and
    "predict a *new* listen" are different questions, so report metrics split by
    whether the target already occurs in the history.

    The source file is 2.2 GB compressed and expands to a quarter of a billion
    rows, which does not fit in memory as a DataFrame on an ordinary machine.
    Pass ``start`` and ``end`` to take a time window. Window by time rather than
    by sampling rows: dropping random events destroys the adjacency that any
    sequential claim rests on, while a window leaves the surviving histories
    intact.

    Rows arrive newest-first within a user, which is the opposite of the
    chronological order every consumer wants. Nothing here re-sorts them --
    ``leave_last_out`` and the temporal split both sort by timestamp themselves,
    and rewriting the file order would destroy the only evidence available about
    how tied events were originally recorded.

    The release also ships 26 audio, video, lyric and metadata feature sets that
    this adapter does not download; the catalog carries item IDs only.
    """

    name = "music4all-onion"
    source_page = RECORD
    timestamp_precision = "seconds"

    events_file = "userid_trackid_timestamp.tsv.bz2"
    url = f"{RECORD}/files/{events_file}?download=1"

    def __init__(
        self,
        data_dir="data",
        *,
        start: str | None = None,
        end: str | None = None,
        metadata_text_fields=None,
        min_entity_text_words: int = 0,
        show_progress: bool = True,
    ) -> None:
        super().__init__(
            data_dir,
            metadata_text_fields=metadata_text_fields,
            min_entity_text_words=min_entity_text_words,
            show_progress=show_progress,
        )
        self.start = start
        self.end = end
        # Half-open, so adjacent windows partition the log rather than sharing
        # the events on their boundary.
        self._start_seconds = None if start is None else pd.Timestamp(start, tz="UTC").timestamp()
        self._end_seconds = None if end is None else pd.Timestamp(end, tz="UTC").timestamp()

    def download(self) -> None:
        download(self.url, self.root / self.events_file, show_progress=self.show_progress)

    def _frames(self):
        for frame in pd.read_csv(
            self.root / self.events_file,
            sep="\t",
            dtype={"user_id": str, "track_id": str},
            chunksize=1_000_000,
        ):
            # "2013-01-27 21:42:38", so unix_seconds parses it correctly here.
            seconds = unix_seconds(frame["timestamp"])
            keep = pd.Series(True, index=frame.index)
            if self._start_seconds is not None:
                keep &= seconds >= self._start_seconds
            if self._end_seconds is not None:
                keep &= seconds < self._end_seconds
            yield pd.DataFrame(
                {
                    "user_id": frame["user_id"][keep],
                    "item_id": frame["track_id"][keep],
                    "value": 1.0,
                    "timestamp": seconds[keep],
                }
            )

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(
            self.root / self.events_file,
            self._frames,
            version=window_version(self.start, self.end),
        )
        self.finish(interactions, pd.DataFrame({"item_id": pd.Series(dtype=str)}))
