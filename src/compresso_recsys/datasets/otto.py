"""OTTO ecommerce sessions, downloaded by hand from Kaggle."""
from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections.abc import Sequence

import pandas as pd

from ._download import cached_interactions
from ._public import PublicDataset

#: Every event type the log records.
EVENT_TYPES = ("clicks", "carts", "orders")

#: Clicks are the browsing signal; carts and orders are a different act on a
#: different timescale. The canonical schema has one event kind, so recording
#: all three in the same column would say a purchase and a page view are the
#: same event rather than making the data richer. Pass ``events`` to override
#: that default when they are the same event to you.
KEPT_EVENTS = ("clicks",)

#: Session ids sit first on every line, so a sampled-out session can be skipped
#: without paying for a full JSON parse. Falls back to parsing when it does not
#: match, so a change in key order costs speed rather than correctness.
SESSION_PREFIX = re.compile(r'^\{"session":\s*(\d+)')

#: The competition and the standalone release name the same file differently.
EVENT_FILES = (
    "train.jsonl",
    "train.jsonl.gz",
    "otto-recsys-train.jsonl",
    "otto-recsys-train.jsonl.gz",
)


def _event_selection(events: Sequence[str]) -> tuple[str, ...]:
    """Normalise and check an event-type selection.

    Rejecting an unknown name matters more here than it looks: the filter is a
    plain membership test, so a typo would not raise -- it would yield an empty
    log and surface much later as a dataset with no interactions.
    """
    selection = tuple(dict.fromkeys(str(event) for event in events))
    if not selection:
        raise ValueError("events must name at least one event type")
    unknown = [event for event in selection if event not in EVENT_TYPES]
    if unknown:
        raise ValueError(f"unknown event types {unknown}; choose from {list(EVENT_TYPES)}")
    return selection


class OTTO(PublicDataset):
    """Sessions with real boundaries: 12.9M sessions, 1.8M items, 216M events.

    Every other registered dataset either has no session structure or requires
    inferring one from inter-arrival gaps. Here the boundaries are recorded, so
    a session is a fact about the data rather than a threshold someone chose.

    Sessions are the rows, so ``user_id`` holds a session id. There is no
    identity linking one session to the next and no item metadata whatsoever --
    this dataset can support sequential work and nothing else.

    Parsing is a Python loop over nested JSON, which on the full file takes long
    enough to be worth doing once; the canonical columns are then cached as
    parquet and later runs skip it.

    The full log is around 194 million clicks, which does not fit in memory as a
    DataFrame on an ordinary machine. ``session_sample`` keeps a deterministic
    fraction of sessions, chosen by hashing the session id::

        OTTO(session_sample=0.1)

    Sample sessions rather than rows or a time window. Dropping random events
    would destroy the adjacency inside a session, and a time window would cut
    sessions in half; keeping or dropping whole sessions leaves every surviving
    history exactly as it was. Hashing rather than taking the first N lines
    matters too, because the file is ordered by session id and the earliest ids
    are the longest-running sessions.

    The competition's ``test.jsonl`` is its held-out split and is not read; this
    adapter builds its own splits from ``train.jsonl``.
    """

    name = "otto"
    source_page = "https://github.com/otto-de/recsys-dataset"
    timestamp_precision = "milliseconds"

    #: Kaggle serves both the competition and the standalone dataset to
    #: signed-in users only, so there is nothing to fetch over plain HTTP.
    download_command = "kaggle datasets download -d otto/recsys-dataset"

    def __init__(
        self,
        data_dir="data",
        *,
        session_sample: float | None = None,
        events: Sequence[str] = KEPT_EVENTS,
        metadata_text_fields=None,
        min_entity_text_words: int = 0,
        show_progress: bool = True,
    ) -> None:
        if session_sample is not None and not 0.0 < session_sample <= 1.0:
            raise ValueError(f"session_sample must lie in (0, 1], got {session_sample!r}")
        self.session_sample = session_sample
        self.events = _event_selection(events)
        super().__init__(
            data_dir,
            metadata_text_fields=metadata_text_fields,
            min_entity_text_words=min_entity_text_words,
            show_progress=show_progress,
        )

    def _keeps(self, session: str) -> bool:
        """Deterministic membership, stable across runs and machines.

        ``hash()`` is salted per process, so a sampled dataset would differ
        between runs and the parquet cache would be wrong the moment it was
        reused.
        """
        if self.session_sample is None:
            return True
        digest = hashlib.sha256(session.encode()).digest()
        return int.from_bytes(digest[:4], "big") < self.session_sample * 2**32

    def events_path(self):
        for name in EVENT_FILES:
            candidate = self.root / name
            if candidate.is_file():
                return candidate
        return None

    def download(self) -> None:
        """Check for the manually placed export rather than fetching anything.

        Raising beats warning: ``prepare`` would otherwise continue and report a
        missing file from inside the reader instead of the manual step that is
        actually required.
        """
        if self.events_path() is None:
            raise FileNotFoundError(
                f"{self.name} cannot be downloaded automatically. Run "
                f"`{self.download_command}` with Kaggle credentials, or download "
                f"from {self.source_page}, and place one of "
                f"{', '.join(EVENT_FILES)} in {self.root}."
            )

    def _records(self, path):
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                match = SESSION_PREFIX.match(line)
                if match is not None and not self._keeps(match.group(1)):
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid record in {path}, line {number}") from exc
                if match is None and not self._keeps(str(record["session"])):
                    continue
                yield record

    def _frames(self, chunk_rows: int = 1_000_000):
        sessions: list[str] = []
        items: list[str] = []
        times: list[float] = []
        for record in self._records(self.events_path()):
            session = str(record["session"])
            for event in record["events"]:
                if event["type"] not in self.events:
                    continue
                sessions.append(session)
                items.append(str(event["aid"]))
                # Milliseconds as an integer, so no datetime parsing is involved.
                times.append(event["ts"] / 1000.0)
            # One session is never split across chunks: a history divided
            # between two writes would still reassemble, but the boundary is
            # free to avoid and cheap to keep.
            if len(sessions) >= chunk_rows:
                yield self._frame(sessions, items, times)
                sessions, items, times = [], [], []
        if sessions:
            yield self._frame(sessions, items, times)

    @staticmethod
    def _frame(sessions, items, times) -> pd.DataFrame:
        return pd.DataFrame(
            {"user_id": sessions, "item_id": items, "value": 1.0, "timestamp": times}
        )

    def _cache_version(self) -> int:
        """Cache key for the options that change which rows are kept.

        The sample and the event filter both select rows, and the cache is keyed
        on the source file alone, so they have to enter the key or a second run
        would silently read the first one's parquet. Only options that *differ*
        from their default are folded in, which means adding a new option never
        invalidates a cache built before that option existed -- worth the small
        asymmetry on a source that takes a Python JSON loop to re-parse.
        """
        signature = f"1|{self.session_sample}"
        if self.events != KEPT_EVENTS:
            signature += f"|events={self.events}"
        return int(hashlib.sha256(signature.encode()).hexdigest()[:8], 16)

    def prepare(self) -> None:
        self.download()
        interactions = cached_interactions(
            self.events_path(),
            self._frames,
            version=self._cache_version(),
        )
        self.finish(interactions, pd.DataFrame({"item_id": pd.Series(dtype=str)}))
