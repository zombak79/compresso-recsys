"""Small, atomic download and parsing helpers for public dataset adapters."""
from __future__ import annotations

import ast
import gzip
import json
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def download(url: str, path: Path, *, show_progress: bool = True, opener=None) -> Path:
    """Reuse local archives; never promote an interrupted download to the cache."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".part", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as out, (opener or urlopen)(url, timeout=60) as response:
            if show_progress:
                print(f"Downloading {url} to {path}", flush=True)
            expected = response.headers.get("Content-Length")
            received = 0
            while block := response.read(1024 * 1024):
                out.write(block)
                received += len(block)
                if show_progress and received % (64 * 1024 * 1024) == 0:
                    total = f" / {int(expected) / 1024**2:.0f} MiB" if expected else ""
                    print(f"  {received / 1024**2:.0f} MiB{total}", flush=True)
            if expected is not None and received != int(expected):
                raise OSError(f"Incomplete download of {url}: {received}/{expected} bytes")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def gzip_records(path: Path):
    """Steam releases contain JSON or Python-literal dictionaries, never code."""
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    record = ast.literal_eval(line)
                if not isinstance(record, dict):
                    raise ValueError("expected a dictionary")
            except (ValueError, SyntaxError) as exc:
                raise ValueError(f"Invalid record in {path}, line {number}") from exc
            yield record


def unix_seconds(values: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="raise")
    # Explicit units avoid pandas-version-dependent datetime resolutions.
    return (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def _adopt_legacy_cache(legacy: Path, path: Path, signature: dict) -> bool:
    """Move a cache written under the unsuffixed name to its versioned name.

    Caches built before versions had their own files all used the unsuffixed
    name. One whose marker matches is exactly the selection being asked for, so
    it is renamed rather than rebuilt -- on OTTO that saves a full JSON parse.
    """
    legacy_marker = legacy.with_suffix(".json")
    try:
        if not legacy.exists() or json.loads(legacy_marker.read_text()) != signature:
            return False
        os.replace(legacy, path)
        os.replace(legacy_marker, path.with_suffix(".json"))
    except (ValueError, OSError):
        return False
    return True


def cached_interactions(source: Path, frames, *, version: int = 1) -> pd.DataFrame:
    """Cache only canonical columns, parsing large sources in bounded batches.

    ``frames`` is a callable so a cache hit never opens/decompresses the source.
    The final public DataFrame is still held in memory.

    Each ``version`` gets its own file, so two selections of one source never
    overwrite each other: concurrent builds with different adapter options
    cannot leave one selection's rows under the other's marker, and switching
    back to an earlier selection is a cache hit rather than a re-parse.
    ``version=1`` keeps the original unsuffixed name.
    """
    legacy = source.with_name(source.name + ".interactions.parquet")
    path = legacy if version == 1 else source.with_name(
        f"{source.name}.v{version:x}.interactions.parquet"
    )
    marker = path.with_suffix(".json")
    stat = source.stat()
    signature = {"version": version, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if path.exists() and marker.exists():
        try:
            if json.loads(marker.read_text()) == signature:
                return pd.read_parquet(path)
        except (ValueError, OSError):
            pass
    if path != legacy and _adopt_legacy_cache(legacy, path, signature):
        return pd.read_parquet(path)
    schema = pa.schema([
        ("user_id", pa.string()), ("item_id", pa.string()),
        ("value", pa.float64()), ("timestamp", pa.float64()),
    ])
    fd, temporary = tempfile.mkstemp(prefix=path.name, suffix=".part", dir=path.parent)
    os.close(fd)
    try:
        with pq.ParquetWriter(temporary, schema, compression="snappy") as writer:
            for frame in frames():
                if frame.empty:
                    continue
                if frame[["user_id", "item_id", "value"]].isna().any().any():
                    raise ValueError(f"Missing user_id, item_id, or value in {source}")
                frame = frame.copy()
                frame["user_id"] = frame["user_id"].astype(str)
                frame["item_id"] = frame["item_id"].astype(str)
                writer.write_table(pa.Table.from_pandas(frame, schema=schema, preserve_index=False))
        os.replace(temporary, path)
        marker.write_text(json.dumps(signature))
    finally:
        Path(temporary).unlink(missing_ok=True)
    return pd.read_parquet(path)
