"""Side-information matrices assembled for a checkpoint's entity tables."""

from __future__ import annotations


import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.datasets import Goodbooks, MovieLens20M


def _build_genre_tag_matrix(ds, item_ids: np.ndarray):
    metadata = ds.get_item_metadata()
    if "genres" not in metadata.columns:
        return None, None

    item_ids = np.asarray(item_ids).astype(str)
    item_to_genres = dict(zip(metadata["item_id"].astype(str), metadata["genres"].astype(str)))
    rows: list[int] = []
    tag_values: list[str] = []
    tag_to_col: dict[str, int] = {}
    cols: list[int] = []

    for row, item_id in enumerate(item_ids.tolist()):
        raw = item_to_genres.get(item_id)
        if raw is None or pd.isna(raw) or raw == "nan":
            continue
        for tag in raw.split("|"):
            tag = tag.strip()
            if not tag or tag == "(no genres listed)":
                continue
            col = tag_to_col.get(tag)
            if col is None:
                col = len(tag_values)
                tag_to_col[tag] = col
                tag_values.append(tag)
            rows.append(row)
            cols.append(col)

    if not tag_values:
        return None, None
    data = np.ones(len(rows), dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(item_ids), len(tag_values)), dtype=np.float32)
    return matrix, np.asarray(tag_values, dtype=str)


def _build_ml20m_user_tag_matrix(data_dir: str, item_ids: np.ndarray, *, min_count: int):
    if min_count < 1:
        raise ValueError("annotation_min_count must be >= 1")
    ml20m = MovieLens20M(data_dir=data_dir)
    ml20m.download()
    tags_path = ml20m.root / "ml-20m" / "tags.csv"
    if not tags_path.exists():
        raise FileNotFoundError(f"Missing ML20M tags file: {tags_path}")

    tags = pd.read_csv(tags_path, usecols=["movieId", "tag"])
    tags = tags.dropna(subset=["movieId", "tag"])
    tags["item_id"] = tags["movieId"].astype(str)
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
    tags = tags[tags["tag"] != ""].copy()

    tag_counts = tags.groupby("tag").size()
    keep_tags = set(tag_counts[tag_counts >= min_count].index.tolist())
    tags = tags[tags["tag"].isin(keep_tags)]
    if tags.empty:
        return None, None

    item_ids = np.asarray(item_ids).astype(str)
    row_by_item = {item_id: row for row, item_id in enumerate(item_ids.tolist())}
    tag_names = sorted(tags["tag"].unique().tolist())
    col_by_tag = {tag: col for col, tag in enumerate(tag_names)}

    tags = tags[tags["item_id"].isin(row_by_item)]
    if tags.empty:
        return None, None

    grouped = tags.groupby(["item_id", "tag"]).size().reset_index(name="count")
    rows = grouped["item_id"].map(row_by_item).to_numpy(dtype=np.int64)
    cols = grouped["tag"].map(col_by_tag).to_numpy(dtype=np.int64)
    data = grouped["count"].to_numpy(dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(item_ids), len(tag_names)), dtype=np.float32)
    return matrix, np.asarray(tag_names, dtype=str)


def _build_goodbooks_user_tag_matrix(ds: Goodbooks, item_ids: np.ndarray, *, min_count: int):
    if min_count < 1:
        raise ValueError("annotation_min_count must be >= 1")
    ds.download()
    books_path = ds.root / "books.csv"
    book_tags_path = ds.root / "book_tags.csv"
    tags_path = ds.root / "tags.csv"
    for path in (books_path, book_tags_path, tags_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Goodbooks tag source file: {path}")

    books = pd.read_csv(books_path, usecols=["book_id", "goodreads_book_id"])
    book_tags = pd.read_csv(book_tags_path, usecols=["goodreads_book_id", "tag_id", "count"])
    tags = pd.read_csv(tags_path, usecols=["tag_id", "tag_name"])

    book_tags = book_tags.dropna(subset=["goodreads_book_id", "tag_id", "count"])
    book_tags["count"] = book_tags["count"].astype(float)
    book_tags = book_tags[book_tags["count"] > 0].copy()
    if book_tags.empty:
        return None, None

    tag_counts = book_tags.groupby("tag_id")["count"].sum()
    keep_tag_ids = set(tag_counts[tag_counts >= min_count].index.tolist())
    book_tags = book_tags[book_tags["tag_id"].isin(keep_tag_ids)]
    if book_tags.empty:
        return None, None

    item_ids = np.asarray(item_ids).astype(str)
    row_by_item = {item_id: row for row, item_id in enumerate(item_ids.tolist())}

    books["item_id"] = books["book_id"].astype(str)
    id_map = books[["goodreads_book_id", "item_id"]].copy()
    book_tags = book_tags.merge(id_map, on="goodreads_book_id", how="inner")
    book_tags = book_tags[book_tags["item_id"].isin(row_by_item)]
    if book_tags.empty:
        return None, None

    tags["tag_name"] = tags["tag_name"].astype(str).str.strip().str.lower()
    tags = tags[tags["tag_name"] != ""].copy()
    book_tags = book_tags.merge(tags, on="tag_id", how="inner")
    if book_tags.empty:
        return None, None

    tag_names = sorted(book_tags["tag_name"].unique().tolist())
    col_by_tag = {tag: col for col, tag in enumerate(tag_names)}
    grouped = book_tags.groupby(["item_id", "tag_name"], as_index=False)["count"].sum()
    rows = grouped["item_id"].map(row_by_item).to_numpy(dtype=np.int64)
    cols = grouped["tag_name"].map(col_by_tag).to_numpy(dtype=np.int64)
    data = grouped["count"].to_numpy(dtype=np.float32)
    matrix = csr_matrix((data, (rows, cols)), shape=(len(item_ids), len(tag_names)), dtype=np.float32)
    return matrix, np.asarray(tag_names, dtype=str)


def _build_entity_tag_matrix(args, ds, item_ids: np.ndarray):
    if args.annotation_source == "none":
        return None, None, None
    if args.annotation_source == "genres":
        matrix, names = _build_genre_tag_matrix(ds, item_ids)
        return matrix, names, "genres" if matrix is not None else None
    if args.annotation_source == "ml20m_tags":
        matrix, names = _build_ml20m_user_tag_matrix(
            args.data_dir,
            item_ids,
            min_count=args.annotation_min_count,
        )
        return matrix, names, "ml20m_tags" if matrix is not None else None
    if args.annotation_source == "goodbooks_tags":
        if not isinstance(ds, Goodbooks):
            raise ValueError("--annotation_source goodbooks_tags can only be used with --dataset goodbooks")
        matrix, names = _build_goodbooks_user_tag_matrix(
            ds,
            item_ids,
            min_count=args.annotation_min_count,
        )
        return matrix, names, "goodbooks_tags" if matrix is not None else None
    raise ValueError(f"Unsupported annotation_source: {args.annotation_source}")


def _to_sparse_matrix_for_items(df: pd.DataFrame, item_ids: np.ndarray):
    return _to_sparse_matrix_for_items_with_users(df, item_ids)[0]


def _to_sparse_matrix_for_items_with_users(df: pd.DataFrame, item_ids: np.ndarray):
    users = pd.Index(sorted(df["user_id"].astype(str).unique()))
    items = pd.Index(np.asarray(item_ids).astype(str))
    if len(users) == 0:
        return csr_matrix((0, len(items)), dtype=np.float32), np.asarray([], dtype=str)

    u_codes = pd.Categorical(df["user_id"].astype(str), categories=users).codes
    i_codes = pd.Categorical(df["item_id"].astype(str), categories=items).codes
    valid = (u_codes >= 0) & (i_codes >= 0)
    vals = df["value"].astype(float).to_numpy()[valid]
    matrix = csr_matrix(
        (vals, (u_codes[valid], i_codes[valid])),
        shape=(len(users), len(items)),
        dtype=np.float32,
    )
    return matrix, users.to_numpy(dtype=str)
