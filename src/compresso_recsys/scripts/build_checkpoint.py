from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.checkpoint import save_recsys_split, update_checkpoint
from compresso_recsys.datasets import AmazonReviews2023, Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.retrieval import (
    build_eval_holdout,
    build_item_cold_holdout,
    build_leave_last_out_holdout,
    build_temporal_holdout,
)


@dataclass(frozen=True)
class DatasetSpec:
    cls: type
    checkpoint_path: str
    seed: int
    val_users: int
    test_users: int
    min_user_support: int = 5
    item_min_support: int = 1
    min_value_to_keep: float = 4.0
    set_all_values_to: float = 1.0


DATASETS = {
    "goodbooks": DatasetSpec(Goodbooks, "artifacts/goodbooks/recsys_checkpoint.zip", seed=0, val_users=1000, test_users=2500),
    "ml1m": DatasetSpec(MovieLens1M, "artifacts/ml1m/recsys_checkpoint.zip", seed=42, val_users=500, test_users=1000),
    "ml20m": DatasetSpec(MovieLens20M, "artifacts/ml20m/recsys_checkpoint.zip", seed=42, val_users=2500, test_users=5000),
    "amazon2023": DatasetSpec(
        AmazonReviews2023,
        "artifacts/amazon2023/{amazon_category}/recsys_checkpoint.zip",
        seed=42,
        val_users=2500,
        test_users=5000,
        min_user_support=20,
        item_min_support=20,
        min_value_to_keep=4.0,
        set_all_values_to=1.0,
    ),
}


def _metadata_text_fields_arg(value: str | list[str] | tuple[str, ...] | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return ",".join(str(field) for field in value)


class _CheckpointProgress:
    def __init__(self, *, enabled: bool, total: int) -> None:
        self.enabled = enabled
        self.current = False
        self.bar: Any = None
        if not enabled:
            return
        try:
            from tqdm.auto import tqdm
        except Exception:  # pragma: no cover - optional dependency
            return
        self.bar = tqdm(total=total, unit="step", desc="Building checkpoint")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.bar is not None:
            if self.current and exc_type is None:
                self.bar.update(1)
            self.bar.close()

    def step(self, message: str) -> None:
        if not self.enabled:
            return
        if self.bar is None:
            print(f"[compresso-recsys] {message}", flush=True)
            return
        if self.current:
            self.bar.update(1)
        self.current = True
        self.bar.set_description_str(message)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True, choices=sorted(DATASETS))
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--checkpoint_path", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--val_users", type=int, default=None)
    p.add_argument("--test_users", type=int, default=None)
    p.add_argument("--min_user_support", type=int, default=None)
    p.add_argument("--item_min_support", type=int, default=None)
    p.add_argument("--min_value_to_keep", type=float, default=None)
    p.add_argument("--set_all_values_to", type=float, default=None)
    p.add_argument("--eval_fold", type=int, default=0, choices=[0, 1])
    p.add_argument(
        "--split_mode",
        type=str,
        default="user_split",
        choices=["user_split", "item_split", "leave_last_out", "temporal"],
    )
    p.add_argument("--val_items", type=int, default=None, help="Number of cold validation items for item_split.")
    p.add_argument("--test_items", type=int, default=None, help="Number of cold test items for item_split.")
    p.add_argument("--item_val_frac", type=float, default=0.05, help="Cold validation item fraction for item_split.")
    p.add_argument("--item_test_frac", type=float, default=0.10, help="Cold test item fraction for item_split.")
    p.add_argument("--temporal_test_frac", type=float, default=0.10, help="Global latest-interaction fraction for temporal split.")
    p.add_argument("--min_source_items", type=int, default=1)
    p.add_argument("--min_target_items", type=int, default=1)
    p.add_argument(
        "--amazon_category",
        type=str,
        default="Toys_and_Games",
        help="Amazon Reviews 2023 category, e.g. Toys_and_Games, Electronics, Clothing_Shoes_and_Jewelry.",
    )
    p.add_argument(
        "--metadata_text_fields",
        type=str,
        default=None,
        help="Comma-separated metadata fields joined into entity_text for text-aware datasets.",
    )
    p.add_argument(
        "--min_entity_text_words",
        type=int,
        default=30,
        help="Drop items whose constructed entity_text has fewer words. Mostly useful for Amazon 2023.",
    )
    p.add_argument(
        "--include_image_urls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Amazon product image_url/image_urls columns in checkpoint metadata.",
    )
    p.add_argument(
        "--annotation_source",
        type=str,
        default="genres",
        choices=["genres", "ml20m_tags", "goodbooks_tags", "none"],
    )
    p.add_argument("--annotation_min_count", type=int, default=100)
    p.add_argument(
        "--show_progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show download and checkpoint-building progress. Use --no-show_progress to disable.",
    )
    return p.parse_args()


def _build_args(
    *,
    dataset: str,
    data_dir: str = "data",
    checkpoint_path: str | None = None,
    seed: int | None = None,
    val_users: int | None = None,
    test_users: int | None = None,
    min_user_support: int | None = None,
    item_min_support: int | None = None,
    min_value_to_keep: float | None = None,
    set_all_values_to: float | None = None,
    eval_fold: int = 0,
    split_mode: str = "user_split",
    val_items: int | None = None,
    test_items: int | None = None,
    item_val_frac: float = 0.05,
    item_test_frac: float = 0.10,
    temporal_test_frac: float = 0.10,
    min_source_items: int = 1,
    min_target_items: int = 1,
    amazon_category: str = "Toys_and_Games",
    metadata_text_fields: str | list[str] | tuple[str, ...] | None = None,
    min_entity_text_words: int = 30,
    include_image_urls: bool = False,
    annotation_source: str = "genres",
    annotation_min_count: int = 100,
    show_progress: bool = True,
) -> argparse.Namespace:
    if dataset not in DATASETS:
        choices = ", ".join(sorted(DATASETS))
        raise ValueError(f"dataset must be one of {{{choices}}}, got {dataset!r}")
    if eval_fold not in (0, 1):
        raise ValueError(f"eval_fold must be 0 or 1, got {eval_fold!r}")
    if split_mode not in {"user_split", "item_split", "leave_last_out", "temporal"}:
        raise ValueError(f"Unsupported split_mode: {split_mode!r}")
    if annotation_source not in {"genres", "ml20m_tags", "goodbooks_tags", "none"}:
        raise ValueError(f"Unsupported annotation_source: {annotation_source!r}")
    return argparse.Namespace(
        dataset=dataset,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        seed=seed,
        val_users=val_users,
        test_users=test_users,
        min_user_support=min_user_support,
        item_min_support=item_min_support,
        min_value_to_keep=min_value_to_keep,
        set_all_values_to=set_all_values_to,
        eval_fold=eval_fold,
        split_mode=split_mode,
        val_items=val_items,
        test_items=test_items,
        item_val_frac=item_val_frac,
        item_test_frac=item_test_frac,
        temporal_test_frac=temporal_test_frac,
        min_source_items=min_source_items,
        min_target_items=min_target_items,
        amazon_category=amazon_category,
        metadata_text_fields=_metadata_text_fields_arg(metadata_text_fields),
        min_entity_text_words=min_entity_text_words,
        include_image_urls=include_image_urls,
        annotation_source=annotation_source,
        annotation_min_count=annotation_min_count,
        show_progress=show_progress,
    )


def _resolve_args(args):
    spec = DATASETS[args.dataset]
    args.checkpoint_path = args.checkpoint_path or spec.checkpoint_path.format(
        amazon_category=args.amazon_category,
    )
    args.seed = spec.seed if args.seed is None else args.seed
    args.val_users = spec.val_users if args.val_users is None else args.val_users
    args.test_users = spec.test_users if args.test_users is None else args.test_users
    args.min_user_support = spec.min_user_support if args.min_user_support is None else args.min_user_support
    args.item_min_support = spec.item_min_support if args.item_min_support is None else args.item_min_support
    args.min_value_to_keep = spec.min_value_to_keep if args.min_value_to_keep is None else args.min_value_to_keep
    args.set_all_values_to = spec.set_all_values_to if args.set_all_values_to is None else args.set_all_values_to
    return args, spec


def _make_dataset(args, spec: DatasetSpec):
    default_fields = getattr(spec.cls, "default_text_fields", ())
    fields = (
        [field.strip() for field in args.metadata_text_fields.split(",") if field.strip()]
        if args.metadata_text_fields
        else list(default_fields)
    )
    if not fields:
        raise ValueError("--metadata_text_fields must contain at least one field")
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
        if raw is None or raw == "nan":
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


def _split_item_ids_random(item_ids: np.ndarray, *, args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    item_ids = np.asarray(item_ids).astype(str)
    n_items = len(item_ids)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n_items)

    n_val = args.val_items if args.val_items is not None else int(np.ceil(n_items * args.item_val_frac))
    n_test = args.test_items if args.test_items is not None else int(np.ceil(n_items * args.item_test_frac))
    n_val = max(0, int(n_val))
    n_test = max(0, int(n_test))
    if n_val + n_test >= n_items:
        raise ValueError("Cold val/test items must leave at least one train item")

    val_idx = np.sort(perm[:n_val])
    test_idx = np.sort(perm[n_val : n_val + n_test])
    train_idx = np.sort(perm[n_val + n_test :])
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def _build_user_split(args, ds, proc_df):
    split = ds.split_users_strong_generalization(
        val_users=args.val_users,
        test_users=args.test_users,
        min_user_support=1,
        random_state=args.seed,
        interactions=proc_df,
    )
    x_train, train_user_index, item_ids = ds.to_sparse_matrix(split.train)
    val_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.val,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_fold=args.eval_fold,
    )
    test_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.test,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_fold=args.eval_fold,
    )
    return {
        "item_ids": test_holdout["item_ids"],
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": val_holdout,
        "test_holdout": test_holdout,
        "train_item_indices": None,
        "val_item_indices": np.array([], dtype=np.int64),
        "test_item_indices": np.array([], dtype=np.int64),
        "train_user_ids": np.asarray(train_user_index).astype(str),
        "val_user_ids": np.asarray(sorted(split.val["user_id"].astype(str).unique())),
        "test_user_ids": np.asarray(sorted(split.test["user_id"].astype(str).unique())),
        "extra_metadata": {
            "has_user_partitions": True,
            "has_item_partitions": False,
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": "Random user split; timestamps are not used to prevent future-to-past leakage.",
        },
    }


def _build_item_split(args, proc_df):
    item_ids = np.array(sorted(proc_df["item_id"].astype(str).unique()))
    train_idx, val_idx, test_idx = _split_item_ids_random(item_ids, args=args)
    train_items = set(item_ids[train_idx].tolist())
    val_items = set(item_ids[val_idx].tolist())
    test_items = set(item_ids[test_idx].tolist())
    train_df = proc_df[proc_df["item_id"].astype(str).isin(train_items)].copy()
    x_train, train_user_ids = _to_sparse_matrix_for_items_with_users(train_df, item_ids)
    val_holdout = build_item_cold_holdout(
        item_ids=item_ids,
        interactions=proc_df,
        source_item_ids=train_items,
        target_item_ids=val_items,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    test_holdout = build_item_cold_holdout(
        item_ids=item_ids,
        interactions=proc_df,
        source_item_ids=train_items,
        target_item_ids=test_items,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": val_holdout,
        "test_holdout": test_holdout,
        "train_item_indices": train_idx,
        "val_item_indices": val_idx,
        "test_item_indices": test_idx,
        "train_user_ids": train_user_ids,
        "val_user_ids": None,
        "test_user_ids": None,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": True,
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": "Random item split; timestamps are not used to prevent future-to-past leakage.",
            "item_val_frac": args.item_val_frac,
            "item_test_frac": args.item_test_frac,
            "val_items": int(len(val_idx)),
            "test_items": int(len(test_idx)),
        },
    }


def _build_leave_last_out_split(args, proc_df):
    item_ids = np.array(sorted(proc_df["item_id"].astype(str).unique()))
    holdout = build_leave_last_out_holdout(
        item_ids=item_ids,
        interactions=proc_df,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    target_items = sorted({int(i) for row in holdout["target_indices"] for i in row.tolist()})
    target_idx = np.asarray(target_items, dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(len(item_ids), dtype=np.int64), target_idx, assume_unique=False)
    train_items = set(item_ids[train_idx].tolist())
    train_df = proc_df[proc_df["item_id"].astype(str).isin(train_items)].copy()
    x_train, train_user_ids = _to_sparse_matrix_for_items_with_users(train_df, item_ids)
    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": holdout,
        "test_holdout": holdout,
        "train_item_indices": train_idx,
        "val_item_indices": target_idx,
        "test_item_indices": target_idx,
        "train_user_ids": train_user_ids,
        "val_user_ids": None,
        "test_user_ids": None,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": False,
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": "Leave-last-out is chronological within each user but can leak global future information across users.",
            "cold_target_items": int(len(target_idx)),
        },
    }


def _build_temporal_split(args, proc_df):
    item_ids = np.array(sorted(proc_df["item_id"].astype(str).unique()))
    holdout = build_temporal_holdout(
        item_ids=item_ids,
        interactions=proc_df,
        test_frac=args.temporal_test_frac,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    target_items = sorted({int(i) for row in holdout["target_indices"] for i in row.tolist()})
    target_idx = np.asarray(target_items, dtype=np.int64)
    train_idx = np.setdiff1d(np.arange(len(item_ids), dtype=np.int64), target_idx, assume_unique=False)
    train_items = set(item_ids[train_idx].tolist())
    train_df = proc_df[proc_df["item_id"].astype(str).isin(train_items)].copy()
    x_train, train_user_ids = _to_sparse_matrix_for_items_with_users(train_df, item_ids)
    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": holdout,
        "test_holdout": holdout,
        "train_item_indices": train_idx,
        "val_item_indices": target_idx,
        "test_item_indices": target_idx,
        "train_user_ids": train_user_ids,
        "val_user_ids": None,
        "test_user_ids": None,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": False,
            "is_temporal": True,
            "is_future_blind": True,
            "leakage_note": "Temporal split uses global timestamp cutoffs so training interactions precede evaluation targets.",
            "temporal_test_frac": args.temporal_test_frac,
            "timestamp_cutoff": holdout.get("timestamp_cutoff"),
            "cold_target_items": int(len(target_idx)),
        },
    }


def _normalize_amazon_split_df(df: pd.DataFrame, args, *, valid_items: set[str]) -> pd.DataFrame:
    out = df.rename(columns={"parent_asin": "item_id", "rating": "value"}).copy()
    out["user_id"] = out["user_id"].astype(str)
    out["item_id"] = out["item_id"].astype(str)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out["timestamp"] = pd.to_numeric(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["user_id", "item_id", "value"])
    out = out[out["item_id"].isin(valid_items)].copy()
    if args.min_value_to_keep is not None:
        out = out[out["value"] >= float(args.min_value_to_keep)].copy()
    if args.set_all_values_to is not None:
        out["value"] = float(args.set_all_values_to)
    return out


def _history_to_indices(raw_history, item_to_idx: dict[str, int]) -> np.ndarray:
    if raw_history is None or (isinstance(raw_history, float) and pd.isna(raw_history)):
        return np.array([], dtype=np.int64)
    if isinstance(raw_history, str):
        items = raw_history.split()
    elif isinstance(raw_history, (list, tuple)):
        items = [str(x) for x in raw_history]
    else:
        items = str(raw_history).split()
    indices = sorted({item_to_idx[item] for item in items if item in item_to_idx})
    return np.asarray(indices, dtype=np.int64)


def _build_amazon_predefined_temporal_split(args, ds: AmazonReviews2023, proc_df: pd.DataFrame):
    raw_splits = ds.load_timestamp_splits_with_history()
    valid_items = set(proc_df["item_id"].astype(str))
    train_df = _normalize_amazon_split_df(raw_splits["train"], args, valid_items=valid_items)
    valid_df = _normalize_amazon_split_df(raw_splits["valid"], args, valid_items=valid_items)
    test_df = _normalize_amazon_split_df(raw_splits["test"], args, valid_items=valid_items)

    item_ids = np.array(sorted(valid_items))
    item_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}
    train_items = set(train_df["item_id"].astype(str).unique())
    train_idx = np.asarray([item_to_idx[item] for item in sorted(train_items) if item in item_to_idx], dtype=np.int64)

    x_train, train_user_ids = _to_sparse_matrix_for_items_with_users(train_df, item_ids)

    def _holdout(df: pd.DataFrame) -> dict[str, object]:
        source_indices: list[np.ndarray] = []
        target_indices: list[np.ndarray] = []
        user_ids: list[str] = []
        target_item_idx: set[int] = set()
        for _, row in df.iterrows():
            target_item = str(row["item_id"])
            target_idx = item_to_idx.get(target_item)
            if target_idx is None or target_item in train_items:
                continue
            src = _history_to_indices(row.get("history", ""), item_to_idx)
            src = np.asarray([idx for idx in src.tolist() if item_ids[idx] in train_items], dtype=np.int64)
            if len(src) >= args.min_source_items:
                source_indices.append(src)
                target_indices.append(np.asarray([target_idx], dtype=np.int64))
                user_ids.append(str(row["user_id"]))
                target_item_idx.add(target_idx)
        return {
            "item_ids": item_ids,
            "source_indices": source_indices,
            "target_indices": target_indices,
            "user_ids": np.asarray(user_ids, dtype=str),
            "target_item_indices": np.asarray(sorted(target_item_idx), dtype=np.int64),
        }

    val_holdout = _holdout(valid_df)
    test_holdout = _holdout(test_df)
    val_idx = val_holdout.pop("target_item_indices")
    test_idx = test_holdout.pop("target_item_indices")
    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": val_holdout,
        "test_holdout": test_holdout,
        "train_item_indices": np.sort(train_idx),
        "val_item_indices": val_idx,
        "test_item_indices": test_idx,
        "train_user_ids": train_user_ids,
        "val_user_ids": None,
        "test_user_ids": None,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": False,
            "is_temporal": True,
            "is_future_blind": True,
            "leakage_note": "Amazon predefined temporal split uses McAuley's timestamp split with histories.",
            "amazon_predefined_split": "0core_timestamp_w_his",
            "val_items": int(len(val_idx)),
            "test_items": int(len(test_idx)),
        },
    }


def _build_split_payload(args, ds, proc_df):
    if args.split_mode == "user_split":
        return _build_user_split(args, ds, proc_df)
    if args.split_mode == "item_split":
        return _build_item_split(args, proc_df)
    if args.split_mode == "leave_last_out":
        return _build_leave_last_out_split(args, proc_df)
    if args.split_mode == "temporal":
        if isinstance(ds, AmazonReviews2023):
            return _build_amazon_predefined_temporal_split(args, ds, proc_df)
        return _build_temporal_split(args, proc_df)
    raise ValueError(f"Unsupported split_mode: {args.split_mode}")


def _build_recsys_checkpoint_from_args(args) -> Path:
    args, spec = _resolve_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with _CheckpointProgress(enabled=getattr(args, "show_progress", True), total=6) as progress:
        progress.step("Loading interactions")
        ds = _make_dataset(args, spec)
        raw_df = ds.get_interactions()

        progress.step("Preprocessing interactions")
        proc_df = ds.preprocess_interactions_for_recsys(
            raw_df,
            min_value_to_keep=args.min_value_to_keep,
            user_min_support=args.min_user_support,
            item_min_support=args.item_min_support,
            set_all_values_to=args.set_all_values_to,
        )

        progress.step(f"Building {args.split_mode} split")
        split_payload = _build_split_payload(args, ds, proc_df)
        item_ids = split_payload["item_ids"]
        val_holdout = split_payload["val_holdout"]
        test_holdout = split_payload["test_holdout"]
        train_item_indices = split_payload.get("train_item_indices")
        val_item_indices = split_payload.get("val_item_indices")
        test_item_indices = split_payload.get("test_item_indices")
        train_item_count = int(len(train_item_indices)) if train_item_indices is not None else int(len(item_ids))
        val_item_count = int(len(val_item_indices)) if val_item_indices is not None else 0
        test_item_count = int(len(test_item_indices)) if test_item_indices is not None else 0

        progress.step("Building annotations")
        entity_tag_matrix, tag_names, annotation_name = _build_entity_tag_matrix(args, ds, item_ids)

        progress.step("Loading item metadata")
        entity_metadata = ds.get_item_metadata()

        progress.step("Writing checkpoint")
        with update_checkpoint(args.checkpoint_path) as root:
            save_recsys_split(
                root,
                item_ids=item_ids,
                x_train=split_payload["x_train"],
                val_source_indices=val_holdout["source_indices"],
                val_target_indices=val_holdout["target_indices"],
                test_source_indices=test_holdout["source_indices"],
                test_target_indices=test_holdout["target_indices"],
                train_source_matrix=split_payload.get("train_source_matrix"),
                train_target_matrix=split_payload.get("train_target_matrix"),
                train_user_ids=split_payload.get("train_user_ids"),
                val_user_ids=split_payload.get("val_user_ids"),
                test_user_ids=split_payload.get("test_user_ids"),
                val_eval_user_ids=val_holdout.get("user_ids"),
                test_eval_user_ids=test_holdout.get("user_ids"),
                train_item_indices=train_item_indices,
                val_item_indices=val_item_indices,
                test_item_indices=test_item_indices,
                entity_tag_matrix=entity_tag_matrix,
                tag_names=tag_names,
                entity_metadata=entity_metadata,
                metadata={
                    "dataset": args.dataset,
                    "seed": args.seed,
                    "val_users": args.val_users,
                    "test_users": args.test_users,
                    "min_user_support": args.min_user_support,
                    "item_min_support": args.item_min_support,
                    "min_value_to_keep": args.min_value_to_keep,
                    "set_all_values_to": args.set_all_values_to,
                    "eval_fold": args.eval_fold,
                    "split_mode": args.split_mode,
                    "min_source_items": args.min_source_items,
                    "min_target_items": args.min_target_items,
                    "train_items": train_item_count,
                    "val_cold_items": val_item_count,
                    "test_cold_items": test_item_count,
                    "n_train_users": int(len(split_payload["train_user_ids"])) if split_payload.get("train_user_ids") is not None else None,
                    "n_val_users": int(len(split_payload["val_user_ids"])) if split_payload.get("val_user_ids") is not None else None,
                    "n_test_users": int(len(split_payload["test_user_ids"])) if split_payload.get("test_user_ids") is not None else None,
                    "n_val_eval_users": int(len(val_holdout["source_indices"])),
                    "n_test_eval_users": int(len(test_holdout["source_indices"])),
                    "split_files": {
                        "train_source_matrix": "data/train_source_matrix.npz",
                        "train_target_matrix": "data/train_target_matrix.npz",
                        "val_source_matrix": "data/val_source_matrix.npz",
                        "val_target_matrix": "data/val_target_matrix.npz",
                        "test_source_matrix": "data/test_source_matrix.npz",
                        "test_target_matrix": "data/test_target_matrix.npz",
                        "train_user_ids": "data/train_user_ids.npy",
                        "val_user_ids": "data/val_user_ids.npy",
                        "test_user_ids": "data/test_user_ids.npy",
                        "val_eval_user_ids": "data/val_eval_user_ids.npy",
                        "test_eval_user_ids": "data/test_eval_user_ids.npy",
                    },
                    **split_payload["extra_metadata"],
                    "annotation_source": args.annotation_source,
                    "annotation_min_count": args.annotation_min_count,
                    "amazon_category": args.amazon_category if args.dataset == "amazon2023" else None,
                    "metadata_text_fields": (
                        [field.strip() for field in args.metadata_text_fields.split(",") if field.strip()]
                        if args.metadata_text_fields
                        else list(getattr(spec.cls, "default_text_fields", ()))
                    ),
                    "min_entity_text_words": args.min_entity_text_words,
                    "include_image_urls": bool(getattr(args, "include_image_urls", False)),
                    "annotations": {
                        "entity_tags": annotation_name,
                        "n_tags": int(len(tag_names)) if tag_names is not None else 0,
                        "entity_metadata": True,
                    },
                },
            )
    return Path(args.checkpoint_path)


def build_recsys_checkpoint(
    *,
    dataset: str,
    data_dir: str = "data",
    checkpoint_path: str | None = None,
    seed: int | None = None,
    val_users: int | None = None,
    test_users: int | None = None,
    min_user_support: int | None = None,
    item_min_support: int | None = None,
    min_value_to_keep: float | None = None,
    set_all_values_to: float | None = None,
    eval_fold: int = 0,
    split_mode: str = "user_split",
    val_items: int | None = None,
    test_items: int | None = None,
    item_val_frac: float = 0.05,
    item_test_frac: float = 0.10,
    temporal_test_frac: float = 0.10,
    min_source_items: int = 1,
    min_target_items: int = 1,
    amazon_category: str = "Toys_and_Games",
    metadata_text_fields: str | list[str] | tuple[str, ...] | None = None,
    min_entity_text_words: int = 30,
    include_image_urls: bool = False,
    annotation_source: str = "genres",
    annotation_min_count: int = 100,
    show_progress: bool = True,
) -> Path:
    """Build a recommender-system split checkpoint and return its path."""
    args = _build_args(
        dataset=dataset,
        data_dir=data_dir,
        checkpoint_path=checkpoint_path,
        seed=seed,
        val_users=val_users,
        test_users=test_users,
        min_user_support=min_user_support,
        item_min_support=item_min_support,
        min_value_to_keep=min_value_to_keep,
        set_all_values_to=set_all_values_to,
        eval_fold=eval_fold,
        split_mode=split_mode,
        val_items=val_items,
        test_items=test_items,
        item_val_frac=item_val_frac,
        item_test_frac=item_test_frac,
        temporal_test_frac=temporal_test_frac,
        min_source_items=min_source_items,
        min_target_items=min_target_items,
        amazon_category=amazon_category,
        metadata_text_fields=metadata_text_fields,
        min_entity_text_words=min_entity_text_words,
        include_image_urls=include_image_urls,
        annotation_source=annotation_source,
        annotation_min_count=annotation_min_count,
        show_progress=show_progress,
    )
    return _build_recsys_checkpoint_from_args(args)


def main():
    args = parse_args()
    path = _build_recsys_checkpoint_from_args(args)
    print(f"Saved {args.dataset} data split checkpoint to: {path}")


if __name__ == "__main__":
    main()
