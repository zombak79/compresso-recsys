from __future__ import annotations

import argparse
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.checkpoint import (
    _indices_to_csr,
    save_recsys_split,
    update_checkpoint,
)
from compresso_recsys.datasets import AmazonReviews2023, Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.sequences import ItemSequences
from compresso_recsys.retrieval import (
    LEAVE_LAST_OUT_MIN_HISTORY,
    LEAVE_LAST_OUT_STAGES,
    build_eval_holdout,
    build_item_cold_holdout,
    build_leave_last_out_holdout,
    leave_last_out_histories,
    leave_last_out_stage_slices,
)


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

    def detail(self, message: str) -> None:
        """Update the active step label without advancing the progress bar."""
        if not self.enabled:
            return
        if self.bar is None:
            print(f"[compresso-recsys] {message}", flush=True)
            return
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
    p.add_argument("--eval_draws", type=int, default=5)
    p.add_argument("--eval_holdout_frac", type=float, default=0.2)
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
    p.add_argument("--temporal_test_frac", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--temporal_period_hours",
        type=float,
        default=DEFAULT_TEMPORAL_PERIOD_HOURS,
        help="Width in hours of each train/validation/test temporal target window.",
    )
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
    eval_draws: int = 5,
    eval_holdout_frac: float = 0.2,
    split_mode: str = "user_split",
    val_items: int | None = None,
    test_items: int | None = None,
    item_val_frac: float = 0.05,
    item_test_frac: float = 0.10,
    temporal_test_frac: float | None = None,
    temporal_period_hours: float = DEFAULT_TEMPORAL_PERIOD_HOURS,
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
    if eval_draws < 1:
        raise ValueError(f"eval_draws must be >= 1, got {eval_draws!r}")
    if not 0.0 < eval_holdout_frac < 1.0:
        raise ValueError(
            f"eval_holdout_frac must be strictly between 0 and 1, "
            f"got {eval_holdout_frac!r}"
        )
    if split_mode not in {"user_split", "item_split", "leave_last_out", "temporal"}:
        raise ValueError(f"Unsupported split_mode: {split_mode!r}")
    if annotation_source not in {"genres", "ml20m_tags", "goodbooks_tags", "none"}:
        raise ValueError(f"Unsupported annotation_source: {annotation_source!r}")
    if (
        isinstance(temporal_period_hours, bool)
        or not np.isfinite(temporal_period_hours)
        or temporal_period_hours <= 0
    ):
        raise ValueError("temporal_period_hours must be finite and > 0")
    if temporal_test_frac is not None:
        warnings.warn(
            "temporal_test_frac is deprecated and ignored; use "
            "temporal_period_hours instead",
            DeprecationWarning,
            stacklevel=2,
        )
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
        eval_draws=eval_draws,
        eval_holdout_frac=eval_holdout_frac,
        split_mode=split_mode,
        val_items=val_items,
        test_items=test_items,
        item_val_frac=item_val_frac,
        item_test_frac=item_test_frac,
        temporal_test_frac=temporal_test_frac,
        temporal_period_hours=float(temporal_period_hours),
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
        eval_draws=args.eval_draws,
        eval_holdout_frac=args.eval_holdout_frac,
    )
    test_holdout = build_eval_holdout(
        train_item_ids=item_ids,
        eval_interactions=split.test,
        min_user_support=args.min_user_support,
        random_state=args.seed,
        eval_draws=args.eval_draws,
        eval_holdout_frac=args.eval_holdout_frac,
    )
    catalog_item_ids = test_holdout["item_ids"]
    return {
        "item_ids": catalog_item_ids,
        "x_train": x_train,
        "train_source_matrix": x_train,
        "train_target_matrix": x_train,
        "val_holdout": val_holdout,
        "test_holdout": test_holdout,
        # Every item is present while training and no later phase introduces new
        # ones, so training spans the catalog and validation/test add nothing.
        # Written out explicitly instead of left as None so that every split mode
        # stores all three partitions and none of them has to be inferred.
        "train_item_indices": np.arange(len(catalog_item_ids), dtype=np.int64),
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
    """Chronological per-user holdout with the catalog left intact.

    Each user's last interaction is the test target, the one before it the
    validation target, and the one before that the training target. Sources are
    the corresponding prefixes.

    Nothing is stripped from training. Item partitions are *observed* rather than
    imposed: an item lands in the validation or test partition only when every
    one of its occurrences happens to fall in a held-out tail, which on dense
    data means the partitions come out empty and on sparse data means they hold
    the genuinely new items.
    """
    item_ids = np.array(sorted(proc_df["item_id"].astype(str).unique()))
    histories, user_ids = leave_last_out_histories(
        item_ids=item_ids,
        interactions=proc_df,
        min_history=LEAVE_LAST_OUT_MIN_HISTORY,
    )
    if len(user_ids) == 0:
        raise ValueError(
            f"leave_last_out needs users with at least "
            f"{LEAVE_LAST_OUT_MIN_HISTORY} interactions; none qualified"
        )

    stages: dict[str, dict[str, list[np.ndarray]]] = {}
    ordered_sources: dict[str, list[np.ndarray]] = {}
    for stage in LEAVE_LAST_OUT_STAGES:
        sources, targets, in_order = [], [], []
        for history in histories:
            source, target = leave_last_out_stage_slices(history, stage)
            # Two views of the same events, taken in one pass: the matrix wants a
            # set, the sequence wants the order. Deriving one from the other later
            # is impossible in the direction that matters.
            sources.append(np.unique(source))
            targets.append(np.unique(target))
            in_order.append(source)
        stages[stage] = {"source_indices": sources, "target_indices": targets}
        ordered_sources[stage] = in_order

    n_items = len(item_ids)
    train_source = _indices_to_csr(stages["train"]["source_indices"], n_cols=n_items)
    train_target = _indices_to_csr(stages["train"]["target_indices"], n_cols=n_items)
    # The same relationship temporal uses: the training window is the pair's
    # union, and a symmetric model trains on that.
    x_train = train_source.maximum(train_target).tocsr()
    # Items first seen in each phase, exactly as the temporal stages compute it.
    def _observed(stage: str) -> np.ndarray:
        rows = stages[stage]["source_indices"] + stages[stage]["target_indices"]
        return np.unique(np.concatenate(rows)) if rows else np.array([], dtype=np.int64)

    train_item_indices = _observed("train")
    val_item_indices = np.setdiff1d(_observed("val"), train_item_indices)
    test_item_indices = np.setdiff1d(
        _observed("test"), np.union1d(train_item_indices, val_item_indices)
    )

    # The training window in order, which is what a sequential model trains on:
    # it shifts internally, so handing over only the source would discard the
    # last transition the matrix pair encodes explicitly.
    train_window = [history[:-2] for history in histories]
    sequences = {
        "x_train_sequences": ItemSequences.from_rows(train_window, n_items=n_items),
        "train_source_sequences": ItemSequences.from_rows(
            ordered_sources["train"], n_items=n_items
        ),
        "val_source_sequences": ItemSequences.from_rows(
            ordered_sources["val"], n_items=n_items
        ),
        "test_source_sequences": ItemSequences.from_rows(
            ordered_sources["test"], n_items=n_items
        ),
    }

    holdouts = {
        stage: {
            "item_ids": item_ids,
            "source_indices": stages[stage]["source_indices"],
            "target_indices": stages[stage]["target_indices"],
            "user_ids": user_ids,
        }
        for stage in LEAVE_LAST_OUT_STAGES
    }

    return {
        "item_ids": item_ids,
        "x_train": x_train,
        "train_source_matrix": train_source,
        "train_target_matrix": train_target,
        **sequences,
        "val_holdout": holdouts["val"],
        "test_holdout": holdouts["test"],
        "train_holdout": holdouts["train"],
        "train_item_indices": train_item_indices,
        "val_item_indices": val_item_indices,
        "test_item_indices": test_item_indices,
        "train_user_ids": user_ids,
        "val_user_ids": user_ids,
        "test_user_ids": user_ids,
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": bool(val_item_indices.size or test_item_indices.size),
            "is_temporal": False,
            "is_future_blind": False,
            "leakage_note": (
                "Leave-last-out is chronological within each user but not "
                "globally future-blind: another user's training interactions may "
                "post-date this user's test target."
            ),
            "min_history": LEAVE_LAST_OUT_MIN_HISTORY,
            "eligible_users": int(len(user_ids)),
            "new_val_items": int(val_item_indices.size),
            "new_test_items": int(test_item_indices.size),
        },
    }


def _csr_row_indices(matrix: csr_matrix) -> list[np.ndarray]:
    """Per-row column indices as read-only views into ``matrix.indices``.

    The split returned by the builders keeps ``matrix`` alongside this list, so
    copying every row would duplicate the whole index buffer while the original
    stays alive: 80 MB of the 200 MB these lists cost at a million users with
    twenty interactions each. Slices share that buffer instead.

    The views are marked read-only because they alias the matrix, and writing
    through one would silently corrupt the other. Consumers do not need to
    write: both ``_indices_to_csr`` and ``_as_obj_array`` convert to int64,
    and the retrieval helpers concatenate, each producing fresh writable arrays.
    """
    indices = matrix.indices
    indptr = matrix.indptr
    rows: list[np.ndarray] = []
    for row in range(matrix.shape[0]):
        view = indices[indptr[row] : indptr[row + 1]]
        view.flags.writeable = False
        rows.append(view)
    return rows


def _filter_temporal_pair(
    source: csr_matrix,
    target: csr_matrix,
    *,
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    inherited_items: int,
    min_user_support: int,
    item_min_support: int,
    min_source_items: int,
    min_target_items: int,
    stage: str,
) -> tuple[csr_matrix, csr_matrix, np.ndarray, np.ndarray, dict[str, int]]:
    if source.shape != target.shape:
        raise ValueError(f"{stage} source and target shapes must match")
    if source.shape != (len(user_ids), len(item_ids)):
        raise ValueError(f"{stage} matrix shape does not match its IDs")

    initial_users = int(source.shape[0])
    initial_items = int(source.shape[1])
    initial_new_items = initial_items - int(inherited_items)
    initial_source_interactions = int(source.nnz)
    initial_target_interactions = int(target.nnz)
    iterations = 0
    while True:
        iterations += 1
        combined = source.maximum(target)
        row_keep = (
            (source.getnnz(axis=1) >= min_source_items)
            & (target.getnnz(axis=1) >= min_target_items)
            & (combined.getnnz(axis=1) >= min_user_support)
        )
        if not bool(row_keep.any()):
            raise ValueError(
                f"{stage} temporal window has no users after support filtering"
            )
        rows_changed = not bool(row_keep.all())
        if rows_changed:
            source = source[row_keep].tocsr()
            target = target[row_keep].tocsr()
            user_ids = user_ids[row_keep]
            combined = source.maximum(target)

        item_support = np.asarray(combined.getnnz(axis=0)).ravel()
        column_keep = np.ones(source.shape[1], dtype=bool)
        column_keep[inherited_items:] = (
            item_support[inherited_items:] >= item_min_support
        )
        columns_changed = not bool(column_keep.all())
        if columns_changed:
            source = source[:, column_keep].tocsr()
            target = target[:, column_keep].tocsr()
            item_ids = item_ids[column_keep]

        if not rows_changed and not columns_changed:
            break

    stats = {
        "initial_users": initial_users,
        "users": int(source.shape[0]),
        "initial_items": initial_items,
        "items": int(source.shape[1]),
        "inherited_items": int(inherited_items),
        "initial_new_items": int(initial_new_items),
        "new_items": int(source.shape[1] - inherited_items),
        "initial_source_interactions": initial_source_interactions,
        "initial_target_interactions": initial_target_interactions,
        "source_interactions": int(source.nnz),
        "target_interactions": int(target.nnz),
        "support_iterations": int(iterations),
    }
    return source, target, user_ids, item_ids, stats


def _sequences_from_temporal_codes(
    *,
    event_mask: np.ndarray,
    global_user_codes: np.ndarray,
    global_item_codes: np.ndarray,
    timestamps: np.ndarray,
    user_lookup: np.ndarray,
    item_lookup: np.ndarray,
    n_rows: int,
    n_items: int,
) -> ItemSequences:
    """Chronological histories for the events a mask selects.

    The matrix twin of this drops order and merges duplicates; both read the same
    masked events, so the two views describe the same interactions rather than
    two things that happen to look alike.

    Sorting is by ``(row, timestamp)`` with a stable kind, so events sharing a
    timestamp keep the order the source data gave them rather than an arbitrary
    one.
    """
    if n_rows == 0:
        return ItemSequences.from_rows([], n_items=n_items)

    rows = user_lookup[global_user_codes[event_mask]]
    cols = item_lookup[global_item_codes[event_mask]]
    times = timestamps[event_mask]
    keep = (rows >= 0) & (cols >= 0)
    rows, cols, times = rows[keep], cols[keep], times[keep]

    order = np.lexsort((times, rows))
    rows, cols = rows[order], cols[order]

    counts = np.bincount(rows, minlength=n_rows)
    indptr = np.concatenate(([0], np.cumsum(counts)))
    return ItemSequences(values=cols, indptr=indptr, n_items=n_items)


def _matrix_from_temporal_codes(
    *,
    event_mask: np.ndarray,
    global_user_codes: np.ndarray,
    global_item_codes: np.ndarray,
    values: np.ndarray,
    user_lookup: np.ndarray,
    item_lookup: np.ndarray,
    shape: tuple[int, int],
) -> csr_matrix:
    if not bool(event_mask.any()) or shape[0] == 0 or shape[1] == 0:
        return csr_matrix(shape, dtype=np.float32)

    rows = user_lookup[global_user_codes[event_mask]]
    cols = item_lookup[global_item_codes[event_mask]]
    valid = (rows >= 0) & (cols >= 0)
    matrix = csr_matrix(
        (values[event_mask][valid], (rows[valid], cols[valid])),
        shape=shape,
        dtype=np.float32,
    )
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return matrix


def _temporal_user_upper_bound(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    global_user_codes: np.ndarray,
    n_users: int,
    min_user_support: int,
    min_source_items: int,
    min_target_items: int,
) -> tuple[np.ndarray, int]:
    """Reject users that cannot meet support before allocating tall CSRs.

    Event counts are an upper bound on distinct nonzero item counts. Keeping a
    user here does not guarantee eligibility, but rejecting one is always safe;
    the exact fixed-point filter still runs on the resulting sparse matrices.
    """
    source_counts = np.bincount(
        global_user_codes[source_mask], minlength=n_users
    )
    target_counts = np.bincount(
        global_user_codes[target_mask], minlength=n_users
    )
    keep = source_counts >= min_source_items
    keep &= target_counts >= min_target_items
    source_counts += target_counts
    initial_users = int(np.count_nonzero(source_counts > 0))
    keep &= source_counts >= min_user_support
    return np.flatnonzero(keep).astype(np.int64, copy=False), initial_users


def _timestamps_in_seconds(values: pd.Series) -> np.ndarray:
    timestamps = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(timestamps)
    if not bool(finite.any()):
        raise ValueError("temporal split requires non-empty timestamp values")
    magnitude = float(np.max(np.abs(timestamps[finite])))
    if magnitude >= 1e17:
        timestamps /= 1e9
    elif magnitude >= 1e14:
        timestamps /= 1e6
    elif magnitude >= 1e11:
        timestamps /= 1e3
    return timestamps


def _build_temporal_stage(
    *,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    global_user_codes: np.ndarray,
    global_item_codes: np.ndarray,
    global_user_ids: np.ndarray,
    global_item_ids: np.ndarray,
    values: np.ndarray,
    timestamps: np.ndarray,
    inherited_item_codes: np.ndarray,
    args,
    stage: str,
) -> dict[str, object]:
    observed_item_codes = np.unique(
        global_item_codes[source_mask | target_mask]
    )
    inherited_item_codes = np.asarray(inherited_item_codes, dtype=np.int64)
    new_item_codes = np.setdiff1d(
        observed_item_codes,
        inherited_item_codes,
        assume_unique=True,
    )
    item_codes = np.concatenate((inherited_item_codes, new_item_codes))

    user_codes, initial_users = _temporal_user_upper_bound(
        source_mask=source_mask,
        target_mask=target_mask,
        global_user_codes=global_user_codes,
        n_users=len(global_user_ids),
        min_user_support=args.min_user_support,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
    )
    if len(user_codes) == 0:
        raise ValueError(
            f"{stage} temporal window has no users after support filtering"
        )

    user_lookup = np.full(len(global_user_ids), -1, dtype=np.int64)
    user_lookup[user_codes] = np.arange(len(user_codes), dtype=np.int64)
    eligible_users = np.zeros(len(global_user_ids), dtype=bool)
    eligible_users[user_codes] = True
    matrix_source_mask = source_mask & eligible_users[global_user_codes]
    matrix_target_mask = target_mask & eligible_users[global_user_codes]
    item_lookup = np.full(len(global_item_ids), -1, dtype=np.int64)
    item_lookup[item_codes] = np.arange(len(item_codes), dtype=np.int64)
    shape = (len(user_codes), len(item_codes))
    source = _matrix_from_temporal_codes(
        event_mask=matrix_source_mask,
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        values=values,
        user_lookup=user_lookup,
        item_lookup=item_lookup,
        shape=shape,
    )
    target = _matrix_from_temporal_codes(
        event_mask=matrix_target_mask,
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        values=values,
        user_lookup=user_lookup,
        item_lookup=item_lookup,
        shape=shape,
    )
    user_ids = global_user_ids[user_codes]
    item_ids = global_item_ids[item_codes]
    source, target, user_ids, item_ids, stats = _filter_temporal_pair(
        source,
        target,
        user_ids=user_ids,
        item_ids=item_ids,
        inherited_items=len(inherited_item_codes),
        min_user_support=args.min_user_support,
        item_min_support=args.item_min_support,
        min_source_items=args.min_source_items,
        min_target_items=args.min_target_items,
        stage=stage,
    )
    stats["initial_users"] = initial_users
    stats["prefiltered_users"] = int(len(user_codes))
    retained_item_codes = pd.Index(global_item_ids).get_indexer(item_ids)

    # _filter_temporal_pair drops users and items, so the lookups built above no
    # longer describe the returned matrices. Rebuild them from what survived, or
    # the sequence rows would address a row space the matrices no longer have.
    retained_user_codes = pd.Index(global_user_ids).get_indexer(user_ids)
    final_user_lookup = np.full(len(global_user_ids), -1, dtype=np.int64)
    final_user_lookup[retained_user_codes] = np.arange(len(user_ids), dtype=np.int64)
    final_item_lookup = np.full(len(global_item_ids), -1, dtype=np.int64)
    final_item_lookup[retained_item_codes] = np.arange(len(item_ids), dtype=np.int64)

    def _stage_sequences(mask: np.ndarray) -> ItemSequences:
        return _sequences_from_temporal_codes(
            event_mask=mask,
            global_user_codes=global_user_codes,
            global_item_codes=global_item_codes,
            timestamps=timestamps,
            user_lookup=final_user_lookup,
            item_lookup=final_item_lookup,
            n_rows=len(user_ids),
            n_items=len(item_ids),
        )

    return {
        "source": source,
        "target": target,
        "source_sequences": _stage_sequences(source_mask),
        # The stage's whole window, source and target together. Each stage is
        # filtered independently, so a window sequence taken from a later stage
        # would address a different row and column space than this stage's
        # matrices.
        "window_sequences": _stage_sequences(source_mask | target_mask),
        "user_ids": user_ids,
        "item_ids": item_ids,
        "item_codes": retained_item_codes.astype(np.int64, copy=False),
        "stats": stats,
    }


def _build_temporal_split(args, proc_df, progress: _CheckpointProgress | None = None):
    if "timestamp" not in proc_df.columns:
        raise ValueError("temporal split requires a timestamp column")
    timestamps = _timestamps_in_seconds(proc_df["timestamp"])
    finite = np.isfinite(timestamps)
    if not bool(finite.any()):
        raise ValueError("temporal split requires non-empty timestamp values")
    timestamps = timestamps[finite]
    values = proc_df.loc[finite, "value"].to_numpy(dtype=np.float32)
    global_user_codes, global_user_ids = pd.factorize(
        proc_df.loc[finite, "user_id"], sort=True
    )
    global_item_codes, global_item_ids = pd.factorize(
        proc_df.loc[finite, "item_id"], sort=True
    )
    global_user_codes = global_user_codes.astype(np.int64, copy=False)
    global_item_codes = global_item_codes.astype(np.int64, copy=False)
    global_user_ids = np.asarray(global_user_ids, dtype=object)
    global_item_ids = np.asarray(global_item_ids, dtype=object)

    period_seconds = float(args.temporal_period_hours) * 60.0 * 60.0
    timestamp_min = float(timestamps.min())
    timestamp_max = float(timestamps.max())
    train_target_start = timestamp_max - 3.0 * period_seconds
    validation_target_start = timestamp_max - 2.0 * period_seconds
    test_target_start = timestamp_max - period_seconds
    if train_target_start <= timestamp_min:
        span_hours = (timestamp_max - timestamp_min) / 3600.0
        raise ValueError(
            "temporal_period_hours requires three target windows shorter than "
            f"the available {span_hours:.3f}-hour timestamp span"
        )

    if progress is not None:
        progress.detail("Building temporal split: train")
    train_stage = _build_temporal_stage(
        source_mask=timestamps < train_target_start,
        target_mask=(timestamps >= train_target_start)
        & (timestamps < validation_target_start),
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        global_user_ids=global_user_ids,
        global_item_ids=global_item_ids,
        values=values,
        timestamps=timestamps,
        inherited_item_codes=np.asarray([], dtype=np.int64),
        args=args,
        stage="train",
    )
    if progress is not None:
        progress.detail("Building temporal split: validation")
    val_stage = _build_temporal_stage(
        source_mask=timestamps < validation_target_start,
        target_mask=(timestamps >= validation_target_start)
        & (timestamps < test_target_start),
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        global_user_ids=global_user_ids,
        global_item_ids=global_item_ids,
        values=values,
        timestamps=timestamps,
        inherited_item_codes=train_stage["item_codes"],
        args=args,
        stage="validation",
    )
    if progress is not None:
        progress.detail("Building temporal split: test")
    test_stage = _build_temporal_stage(
        source_mask=timestamps < test_target_start,
        target_mask=timestamps >= test_target_start,
        global_user_codes=global_user_codes,
        global_item_codes=global_item_codes,
        global_user_ids=global_user_ids,
        global_item_ids=global_item_ids,
        values=values,
        timestamps=timestamps,
        inherited_item_codes=val_stage["item_codes"],
        args=args,
        stage="test",
    )

    train_item_ids = np.asarray(train_stage["item_ids"]).astype(str)
    val_item_ids = np.asarray(val_stage["item_ids"]).astype(str)
    test_item_ids = np.asarray(test_stage["item_ids"]).astype(str)
    train_source = train_stage["source"]
    train_target = train_stage["target"]
    val_source = val_stage["source"]
    val_target = val_stage["target"]
    test_source = test_stage["source"]
    test_target = test_stage["target"]
    x_train = train_source.maximum(train_target).tocsr()

    # The training window in order. For both chronological modes the validation
    # source is that same window, so these two agree exactly, mirroring x_train
    # and val_source_matrix on the matrix side.
    sequences = {
        # x_train is the train stage's window, so its sequence must come from the
        # same stage: val_stage covers the same events but in its own filtered
        # row and column space.
        "x_train_sequences": train_stage["window_sequences"],
        "train_source_sequences": train_stage["source_sequences"],
        "val_source_sequences": val_stage["source_sequences"],
        "test_source_sequences": test_stage["source_sequences"],
    }

    train_count = len(train_item_ids)
    val_count = len(val_item_ids)
    test_count = len(test_item_ids)
    return {
        "item_ids": test_item_ids,
        "train_item_ids": train_item_ids,
        "val_item_ids": val_item_ids,
        "test_item_ids": test_item_ids,
        "x_train": x_train,
        "train_source_matrix": train_source,
        "train_target_matrix": train_target,
        **sequences,
        "val_source_matrix": val_source,
        "val_target_matrix": val_target,
        "test_source_matrix": test_source,
        "test_target_matrix": test_target,
        "val_holdout": {
            "source_indices": _csr_row_indices(val_source),
            "target_indices": _csr_row_indices(val_target),
            "user_ids": val_stage["user_ids"],
        },
        "test_holdout": {
            "source_indices": _csr_row_indices(test_source),
            "target_indices": _csr_row_indices(test_target),
            "user_ids": test_stage["user_ids"],
        },
        "train_item_indices": np.arange(train_count, dtype=np.int64),
        "val_item_indices": np.arange(train_count, val_count, dtype=np.int64),
        "test_item_indices": np.arange(val_count, test_count, dtype=np.int64),
        "train_user_ids": train_stage["user_ids"],
        "val_user_ids": val_stage["user_ids"],
        "test_user_ids": test_stage["user_ids"],
        "extra_metadata": {
            "has_user_partitions": False,
            "has_item_partitions": False,
            "has_stage_item_spaces": True,
            "is_temporal": True,
            "is_future_blind": True,
            "leakage_note": (
                "Temporal targets follow expanding histories. Item support may "
                "use a complete target window only to define benchmark eligibility."
            ),
            "temporal_period_hours": float(args.temporal_period_hours),
            "timestamp_unit": "unix_seconds",
            "timestamp_min": timestamp_min,
            "timestamp_max": timestamp_max,
            "train_target_start": train_target_start,
            "validation_target_start": validation_target_start,
            "test_target_start": test_target_start,
            "train_stage": train_stage["stats"],
            "validation_stage": val_stage["stats"],
            "test_stage": test_stage["stats"],
            "val_cold_items": int(val_count - train_count),
            "test_new_items": int(test_count - val_count),
            "test_model_cold_items": int(test_count - train_count),
        },
    }


def _distinct_eval_users(holdout) -> int | None:
    """How many distinct users a holdout evaluates, or ``None`` if unrecorded.

    Not the same as its row count. ``eval_draws`` above 1 gives each user one
    row per draw and tiles the identifiers to match.
    """
    user_ids = holdout.get("user_ids")
    if user_ids is None:
        return None
    return int(np.unique(np.asarray(user_ids)).shape[0])


def _build_split_payload(args, ds, proc_df, progress: _CheckpointProgress | None = None):
    if args.split_mode == "user_split":
        return _build_user_split(args, ds, proc_df)
    if args.split_mode == "item_split":
        return _build_item_split(args, proc_df)
    if args.split_mode == "leave_last_out":
        return _build_leave_last_out_split(args, proc_df)
    if args.split_mode == "temporal":
        return _build_temporal_split(args, proc_df, progress=progress)
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
        temporal = args.split_mode == "temporal"
        proc_df = ds.preprocess_interactions_for_recsys(
            raw_df,
            min_value_to_keep=args.min_value_to_keep,
            user_min_support=1 if temporal else args.min_user_support,
            item_min_support=1 if temporal else args.item_min_support,
            set_all_values_to=args.set_all_values_to,
        )

        progress.step(f"Building {args.split_mode} split")
        split_payload = _build_split_payload(args, ds, proc_df, progress=progress)
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
                train_item_ids=split_payload.get("train_item_ids"),
                val_item_ids=split_payload.get("val_item_ids"),
                test_item_ids=split_payload.get("test_item_ids"),
                val_source_indices=val_holdout["source_indices"],
                val_target_indices=val_holdout["target_indices"],
                test_source_indices=test_holdout["source_indices"],
                test_target_indices=test_holdout["target_indices"],
                train_source_matrix=split_payload.get("train_source_matrix"),
                train_target_matrix=split_payload.get("train_target_matrix"),
                val_source_matrix=split_payload.get("val_source_matrix"),
                val_target_matrix=split_payload.get("val_target_matrix"),
                test_source_matrix=split_payload.get("test_source_matrix"),
                test_target_matrix=split_payload.get("test_target_matrix"),
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
                    "eval_draws": args.eval_draws,
                    "eval_holdout_frac": args.eval_holdout_frac,
                    "split_mode": args.split_mode,
                    "min_source_items": args.min_source_items,
                    "min_target_items": args.min_target_items,
                    "train_items": train_item_count,
                    "val_cold_items": val_item_count,
                    "test_cold_items": test_item_count,
                    "n_train_users": int(len(split_payload["train_user_ids"])) if split_payload.get("train_user_ids") is not None else None,
                    "n_val_users": int(len(split_payload["val_user_ids"])) if split_payload.get("val_user_ids") is not None else None,
                    "n_test_users": int(len(split_payload["test_user_ids"])) if split_payload.get("test_user_ids") is not None else None,
                    # Rows and users differ once a protocol draws a user more
                    # than once: at eval_draws=5 the row count is five times the
                    # user count, and recording only the former under a name
                    # saying "users" overstated the evaluation by that factor.
                    "n_val_eval_rows": int(len(val_holdout["source_indices"])),
                    "n_test_eval_rows": int(len(test_holdout["source_indices"])),
                    "n_val_eval_users": _distinct_eval_users(val_holdout),
                    "n_test_eval_users": _distinct_eval_users(test_holdout),
                    "split_files": {
                        "train_source_matrix": "data/train_source_matrix.npz",
                        "train_target_matrix": "data/train_target_matrix.npz",
                        "val_source_matrix": "data/val_source_matrix.npz",
                        "val_target_matrix": "data/val_target_matrix.npz",
                        "test_source_matrix": "data/test_source_matrix.npz",
                        "test_target_matrix": "data/test_target_matrix.npz",
                        "train_item_ids": "data/train_item_ids.npy",
                        "val_item_ids": "data/val_item_ids.npy",
                        "test_item_ids": "data/test_item_ids.npy",
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
    eval_draws: int = 5,
    eval_holdout_frac: float = 0.2,
    split_mode: str = "user_split",
    val_items: int | None = None,
    test_items: int | None = None,
    item_val_frac: float = 0.05,
    item_test_frac: float = 0.10,
    temporal_test_frac: float | None = None,
    temporal_period_hours: float = DEFAULT_TEMPORAL_PERIOD_HOURS,
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
        eval_draws=eval_draws,
        eval_holdout_frac=eval_holdout_frac,
        split_mode=split_mode,
        val_items=val_items,
        test_items=test_items,
        item_val_frac=item_val_frac,
        item_test_frac=item_test_frac,
        temporal_test_frac=temporal_test_frac,
        temporal_period_hours=temporal_period_hours,
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
