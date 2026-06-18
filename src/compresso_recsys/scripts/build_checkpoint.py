from __future__ import annotations

import argparse
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from compresso_recsys.checkpoint import save_recsys_split, update_checkpoint
from compresso_recsys.datasets import Goodbooks, MovieLens1M, MovieLens20M
from compresso_recsys.retrieval import build_eval_holdout


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
}


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
        "--annotation_source",
        type=str,
        default="genres",
        choices=["genres", "ml20m_tags", "goodbooks_tags", "none"],
    )
    p.add_argument("--annotation_min_count", type=int, default=100)
    return p.parse_args()


def _resolve_args(args):
    spec = DATASETS[args.dataset]
    args.checkpoint_path = args.checkpoint_path or spec.checkpoint_path
    args.seed = spec.seed if args.seed is None else args.seed
    args.val_users = spec.val_users if args.val_users is None else args.val_users
    args.test_users = spec.test_users if args.test_users is None else args.test_users
    args.min_user_support = spec.min_user_support if args.min_user_support is None else args.min_user_support
    args.item_min_support = spec.item_min_support if args.item_min_support is None else args.item_min_support
    args.min_value_to_keep = spec.min_value_to_keep if args.min_value_to_keep is None else args.min_value_to_keep
    args.set_all_values_to = spec.set_all_values_to if args.set_all_values_to is None else args.set_all_values_to
    return args, spec


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


def main():
    args, spec = _resolve_args(parse_args())
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds = spec.cls(data_dir=args.data_dir)
    raw_df = ds.get_interactions()
    proc_df = ds.preprocess_interactions_for_recsys(
        raw_df,
        min_value_to_keep=args.min_value_to_keep,
        user_min_support=args.min_user_support,
        item_min_support=args.item_min_support,
        set_all_values_to=args.set_all_values_to,
    )
    split = ds.split_users_strong_generalization(
        val_users=args.val_users,
        test_users=args.test_users,
        min_user_support=1,
        random_state=args.seed,
        interactions=proc_df,
    )
    x_train, _, item_ids = ds.to_sparse_matrix(split.train)
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
    entity_tag_matrix, tag_names, annotation_name = _build_entity_tag_matrix(args, ds, test_holdout["item_ids"])
    entity_metadata = ds.get_item_metadata()

    with update_checkpoint(args.checkpoint_path) as root:
        save_recsys_split(
            root,
            item_ids=test_holdout["item_ids"],
            x_train=x_train,
            val_source_indices=val_holdout["source_indices"],
            val_target_indices=val_holdout["target_indices"],
            test_source_indices=test_holdout["source_indices"],
            test_target_indices=test_holdout["target_indices"],
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
                "annotation_source": args.annotation_source,
                "annotation_min_count": args.annotation_min_count,
                "annotations": {
                    "entity_tags": annotation_name,
                    "n_tags": int(len(tag_names)) if tag_names is not None else 0,
                    "entity_metadata": True,
                },
            },
        )
    print(f"Saved {args.dataset} data split checkpoint to: {args.checkpoint_path}")


if __name__ == "__main__":
    main()
