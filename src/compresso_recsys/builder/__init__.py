"""Build a Compresso recsys checkpoint from a named public dataset.

This package is the former ``builder`` module, split along the seams its own
call graph already had: the dataset registry, the command line, the side
information matrices, and the split strategies. What stays here is the part
that orchestrates them, plus the public entry point.

Every name the module exported is re-exported, so existing
``from compresso_recsys.builder import ...`` lines keep working.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from compresso_recsys.checkpoint import (
    save_recsys_split,
    update_checkpoint,
)
from compresso_recsys.checkpoint.io import _indices_to_csr
from compresso_recsys.datasets import Steam, NetflixPrize, TasteProfile, Gowalla
from compresso_recsys.datasets._public import PublicDataset

# Re-exported because the flat module's namespace carried them and callers
# read them from here -- examples/validation/amazon_profile.py does.
from compresso_recsys.retrieval import (
    LEAVE_LAST_OUT_MIN_HISTORY,
    LEAVE_LAST_OUT_STAGES,
)
from compresso_recsys.builder.cli import (
    _build_args,
    _metadata_text_fields_arg,
    _resolve_args,
    parse_args,
)
from compresso_recsys.builder.features import (
    _build_entity_tag_matrix,
    _build_genre_tag_matrix,
    _build_goodbooks_user_tag_matrix,
    _build_ml20m_user_tag_matrix,
    _to_sparse_matrix_for_items,
    _to_sparse_matrix_for_items_with_users,
)
from compresso_recsys.builder.progress import _CheckpointProgress
from compresso_recsys.builder.specs import (
    DATASETS,
    DEFAULT_TEMPORAL_PERIOD_HOURS,
    DatasetSpec,
    _make_dataset,
    _temporal_period_hours,
)
from compresso_recsys.builder.splits.basic import (
    _build_item_split,
    _build_leave_last_out_split,
    _build_official_split,
    _build_user_split,
    _split_item_ids_random,
)
from compresso_recsys.builder.splits.temporal import (
    _build_temporal_split,
    _build_temporal_stage,
    _csr_row_indices,
    _filter_temporal_pair,
    _matrix_from_temporal_codes,
    _sequences_from_temporal_codes,
    _temporal_user_upper_bound,
    _timestamps_in_seconds,
)

__all__ = ["build_recsys_checkpoint", "DATASETS", "DatasetSpec", "parse_args"]


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
    if args.split_mode == "official":
        return _build_official_split(args, ds, proc_df)
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
        if raw_df.empty:
            raise ValueError(f"{args.dataset} has no interactions after metadata filtering")
        # Collapse repeated pairs only for CF splits. Ordered protocols retain
        # every check-in/review event, including repeat visits on different days.
        if isinstance(ds, PublicDataset) and args.split_mode in {"user_split", "item_split"}:
            raw_df = raw_df.drop_duplicates(["user_id", "item_id"], keep="first")

        progress.step("Preprocessing interactions")
        temporal = args.split_mode == "temporal"
        preprocessing_df = raw_df[raw_df.source_split == "train"] if args.split_mode == "official" else raw_df
        proc_df = ds.preprocess_interactions_for_recsys(
            preprocessing_df,
            min_value_to_keep=args.min_value_to_keep,
            user_min_support=1 if temporal else args.min_user_support,
            item_min_support=1 if temporal else args.item_min_support,
            set_all_values_to=args.set_all_values_to,
        )
        if proc_df.empty:
            raise ValueError(f"{args.dataset} has no interactions after preprocessing; lower support/text thresholds")

        progress.step(f"Building {args.split_mode} split")
        split_payload = _build_split_payload(args, ds, proc_df, progress=progress)
        item_ids = split_payload["item_ids"]
        val_holdout = split_payload["val_holdout"]
        test_holdout = split_payload["test_holdout"]
        warm_item_indices = split_payload.get("warm_item_indices")
        val_cold_item_indices = split_payload.get("val_cold_item_indices")
        test_cold_item_indices = split_payload.get("test_cold_item_indices")
        train_item_count = int(len(warm_item_indices)) if warm_item_indices is not None else int(len(item_ids))
        val_item_count = int(len(val_cold_item_indices)) if val_cold_item_indices is not None else 0
        test_item_count = int(len(test_cold_item_indices)) if test_cold_item_indices is not None else 0

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
                x_train_sequences=split_payload.get("x_train_sequences"),
                train_source_sequences=split_payload.get("train_source_sequences"),
                val_source_sequences=split_payload.get("val_source_sequences"),
                test_source_sequences=split_payload.get("test_source_sequences"),
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
                warm_item_indices=warm_item_indices,
                val_cold_item_indices=val_cold_item_indices,
                test_cold_item_indices=test_cold_item_indices,
                entity_tag_matrix=entity_tag_matrix,
                tag_names=tag_names,
                entity_metadata=entity_metadata,
                metadata={
                    "dataset": args.dataset,
                    "source_page": getattr(ds, "source_page", None),
                    "timestamp_precision": getattr(ds, "timestamp_precision", None),
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
                    # Listed only when the split mode produced them, so the
                    # registry says what a checkpoint holds rather than what the
                    # format allows.
                    "sequence_files": {
                        name: f"data/{name}.npz"
                        for name in (
                            "x_train_sequences",
                            "train_source_sequences",
                            "val_source_sequences",
                            "test_source_sequences",
                        )
                        if split_payload.get(name) is not None
                    },
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
            if getattr(args, "multimodal_features", None) is not None:
                from compresso_recsys.multimodal import import_multimodal_embeddings
                import_multimodal_embeddings(
                    root, dataset=args.dataset, features=args.multimodal_features,
                    data_dir=args.data_dir, show_progress=getattr(args, "show_progress", True),
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
    eval_draws: int = 1,
    eval_holdout_frac: float = 0.2,
    split_mode: str = "user_split",
    val_items: int | None = None,
    test_items: int | None = None,
    item_val_frac: float = 0.05,
    item_test_frac: float = 0.10,
    temporal_test_frac: float | None = None,
    temporal_period_hours: float | None = None,
    min_source_items: int = 1,
    min_target_items: int = 1,
    amazon_category: str = "Toys_and_Games",
    metadata_text_fields: str | list[str] | tuple[str, ...] | None = None,
    min_entity_text_words: int | None = None,
    include_image_urls: bool = False,
    annotation_source: str = "genres",
    annotation_min_count: int = 100,
    show_progress: bool = True,
    multimodal_features: str | list[str] | None = None,
) -> Path:
    """Build a recommender-system split checkpoint and return its path.

    ``temporal_period_hours=None`` uses 720 hours for Gowalla and 8136 for
    other datasets. An explicit positive period overrides that default.
    """
    args = _build_args(
        multimodal_features=multimodal_features,
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
