"""Argument parsing, and the resolution of a build's effective settings.

Separate from the build itself so that reading how an option reaches a stage
does not mean scrolling past the stage that consumes it.
"""

from __future__ import annotations

import argparse
import warnings


from compresso_recsys.datasets import AmazonReviews2023
from compresso_recsys.builder.specs import DATASETS, _temporal_period_hours


def _metadata_text_fields_arg(value: str | list[str] | tuple[str, ...] | None) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return ",".join(str(field) for field in value)


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
    p.add_argument("--eval_draws", type=int, default=1)
    p.add_argument("--multimodal_features", default=None,
                   help="Optional comma-separated SWAP features, e.g. text/minilm,image/resnet152")
    p.add_argument("--eval_holdout_frac", type=float, default=0.2)
    p.add_argument(
        "--split_mode",
        type=str,
        default="user_split",
        choices=["user_split", "item_split", "leave_last_out", "temporal", "official"],
    )
    p.add_argument("--val_items", type=int, default=None, help="Number of cold validation items for item_split.")
    p.add_argument("--test_items", type=int, default=None, help="Number of cold test items for item_split.")
    p.add_argument("--item_val_frac", type=float, default=0.05, help="Cold validation item fraction for item_split.")
    p.add_argument("--item_test_frac", type=float, default=0.10, help="Cold test item fraction for item_split.")
    p.add_argument("--temporal_test_frac", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument(
        "--temporal_period_hours",
        type=float,
        default=None,
        help="Width in hours of each train/validation/test temporal target window "
             "(default: 720 for Gowalla, 8136 otherwise).",
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
        help="Comma-separated metadata fields joined into entity_text; Amazon defaults vary by category and support paths such as details.Brand.",
    )
    p.add_argument(
        "--min_entity_text_words",
        type=int,
        default=None,
        help="Minimum item text words. Defaults to 30 for existing datasets, 0 for Steam/Netflix/Taste Profile/Gowalla.",
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
    if split_mode not in {"user_split", "item_split", "leave_last_out", "temporal", "official"}:
        raise ValueError(f"Unsupported split_mode: {split_mode!r}")
    if annotation_source not in {"genres", "ml20m_tags", "goodbooks_tags", "none"}:
        raise ValueError(f"Unsupported annotation_source: {annotation_source!r}")
    temporal_period_hours = _temporal_period_hours(dataset, temporal_period_hours)
    if temporal_test_frac is not None:
        warnings.warn(
            "temporal_test_frac is deprecated and ignored; use "
            "temporal_period_hours instead",
            DeprecationWarning,
            stacklevel=2,
        )
    return argparse.Namespace(
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
    args.temporal_period_hours = _temporal_period_hours(args.dataset, args.temporal_period_hours)
    if args.split_mode == "official" and args.dataset != "dbbook":
        raise ValueError("official split is supported only for dbbook")
    if getattr(args, "multimodal_features", None) is not None:
        from compresso_recsys.multimodal import _selection
        _selection(args.dataset, args.multimodal_features)
    if args.dataset == "amazon2023":
        args.amazon_category = AmazonReviews2023.normalize_category(args.amazon_category)
    args.checkpoint_path = args.checkpoint_path or spec.checkpoint_path.format(
        amazon_category=args.amazon_category,
    )
    if args.dataset == "amazon2023":
        from compresso_recsys.datasets._amazon_defaults import AMAZON_SPLIT_DEFAULTS

        profile = AMAZON_SPLIT_DEFAULTS.get(args.amazon_category, {}).get(args.split_mode, {})
        for name, value in profile.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
        if args.metadata_text_fields is None:
            args.metadata_text_fields = ",".join(AmazonReviews2023.text_fields_for_category(args.amazon_category))
        elif not any(field.strip() for field in args.metadata_text_fields.split(",")):
            raise ValueError("--metadata_text_fields must contain at least one field")
    args.seed = spec.seed if args.seed is None else args.seed
    args.val_users = spec.val_users if args.val_users is None else args.val_users
    args.test_users = spec.test_users if args.test_users is None else args.test_users
    args.min_user_support = spec.min_user_support if args.min_user_support is None else args.min_user_support
    args.item_min_support = spec.item_min_support if args.item_min_support is None else args.item_min_support
    args.min_value_to_keep = spec.min_value_to_keep if args.min_value_to_keep is None else args.min_value_to_keep
    args.set_all_values_to = spec.set_all_values_to if args.set_all_values_to is None else args.set_all_values_to
    args.min_entity_text_words = spec.min_entity_text_words if args.min_entity_text_words is None else args.min_entity_text_words
    if args.split_mode in {"leave_last_out", "temporal"} and not getattr(spec.cls, "has_timestamps", True):
        raise ValueError(f"{args.dataset} has no interaction timestamps; use user_split or item_split")
    return args, spec
