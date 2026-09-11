from __future__ import annotations

import pytest

import compresso_recsys as cr
from compresso_recsys.builder import DATASETS, _build_args, _make_dataset, _resolve_args


def test_build_recsys_checkpoint_is_public_function():
    assert cr.build_recsys_checkpoint.__name__ == "build_recsys_checkpoint"


@pytest.mark.parametrize("dataset", sorted(DATASETS))
@pytest.mark.parametrize("period", [None, 96, 8136])
def test_temporal_period_defaults_and_overrides_match_api_and_cli(monkeypatch, dataset, period):
    from compresso_recsys import builder

    expected = (720 if dataset == "gowalla" else 8136) if period is None else period
    options = {} if period is None else {"temporal_period_hours": period}
    api_args = _build_args(dataset=dataset, **options)
    assert api_args.temporal_period_hours == expected
    monkeypatch.setattr(builder, "_build_recsys_checkpoint_from_args",
                        lambda args: _resolve_args(args)[0].temporal_period_hours)
    assert cr.build_recsys_checkpoint(dataset=dataset, **options) == expected
    cli = ["build-checkpoint", "--dataset", dataset]
    if period is not None:
        cli += ["--temporal_period_hours", str(period)]
    monkeypatch.setattr("sys.argv", cli)
    assert _resolve_args(builder.parse_args())[0].temporal_period_hours == expected


def test_eval_draws_defaults_to_one_in_api_and_cli(monkeypatch):
    import inspect
    from compresso_recsys.builder import parse_args

    assert _build_args(dataset="ml1m").eval_draws == 1
    assert inspect.signature(cr.build_recsys_checkpoint).parameters["eval_draws"].default == 1
    monkeypatch.setattr("sys.argv", ["build-checkpoint", "--dataset", "ml1m"])
    assert parse_args().eval_draws == 1
    monkeypatch.setattr("sys.argv", ["build-checkpoint", "--dataset", "ml1m", "--eval_draws", "5"])
    assert parse_args().eval_draws == 5
    assert _build_args(dataset="ml1m", eval_draws=5).eval_draws == 5


def test_build_recsys_checkpoint_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="dataset must be one of"):
        cr.build_recsys_checkpoint(dataset="unknown")


def test_build_checkpoint_args_accept_python_metadata_field_list():
    args = _build_args(dataset="ml1m", metadata_text_fields=["title", "genres"])

    assert args.metadata_text_fields == "title,genres"


@pytest.mark.parametrize("dataset", sorted(DATASETS))
def test_every_dataset_can_be_constructed_with_builder_defaults(dataset, tmp_path):
    args, spec = _resolve_args(_build_args(dataset=dataset, data_dir=str(tmp_path)))
    ds = _make_dataset(args, spec)
    assert isinstance(ds, spec.cls)
    if dataset == "amazon2023":
        assert ds.metadata_text_fields == ds.text_fields_for_category("Toys_and_Games")
        assert ds.category == "Toys_and_Games"


def test_amazon_builder_accepts_custom_metadata_fields(tmp_path):
    args, spec = _resolve_args(_build_args(
        dataset="amazon2023", data_dir=str(tmp_path), metadata_text_fields=["title", "store"],
    ))
    assert _make_dataset(args, spec).metadata_text_fields == ("title", "store")


def test_amazon_defaults_use_measured_toys_profile_and_preserve_overrides():
    args, _ = _resolve_args(_build_args(dataset="amazon2023"))
    assert (args.min_user_support, args.item_min_support) == (6, 22)
    assert (args.val_users, args.test_users) == (5000, 5000)
    assert args.min_entity_text_words == 0
    assert args.min_value_to_keep is None
    custom, _ = _resolve_args(_build_args(dataset="amazon2023", min_user_support=20, item_min_support=20,
                                        val_users=2500, test_users=5000, min_entity_text_words=30,
                                        min_value_to_keep=4.0))
    assert (custom.min_user_support, custom.item_min_support) == (20, 20)
    assert (custom.val_users, custom.test_users) == (2500, 5000)
    assert custom.min_entity_text_words == 30
    assert custom.min_value_to_keep == 4.0


def test_amazon_default_retains_every_rating_and_binarizes_it(tmp_path):
    import pandas as pd

    frame = pd.DataFrame({"user_id": ["u"] * 5, "item_id": list("abcde"),
                          "value": [1., 2., 3., 4., 5.], "timestamp": range(5)})
    # Isolate the default rating policy from support filtering on this tiny fixture.
    args, spec = _resolve_args(_build_args(dataset="amazon2023", data_dir=str(tmp_path),
                                          min_user_support=1, item_min_support=1))
    dataset = _make_dataset(args, spec)
    result = dataset.preprocess_interactions_for_recsys(
        frame, min_value_to_keep=args.min_value_to_keep,
        user_min_support=args.min_user_support, item_min_support=args.item_min_support,
        set_all_values_to=args.set_all_values_to,
    )
    assert result.item_id.tolist() == list("abcde")
    assert result.value.tolist() == [1.] * 5
    assert frame.value.tolist() == [1., 2., 3., 4., 5.]  # Raw ratings stay intact.


@pytest.mark.parametrize("category,support,heldout", [
    ("All_Beauty", (2, 2), 2263),
    ("Amazon_Fashion", (2, 4), 3571),
    ("Digital_Music", (2, 2), 278),
])
def test_measured_amazon_profiles_are_split_specific_and_preserve_overrides(category, support, heldout):
    args, _ = _resolve_args(_build_args(dataset="amazon2023", amazon_category=category))
    assert (args.min_user_support, args.item_min_support) == support
    assert (args.val_users, args.test_users) == (heldout, heldout)
    assert args.eval_draws == 1
    assert args.min_value_to_keep is None
    custom, _ = _resolve_args(_build_args(dataset="amazon2023", amazon_category=category,
        min_user_support=7, item_min_support=9, val_users=123, test_users=456))
    assert (custom.min_user_support, custom.item_min_support) == (7, 9)
    assert (custom.val_users, custom.test_users) == (123, 456)


def test_unprofiled_amazon_category_still_uses_fallback_defaults(monkeypatch):
    from compresso_recsys.datasets._amazon_defaults import AMAZON_SPLIT_DEFAULTS

    monkeypatch.delitem(AMAZON_SPLIT_DEFAULTS, "Toys_and_Games")
    for split in ("user_split", "item_split", "leave_last_out", "temporal"):
        args, _ = _resolve_args(_build_args(dataset="amazon2023", amazon_category="Toys_and_Games", split_mode=split))
        assert (args.min_user_support, args.item_min_support) == (5, 1)
        assert (args.val_users, args.test_users) == (100, 200)


def test_build_checkpoint_show_progress_defaults_true_and_can_be_disabled():
    default_args = _build_args(dataset="ml1m")
    quiet_args = _build_args(dataset="ml1m", show_progress=False)

    assert default_args.show_progress is True
    assert quiet_args.show_progress is False


def test_build_checkpoint_include_image_urls_defaults_false_and_can_be_enabled():
    default_args = _build_args(dataset="amazon2023")
    image_args = _build_args(dataset="amazon2023", include_image_urls=True)

    assert default_args.include_image_urls is False
    assert image_args.include_image_urls is True


def test_csr_row_indices_are_read_only_views_over_the_matrix():
    """The split keeps the matrix, so rows alias it instead of copying it.

    Copying every row duplicated the whole index buffer while the original
    stayed alive in the same returned dict. Views are read-only because a write
    through one would corrupt the other.
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    from compresso_recsys.builder import _csr_row_indices

    matrix = csr_matrix(
        np.array(
            [[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 0]],
            dtype=np.float32,
        )
    )

    rows = _csr_row_indices(matrix)

    assert [row.tolist() for row in rows] == [[0, 2], [], [0, 1, 2, 3], [1]]
    assert all(not row.flags.writeable for row in rows)
    assert np.shares_memory(rows[0], matrix.indices)
    with pytest.raises(ValueError):
        rows[0][0] = 99


def test_csr_row_indices_survive_conversion_by_its_consumers():
    """Consumers convert or concatenate, so read-only views reach them intact."""
    import numpy as np
    from scipy.sparse import csr_matrix

    from compresso_recsys.builder import _csr_row_indices
    from compresso_recsys.checkpoint import _as_obj_array, _indices_to_csr

    dense = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
    matrix = csr_matrix(dense)
    rows = _csr_row_indices(matrix)

    rebuilt = _indices_to_csr(rows, n_cols=3)
    assert np.array_equal(rebuilt.toarray(), dense)

    stored = _as_obj_array(rows)
    assert [np.asarray(v).tolist() for v in stored] == [[0, 2], [1], []]
    # The conversion produces fresh writable arrays rather than aliasing.
    assert all(np.asarray(v).flags.writeable for v in stored)
