from __future__ import annotations

import pytest

import compresso_recsys as cr
from compresso_recsys.scripts.build_checkpoint import _build_args


def test_build_recsys_checkpoint_is_public_function():
    assert cr.build_recsys_checkpoint.__name__ == "build_recsys_checkpoint"


def test_build_recsys_checkpoint_rejects_unknown_dataset():
    with pytest.raises(ValueError, match="dataset must be one of"):
        cr.build_recsys_checkpoint(dataset="unknown")


def test_build_checkpoint_args_accept_python_metadata_field_list():
    args = _build_args(dataset="ml1m", metadata_text_fields=["title", "genres"])

    assert args.metadata_text_fields == "title,genres"


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
