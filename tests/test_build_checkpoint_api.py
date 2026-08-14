from __future__ import annotations

import pytest

import compresso_recsys as cr
from compresso_recsys.builder import _build_args


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
