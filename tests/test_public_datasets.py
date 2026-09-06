from __future__ import annotations

import gzip
import io
import json
import tarfile
import zipfile

import pandas as pd
import pytest

from compresso_recsys import Steam, NetflixPrize, TasteProfile, Gowalla, build_recsys_checkpoint
from compresso_recsys.builder import _build_args, _resolve_args
from compresso_recsys.checkpoint import load_recsys_split, read_checkpoint
from compresso_recsys.datasets import _download


def write_gzip(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as out:
        for record in records:
            out.write(record + "\n")


def steam_fixture(tmp_path):
    ds = Steam(data_dir=tmp_path, show_progress=False)
    reviews = [
        {"username": f"u{user}", "product_id": str(item), "date": f"2017-01-{item + 1:02d}",
         "recommended": item % 2 == 0, "text": "not needed for interactions"}
        for user in range(8) for item in range(12)
    ]
    write_gzip(ds.root / "steam_reviews.json.gz", [repr(row) for row in reviews])
    write_gzip(ds.root / "steam_games.json.gz", [json.dumps({
        "id": str(item), "app_name": f"Game {item}", "genres": ["Action", "Indie"],
        "tags": ["Adventure"], "developer": "Example Studio", "price": "Free To Play",
    }) for item in range(11)])  # Reviewed item 11 has no metadata.
    return ds


def test_steam_original_ids_dates_negative_reviews_and_missing_metadata(tmp_path):
    ds = steam_fixture(tmp_path)
    interactions = ds.get_interactions()
    metadata = ds.get_item_metadata().set_index("item_id")
    assert len(interactions) == 96  # Negative reviews are also interactions.
    assert interactions.value.eq(1).all()
    assert interactions.item_id.iloc[0] == "0"
    assert interactions.timestamp.iloc[0] == pd.Timestamp("2017-01-01", tz="UTC").timestamp()
    assert metadata.loc["0", "genres"] == "Action|Indie"
    assert "Example Studio" in metadata.loc["0", "entity_text"]
    assert "11" in metadata.index
    assert metadata.loc["11", "entity_text"] == ""
    # A warm cache must neither parse the reviews nor require the network.
    again = Steam(data_dir=tmp_path, show_progress=False)
    again._review_frames = lambda: pytest.fail("unexpected reparse")
    pd.testing.assert_frame_equal(again.get_interactions(), interactions)


def test_steam_explicit_text_filter_and_cache_invalidation(tmp_path):
    ds = steam_fixture(tmp_path)
    ds.prepare()
    filtered = Steam(data_dir=tmp_path, min_entity_text_words=1, show_progress=False)
    assert "11" not in set(filtered.get_interactions().item_id)
    write_gzip(ds.root / "steam_reviews.json.gz", [json.dumps({
        "username": "new", "product_id": "0", "date": "2018-01-01",
    })])
    assert Steam(data_dir=tmp_path).get_interactions().user_id.tolist() == ["new"]


@pytest.mark.parametrize("mode", ["user_split", "item_split", "leave_last_out", "temporal"])
def test_steam_builds_checkpoint_with_metadata_and_all_split_modes(tmp_path, mode):
    ds = steam_fixture(tmp_path)
    path = build_recsys_checkpoint(
        dataset="steam", data_dir=tmp_path, checkpoint_path=str(tmp_path / f"{mode}.zip"),
        split_mode=mode, val_users=2, test_users=2, val_items=2, test_items=2,
        min_user_support=1, item_min_support=1, eval_draws=1,
        temporal_period_hours=48, show_progress=False,
    )
    with read_checkpoint(path) as root:
        split = load_recsys_split(root)
    assert len(split["item_ids"]) > 0
    assert split["entity_metadata"] is not None
    assert "entity_text" in split["entity_metadata"]
    assert split["entity_tag_matrix"] is not None
    if mode == "item_split":
        cold_ids = set(split["item_ids"][split["test_cold_item_indices"]])
        assert cold_ids <= set(split["entity_metadata"].item_id)
    if mode in {"temporal", "leave_last_out"}:
        assert split["train_source_sequences"] is not None


def test_taste_profile_preserves_counts_and_has_no_timestamps(tmp_path):
    ds = TasteProfile(data_dir=tmp_path, show_progress=False)
    with zipfile.ZipFile(ds.root / "train_triplets.txt.zip", "w") as archive:
        archive.writestr("nested/train_triplets.txt", "u1\ts1\t8\nu2\ts2\t1\n")
    interactions = ds.get_interactions()
    assert interactions.value.tolist() == [8.0, 1.0]
    assert interactions.timestamp.isna().all()
    assert ds.get_item_metadata().item_id.tolist() == ["s1", "s2"]
    for mode in ("temporal", "leave_last_out"):
        with pytest.raises(ValueError, match="no interaction timestamps"):
            build_recsys_checkpoint(dataset="taste-profile", data_dir=tmp_path, split_mode=mode)


def test_gowalla_retains_repeat_events_and_coordinates(tmp_path):
    ds = Gowalla(data_dir=tmp_path, show_progress=False)
    write_gzip(ds.root / "loc-gowalla_totalCheckins.txt.gz", [
        "u1\t2010-01-02T00:00:00Z\t50.0\t14.0\tp1",
        "u1\t2010-01-01T00:00:00Z\t50.1\t14.1\tp1",
        "u2\t2010-01-01T00:00:00Z\t51.0\t15.0\tp2",
    ])
    assert ds.get_interactions().item_id.tolist() == ["p1", "p1", "p2"]
    meta = ds.get_item_metadata().set_index("item_id")
    assert meta.loc["p1", "latitude"] == 50.0
    assert meta.loc["p1", "entity_text"] == ""


def add_tar_file(archive, name, data):
    member = tarfile.TarInfo(name)
    member.size = len(data)
    archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("nested", [False, True])
def test_netflix_nested_and_flat_archives(tmp_path, nested):
    ds = NetflixPrize(data_dir=tmp_path, show_progress=False)
    contents = b"1:\n123,5,2005-01-01\n456,2,2005-01-02\n"
    with tarfile.open(ds.root / "nf_prize_dataset.tar.gz", "w:gz") as archive:
        if nested:
            inner_bytes = io.BytesIO()
            with tarfile.open(fileobj=inner_bytes, mode="w") as inner:
                add_tar_file(inner, "training_set/mv_0000001.txt", contents)
            add_tar_file(archive, "download/training_set.tar", inner_bytes.getvalue())
        else:
            add_tar_file(archive, "training_set/mv_0000001.txt", contents)
        add_tar_file(archive, "download/movie_titles.txt", b"1,2001,A title, with comma\n2,NULL,Other\n")
    interactions = ds.get_interactions()
    assert interactions.value.tolist() == [5.0, 2.0]
    assert interactions.user_id.tolist() == ["123", "456"]
    assert interactions.timestamp.diff().iloc[1] == 86400
    meta = ds.get_item_metadata().set_index("item_id")
    assert meta.loc["1", "title"] == "A title, with comma"
    assert meta.loc["2", "release_year"] == ""


def test_new_defaults_preserve_implicit_interactions_and_short_metadata():
    for dataset in ("steam", "gowalla", "taste-profile"):
        args, _ = _resolve_args(_build_args(dataset=dataset))
        assert args.min_value_to_keep is None
        assert args.min_entity_text_words == 0
    for dataset in ("ml1m", "ml20m", "goodbooks", "amazon2023"):
        args, _ = _resolve_args(_build_args(dataset=dataset))
        assert args.min_entity_text_words == 30
        assert args.min_value_to_keep == 4.0


def test_download_does_not_cache_partial_responses(tmp_path, monkeypatch):
    response = io.BytesIO(b"partial")
    response.headers = {"Content-Length": "100"}
    monkeypatch.setattr(_download, "urlopen", lambda *args, **kwargs: response)
    with pytest.raises(OSError, match="Incomplete download"):
        _download.download("https://example.invalid/data", tmp_path / "archive", show_progress=False)
    assert list(tmp_path.iterdir()) == []


def test_python_literal_parser_never_evaluates_code(tmp_path):
    source = tmp_path / "bad.gz"
    write_gzip(source, ["__import__('os').getcwd()"])
    with pytest.raises(ValueError, match="line 1"):
        list(_download.gzip_records(source))
