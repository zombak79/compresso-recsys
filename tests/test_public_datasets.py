from __future__ import annotations

import gzip
import io
import json
import sys
from dataclasses import replace
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

from compresso_recsys import (
    Steam, NetflixPrize, TasteProfile, Gowalla, RetailRocket, Music4AllOnion, OTTO,
    Yambda, build_recsys_checkpoint,
)
from compresso_recsys import builder
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


def retailrocket_fixture(tmp_path):
    """Views and carts for eight visitors, with repeats and one item seen twice."""
    ds = RetailRocket(data_dir=tmp_path, show_progress=False)
    rows = [
        {"timestamp": 1433221332117 + visitor * 1000 + step * 3_600_000,
         "visitorid": visitor,
         # Every fourth event is a cart, so the view filter has something to drop.
         "event": "addtocart" if step % 4 == 3 else "view",
         "itemid": 100 + step % 5,
         "transactionid": 7 if step % 4 == 3 else ""}
        for visitor in range(8) for step in range(8)
    ]
    pd.DataFrame(rows).to_csv(ds.root / "events.csv", index=False)
    return ds


def test_retailrocket_keeps_views_repeats_and_millisecond_timestamps(tmp_path):
    interactions = retailrocket_fixture(tmp_path).get_interactions()
    assert len(interactions) == 48  # 64 events less the 16 add-to-carts.
    assert interactions.value.eq(1).all()
    # Milliseconds, not the nanoseconds pandas assumes for a bare integer.
    assert interactions.timestamp.iloc[0] == 1433221332.117
    first = interactions[interactions.user_id == "0"]
    assert first.item_id.tolist() == ["100", "101", "102", "104", "100", "101"]
    assert len(first.drop_duplicates(["user_id", "item_id"])) < len(first)


def test_retailrocket_names_the_manual_step_when_the_export_is_absent(tmp_path):
    ds = RetailRocket(data_dir=tmp_path, show_progress=False)
    with pytest.raises(FileNotFoundError, match="cannot be downloaded automatically"):
        ds.get_interactions()


@pytest.mark.parametrize("mode", ["user_split", "item_split", "leave_last_out", "temporal"])
def test_retailrocket_builds_checkpoint_without_item_metadata(tmp_path, mode):
    retailrocket_fixture(tmp_path)
    path = build_recsys_checkpoint(
        dataset="retailrocket", data_dir=tmp_path, checkpoint_path=str(tmp_path / f"{mode}.zip"),
        split_mode=mode, val_users=2, test_users=2, val_items=1, test_items=1,
        min_user_support=1, item_min_support=1, eval_draws=1,
        temporal_period_hours=1, show_progress=False,
    )
    with read_checkpoint(path) as root:
        split = load_recsys_split(root)
    assert len(split["item_ids"]) > 0


def music4all_fixture(tmp_path, **kwargs):
    """Newest-first rows, as the release stores them, with replays."""
    ds = Music4AllOnion(data_dir=tmp_path, show_progress=False, **kwargs)
    rows = [
        {"user_id": f"u{user}", "track_id": f"t{step % 4}",
         "timestamp": f"2013-01-{2 + user:02d} 0{7 - step}:00:00"}
        for user in range(6) for step in range(6)
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(ds.root / ds.events_file, sep="\t", index=False, compression="bz2")
    return ds


def test_music4all_parses_datetimes_and_keeps_replays(tmp_path):
    interactions = music4all_fixture(tmp_path).get_interactions()
    assert len(interactions) == 36
    assert interactions.timestamp.iloc[0] == pd.Timestamp("2013-01-02 07:00:00", tz="UTC").timestamp()
    # Source order is newest-first and nothing re-sorts it.
    assert interactions.timestamp.iloc[1] < interactions.timestamp.iloc[0]
    assert len(interactions.drop_duplicates(["user_id", "item_id"])) < len(interactions)


def test_music4all_window_filters_and_does_not_reuse_the_other_window_cache(tmp_path):
    music4all_fixture(tmp_path).get_interactions()
    windowed = music4all_fixture(tmp_path, start="2013-01-04", end="2013-01-06")
    interactions = windowed.get_interactions()
    assert set(interactions.user_id) == {"u2", "u3"}
    # The unwindowed cache must not leak back in when the window is removed.
    assert len(music4all_fixture(tmp_path).get_interactions()) == 36


def yambda_fixture(tmp_path, **kwargs):
    ds = Yambda(data_dir=tmp_path, show_progress=False, **kwargs)
    frame = pd.DataFrame({
        "uid": np.repeat(np.arange(6, dtype="uint32"), 6),
        "item_id": np.tile(np.arange(4, dtype="uint32"), 9),
        # Relative seconds on a five-second grid, as the release publishes them.
        "timestamp": np.arange(36, dtype="uint32") * 5 + 120,
        "is_organic": np.tile(np.array([1, 0], dtype="uint8"), 18),
    })
    frame.to_parquet(ds.events_path, index=False)
    return ds


def test_yambda_keeps_relative_seconds_and_filters_organic(tmp_path):
    interactions = yambda_fixture(tmp_path).get_interactions()
    assert len(interactions) == 36
    # 120 seconds into the log, not 1970-01-01 plus two minutes' worth of nanoseconds.
    assert interactions.timestamp.iloc[0] == 120.0
    assert (interactions.timestamp % 5 == 0).all()
    organic = yambda_fixture(tmp_path, organic_only=True).get_interactions()
    assert len(organic) == 18


def test_yambda_rejects_an_unknown_variant(tmp_path):
    with pytest.raises(ValueError, match="variant must be one of"):
        Yambda(data_dir=tmp_path, variant="42m")


def otto_fixture(tmp_path, name="train.jsonl"):
    ds = OTTO(data_dir=tmp_path, show_progress=False)
    lines = [
        json.dumps({"session": session, "events": [
            {"aid": 100 + step % 5, "ts": 1661724000000 + session * 1000 + step * 60_000,
             # Every fourth event is an order, so the click filter has work to do.
             "type": "orders" if step % 4 == 3 else "clicks"}
            for step in range(8)]})
        for session in range(8)
    ]
    (ds.root / name).write_text("\n".join(lines) + "\n")
    return ds


@pytest.mark.parametrize("name", ["train.jsonl", "otto-recsys-train.jsonl"])
def test_otto_flattens_sessions_and_keeps_clicks(tmp_path, name):
    interactions = otto_fixture(tmp_path, name=name).get_interactions()
    assert len(interactions) == 48  # 64 events less the 16 orders.
    assert interactions.user_id.iloc[0] == "0"
    assert interactions.timestamp.iloc[0] == 1661724000.0
    assert len(interactions.drop_duplicates(["user_id", "item_id"])) < len(interactions)


def test_otto_session_sample_is_deterministic_and_keeps_whole_sessions(tmp_path):
    otto_fixture(tmp_path)
    full = OTTO(data_dir=tmp_path, show_progress=False).get_interactions()
    sampled = OTTO(data_dir=tmp_path, session_sample=0.5, show_progress=False).get_interactions()
    kept = set(sampled.user_id)
    assert 0 < len(kept) < full.user_id.nunique()
    # A kept session keeps every one of its events, and a dropped one none.
    for session in kept:
        assert (sampled.user_id == session).sum() == (full.user_id == session).sum()
    # Stable across instances, and the unsampled cache is not reused.
    again = OTTO(data_dir=tmp_path, session_sample=0.5, show_progress=False).get_interactions()
    assert set(again.user_id) == kept
    assert OTTO(data_dir=tmp_path, show_progress=False).get_interactions().user_id.nunique() == 8


def test_otto_rejects_a_sample_outside_the_unit_interval(tmp_path):
    for bad in (0.0, 1.5, -0.1):
        with pytest.raises(ValueError, match="session_sample must lie"):
            OTTO(data_dir=tmp_path, session_sample=bad)


def test_otto_names_the_manual_step_when_no_export_is_present(tmp_path):
    with pytest.raises(FileNotFoundError, match="cannot be downloaded automatically"):
        OTTO(data_dir=tmp_path, show_progress=False).get_interactions()


#: OTTO and Music4All are registered with a sample and a window sized for their
#: real logs, which would leave these few-dozen-row fixtures empty. Turning a
#: registered default off is exactly what a caller with a small extract does.
WHOLE_FIXTURE = {
    "music4all-onion": {"start": None, "end": None},
    "otto": {"session_sample": None},
    "yambda": {"user_sample": None},
}


@pytest.mark.parametrize("dataset", ["music4all-onion", "otto", "yambda"])
def test_new_sequential_datasets_build_checkpoints(tmp_path, dataset):
    {"music4all-onion": music4all_fixture, "otto": otto_fixture,
     "yambda": yambda_fixture}[dataset](tmp_path)
    path = build_recsys_checkpoint(
        dataset=dataset, data_dir=tmp_path, checkpoint_path=str(tmp_path / f"{dataset}.zip"),
        split_mode="leave_last_out", val_users=2, test_users=2,
        min_user_support=1, item_min_support=1, eval_draws=1, show_progress=False,
        dataset_options=WHOLE_FIXTURE[dataset],
    )
    with read_checkpoint(path) as root:
        split = load_recsys_split(root)
    assert split["x_train_sequences"] is not None


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
    for dataset in ("steam", "gowalla", "taste-profile", "retailrocket",
                    "music4all-onion", "otto", "yambda"):
        args, _ = _resolve_args(_build_args(dataset=dataset))
        assert args.min_value_to_keep is None
        assert args.min_entity_text_words == 0
    for dataset in ("ml1m", "ml20m", "goodbooks"):
        args, _ = _resolve_args(_build_args(dataset=dataset))
        assert args.min_entity_text_words == 30
        assert args.min_value_to_keep == 4.0
    amazon, _ = _resolve_args(_build_args(dataset="amazon2023"))
    assert amazon.min_entity_text_words == 0
    assert amazon.min_value_to_keep is None


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


def test_retailrocket_event_selection_changes_rows_and_is_not_cached_across_selections(tmp_path):
    views = retailrocket_fixture(tmp_path).get_interactions()
    both = RetailRocket(data_dir=tmp_path, events=("view", "addtocart"), show_progress=False)
    # The cache is keyed on the source file, so a stale hit here would silently
    # return the view-only rows for a selection that asked for carts as well.
    assert len(both.get_interactions()) == 64
    assert len(views) == 48
    assert len(retailrocket_fixture(tmp_path).get_interactions()) == 48


def test_retailrocket_rejects_an_unknown_event_type():
    with pytest.raises(ValueError, match="unknown event types"):
        RetailRocket(events=("veiw",))
    with pytest.raises(ValueError, match="at least one event type"):
        RetailRocket(events=())


def test_otto_event_selection_changes_rows_and_is_not_cached_across_selections(tmp_path):
    otto_fixture(tmp_path)
    clicks = OTTO(data_dir=tmp_path, show_progress=False).get_interactions()
    assert len(clicks) == 48  # 64 events less the 16 orders.
    # A stale cache hit here would return the click-only rows for a selection
    # that asked for orders as well; the cache is keyed on the source file.
    with_orders = OTTO(data_dir=tmp_path, events=("clicks", "orders"), show_progress=False)
    assert len(with_orders.get_interactions()) == 64
    assert len(OTTO(data_dir=tmp_path, show_progress=False).get_interactions()) == 48


def test_otto_rejects_an_unknown_event_type():
    with pytest.raises(ValueError, match="unknown event types"):
        OTTO(events=("click",))


@pytest.mark.parametrize(
    "dataset, options, attribute, expected",
    [
        ("yambda", {"variant": "500m"}, "variant", "500m"),
        ("yambda", {"organic_only": True}, "organic_only", True),
        ("otto", {"session_sample": 0.25}, "session_sample", 0.25),
        ("otto", {"events": ("clicks", "carts")}, "events", ("clicks", "carts")),
        ("retailrocket", {"events": ["view", "transaction"]}, "events", ("view", "transaction")),
        ("music4all-onion", {"start": "2013-01-04"}, "start", "2013-01-04"),
    ],
)
def test_dataset_options_reach_the_adapter(tmp_path, dataset, options, attribute, expected):
    args, spec = _resolve_args(_build_args(dataset=dataset, data_dir=str(tmp_path),
                                           dataset_options=options, show_progress=False))
    assert getattr(builder._make_dataset(args, spec), attribute) == expected


def test_dataset_options_are_coerced_from_command_line_strings(tmp_path):
    args = _build_args(dataset="otto", data_dir=str(tmp_path), show_progress=False)
    # argparse collects repeated --dataset_option flags as raw key=value strings.
    args.dataset_options = ["session_sample=0.5", "events=clicks,carts"]
    resolved, spec = _resolve_args(args)
    assert resolved.dataset_options == {"session_sample": 0.5, "events": ("clicks", "carts")}
    dataset = builder._make_dataset(resolved, spec)
    assert dataset.session_sample == 0.5 and dataset.events == ("clicks", "carts")


def test_unknown_dataset_option_names_the_options_the_dataset_accepts(tmp_path):
    with pytest.raises(ValueError, match="variant, organic_only"):
        _resolve_args(_build_args(dataset="yambda", data_dir=str(tmp_path),
                                  dataset_options={"varient": "500m"}))


def test_dataset_option_rejects_a_value_the_adapter_cannot_take(tmp_path):
    with pytest.raises(ValueError, match="invalid value for dataset option"):
        _resolve_args(_build_args(dataset="otto", data_dir=str(tmp_path),
                                  dataset_options={"session_sample": "half"}))
    with pytest.raises(ValueError, match="key=value"):
        args = _build_args(dataset="yambda", data_dir=str(tmp_path))
        args.dataset_options = ["variant"]
        _resolve_args(args)


def test_builder_managed_arguments_are_not_settable_as_dataset_options(tmp_path):
    # Otherwise one build could be told two different things about the same knob.
    for key in ("data_dir", "min_entity_text_words", "show_progress"):
        with pytest.raises(ValueError, match="unknown dataset option"):
            _resolve_args(_build_args(dataset="otto", data_dir=str(tmp_path),
                                      dataset_options={key: 1}))


def test_checkpoint_manifest_records_the_dataset_options_that_built_it(tmp_path):
    retailrocket_fixture(tmp_path)
    path = build_recsys_checkpoint(
        dataset="retailrocket", data_dir=tmp_path,
        checkpoint_path=str(tmp_path / "options.zip"), split_mode="user_split",
        val_users=2, test_users=2, min_user_support=1, item_min_support=1,
        eval_draws=1, show_progress=False,
        dataset_options={"events": ("view", "addtocart")},
    )
    with read_checkpoint(path) as root:
        manifest = json.loads((root / "manifest.json").read_text())
    recorded = json.dumps(manifest)
    assert '"dataset_options"' in recorded and '"addtocart"' in recorded


def test_cli_accepts_repeated_dataset_option_flags(monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["build", "--dataset", "yambda", "--dataset_option", "variant=500m",
         "--dataset_option", "organic_only=true"],
    )
    assert builder.parse_args().dataset_options == ["variant=500m", "organic_only=true"]


def test_spec_level_default_options_apply_and_yield_to_an_explicit_one(tmp_path, monkeypatch):
    # No dataset registers one today, so the mechanism is exercised by injecting
    # a spec rather than by depending on a default that may never be set.
    monkeypatch.setitem(builder.DATASETS, "otto",
                        replace(builder.DATASETS["otto"], dataset_options={"session_sample": 0.5}))

    args, spec = _resolve_args(_build_args(dataset="otto", data_dir=str(tmp_path), show_progress=False))
    assert args.dataset_options == {"session_sample": 0.5}
    assert builder._make_dataset(args, spec).session_sample == 0.5

    args, spec = _resolve_args(_build_args(dataset="otto", data_dir=str(tmp_path),
                                           dataset_options={"session_sample": 0.25},
                                           show_progress=False))
    assert args.dataset_options == {"session_sample": 0.25}
    assert builder._make_dataset(args, spec).session_sample == 0.25


def test_spec_level_and_caller_options_merge_by_key(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.DATASETS, "otto",
                        replace(builder.DATASETS["otto"],
                                dataset_options={"session_sample": 0.5, "events": ("clicks",)}))
    args, spec = _resolve_args(_build_args(dataset="otto", data_dir=str(tmp_path),
                                           dataset_options={"events": ("clicks", "orders")},
                                           show_progress=False))
    # The untouched default survives alongside the overridden one.
    assert args.dataset_options == {"session_sample": 0.5, "events": ("clicks", "orders")}


def test_a_typo_in_a_registered_spec_is_rejected_like_any_other(tmp_path, monkeypatch):
    monkeypatch.setitem(builder.DATASETS, "otto",
                        replace(builder.DATASETS["otto"], dataset_options={"sesion_sample": 0.5}))
    with pytest.raises(ValueError, match="unknown dataset option"):
        _resolve_args(_build_args(dataset="otto", data_dir=str(tmp_path), show_progress=False))


#: Window each dataset is registered with. Every one of these logs is far too
#: short for the 339-day global default to fit three target windows.
REGISTERED_TEMPORAL_WINDOWS = {
    "retailrocket": 14 * 24,
    "otto": 2 * 24,
    "yambda": 30 * 24,
    "music4all-onion": 30 * 24,
    "gowalla": 30 * 24,
}


def test_temporal_window_comes_from_the_dataset_it_is_registered_with():
    for name, expected in REGISTERED_TEMPORAL_WINDOWS.items():
        args, _ = _resolve_args(_build_args(dataset=name, split_mode="temporal",
                                            show_progress=False))
        assert args.temporal_period_hours == expected, name
    # Datasets measured before the field existed must keep the global value, or
    # their published tables describe a window nobody would reproduce.
    for name in ("ml1m", "ml20m", "steam", "amazon2023"):
        args, _ = _resolve_args(_build_args(dataset=name, split_mode="temporal",
                                            show_progress=False))
        assert args.temporal_period_hours == builder.DEFAULT_TEMPORAL_PERIOD_HOURS, name



def test_an_explicit_temporal_window_beats_the_registered_one():
    args, _ = _resolve_args(_build_args(dataset="retailrocket", split_mode="temporal",
                                        temporal_period_hours=72, show_progress=False))
    assert args.temporal_period_hours == 72


def test_a_registered_temporal_window_reaches_a_temporal_build(tmp_path, monkeypatch):
    retailrocket_fixture(tmp_path)
    # The fixture spans hours, not months, so register a window that fits it.
    monkeypatch.setitem(builder.DATASETS, "retailrocket",
                        replace(builder.DATASETS["retailrocket"], temporal_period_hours=1))
    path = build_recsys_checkpoint(
        dataset="retailrocket", data_dir=tmp_path,
        checkpoint_path=str(tmp_path / "temporal.zip"), split_mode="temporal",
        val_users=2, test_users=2, min_user_support=1, item_min_support=1,
        eval_draws=1, show_progress=False,
    )
    with read_checkpoint(path) as root:
        manifest = json.loads((root / "manifest.json").read_text())
    assert '"temporal_period_hours": 1.0' in json.dumps(manifest)


def test_an_unusable_temporal_window_still_reports_the_span_it_needed(tmp_path):
    retailrocket_fixture(tmp_path)
    # The registered 14-day window cannot fit three periods into a 6-hour log.
    with pytest.raises(ValueError, match="three target windows shorter than"):
        build_recsys_checkpoint(
            dataset="retailrocket", data_dir=tmp_path,
            checkpoint_path=str(tmp_path / "nope.zip"), split_mode="temporal",
            val_users=2, test_users=2, min_user_support=1, item_min_support=1,
            eval_draws=1, show_progress=False,
        )


def test_an_invalid_temporal_window_is_still_rejected():
    for bad in (0, -5, float("nan"), True):
        with pytest.raises(ValueError, match="finite and > 0"):
            _build_args(dataset="ml1m", temporal_period_hours=bad)


def test_registered_defaults_are_the_ones_the_measured_tables_describe():
    # These two exist because the full logs do not fit in memory; if either is
    # changed the published tables describe a dataset nobody can rebuild.
    assert builder.DATASETS["otto"].dataset_options == {"session_sample": 0.1}
    assert builder.DATASETS["music4all-onion"].dataset_options == {
        "start": "2014-01-01", "end": "2015-01-01"}
    assert builder.DATASETS["yambda"].dataset_options == {"user_sample": 0.2}
    # Retailrocket is small enough to measure whole, so it samples nothing.
    assert builder.DATASETS["retailrocket"].dataset_options == {}


def test_yambda_user_sample_is_deterministic_and_keeps_whole_users(tmp_path):
    whole = yambda_fixture(tmp_path).get_interactions()
    sampled = Yambda(data_dir=tmp_path, user_sample=0.5, show_progress=False).get_interactions()
    again = Yambda(data_dir=tmp_path, user_sample=0.5, show_progress=False).get_interactions()
    assert 0 < len(sampled) < len(whole)
    assert sorted(sampled.user_id.unique()) == sorted(again.user_id.unique())
    # Whole users, never a truncated history.
    per_user = whole.groupby("user_id").size()
    assert all(len(sampled[sampled.user_id == u]) == per_user[u] for u in sampled.user_id.unique())
    # And the unsampled cache must not be the one the sample reads back.
    assert len(yambda_fixture(tmp_path).get_interactions()) == len(whole)


def test_every_registered_temporal_window_fits_its_measured_span():
    """Three target windows must fit inside the log, or temporal cannot build.

    The spans are the ones the measurement runs reported, so a later change to
    a window or a sample that shortens a log fails here rather than in a build.
    """
    measured_span_hours = {"retailrocket": 3312.0, "otto": 672.0,
                           "yambda": 7222.221, "music4all-onion": 8760.0,
                           "gowalla": 15024.074}
    for dataset, span in measured_span_hours.items():
        args, _ = _resolve_args(_build_args(dataset=dataset, split_mode="temporal",
                                            show_progress=False))
        assert 3 * args.temporal_period_hours < span, dataset


def test_yambda_holds_out_fewer_users_than_its_sample_retains():
    # The sample measures 1,800 users; asking for more than that made
    # user_split fail outright rather than produce a small evaluation set.
    spec = builder.DATASETS["yambda"]
    assert spec.val_users + spec.test_users < 1800


def test_a_registered_subset_says_so_and_names_the_way_out(capsys):
    _resolve_args(_build_args(dataset="otto"))
    notice = capsys.readouterr().out
    assert "reading a registered subset" in notice
    assert "session_sample=0.1" in notice
    assert "--dataset_option session_sample=1.0" in notice


def test_a_windowed_default_names_both_bounds_to_clear(capsys):
    _resolve_args(_build_args(dataset="music4all-onion"))
    notice = capsys.readouterr().out
    assert "--dataset_option end= --dataset_option start=" in notice


@pytest.mark.parametrize("dataset", ["retailrocket", "ml1m", "gowalla"])
def test_datasets_that_read_everything_stay_quiet(capsys, dataset):
    _resolve_args(_build_args(dataset=dataset))
    assert "registered subset" not in capsys.readouterr().out


def test_choosing_the_sample_yourself_is_not_warned_about(capsys):
    # An explicit choice is deliberate; only an unnoticed default needs saying.
    _resolve_args(_build_args(dataset="otto", dataset_options={"session_sample": 0.5}))
    assert "registered subset" not in capsys.readouterr().out


def test_a_quiet_build_does_not_print_the_notice(capsys):
    _resolve_args(_build_args(dataset="otto", show_progress=False))
    assert capsys.readouterr().out == ""


def test_an_empty_command_line_value_clears_an_optional_option():
    # The only way to turn off a registered window from the command line.
    args = _build_args(dataset="music4all-onion", show_progress=False)
    args.dataset_options = ["start=", "end="]
    resolved, _ = _resolve_args(args)
    assert resolved.dataset_options == {"start": None, "end": None}


def test_clearing_does_not_swallow_none_as_a_literal_value():
    # Only the empty string means "unset"; "none" stays a string an adapter
    # could legitimately want.
    args = _build_args(dataset="music4all-onion", show_progress=False)
    args.dataset_options = ["start=none"]
    resolved, _ = _resolve_args(args)
    assert resolved.dataset_options["start"] == "none"
