from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest
from scipy.sparse import csr_matrix

import compresso_recsys as cr
from compresso_recsys import builder


def checkpoint(path, dataset="ml1m"):
    with cr.update_checkpoint(path) as root:
        cr.save_recsys_split(root, item_ids=np.array(["20", "10", "30"]),
                            x_train=csr_matrix([[1., 0., 1.]]),
                            val_source_indices=[np.array([0])], val_target_indices=[np.array([1])],
                            test_source_indices=[np.array([0])], test_target_indices=[np.array([2])],
                            metadata={"dataset": dataset})


def test_embeddings_roundtrip_and_alignment(tmp_path):
    path = tmp_path / "checkpoint.zip"
    checkpoint(path)
    with cr.update_checkpoint(path) as root:
        assert cr.list_item_embeddings(root) == {}
        cr.save_item_embeddings(root, "text/test", item_ids=["10", "20"],
                                embeddings=[[1, 2], [3, 4]], available=np.array([True, False]),
                                metadata={"encoder": "test"})
    with cr.read_checkpoint(path) as root:
        result = cr.load_item_embeddings(root, "text/test", item_ids=["20", "30", "10"])
        np.testing.assert_array_equal(result["embeddings"], [[0, 0], [0, 0], [1, 2]])
        assert result["available"].tolist() == [False, False, True]
        assert result["metadata"] == {"encoder": "test"}
        assert cr.load_recsys_split(root)["x_train"].nnz == 2


@pytest.mark.parametrize("kwargs", [
    {"name": "../escape"}, {"item_ids": ["1", "1"]},
    {"embeddings": [[float("nan")], [1]]}, {"embeddings": [[1]]},
    {"available": [1, 0]},
])
def test_invalid_embeddings(tmp_path, kwargs):
    options = dict(name="text/test", item_ids=["1", "2"], embeddings=[[1], [2]])
    options.update(kwargs)
    with pytest.raises(ValueError):
        cr.save_item_embeddings(tmp_path, **options)


def test_json_import(tmp_path):
    path, archive = tmp_path / "checkpoint.zip", tmp_path / "features.zip"
    checkpoint(path)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("json/all-MiniLM-L6-v2.json", json.dumps({"10": [1, 2], "20": [3, 4]}))
    cr.enrich_multimodal_checkpoint(path, dataset="ml1m", features="text/minilm", archive_path=archive)
    with cr.read_checkpoint(path) as root:
        result = cr.load_item_embeddings(root, "text/minilm")
        assert result["item_ids"].tolist() == ["20", "10", "30"]
        np.testing.assert_array_equal(result["embeddings"], [[3, 4], [1, 2], [0, 0]])
        assert result["available"].tolist() == [True, True, False]
        assert result["metadata"]["verified_release"] is False
    before = path.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        cr.enrich_multimodal_checkpoint(path, dataset="dbbook", features="text/minilm", archive_path=archive)
    assert path.read_bytes() == before


def test_lastfm_media_are_mean_pooled(tmp_path):
    path, archive = tmp_path / "checkpoint.zip", tmp_path / "features.zip"
    checkpoint(path, dataset="lfm2k")
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("json/resnet152.json", json.dumps({"10_1": [1, 2], "10_2": [3, 4], "20_1": [5, 6]}))
    cr.enrich_multimodal_checkpoint(path, dataset="lfm2k", features="image/resnet152", archive_path=archive)
    with cr.read_checkpoint(path) as root:
        result = cr.load_item_embeddings(root, "image/resnet152")
        np.testing.assert_array_equal(result["embeddings"], [[5, 6], [2, 3], [0, 0]])
        assert result["metadata"]["pooling"] == "mean by artist ID prefix"


@pytest.mark.parametrize("payload", ['{"10":[1],"10":[2]}', '{"10":[1],"20":[1,2]}', '{"99":[1]}'])
def test_bad_import_is_atomic(tmp_path, payload):
    path, archive = tmp_path / "checkpoint.zip", tmp_path / "features.zip"
    checkpoint(path)
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("json/all-MiniLM-L6-v2.json", payload)
    before = path.read_bytes()
    with pytest.raises(ValueError):
        cr.enrich_multimodal_checkpoint(path, dataset="ml1m", features="text/minilm", archive_path=archive)
    assert path.read_bytes() == before


def dbbook_data(tmp_path):
    folder = tmp_path / "dbbook"
    folder.mkdir()
    with zipfile.ZipFile(folder / "dbbook_interaction_data.zip", "w") as z:
        z.writestr("interaction_data/train.tsv", "".join(
            f"{user}\t{item}\t1\n" for user in range(8) for item in range(8) if item != user))
        z.writestr("interaction_data/test.tsv", "".join(f"{user}\t{user}\t1\n" for user in range(8)))
        z.writestr("interaction_data/DBbook_Items_DBpedia_mapping.tsv",
                   "DBbook_ItemID\tname\tDBpedia_uri\n" + "".join(f"{i}\tBook {i}\turi:{i}\n" for i in range(8)))


@pytest.mark.parametrize("mode", ["user_split", "item_split", "official"])
def test_dbbook_builder(tmp_path, mode):
    dbbook_data(tmp_path)
    dataset = cr.DBbook(tmp_path)
    assert len(dataset.get_official_split()["test"]) == 8
    output = tmp_path / "out.zip"
    cr.build_recsys_checkpoint(dataset="dbbook", data_dir=str(tmp_path), checkpoint_path=str(output),
                               split_mode=mode, min_user_support=1, item_min_support=1,
                               val_users=1, test_users=1, val_items=1, test_items=1,
                               eval_draws=1, show_progress=False)
    with cr.read_checkpoint(output) as root:
        split = cr.load_recsys_split(root)
        assert split["x_train"].nnz > 0
        if mode == "official":
            assert split["test_target_matrix"].nnz == 8
            for user, target in zip(split["val_eval_user_ids"], split["val_target_matrix"]):
                row = list(split["train_user_ids"]).index(user)
                assert split["x_train"][row].multiply(target).nnz == 0
            assert cr.load_manifest(root)["stages"]["data"]["official_test_excluded_interactions"] == 0


def test_lastfm_adapter(tmp_path):
    folder = tmp_path / "lfm2k"
    folder.mkdir()
    with zipfile.ZipFile(folder / "lfm2k_interaction_data.zip", "w") as z:
        z.writestr("interaction_data/user_artists.dat", "userID\tartistID\tweight\n1\t5\t30\n")
        z.writestr("interaction_data/artists.dat", "id\tname\turl\tpictureURL\n5\tArtist\tu\tp\n")
    ds = cr.LastFM2K(tmp_path)
    assert ds.get_interactions().iloc[0].value == 30
    assert ds.get_interactions().timestamp.isna().all()
    assert ds.get_item_metadata().iloc[0].item_id == "5"


@pytest.mark.parametrize("dataset", ["dbbook", "lfm2k"])
@pytest.mark.parametrize("mode", ["temporal", "leave_last_out"])
def test_no_timestamps_rejected_before_download(tmp_path, dataset, mode):
    with pytest.raises(ValueError, match="no interaction timestamps"):
        cr.build_recsys_checkpoint(dataset=dataset, data_dir=str(tmp_path), split_mode=mode)


def test_bad_features_rejected_before_download(tmp_path):
    with pytest.raises(ValueError, match="Unsupported feature"):
        cr.build_recsys_checkpoint(dataset="dbbook", data_dir=str(tmp_path), multimodal_features="video/i3d")


def test_dbbook_negatives_are_not_binarized_into_positives():
    args, _ = builder._resolve_args(builder._build_args(dataset="dbbook"))
    assert args.min_value_to_keep == 1.0


def test_builder_imports_features_into_same_catalog(tmp_path, monkeypatch):
    import hashlib
    import compresso_recsys.multimodal as mm

    dbbook_data(tmp_path)
    cache = tmp_path / "multimodal"
    cache.mkdir()
    archive = cache / "dbbook_mm_json.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("json/all-MiniLM-L6-v2.json", json.dumps({str(i): [float(i)] for i in range(8)}))
    monkeypatch.setitem(mm.ARCHIVE_MD5, "dbbook", hashlib.md5(archive.read_bytes()).hexdigest())
    output = tmp_path / "out.zip"
    cr.build_recsys_checkpoint(dataset="dbbook", data_dir=str(tmp_path), checkpoint_path=str(output),
                               split_mode="item_split", val_items=1, test_items=1,
                               multimodal_features="text/minilm", show_progress=False)
    with cr.read_checkpoint(output) as root:
        split = cr.load_recsys_split(root)
        features = cr.load_item_embeddings(root, "text/minilm")
        np.testing.assert_array_equal(features["item_ids"], split["item_ids"])
        assert features["available"].all()
