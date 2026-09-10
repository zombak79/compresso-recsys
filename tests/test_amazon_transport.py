"""No-network checks for compact transfers and range-resumed source downloads."""
import importlib
import io
import json
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "examples/validation"))
    return importlib.import_module("amazon_download"), importlib.import_module("amazon_hybrid")


class Response(io.BytesIO):
    def __init__(self, data, status, headers):
        super().__init__(data)
        self.status, self.headers = status, headers


@pytest.mark.parametrize("status", [200, 206])
def test_downloader_restarts_ignored_range_or_appends_correct_range(modules, tmp_path, monkeypatch, status):
    download, _ = modules
    ds = download.AmazonReviews2023(data_dir=tmp_path, category="All_Beauty")
    source = ds._hf_source("sample.csv", kind="csv", size=6)
    target = ds.root / source["local_path"]
    target.parent.mkdir(parents=True)
    temporary = target.with_suffix(".csv.tmp")
    temporary.write_bytes(b"abc")
    requests = []
    def response(request, timeout):
        requests.append(request)
        assert request.get_header("Range") == "bytes=3-"
        return Response(b"def" if status == 206 else b"abcdef", status,
                        {"Content-Range": "bytes 3-5/6"} if status == 206 else {"Content-Length": "6"})
    monkeypatch.setattr(download, "urlopen", response)
    download.download_file(ds, source, attempts=1)
    assert target.read_bytes() == b"abcdef"
    assert not temporary.exists()
    download.download_file(ds, source, attempts=1)
    assert len(requests) == 1


def test_downloader_refuses_wrong_resume_range(modules, tmp_path, monkeypatch):
    download, _ = modules
    ds = download.AmazonReviews2023(data_dir=tmp_path, category="All_Beauty")
    source = ds._hf_source("sample.csv", kind="csv", size=6)
    target = ds.root / source["local_path"]
    target.parent.mkdir(parents=True)
    temporary = target.with_suffix(".csv.tmp")
    temporary.write_bytes(b"abc")
    monkeypatch.setattr(download, "urlopen", lambda *a, **k: Response(b"bcdef", 206, {"Content-Range": "bytes 1-5/6"}))
    with pytest.raises(RuntimeError, match="partial data retained"):
        download.download_file(ds, source, attempts=1)
    assert temporary.read_bytes() == b"abc"
    assert not target.exists()


def test_hybrid_publishes_ready_marker_after_input_and_checks_disk(modules, tmp_path, monkeypatch):
    _, hybrid = modules
    directory = tmp_path / "All_Beauty"
    directory.mkdir()
    (directory / "input.parquet").write_bytes(b"prepared data")
    digest = hybrid.profile.sha256(directory / "input.parquet")
    hybrid.profile.write_json(directory / "input.json", {"prepared": {"sha256": digest}})
    remote_calls, transfers = [], []
    def remote(host, command):
        remote_calls.append(command)
        return str(80 * 1024 ** 3)
    def transfer(host, paths, destination):
        transfers.append([path.name for path in paths])
        if paths[0].name == "input.parquet":
            assert not (directory / "READY.json").exists()
    monkeypatch.setattr(hybrid, "remote", remote)
    monkeypatch.setattr(hybrid, "transfer", transfer)
    assert hybrid.publish_input("rtx", Path("/home/user/audit"), "python3", directory, "All_Beauty") == digest
    assert transfers == [["input.parquet", "input.json"], ["READY.json"]]
    assert json.loads((directory / "READY.json").read_text())["sha256"] == digest
    transfers.clear()
    monkeypatch.setattr(hybrid, "remote", lambda *a: str(10 * 1024 ** 3))
    with pytest.raises(RuntimeError, match="20 GiB"):
        hybrid.publish_input("rtx", Path("/home/user/audit"), "python3", directory, "All_Beauty")
    assert not transfers
