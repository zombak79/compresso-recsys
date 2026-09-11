from __future__ import annotations

import io
import ssl
import warnings
import zipfile
from types import SimpleNamespace
from urllib.error import HTTPError, URLError
from urllib.request import HTTPSHandler, Request

import pytest

from compresso_recsys.datasets import _download, _grouplens, movielens1m, movielens20m


URLS = [movielens1m.MovieLens1M.url, movielens20m.MovieLens20M.url]


def certificate_error(code=10):
    error = ssl.SSLCertVerificationError(1, "certificate verification failed")
    error.verify_code = code
    return error


def response(data=b"archive", *, length=None):
    stream = io.BytesIO(data)
    stream.headers = {"Content-Length": str(len(data) if length is None else length)}
    return stream


def raises(error):
    def fail(*args, **kwargs):
        raise error
    return fail


@pytest.mark.parametrize("url", URLS)
def test_valid_certificate_uses_verified_download_without_warning(tmp_path, monkeypatch, url):
    calls = []
    def verified(actual_url, **kwargs):
        calls.append((actual_url, kwargs))
        return response()
    monkeypatch.setattr(_grouplens, "urlopen", verified)
    monkeypatch.setattr(_grouplens, "build_opener", lambda *args: pytest.fail("No bypass needed"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        path = _grouplens.download_movielens_archive(url, tmp_path / "archive.zip")
    assert path.read_bytes() == b"archive"
    assert calls == [(url, {"timeout": 60})]
    assert not caught


@pytest.mark.parametrize("url", URLS)
@pytest.mark.parametrize("wrapped", [False, True])
def test_expiry_warns_and_uses_only_a_local_unverified_context(tmp_path, monkeypatch, url, wrapped):
    error = certificate_error()
    monkeypatch.setattr(_grouplens, "urlopen", raises(URLError(error) if wrapped else error))
    default_context = ssl._create_default_https_context
    import urllib.request
    default_opener = urllib.request._opener
    calls = []
    def build(*handlers):
        https = next(handler for handler in handlers if isinstance(handler, HTTPSHandler))
        assert https._context.verify_mode == ssl.CERT_NONE
        assert https._context.check_hostname is False
        assert any(isinstance(handler, _grouplens._RejectRedirects) for handler in handlers)
        def open_archive(actual_url, **kwargs):
            calls.append((actual_url, kwargs))
            return response()
        return SimpleNamespace(open=open_archive)
    monkeypatch.setattr(_grouplens, "build_opener", build)
    with pytest.warns(RuntimeWarning, match="server identity is not verified"):
        path = _grouplens.download_movielens_archive(url, tmp_path / "archive.zip")
    assert path.read_bytes() == b"archive"
    assert calls == [(url, {"timeout": 60})]
    assert ssl._create_default_https_context is default_context
    assert urllib.request._opener is default_opener
    assert ssl.create_default_context().verify_mode == ssl.CERT_REQUIRED


@pytest.mark.parametrize("url", [
    movielens1m.MovieLens1M.text_descriptions_url,
    "https://example.org/ml-1m.zip",
    "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
    "https://files.grouplens.org.evil.invalid/datasets/movielens/ml-1m.zip",
    "https://files.grouplens.org@evil.invalid/datasets/movielens/ml-1m.zip",
    URLS[0].replace("https:", "http:"),
    URLS[0] + "?redirect=other",
])
def test_expiry_on_any_other_url_never_bypasses_verification(tmp_path, monkeypatch, url):
    error = URLError(certificate_error())
    monkeypatch.setattr(_grouplens, "urlopen", raises(error))
    monkeypatch.setattr(_grouplens, "build_opener", lambda *args: pytest.fail("URL not allowed"))
    with pytest.raises(URLError) as caught:
        _grouplens.download_movielens_archive(url, tmp_path / "archive.zip")
    assert caught.value is error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("error", [
    URLError(certificate_error(9)),   # Not yet valid.
    URLError(certificate_error(18)),  # Self-signed.
    URLError(certificate_error(20)),  # Untrusted issuer.
    URLError(certificate_error(62)),  # Hostname mismatch.
    URLError(certificate_error(None)),
    URLError("certificate has expired"),  # Do not guess from message text.
    URLError(ConnectionRefusedError("refused")),
    HTTPError(URLS[0], 404, "Not Found", {}, None),
])
def test_other_failures_propagate_without_insecure_retry(tmp_path, monkeypatch, error):
    monkeypatch.setattr(_grouplens, "urlopen", raises(error))
    monkeypatch.setattr(_grouplens, "build_opener", lambda *args: pytest.fail("Not an expiry"))
    with pytest.raises(URLError) as caught:
        _grouplens.download_movielens_archive(URLS[0], tmp_path / "archive.zip")
    assert caught.value is error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("url", [URLS[1], "https://example.org/data.zip", URLS[0].replace("https:", "http:")])
@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_unverified_redirects_are_rejected(url, status):
    with pytest.raises(URLError, match="Refusing redirect"):
        _grouplens._RejectRedirects().redirect_request(Request(URLS[0]), None, status, "", {}, url)


def test_cached_archive_never_opens_a_connection(tmp_path, monkeypatch):
    path = tmp_path / "archive.zip"
    path.write_bytes(b"cached")
    monkeypatch.setattr(_grouplens, "urlopen", lambda *args, **kwargs: pytest.fail("Cached"))
    assert _grouplens.download_movielens_archive(URLS[0], path) == path
    assert path.read_bytes() == b"cached"


@pytest.mark.parametrize("failure", ["truncated", "connection"])
def test_failed_fallback_does_not_leave_a_partial_cache(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(_grouplens, "urlopen", raises(URLError(certificate_error())))
    retry = (lambda *args, **kwargs: response(b"partial", length=100)) if failure == "truncated" else raises(URLError("offline"))
    monkeypatch.setattr(_grouplens, "build_opener", lambda *args: SimpleNamespace(open=retry))
    with pytest.warns(RuntimeWarning), pytest.raises(OSError):
        _grouplens.download_movielens_archive(URLS[0], tmp_path / "archive.zip")
    assert list(tmp_path.iterdir()) == []


def test_generic_downloader_remains_verified_on_expiry(tmp_path, monkeypatch):
    error = URLError(certificate_error())
    monkeypatch.setattr(_download, "urlopen", raises(error))
    with pytest.raises(URLError) as caught:
        _download.download(URLS[0], tmp_path / "archive.zip", show_progress=False)
    assert caught.value is error
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("module,cls,folder", [
    (movielens1m, movielens1m.MovieLens1M, "ml-1m"),
    (movielens20m, movielens20m.MovieLens20M, "ml-20m"),
])
def test_loader_limits_workaround_to_archive_and_preserves_metadata_download(tmp_path, monkeypatch, module, cls, folder):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(folder + "/README.txt", "fixture")
    calls = []
    def verified(url, **kwargs):
        calls.append(url)
        raise URLError(certificate_error())
    monkeypatch.setattr(_grouplens, "urlopen", verified)
    monkeypatch.setattr(_grouplens, "build_opener", lambda *args: SimpleNamespace(
        open=lambda *args, **kwargs: response(buffer.getvalue()),
    ))
    metadata_calls = []
    def retrieve(url, path):
        metadata_calls.append(url)
        assert url == cls.text_descriptions_url
        path.write_bytes(b"metadata")
    monkeypatch.setattr(module, "urlretrieve", retrieve)
    dataset = cls(data_dir=tmp_path)
    with pytest.warns(RuntimeWarning):
        dataset.download()
    dataset.download()  # All files are now cached.
    assert calls == [cls.url]
    assert metadata_calls == [cls.text_descriptions_url]
    assert (dataset.root / folder / "README.txt").read_text() == "fixture"
    assert (dataset.root / "item_text_descriptions.feather").read_bytes() == b"metadata"
