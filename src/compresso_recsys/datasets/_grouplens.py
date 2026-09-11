"""Temporary, archive-only workaround for GroupLens's expired TLS certificate."""
from __future__ import annotations

import ssl
import warnings
from pathlib import Path
from urllib.error import URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, build_opener, urlopen

from ._download import download


_ARCHIVE_URLS = frozenset({
    "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "https://files.grouplens.org/datasets/movielens/ml-20m.zip",
})
_CERT_HAS_EXPIRED = 10  # OpenSSL X509_V_ERR_CERT_HAS_EXPIRED.


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never carry an unverified context to another URL, even on this host.
        raise URLError("Refusing redirect during the temporary GroupLens TLS bypass")


def _open_archive(url: str, *, timeout: float):
    # Keep the original verified path active: once GroupLens renews its
    # certificate, every new download automatically uses verification again.
    try:
        return urlopen(url, timeout=timeout)
    except (URLError, ssl.SSLCertVerificationError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else exc
        if (url not in _ARCHIVE_URLS
                or not isinstance(reason, ssl.SSLCertVerificationError)
                or getattr(reason, "verify_code", None) != _CERT_HAS_EXPIRED):
            raise

    # TODO (2026-09-11): recheck BOTH URLs above with normal TLS verification
    # (e.g. curl --fail --head URL, without --insecure). When they work, remove
    # this fallback and use download(url, path, show_progress=False) below.
    # The pre-workaround loader call was urlretrieve(self.url, zip_path).
    # Do not use ssl._create_default_https_context or install_opener: both
    # would weaken unrelated downloads, including beeFormer metadata.
    warnings.warn(
        f"GroupLens's TLS certificate has expired for {url}. Temporarily retrying "
        "this MovieLens archive without certificate verification; server identity "
        "is not verified. Other hosts remain verified. Remove the temporary "
        "workaround in datasets/_grouplens.py once GroupLens renews its certificate.",
        RuntimeWarning,
        stacklevel=3,
    )
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    opener = build_opener(HTTPSHandler(context=context), _RejectRedirects())
    return opener.open(url, timeout=timeout)


def download_movielens_archive(url: str, path: Path) -> Path:
    """Verify first; retry only known MovieLens archives on certificate expiry."""
    return download(url, path, show_progress=False, opener=_open_archive)
