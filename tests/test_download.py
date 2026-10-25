"""UNIT TESTS for download.py.

No real network calls: `monkeypatch` swaps httpx.get for a fake during each test.
That keeps tests fast, offline-proof, and lets us simulate a server error (503) or a
tampered file on purpose.
`tmp_path` is a fresh temporary folder pytest creates per test, so no real files are touched.
"""

import httpx
import pytest

from ai_act_navigator import download as download_module
from ai_act_navigator.download import IntegrityError, download, sha256_of

CONTENT = b"<html>the official text</html>"
GOOD_HASH = sha256_of(CONTENT)


def fake_get(status: int = 200, content: bytes = CONTENT):
    """A stand-in for httpx.get that returns a fixed response without any network."""
    request = httpx.Request("GET", download_module.SOURCE_URL)
    return lambda *args, **kwargs: httpx.Response(status, content=content, request=request)


def test_skips_download_if_the_right_file_exists(tmp_path, monkeypatch):
    dest = tmp_path / "ai_act.html"
    dest.write_bytes(CONTENT)

    def fail(*args, **kwargs):
        raise AssertionError("should not download when the right file exists")

    monkeypatch.setattr(httpx, "get", fail)

    assert download(dest, expected_sha256=GOOD_HASH) == dest


def test_downloads_and_saves_file(tmp_path, monkeypatch):
    dest = tmp_path / "raw" / "ai_act.html"
    monkeypatch.setattr(httpx, "get", fake_get())

    download(dest, expected_sha256=GOOD_HASH)

    assert dest.read_bytes() == CONTENT


def test_replaces_a_damaged_existing_file(tmp_path, monkeypatch):
    dest = tmp_path / "ai_act.html"
    dest.write_bytes(b"half a file")  # e.g. left over from an interrupted download
    monkeypatch.setattr(httpx, "get", fake_get())

    download(dest, expected_sha256=GOOD_HASH)

    assert dest.read_bytes() == CONTENT


def test_rejects_a_file_with_the_wrong_fingerprint(tmp_path, monkeypatch):
    dest = tmp_path / "ai_act.html"
    monkeypatch.setattr(httpx, "get", fake_get(content=b"<html>tampered</html>"))

    with pytest.raises(IntegrityError):
        download(dest, expected_sha256=GOOD_HASH)
    assert not dest.exists()


def test_raises_on_http_error(tmp_path, monkeypatch):
    dest = tmp_path / "ai_act.html"
    monkeypatch.setattr(httpx, "get", fake_get(status=503, content=b""))

    with pytest.raises(httpx.HTTPStatusError):
        download(dest, expected_sha256=GOOD_HASH)
    assert not dest.exists()


def test_source_is_https():
    assert download_module.SOURCE_URL.startswith("https://")
