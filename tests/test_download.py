"""UNIT TESTS for download.py.

No real network calls: `monkeypatch` swaps httpx.get for a fake during each test.
That keeps tests fast, offline-proof, and lets us simulate a server error (503) on purpose.
`tmp_path` is a fresh temporary folder pytest creates per test, so no real files are touched.
"""

import httpx
import pytest

from ai_act_navigator import download as download_module


def test_skips_download_if_file_exists(tmp_path, monkeypatch):
    dest = tmp_path / "ai_act.html"
    dest.write_text("already here")

    def fail(*args, **kwargs):
        raise AssertionError("should not download when the file exists")

    monkeypatch.setattr(httpx, "get", fail)

    assert download_module.download(dest) == dest
    assert dest.read_text() == "already here"


def test_downloads_and_saves_file(tmp_path, monkeypatch):
    dest = tmp_path / "raw" / "ai_act.html"
    request = httpx.Request("GET", download_module.SOURCE_URL)
    monkeypatch.setattr(
        httpx, "get", lambda *a, **kw: httpx.Response(200, content=b"<html/>", request=request)
    )

    download_module.download(dest)

    assert dest.read_bytes() == b"<html/>"


def test_raises_on_http_error(tmp_path, monkeypatch):
    dest = tmp_path / "ai_act.html"
    request = httpx.Request("GET", download_module.SOURCE_URL)
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: httpx.Response(503, request=request))

    with pytest.raises(httpx.HTTPStatusError):
        download_module.download(dest)
    assert not dest.exists()
