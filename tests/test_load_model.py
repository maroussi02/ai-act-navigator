"""UNIT TESTS for how embed.py loads the model. No real model: a fake stands in for the
sentence_transformers library, so these run in milliseconds and never touch the network.
(The tests with the real model are in test_embed.py, marked as integration tests.)
"""

import os
import sys
from types import SimpleNamespace

import pytest

from ai_act_navigator import embed
from ai_act_navigator.embed import MODEL_REVISION, ModelNotAvailableError


@pytest.fixture
def fake_library(monkeypatch):
    """Put a fake sentence_transformers in place and record each load attempt.

    Returns the list of attempts (True = cache only, False = download allowed) and a dict
    to choose which attempts fail.
    """
    calls: list[bool] = []
    fails = {"cache": False, "download": False}

    def fake_model(name, revision, trust_remote_code, local_files_only=False):
        calls.append(local_files_only)
        if local_files_only and fails["cache"]:
            raise OSError("not in the cache")  # what the library raises on a first run
        if not local_files_only and fails["download"]:
            raise OSError("no internet")
        return "cached model" if local_files_only else "downloaded model"

    # load_model imports the library itself, so the fake goes where `import` looks first.
    def fake_cross_encoder(*args, **kwargs):
        return "reranker: " + fake_model(*args, **kwargs)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=fake_model, CrossEncoder=fake_cross_encoder),
    )
    # load_model may set HF_HUB_OFFLINE; monkeypatch puts the original value back afterwards.
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return calls, fails


def test_model_is_pinned_to_an_exact_version():
    assert len(MODEL_REVISION) == 40  # a full git commit hash, not a moving name like "main"


def test_cached_model_loads_without_download(fake_library):
    calls, _ = fake_library
    assert embed.load_model("some/model", "abc", allow_download=True) == "cached model"
    assert calls == [True]


def test_missing_model_is_downloaded_when_allowed(fake_library):
    calls, fails = fake_library
    fails["cache"] = True
    assert embed.load_model("some/model", "abc", allow_download=True) == "downloaded model"
    assert calls == [True, False]  # cache first, then download


def test_offline_mode_never_downloads(fake_library):
    calls, fails = fake_library
    fails["cache"] = True
    with pytest.raises(ModelNotAvailableError, match="Run: uv run ai-act index"):
        embed.load_model("some/model", "abc", allow_download=False)
    assert calls == [True]  # tried the cache only
    assert os.environ["HF_HUB_OFFLINE"] == "1"  # the library's own requests are off too


def test_failed_download_is_a_clear_error(fake_library):
    _, fails = fake_library
    fails.update(cache=True, download=True)
    with pytest.raises(ModelNotAvailableError, match="Check your internet connection"):
        embed.load_model("some/model", "abc", allow_download=True)


def test_a_reranker_loads_with_the_same_offline_rules(fake_library):
    calls, fails = fake_library
    fails["cache"] = True
    with pytest.raises(ModelNotAvailableError, match="reranker model isn't downloaded"):
        embed.load_model("some/reranker", "abc", allow_download=False, kind="reranker")
    fails["cache"] = False
    assert embed.load_model("some/reranker", "abc", True, kind="reranker") == (
        "reranker: cached model"
    )
