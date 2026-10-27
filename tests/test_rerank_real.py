"""INTEGRATION TESTS for rerank.py with the real model and real chunks of the AI Act.

The first run downloads the reranker (about 90 MB) into ~/.cache/huggingface; CI caches it.
Skipped unless the law was downloaded first (CI downloads it before running tests).
"""

import pytest

from ai_act_navigator.chunk import parse
from ai_act_navigator.download import DEFAULT_PATH
from ai_act_navigator.models import Hit
from ai_act_navigator.rerank import Reranker

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="full text not downloaded"),
]


@pytest.fixture(scope="module")
def chunks_by_ref():
    return {c.ref: c for c in parse(DEFAULT_PATH.read_text(encoding="utf-8"))}


def test_the_key_paragraph_comes_first(chunks_by_ref):
    # An AN-003b regression (golden question P6): combined search sent Art. 53(4) and
    # Recital 97 but not Art. 53(1), the paragraph listing the duties, and Claude declined.
    # The hits start in that wrong order; the reranker must put Art. 53(1) first.
    refs = ("Art. 53(4)", "Recital 97", "Recital 3", "Art. 53(1)")
    hits = [Hit(chunks_by_ref[ref], 0.0) for ref in refs]
    question = "What obligations do providers of general-purpose AI models have?"
    assert Reranker().rerank(question, hits)[0].chunk.ref == "Art. 53(1)"


def test_an_unrelated_chunk_ranks_last(chunks_by_ref):
    refs = ("Recital 3", "Art. 53(1)", "Art. 53(4)")
    hits = [Hit(chunks_by_ref[ref], 0.0) for ref in refs]
    question = "What obligations do providers of general-purpose AI models have?"
    assert Reranker().rerank(question, hits)[-1].chunk.ref == "Recital 3"
