"""INTEGRATION TESTS: a question that names an article finds it, on the full real AI Act.

Checks acceptance criterion 1 of AN-003 with the real chunks, the real keyword index and the
real fusion. Meaning search is switched off on purpose: every chunk gets the same vector, so
its ranking is noise, and the named article has to be found by keywords and the
named-article list alone. Free (no Claude, no embedding model) and takes a few seconds.
Skipped unless the law was downloaded first (CI downloads it before running tests).
"""

import pytest

from ai_act_navigator.chunk import parse
from ai_act_navigator.download import DEFAULT_PATH
from ai_act_navigator.golden import load_golden
from ai_act_navigator.refs import article_of
from ai_act_navigator.search import DEFAULT_K, Searcher
from ai_act_navigator.store import VectorStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="full text not downloaded"),
]

EXACT_REF = [q for q in load_golden() if q.type == "exact_ref"]


class UselessEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture(scope="module")
def searcher(tmp_path_factory):
    chunks = parse(DEFAULT_PATH.read_text(encoding="utf-8"))
    store = VectorStore("same-vector-for-all", tmp_path_factory.mktemp("chroma"))
    store.upsert(chunks, [[1.0, 0.0]] * len(chunks))
    return Searcher(UselessEmbedder(), store)


@pytest.mark.parametrize("question", EXACT_REF, ids=[q.id for q in EXACT_REF])
def test_a_named_article_is_among_the_chunks_sent(searcher, question):
    sent = {article_of(hit.chunk.ref) for hit in searcher.search(question.question, k=DEFAULT_K)}
    assert sent & set(question.sources), f"{question.question} -> {sorted(sent)}"
