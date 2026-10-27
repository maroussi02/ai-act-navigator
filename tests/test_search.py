"""UNIT TESTS for search.py: fusing ranked lists, and hybrid search with a fake embedder
and a real store in a temp folder."""

import pytest

from ai_act_navigator.models import Chunk
from ai_act_navigator.search import Searcher, fuse
from ai_act_navigator.store import VectorStore

URL = "https://example.org"
CHATBOT = Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "Tell users.", URL)
FINES = Chunk("art-99-1", "article", "Art. 99(1)", "Penalties", "Fines.", URL)
LITERACY = Chunk("art-4", "article", "Art. 4", "AI literacy", "Train your staff.", URL)


class FakeQueryEmbedder:
    """Every question 'means' chatbots unless it mentions fines."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0, 0.0] if "fine" in text else [1.0, 0.0, 0.0]


@pytest.fixture
def searcher(tmp_path):
    store = VectorStore("fake-model", tmp_path)
    # By meaning, a chatbot question finds CHATBOT (similarity 1.0), then FINES (0.6), then
    # LITERACY (0.0): no ties, so the order is the same on every run. Art. 4's vector is far
    # from every question, so meaning search always ranks it last.
    store.upsert([CHATBOT, FINES, LITERACY], [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    return Searcher(FakeQueryEmbedder(), store)


def test_fuse_prefers_a_chunk_found_by_several_lists():
    # FINES is 2nd in both lists; CHATBOT and LITERACY are 1st in one list each.
    hits = fuse([[CHATBOT, FINES], [LITERACY, FINES]])
    assert [h.chunk for h in hits] == [FINES, CHATBOT, LITERACY]
    assert hits[0].score == pytest.approx(2 / 62)
    assert hits[1].score == pytest.approx(1 / 61)


def test_fuse_of_nothing_is_empty():
    assert fuse([[], []]) == []


def test_vector_mode_is_meaning_only(searcher):
    hits = searcher.search("Must a chatbot say it is AI?", k=2, mode="vector")
    assert [h.chunk.ref for h in hits] == ["Art. 50(1)", "Art. 99(1)"]


def test_hybrid_finds_a_named_article_that_meaning_search_ranks_last(searcher):
    hits = searcher.search("What does article 4 require?", k=1)
    assert [h.chunk.ref for h in hits] == ["Art. 4"]  # keyword + named-article lists agree


def test_hybrid_finds_a_chunk_that_only_keyword_search_matches(searcher):
    # By meaning (fake) this is about chatbots; by keywords only Art. 4 says "staff".
    hits = searcher.search("Who must train their staff?", k=3)
    assert hits[0].chunk.ref == "Art. 4"


def test_unknown_mode_is_an_error(searcher):
    with pytest.raises(ValueError, match="Unknown search mode"):
        searcher.search("q", mode="bm25")
