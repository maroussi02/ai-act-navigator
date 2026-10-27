"""UNIT TESTS for search.py: fusing ranked lists, and hybrid search with a fake embedder
and a real store in a temp folder."""

import pytest

from ai_act_navigator.models import Chunk, Hit
from ai_act_navigator.search import Searcher, fuse, keep_named_articles
from ai_act_navigator.store import VectorStore

URL = "https://example.org"
CHATBOT = Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "Tell users.", URL)
FINES = Chunk("art-99-1", "article", "Art. 99(1)", "Penalties", "Fines.", URL)
LITERACY = Chunk("art-4", "article", "Art. 4", "AI literacy", "Train your staff.", URL)


class FakeQueryEmbedder:
    """Every question 'means' chatbots unless it mentions fines."""

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0, 0.0] if "fine" in text else [1.0, 0.0, 0.0]


class ReverseReranker:
    """Puts the candidates in reverse order, so its effect is easy to see."""

    def rerank(self, question, hits, k=None):
        return [Hit(h.chunk, float(i)) for i, h in enumerate(hits)][::-1][:k]


@pytest.fixture
def store(tmp_path):
    store = VectorStore("fake-model", tmp_path)
    # By meaning, a chatbot question finds CHATBOT (similarity 1.0), then FINES (0.6), then
    # LITERACY (0.0): no ties, so the order is the same on every run. Art. 4's vector is far
    # from every question, so meaning search always ranks it last.
    store.upsert([CHATBOT, FINES, LITERACY], [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    return store


@pytest.fixture
def searcher(store):
    return Searcher(FakeQueryEmbedder(), store, ReverseReranker())


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
    hits = searcher.search("What does article 4 require?", k=1, mode="hybrid")
    assert [h.chunk.ref for h in hits] == ["Art. 4"]  # keyword + named-article lists agree


def test_hybrid_finds_a_chunk_that_only_keyword_search_matches(searcher):
    # By meaning (fake) this is about chatbots; by keywords only Art. 4 says "staff".
    hits = searcher.search("Who must train their staff?", k=3, mode="hybrid")
    assert hits[0].chunk.ref == "Art. 4"


def test_unknown_mode_is_an_error(searcher):
    with pytest.raises(ValueError, match="Unknown search mode"):
        searcher.search("q", mode="bm25")


def test_rerank_mode_reorders_the_hybrid_candidates(searcher):
    hybrid = [h.chunk.ref for h in searcher.search("Must a chatbot say so?", k=3, mode="hybrid")]
    reranked = [h.chunk.ref for h in searcher.search("Must a chatbot say so?", k=3)]
    assert reranked == hybrid[::-1]  # rerank is the default mode


def test_rerank_mode_keeps_a_named_article(searcher):
    # The hybrid list starts with Art. 4; the reverse reranker would push it to the end.
    hits = searcher.search("What does article 4 require?", k=1)
    assert [h.chunk.ref for h in hits] == ["Art. 4"]


def test_rerank_mode_without_a_reranker_is_a_clear_error(store):
    with pytest.raises(ValueError, match="needs a reranker"):
        Searcher(FakeQueryEmbedder(), store).search("q")


def ranked(*refs: str) -> list[Hit]:
    return [Hit(Chunk(r, "article", r, "", "text", URL), 1.0) for r in refs]


def test_a_named_article_replaces_the_lowest_unnamed_hit():
    hits = keep_named_articles(ranked("Art. 102", "Art. 91", "Art. 107", "Art. 4"), {"Art. 4"}, 3)
    assert [h.chunk.ref for h in hits] == ["Art. 102", "Art. 91", "Art. 4"]


def test_a_named_article_already_in_the_top_k_changes_nothing():
    hits = keep_named_articles(ranked("Art. 4(1)", "Art. 9", "Art. 4(2)"), {"Art. 4"}, 2)
    assert [h.chunk.ref for h in hits] == ["Art. 4(1)", "Art. 9"]


def test_two_named_articles_never_push_each_other_out():
    hits = keep_named_articles(
        ranked("Art. 1", "Art. 2", "Art. 4", "Art. 5"), {"Art. 4", "Art. 5"}, 2
    )
    assert [h.chunk.ref for h in hits] == ["Art. 4", "Art. 5"]


def test_a_named_article_not_among_the_hits_is_skipped():
    hits = keep_named_articles(ranked("Art. 1", "Art. 2"), {"Art. 200"}, 2)
    assert [h.chunk.ref for h in hits] == ["Art. 1", "Art. 2"]
