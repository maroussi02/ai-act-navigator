"""UNIT TESTS for rerank.py: a fake model stands in for the cross-encoder (no download)."""

from ai_act_navigator.models import Chunk, Hit
from ai_act_navigator.rerank import RERANK_REVISION, Reranker

URL = "https://example.org"


def hit(ref: str, text: str, title: str = "") -> Hit:
    return Hit(Chunk(ref.lower(), "article", ref, title, text, URL), 0.5)


class FakeModel:
    """Scores a pair by how often "chatbot" appears in the text; records what it was sent."""

    def __init__(self):
        self.pairs = []

    def predict(self, pairs):
        self.pairs = pairs
        return [text.count("chatbot") for _, text in pairs]


HITS = [hit("Art. 50(4)", "Deep fakes."), hit("Art. 50(1)", "A chatbot, a chatbot.")]


def test_reranked_hits_are_best_first_with_the_reranker_score():
    reranked = Reranker(model=FakeModel()).rerank("Must a chatbot say so?", HITS)
    assert [(h.chunk.ref, h.score) for h in reranked] == [("Art. 50(1)", 2.0), ("Art. 50(4)", 0.0)]


def test_only_the_top_k_are_kept():
    assert [h.chunk.ref for h in Reranker(model=FakeModel()).rerank("q", HITS, k=1)] == [
        "Art. 50(1)"
    ]


def test_no_hits_means_no_model_call():
    model = FakeModel()
    assert Reranker(model=model).rerank("q", []) == []
    assert model.pairs == []


def test_the_model_reads_question_and_citation_title_text():
    model = FakeModel()
    Reranker(model=model).rerank("q", [hit("Art. 4", "Train staff.", title="AI literacy")])
    assert model.pairs == [("q", "Art. 4 AI literacy: Train staff.")]


def test_reranker_is_pinned_to_an_exact_version():
    assert len(RERANK_REVISION) == 40  # a full git commit hash, not a moving name like "main"
