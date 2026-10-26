"""UNIT TEST for search.py: a fake query embedder and a real store in a temp folder."""

from ai_act_navigator.models import Chunk
from ai_act_navigator.search import search
from ai_act_navigator.store import VectorStore

URL = "https://example.org"


class FakeQueryEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "chatbot" in text else [0.0, 1.0]


def test_search_returns_the_closest_chunk_first(tmp_path):
    store = VectorStore("fake-model", tmp_path)
    store.upsert(
        [
            Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "Tell users.", URL),
            Chunk("art-99-1", "article", "Art. 99(1)", "Penalties", "Fines.", URL),
        ],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    hits = search("Must a chatbot say it is AI?", FakeQueryEmbedder(), store, k=2)
    assert [h.chunk.ref for h in hits] == ["Art. 50(1)", "Art. 99(1)"]
