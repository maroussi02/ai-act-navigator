"""UNIT TESTS for index.py.

They use a FakeEmbedder instead of the real model: no 130 MB download, runs in a second,
and we choose the vectors ourselves so we know exactly which chunk should be "closest".
The store itself is real Chroma, writing to `tmp_path`, a fresh temporary folder per test.
"""

from ai_act_navigator.index import build_index, text_to_embed
from ai_act_navigator.models import Chunk
from ai_act_navigator.store import VectorStore

URL = "https://example.org"
CHUNKS = [
    Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "Tell users it's an AI.", URL),
    Chunk("art-5-1", "article", "Art. 5(1)", "Prohibited AI practices", "Banned uses.", URL),
    Chunk("rct-1", "recital", "Recital 1", "", "The purpose of this Regulation.", URL),
]


class FakeEmbedder:
    """Gives each chunk its own direction in a 3-number space, based on a keyword."""

    VECTORS = {"Transparency": [1.0, 0.0, 0.0], "Prohibited": [0.0, 1.0, 0.0]}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def _vector(self, text: str) -> list[float]:
        for keyword, vector in self.VECTORS.items():
            if keyword in text:
                return vector
        return [0.0, 0.0, 1.0]


def test_all_chunks_are_indexed_and_findable(tmp_path):
    store = VectorStore("fake-model", tmp_path)
    build_index(CHUNKS, FakeEmbedder(), store)

    assert store.count() == 3
    # A query pointing in the "Prohibited" direction must find Art. 5(1) first.
    assert store.search([0.0, 1.0, 0.0], k=1)[0].chunk.ref == "Art. 5(1)"


def test_text_to_embed_adds_citation_and_title():
    assert text_to_embed(CHUNKS[0]) == "Art. 50(1) Transparency: Tell users it's an AI."
    assert text_to_embed(CHUNKS[2]) == "Recital 1: The purpose of this Regulation."
