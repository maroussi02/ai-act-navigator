"""UNIT TESTS for index.py.

They use a FakeEmbedder instead of the real model: no 130 MB download, runs in a second,
and we choose the vectors ourselves so we know exactly which chunk should be "closest".
Chroma itself is real, but writes to `tmp_path`, a fresh temporary folder per test.
"""

import pytest

from ai_act_navigator.index import build_index, text_to_embed
from ai_act_navigator.models import Chunk

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


@pytest.fixture
def collection(tmp_path):
    return build_index(CHUNKS, FakeEmbedder(), tmp_path)


def test_all_chunks_are_stored(collection):
    assert collection.count() == 3


def test_citation_data_is_stored_with_each_vector(collection):
    stored = collection.get(ids=["art-50-1"])
    assert stored["documents"] == ["Tell users it's an AI."]
    assert stored["metadatas"][0] == {
        "ref": "Art. 50(1)",
        "type": "article",
        "title": "Transparency",
        "url": URL,
    }


def test_running_twice_does_not_create_duplicates(tmp_path):
    build_index(CHUNKS, FakeEmbedder(), tmp_path)
    collection = build_index(CHUNKS, FakeEmbedder(), tmp_path)
    assert collection.count() == 3


def test_chunks_that_no_longer_exist_are_removed(tmp_path):
    build_index(CHUNKS, FakeEmbedder(), tmp_path)
    collection = build_index(CHUNKS[:2], FakeEmbedder(), tmp_path)  # "rct-1" was dropped
    assert sorted(collection.get()["ids"]) == ["art-5-1", "art-50-1"]


def test_closest_chunk_is_found(collection):
    # A query pointing in the "Prohibited" direction must return Art. 5(1) first.
    result = collection.query(query_embeddings=[[0.0, 1.0, 0.0]], n_results=1)
    assert result["metadatas"][0][0]["ref"] == "Art. 5(1)"


def test_text_to_embed_adds_citation_and_title():
    assert text_to_embed(CHUNKS[0]) == "Art. 50(1) Transparency: Tell users it's an AI."
    assert text_to_embed(CHUNKS[2]) == "Recital 1: The purpose of this Regulation."
