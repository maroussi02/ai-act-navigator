"""UNIT TESTS for store.py: real Chroma in a temporary folder, tiny hand-made vectors."""

import pytest

from ai_act_navigator.models import Chunk
from ai_act_navigator.store import IndexMismatchError, VectorStore

URL = "https://example.org"
CHUNKS = [
    Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "Tell users it's an AI.", URL),
    Chunk("art-5-1", "article", "Art. 5(1)", "Prohibited AI practices", "Banned uses.", URL),
]
VECTORS = [[1.0, 0.0], [0.0, 1.0]]


@pytest.fixture
def store(tmp_path):
    store = VectorStore("model-a", tmp_path)
    store.upsert(CHUNKS, VECTORS)
    return store


def test_search_returns_full_chunks_best_first(store):
    hits = store.search([0.0, 1.0], k=2)
    assert [h.chunk for h in hits] == [CHUNKS[1], CHUNKS[0]]  # every field survives the store
    assert hits[0].score == pytest.approx(1.0)  # same direction = same meaning
    assert hits[1].score == pytest.approx(0.0)


def test_rerunning_upsert_creates_no_duplicates(store):
    store.upsert(CHUNKS, VECTORS)
    assert store.count() == 2


def test_chunks_that_no_longer_exist_are_removed(store):
    store.upsert(CHUNKS[:1], VECTORS[:1])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=5)] == ["art-50-1"]


def test_search_refuses_an_index_built_with_another_model(store, tmp_path):
    other = VectorStore("model-b", tmp_path)
    with pytest.raises(IndexMismatchError, match="model-a"):
        other.search([1.0, 0.0])


def test_search_on_an_empty_index_explains_what_to_do(tmp_path):
    with pytest.raises(IndexMismatchError, match="empty"):
        VectorStore("model-a", tmp_path).search([1.0, 0.0])
