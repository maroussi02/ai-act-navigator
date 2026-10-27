"""UNIT TESTS for store.py: real Chroma in a temporary folder, tiny hand-made vectors."""

import math
import random

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


def test_all_chunks_returns_every_stored_chunk(store):
    assert sorted(store.all_chunks(), key=lambda c: c.id) == sorted(CHUNKS, key=lambda c: c.id)


def test_all_chunks_of_an_empty_index_is_a_clear_error(tmp_path):
    with pytest.raises(IndexMismatchError, match="The index is empty"):
        VectorStore("model-a", tmp_path).all_chunks()


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


def test_index_from_an_older_version_asks_for_a_rebuild(tmp_path):
    # Built before the store recorded its model: it has chunks, but no model information.
    store = VectorStore("model-a", tmp_path)
    store._collection.upsert(
        ids=["x"],
        embeddings=[[1.0, 0.0]],
        documents=["t"],
        metadatas=[{"ref": "r", "type": "article", "title": "", "url": "u"}],
    )
    with pytest.raises(IndexMismatchError, match="older version"):
        store.search([1.0, 0.0])


def test_search_is_exact_on_many_vectors(tmp_path):
    """Chroma's own (approximate) index could skip chunks; our search must not."""
    rng = random.Random(0)  # fixed seed: the same "random" vectors on every run
    vectors = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(300)]
    chunks = [Chunk(f"c-{i:03}", "recital", f"Recital {i}", "", "t", URL) for i in range(300)]
    store = VectorStore("model-a", tmp_path)
    store.upsert(chunks, vectors)
    query = [rng.uniform(-1, 1) for _ in range(8)]

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))

    expected = sorted(range(300), key=lambda i: -cosine(query, vectors[i]))[:10]
    assert [h.chunk.id for h in store.search(query, k=10)] == [chunks[i].id for i in expected]


def test_equal_scores_are_ordered_by_id(tmp_path):
    twins = [Chunk(i, "recital", f"Recital {n}", "", "t", URL) for n, i in [(2, "b"), (1, "a")]]
    store = VectorStore("model-a", tmp_path)
    store.upsert(twins, [[1.0, 0.0], [1.0, 0.0]])
    assert [h.chunk.id for h in store.search([1.0, 0.0], k=2)] == ["a", "b"]


def test_search_sees_new_vectors_after_upsert(store):
    assert store.search([1.0, 0.0], k=1)[0].chunk.id == "art-50-1"
    store.upsert(CHUNKS, [[0.0, 1.0], [1.0, 0.0]])  # swap the two vectors
    assert store.search([1.0, 0.0], k=1)[0].chunk.id == "art-5-1"
