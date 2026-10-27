"""The vector store: saves chunk embeddings and finds the closest ones to a question.

It wraps Chroma, a vector database that runs as a library and saves everything as normal
files in a folder (data/chroma/): no account, no server. Only this file knows about Chroma.
The rest of the code talks to VectorStore, so swapping Chroma for another database later
(or for LangChain's vector store wrapper) means changing this one file.

Chroma stores the chunks and vectors, but search doesn't use Chroma's own index. That index
(HNSW) is *approximate*: it trades a little accuracy for speed on millions of vectors, and on
this law it sometimes skipped a chunk (Recital 119 for the chatbot question was missing in
2 of 4 runs with the exact same question vector). With only about 950 chunks, comparing the
question with every vector is exact, gives the same result on every run, and is fast enough:
about 8 ms per search in plain Python (measured on an Apple M5), after a one-time 65 ms load.
"""

import math
from pathlib import Path

import chromadb
from chromadb.config import Settings

from ai_act_navigator.models import Chunk, Hit
from ai_act_navigator.paths import DATA_DIR

CHROMA_PATH = DATA_DIR / "chroma"
COLLECTION_NAME = "ai_act"  # a "collection" is like a table in a normal database


class IndexMismatchError(Exception):
    """The index was built with a different embedding model than the one searching it."""


class VectorStore:
    """Stores chunks with their embeddings and searches them by meaning.

    `embedding_model` names the model whose vectors go in and come out. Vectors from two
    different models live in different "number spaces", so comparing them gives nonsense.
    The store records which model built it and refuses to search with another one.
    """

    def __init__(self, embedding_model: str, path: Path = CHROMA_PATH) -> None:
        self.embedding_model = embedding_model
        client = chromadb.PersistentClient(
            path=str(path),
            # Chroma sends anonymous usage statistics by default. We turn that off (privacy).
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = client.get_or_create_collection(
            COLLECTION_NAME,
            # "cosine": closeness = the angle between two vectors, the standard for text.
            # (Chroma always builds this index; our search doesn't use it, see the top.)
            configuration={"hnsw": {"space": "cosine"}},
            # We always pass our own embeddings, so Chroma never needs its built-in model.
            embedding_function=None,
        )
        # Every chunk with its vector scaled to length 1, read from Chroma on first use.
        # None = not read yet, or out of date after upsert.
        self._rows: list[tuple[Chunk, list[float]]] | None = None

    def count(self) -> int:
        """How many chunks are stored."""
        return self._collection.count()

    def upsert(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Save chunks and their embeddings, replacing the previous contents. Safe to rerun.

        `upsert` = insert, or update if the id already exists, so rerunning creates no
        duplicates. Chunks from an older run that no longer exist are deleted afterwards,
        so search never returns outdated text.
        """
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],  # the original text, returned with results
            # Metadata travels with each vector, so a search result knows how to cite itself.
            metadatas=[
                {"ref": c.ref, "type": c.type, "title": c.title, "url": c.url} for c in chunks
            ],
        )
        # Delete stale entries only after the new ones are saved: if embedding crashed
        # halfway, the old index would still be complete instead of half-empty.
        current_ids = {c.id for c in chunks}
        stale_ids = [i for i in self._collection.get(include=[])["ids"] if i not in current_ids]
        if stale_ids:
            self._collection.delete(ids=stale_ids)
        self._collection.modify(metadata={"embedding_model": self.embedding_model})
        self._rows = None  # the in-memory copy is out of date now

    def search(self, query_embedding: list[float], k: int = 5) -> list[Hit]:
        """The k chunks closest in meaning to the query, best first. Exact: every stored
        vector is compared, so the best chunks can't be skipped (see the top of this file).

        Closeness is the cosine similarity: 1.0 = same direction (same meaning), 0.0 =
        unrelated. Equal scores are ordered by chunk id, so results never vary between runs.
        """
        query = _unit(query_embedding)
        scored = [(_dot(query, vector), chunk) for chunk, vector in self._load()]
        best = sorted(scored, key=lambda pair: (-pair[0], pair[1].id))[:k]
        return [Hit(chunk=chunk, score=similarity) for similarity, chunk in best]

    def all_chunks(self) -> list[Chunk]:
        """Every stored chunk, ordered by id. Keyword search is built from these, so it always
        searches exactly the chunks the vector search holds, in the same order every run."""
        return [chunk for chunk, _ in self._load()]

    def _load(self) -> list[tuple[Chunk, list[float]]]:
        """All chunks with their vectors (length 1), read from Chroma once, ordered by id."""
        self._check_ready()
        if self._rows is None:
            result = self._collection.get(include=["embeddings", "documents", "metadatas"])
            rows = zip(
                result["ids"],
                result["embeddings"],
                result["documents"],
                result["metadatas"],
                strict=True,
            )
            self._rows = sorted(
                # Chroma returns each vector as a numpy array; plain Python floats keep the
                # math below simple to read.
                (
                    (_to_chunk(i, text, meta), _unit([float(x) for x in vector]))
                    for i, vector, text, meta in rows
                ),
                key=lambda row: row[0].id,
            )
        return self._rows

    def _check_ready(self) -> None:
        """Raise IndexMismatchError unless the index is built, with the model searching it."""
        built_with = (self._collection.metadata or {}).get("embedding_model")
        if self.count() == 0:
            raise IndexMismatchError("The index is empty. Build it with: uv run ai-act index")
        if built_with is None:
            raise IndexMismatchError(
                "The index was built by an older version that didn't record its model. "
                "Rebuild it: uv run ai-act index"
            )
        if built_with != self.embedding_model:
            raise IndexMismatchError(
                f"The index was built with {built_with}, but search uses "
                f"{self.embedding_model}. Rebuild it: uv run ai-act index"
            )


def _to_chunk(chunk_id: str, text: str, meta: dict) -> Chunk:
    """Rebuild a Chunk from what Chroma stores: id, text, and the citation metadata."""
    return Chunk(
        id=chunk_id,
        type=meta["type"],
        ref=meta["ref"],
        title=meta["title"],
        text=text,
        url=meta["url"],
    )


def _unit(vector: list[float]) -> list[float]:
    """The vector scaled to length 1, so the cosine similarity is a plain dot product.

    The real embedder already returns length-1 vectors; this keeps search correct for any
    vector (e.g. in tests). An all-zero vector stays all zeros: similarity 0 with everything.
    """
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length else vector


def _dot(a: list[float], b: list[float]) -> float:
    """Sum of a[i] * b[i]: for two length-1 vectors, their cosine similarity."""
    return sum(x * y for x, y in zip(a, b, strict=True))
