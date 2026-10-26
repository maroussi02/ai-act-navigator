"""The vector store: saves chunk embeddings and finds the closest ones to a question.

It wraps Chroma, a vector database that runs as a library and saves everything as normal
files in a folder (data/chroma/): no account, no server. Only this file knows about Chroma.
The rest of the code talks to VectorStore, so swapping Chroma for another database later
(or for LangChain's vector store wrapper) means changing this one file.
"""

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
            configuration={"hnsw": {"space": "cosine"}},
            # We always pass our own embeddings, so Chroma never needs its built-in model.
            embedding_function=None,
        )

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

    def search(self, query_embedding: list[float], k: int = 5) -> list[Hit]:
        """The k chunks closest in meaning to the query, best first."""
        built_with = (self._collection.metadata or {}).get("embedding_model")
        if self.count() == 0:
            raise IndexMismatchError(
                "The index is empty. Build it with: uv run python -m ai_act_navigator.index"
            )
        if built_with is None:
            raise IndexMismatchError(
                "The index was built by an older version that didn't record its model. "
                "Rebuild it: uv run python -m ai_act_navigator.index"
            )
        if built_with != self.embedding_model:
            raise IndexMismatchError(
                f"The index was built with {built_with}, but search uses "
                f"{self.embedding_model}. Rebuild it: uv run python -m ai_act_navigator.index"
            )

        result = self._collection.query(query_embeddings=[query_embedding], n_results=k)
        # Chroma answers for a list of queries; we sent one, so we take element [0] of each.
        rows = zip(
            result["ids"][0],
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            strict=True,
        )
        return [
            Hit(
                chunk=Chunk(
                    id=chunk_id,
                    type=meta["type"],
                    ref=meta["ref"],
                    title=meta["title"],
                    text=text,
                    url=meta["url"],
                ),
                score=1.0 - distance,  # cosine distance -> similarity (1.0 = same meaning)
            )
            for chunk_id, text, meta, distance in rows
        ]
