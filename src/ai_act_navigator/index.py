"""Build the search index: embed every chunk and store it in Chroma.

Chroma is a vector database that runs as a library: it saves everything as normal files
in a folder (data/chroma/). No account, no server. Later (AN-001c) we ask it "which
chunks are closest in meaning to this question?" and it answers in milliseconds.

Run it:  uv run python -m ai_act_navigator.index
"""

import time
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb import Collection
from chromadb.config import Settings

from ai_act_navigator import chunk
from ai_act_navigator.models import Chunk
from ai_act_navigator.paths import DATA_DIR

CHROMA_PATH = DATA_DIR / "chroma"
COLLECTION_NAME = "ai_act"  # a "collection" is like a table in a normal database


class DocumentEmbedder(Protocol):
    """What build_index needs from an embedder: just one method.

    A Protocol describes a shape instead of a specific class. Both the real Embedder and
    the tests' FakeEmbedder fit it, so build_index works with either.
    """

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...


def text_to_embed(c: Chunk) -> str:
    """What we embed for a chunk: citation + title + text.

    "Art. 50(1) Transparency obligations...: Providers shall ensure..." carries more
    meaning than the paragraph text alone, which often doesn't name its topic.
    """
    return f"{c.ref} {c.title}: {c.text}" if c.title else f"{c.ref}: {c.text}"


def open_collection(path: Path = CHROMA_PATH) -> Collection:
    """Open (or create) the collection stored in `path`."""
    client = chromadb.PersistentClient(
        path=str(path),
        # Chroma sends anonymous usage statistics by default. We turn that off (privacy).
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        COLLECTION_NAME,
        # "cosine": closeness = the angle between two vectors, the standard for text.
        configuration={"hnsw": {"space": "cosine"}},
        # We always pass our own embeddings, so Chroma never needs its built-in model.
        embedding_function=None,
    )


def build_index(
    chunks: list[Chunk], embedder: DocumentEmbedder, path: Path = CHROMA_PATH
) -> Collection:
    """Embed all chunks and save them with their citation data. Safe to run again.

    `upsert` = insert, or update if the id already exists. So running this twice
    doesn't create duplicates, it just refreshes the same 821 entries.
    """
    collection = open_collection(path)

    # Remove entries from an older run whose chunk no longer exists (e.g. after the
    # chunking rules changed). Otherwise search could return outdated text.
    current_ids = {c.id for c in chunks}
    stale_ids = [i for i in collection.get(include=[])["ids"] if i not in current_ids]
    if stale_ids:
        collection.delete(ids=stale_ids)

    collection.upsert(
        ids=[c.id for c in chunks],
        embeddings=embedder.embed_documents([text_to_embed(c) for c in chunks]),
        documents=[c.text for c in chunks],  # the original text, returned with search results
        # Metadata travels with each vector, so a search result knows how to cite itself.
        metadatas=[{"ref": c.ref, "type": c.type, "title": c.title, "url": c.url} for c in chunks],
    )
    return collection


def main() -> None:
    """Load the chunks (create them first if missing), embed them and store them."""
    from ai_act_navigator.embed import Embedder  # imported here: loading the model is slow

    if not chunk.OUTPUT_PATH.exists():
        chunk.main()
    chunks = chunk.load_chunks(chunk.OUTPUT_PATH)

    start = time.perf_counter()
    embedder = Embedder()
    truncated = embedder.count_truncated([text_to_embed(c) for c in chunks])
    collection = build_index(chunks, embedder)
    seconds = time.perf_counter() - start

    print(
        f"Indexed {collection.count()} chunks into {CHROMA_PATH} "
        f"({truncated} truncated at {embedder.max_tokens} tokens) in {seconds:.0f}s"
    )


# Runs main() only when started as a program, not when imported by tests.
if __name__ == "__main__":
    main()
