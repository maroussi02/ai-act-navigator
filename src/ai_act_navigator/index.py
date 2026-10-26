"""Build the search index: embed every chunk and save it in the vector store.

Later, search.py asks the store "which chunks are closest in meaning to this question?"
and gets an answer in milliseconds.

Run it:  uv run python -m ai_act_navigator.index
"""

import time
from typing import Protocol

from ai_act_navigator import chunk
from ai_act_navigator.models import Chunk
from ai_act_navigator.store import CHROMA_PATH, VectorStore


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


def build_index(chunks: list[Chunk], embedder: DocumentEmbedder, store: VectorStore) -> None:
    """Embed all chunks and save them in the store, with their citation data."""
    store.upsert(chunks, embedder.embed_documents([text_to_embed(c) for c in chunks]))


def main() -> None:
    """Chunk the law, embed the chunks and store them."""
    from ai_act_navigator.embed import Embedder  # imported here: loading the model is slow

    # Always re-chunk (takes about a second), so the index never uses an outdated
    # chunks.jsonl written by an older version of the chunking rules.
    chunk.main()
    chunks = chunk.load_chunks(chunk.OUTPUT_PATH)

    start = time.perf_counter()
    embedder = Embedder()
    truncated = embedder.count_truncated([text_to_embed(c) for c in chunks])
    store = VectorStore(embedder.model_id)
    build_index(chunks, embedder, store)
    seconds = time.perf_counter() - start

    print(
        f"Indexed {store.count()} chunks into {CHROMA_PATH} "
        f"({truncated} truncated at {embedder.max_tokens} tokens) in {seconds:.0f}s"
    )


# Runs main() only when started as a program, not when imported by tests.
if __name__ == "__main__":
    main()
