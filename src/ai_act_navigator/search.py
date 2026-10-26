"""Find the chunks of the AI Act that are closest in meaning to a question."""

from typing import Protocol

from ai_act_navigator.models import Hit
from ai_act_navigator.store import VectorStore

DEFAULT_K = 5  # how many chunks to send to Claude: enough context, still cheap and focused


class QueryEmbedder(Protocol):
    """What search needs from an embedder. The real Embedder fits; tests use a fake."""

    def embed_query(self, text: str) -> list[float]: ...


def search(
    question: str, embedder: QueryEmbedder, store: VectorStore, k: int = DEFAULT_K
) -> list[Hit]:
    """The k chunks closest in meaning to the question, best first.

    Example: "Does a chatbot have to tell users it is an AI?" -> [Art. 50(1), Art. 50(3), ...]
    """
    return store.search(embedder.embed_query(question), k)
