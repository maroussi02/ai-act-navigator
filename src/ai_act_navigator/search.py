"""Find the chunks of the AI Act that best answer a question ("hybrid search").

Three ranked lists of candidates, merged into one:
1. by meaning (embeddings, store.py): finds "Must a chatbot say it's an AI?" in Art. 50(1),
   even though they share few words;
2. by keywords (BM25, keyword.py): finds exact words and references like "Article 14";
3. the chunks of every article the question names ("What does Article 4 require?"), so a
   named article is a candidate even when both other lists rank it low.

Measured on the golden set (15 questions the AI Act answers, 5 chunks each): meaning alone
found the expected article for 9, all three combined for 14.

mode="vector" is v1's search (meaning only), kept so the v1 baseline can be reproduced.
"""

from typing import Literal, Protocol

from ai_act_navigator.keyword import KeywordIndex
from ai_act_navigator.models import Chunk, Hit
from ai_act_navigator.refs import article_of, mentioned_articles
from ai_act_navigator.store import VectorStore

SearchMode = Literal["hybrid", "vector"]
DEFAULT_MODE: SearchMode = "hybrid"
DEFAULT_K = 5  # how many chunks to send to Claude: enough context, still cheap and focused
CANDIDATES = 50  # how far down each list we look before merging
# The constant from the paper that introduced Reciprocal Rank Fusion (Cormack, Clarke and
# Büttcher, 2009). It softens the gap between rank 1 and rank 2, so one list can't dominate.
RRF_K = 60


class QueryEmbedder(Protocol):
    """What search needs from an embedder. The real Embedder fits; tests use a fake."""

    def embed_query(self, text: str) -> list[float]: ...


def fuse(ranked_lists: list[list[Chunk]], k: int = RRF_K) -> list[Hit]:
    """Merge ranked lists into one with Reciprocal Rank Fusion, best first.

    Each chunk scores 1 / (k + rank) in every list it appears in, and the scores add up.
    Only ranks count, not each method's own scores, which aren't comparable (a cosine
    similarity of 0.8 and a BM25 score of 12 can't be added). A chunk that several lists
    rank fairly high beats one that a single list ranks first.
    Example, k=60: 2nd in two lists = 2/62 = 0.032 > 1st in one list = 1/61 = 0.016.
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for ranking in ranked_lists:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1 / (k + rank)
            chunks[chunk.id] = chunk
    # sorted() is stable: equal scores keep the order they were first seen in.
    best_first = sorted(scores, key=lambda chunk_id: -scores[chunk_id])
    return [Hit(chunk=chunks[chunk_id], score=scores[chunk_id]) for chunk_id in best_first]


class Searcher:
    """Searches the stored chunks. Build it once per run: it prepares the keyword index."""

    def __init__(self, embedder: QueryEmbedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store
        self._keyword = KeywordIndex(store.all_chunks())  # the same chunks as the vector index

    def search(
        self, question: str, k: int = DEFAULT_K, mode: SearchMode = DEFAULT_MODE
    ) -> list[Hit]:
        """The k chunks that best answer the question, best first.

        Example: "What does Article 4 require?" -> [Art. 4, ...]
        """
        query = self._embedder.embed_query(question)
        if mode == "vector":
            return self._store.search(query, k)
        if mode != "hybrid":
            raise ValueError(f"Unknown search mode: {mode!r}")

        depth = min(CANDIDATES, self._store.count())
        by_meaning = [hit.chunk for hit in self._store.search(query, depth)]
        by_keyword = self._keyword.search(question)  # every chunk sharing a word, best first
        # Every chunk of a named article contains its reference ("art_4") in the keyword
        # index, so filtering the keyword list gives them all, the most relevant first.
        named = mentioned_articles(question)
        of_named_articles = [c for c in by_keyword if article_of(c.ref) in named]
        return fuse([by_meaning, by_keyword[:CANDIDATES], of_named_articles])[:k]
