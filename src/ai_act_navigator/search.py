"""Find the chunks of the AI Act that best answer a question.

Step 1, hybrid search: three ranked lists of candidates, merged into one:
1. by meaning (embeddings, store.py): finds "Must a chatbot say it's an AI?" in Art. 50(1),
   even though they share few words;
2. by keywords (BM25, keyword.py): finds exact words and references like "Article 14";
3. the chunks of every article the question names ("What does Article 4 require?"), so a
   named article is a candidate even when both other lists rank it low.

Step 2, reranking (rerank.py): a slower, more precise model reorders the top 30 candidates,
so the paragraph that answers the question comes first. A named article keeps its best
chunk among the chunks sent (keep_named_articles).

Measured on the golden set (15 questions the AI Act answers, 5 chunks each): meaning alone
found the expected article for 9, hybrid for 14. Reranking keeps 14 and brings back 3 of
the 4 key paragraphs hybrid had pushed out of the top 5.

Modes, so every step's score can be reproduced: "vector" (v1), "hybrid" (AN-003b),
"rerank" (hybrid + reranker, the default).
"""

from typing import Literal, Protocol

from ai_act_navigator.keyword import KeywordIndex
from ai_act_navigator.models import Chunk, Hit
from ai_act_navigator.refs import article_of, mentioned_articles
from ai_act_navigator.store import VectorStore

SearchMode = Literal["rerank", "hybrid", "vector"]
DEFAULT_MODE: SearchMode = "rerank"
DEFAULT_K = 5  # how many chunks to send to Claude: enough context, still cheap and focused
CANDIDATES = 50  # how far down each list we look before merging
# How many hybrid candidates the reranker rescores. On the golden set, the key paragraphs
# hybrid missed were at ranks 6 to 17; 30 leaves room. Reranking time grows with this number.
RERANK_CANDIDATES = 30
# The constant from the paper that introduced Reciprocal Rank Fusion (Cormack, Clarke and
# Büttcher, 2009). It softens the gap between rank 1 and rank 2, so one list can't dominate.
RRF_K = 60


class QueryEmbedder(Protocol):
    """What search needs from an embedder. The real Embedder fits; tests use a fake."""

    def embed_query(self, text: str) -> list[float]: ...


class HitReranker(Protocol):
    """What search needs from a reranker. The real Reranker fits; tests use a fake."""

    def rerank(self, question: str, hits: list[Hit], k: int | None = None) -> list[Hit]: ...


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


def keep_named_articles(ranked: list[Hit], named: set[str], k: int) -> list[Hit]:
    """The top k hits, but every named article that is among the hits keeps one place.

    Why: a reranker judges wording, and doesn't know that "Article 4" in a question points
    to the chunk cited as "Art. 4". On the golden set it pushed Art. 4 and Art. 113 out of
    the top 5 for "What does Article 4 require?" and "What does Article 113 say?".
    A missing article's best-ranked chunk replaces the lowest-ranked chunk that belongs to
    no named article. Named articles that aren't among the hits at all are skipped.
    Example, k=3, named {"Art. 4"}: [Art. 102, Art. 91, Art. 107, Art. 4, ...]
                                 -> [Art. 102, Art. 91, Art. 4]
    """
    top = ranked[:k]
    for article in sorted(named):  # sorted, so the result is the same on every run
        if any(article_of(hit.chunk.ref) == article for hit in top):
            continue
        best = next((hit for hit in ranked if article_of(hit.chunk.ref) == article), None)
        if best is None:
            continue
        if len(top) < k:
            top.append(best)
            continue
        replaceable = [i for i, hit in enumerate(top) if article_of(hit.chunk.ref) not in named]
        if replaceable:  # else every place already holds a named article
            top[replaceable[-1]] = best
    # Replacing can break the order (a named chunk lands where an unnamed one was), so put
    # the result back in the reranker's order: best first.
    position = {hit.chunk.id: i for i, hit in enumerate(ranked)}
    return sorted(top, key=lambda hit: position[hit.chunk.id])


class Searcher:
    """Searches the stored chunks. Build it once per run: it prepares the keyword index.

    `reranker` is only needed for mode "rerank"; loading it takes a moment, so the CLI
    only loads it when that mode is used.
    """

    def __init__(
        self, embedder: QueryEmbedder, store: VectorStore, reranker: HitReranker | None = None
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._keyword = KeywordIndex(store.all_chunks())  # the same chunks as the vector index

    def search(
        self, question: str, k: int = DEFAULT_K, mode: SearchMode = DEFAULT_MODE
    ) -> list[Hit]:
        """The k chunks that best answer the question, best first.

        Example: "What does Article 4 require?" -> [Art. 4, ...]
        """
        if mode not in ("rerank", "hybrid", "vector"):
            raise ValueError(f"Unknown search mode: {mode!r}")
        if mode == "rerank" and self._reranker is None:
            raise ValueError('Search mode "rerank" needs a reranker; none was given.')

        query = self._embedder.embed_query(question)
        if mode == "vector":
            return self._store.search(query, k)

        depth = min(CANDIDATES, self._store.count())
        by_meaning = [hit.chunk for hit in self._store.search(query, depth)]
        by_keyword = self._keyword.search(question)  # every chunk sharing a word, best first
        # Every chunk of a named article contains its reference ("art_4") in the keyword
        # index, so filtering the keyword list gives them all, the most relevant first.
        named = mentioned_articles(question)
        of_named_articles = [c for c in by_keyword if article_of(c.ref) in named]
        fused = fuse([by_meaning, by_keyword[:CANDIDATES], of_named_articles])
        if mode == "hybrid":
            return fused[:k]

        reranked = self._reranker.rerank(question, fused[:RERANK_CANDIDATES])
        return keep_named_articles(reranked, named, k)
