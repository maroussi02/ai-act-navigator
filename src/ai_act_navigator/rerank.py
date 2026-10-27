"""Rerank search candidates: put the chunk that best answers the question first.

Search (search.py) is fast but rough: it compares the question with each chunk separately.
A reranker ("cross-encoder") reads the question and one chunk *together* and scores how well
that chunk answers it. That's much slower, so it only rescores the top candidates search
found (30), not the whole law.

Why it's needed: combined search found the right *article* for 14 of 15 golden questions,
but sometimes not its key *paragraph*. "What obligations do providers of general-purpose AI
models have?" got Art. 53(2) to 53(4) but not Art. 53(1), which lists the duties, and Claude
declined. The reranker puts Art. 53(1) back in the top 5. It doesn't fix every case: for
"Does a chatbot have to tell users it is an AI?" it still prefers Art. 50(4) (deepfakes)
over Art. 50(1) (chatbots).

The model runs locally, like the embedding model: `ai-act index` downloads it once (about
90 MB) and `ai-act ask` loads it offline, so no text leaves your machine.
"""

from typing import Protocol

from ai_act_navigator.embed import load_model
from ai_act_navigator.index import text_to_embed
from ai_act_navigator.models import Hit

# Small and fast: about 90 MB, and 0.14 s to score 30 chunks (measured on an Apple M5,
# where the library uses the GPU; on a CPU-only machine expect it to be slower). Apache-2.0.
# Chosen over BAAI/bge-reranker-base (1.1 GB, 0.87 s on the same machine): on the golden set
# both brought back 3 of the 4 key paragraphs that combined search had pushed out of the
# top 5, so the smaller one wins.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"

# Exact version (git commit) on Hugging Face, for the same reasons as the embedding model's
# MODEL_REVISION in embed.py: reproducible, and a later change to the repo can't silently
# swap what we run. Update it on purpose, never by accident.
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


class PairScorer(Protocol):
    """What the reranker needs from the model: a score for each (question, text) pair.

    The real CrossEncoder fits; tests use a fake, so they need no download.
    """

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class Reranker:
    """Wraps the reranking model so the rest of the code doesn't depend on the library."""

    def __init__(
        self,
        allow_download: bool = True,
        model: PairScorer | None = None,  # tests pass a fake; normally the real one loads
    ) -> None:
        self.model_id = f"{RERANK_MODEL}@{RERANK_REVISION}"  # recorded in eval results
        self._model = model or load_model(
            RERANK_MODEL, RERANK_REVISION, allow_download, kind="reranker"
        )

    def rerank(self, question: str, hits: list[Hit], k: int | None = None) -> list[Hit]:
        """The hits reordered so the best answer comes first; only the top k if k is given.

        Each chunk is read as citation + title + text, the same text search indexes, so
        the model also sees e.g. "Art. 50(1) Transparency obligations...".
        Hit.score becomes the reranker's score: higher = better, only useful for ordering.
        """
        if not hits:
            return []
        scores = self._model.predict([(question, text_to_embed(hit.chunk)) for hit in hits])
        rescored = [
            Hit(chunk=hit.chunk, score=float(s)) for hit, s in zip(hits, scores, strict=True)
        ]
        # sorted() is stable: equal scores keep search's order.
        return sorted(rescored, key=lambda hit: -hit.score)[:k]
