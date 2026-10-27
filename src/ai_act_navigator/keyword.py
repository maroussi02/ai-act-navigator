"""Keyword search (BM25): find the chunks that share the question's exact words.

Meaning search (embeddings) is good at "Does a chatbot have to say it's an AI?" but weak at
exact references: "What does Article 4 require?" lands on other articles that merely sound
similar. Keyword search is the opposite, so search.py combines both.

BM25 is the classic keyword-ranking formula, used by search engines like Lucene. For each
question word that appears in a chunk, it adds:
- more if the word is rare in the whole law (IDF): "biometric" says more than "shall";
- more if the word appears often in the chunk, but with diminishing returns (k1);
- a bit less if the chunk is long, so long chunks don't win just by having many words (b).

It runs in memory: building the index for the whole law takes a few hundredths of a second,
so nothing is saved to disk.
"""

import math
import re
from collections import Counter

from ai_act_navigator.index import text_to_embed
from ai_act_navigator.models import Chunk
from ai_act_navigator.refs import MENTION

# The usual textbook values. Not tuned on the golden set, so the score isn't fitted to it.
K1 = 1.5  # how fast repeating a word stops adding to the score
B = 0.75  # how much long chunks are penalised (0 = not at all, 1 = fully)

_WORD = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase words, with each article or annex reference as one word.

    Example: "What does Article 4 require?" -> ["what", "does", "art_4", "require"].
    "Art. 4", "article 4" and "Articles 4" all become "art_4", so a question matches the
    chunk's citation however it is spelled. Without this, "4" alone would match every
    "paragraph 4" in the law.
    """

    def as_one_word(match: re.Match[str]) -> str:
        article, annex = match.groups()
        return f" art_{article} " if article else f" annex_{annex.lower()} "

    return _WORD.findall(MENTION.sub(as_one_word, text).lower())


class KeywordIndex:
    """All chunks, prepared for BM25 search."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        # The same text the embedding index uses: citation + title + text.
        words_per_chunk = [tokenize(text_to_embed(c)) for c in chunks]
        self._counts = [Counter(words) for words in words_per_chunk]
        self._lengths = [len(words) for words in words_per_chunk]
        self._average_length = sum(self._lengths) / len(chunks) if chunks else 0.0
        # In how many chunks each word appears. Rare words get a higher weight (IDF).
        # The "1 +" inside the log keeps every weight above 0, even for a word in every chunk.
        chunks_with_word = Counter(word for words in words_per_chunk for word in set(words))
        n = len(chunks)
        self._idf = {
            word: math.log(1 + (n - df + 0.5) / (df + 0.5)) for word, df in chunks_with_word.items()
        }

    def search(self, question: str, n: int | None = None) -> list[Chunk]:
        """Chunks sharing at least one word with the question, best first (all if n is None).

        Chunks with no word in common are left out, so the list can be shorter than n.
        Ties keep the order the chunks were given in.
        """
        words = [w for w in tokenize(question) if w in self._idf]  # unknown words score 0
        scored = []
        for i, (counts, length) in enumerate(zip(self._counts, self._lengths, strict=True)):
            score = 0.0
            for word in words:
                count = counts[word]
                if count:
                    length_factor = 1 - B + B * length / self._average_length
                    score += self._idf[word] * count * (K1 + 1) / (count + K1 * length_factor)
            if score > 0:
                scored.append((score, i))
        # sorted() is stable: equal scores stay in their original order.
        best = sorted(scored, key=lambda pair: -pair[0])[:n]
        return [self._chunks[i] for _, i in best]
