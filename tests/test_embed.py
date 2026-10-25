"""INTEGRATION TESTS for embed.py with the real model.

The first run downloads the model (about 130 MB) into ~/.cache/huggingface; CI caches it.
Skip these locally with:  uv run pytest -m "not integration"
"""

import pytest

from ai_act_navigator.embed import Embedder

pytestmark = pytest.mark.integration

ART_50_1 = (
    "Art. 50(1) Transparency obligations for providers and deployers of certain AI systems: "
    "Providers shall ensure that AI systems intended to interact directly with natural persons "
    "are designed and developed in such a way that the natural persons concerned are informed "
    "that they are interacting with an AI system."
)
UNRELATED = "The committee shall adopt its rules of procedure by a simple majority of its members."


@pytest.fixture(scope="module")
def embedder() -> Embedder:
    return Embedder()


def similarity(a: list[float], b: list[float]) -> float:
    """Vectors are normalized (length 1), so the dot product is the cosine similarity."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_question_is_closer_to_the_matching_article(embedder):
    question = embedder.embed_query("Does a chatbot have to tell users it is an AI?")
    article, unrelated = embedder.embed_documents([ART_50_1, UNRELATED])
    assert similarity(question, article) > similarity(question, unrelated)


def test_long_texts_are_counted_as_truncated(embedder):
    assert embedder.count_truncated(["short text", "word " * 2000]) == 1
