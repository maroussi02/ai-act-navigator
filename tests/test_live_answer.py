"""PAID SMOKE TEST: one real question to Claude (about $0.01). Never runs in CI.

Run it by hand:  uv run pytest -m paid
Needs ANTHROPIC_API_KEY in .env and a built index (uv run python -m ai_act_navigator.index).
"""

import pytest
from anthropic import Anthropic
from dotenv import load_dotenv

from ai_act_navigator.answer import answer
from ai_act_navigator.embed import Embedder
from ai_act_navigator.paths import PROJECT_ROOT
from ai_act_navigator.search import search
from ai_act_navigator.store import VectorStore

pytestmark = pytest.mark.paid


def test_chatbot_question_cites_article_50_1():
    load_dotenv(PROJECT_ROOT / ".env")
    embedder = Embedder()
    question = "Does a chatbot have to tell users it is an AI?"

    result = answer(
        question, search(question, embedder, VectorStore(embedder.model_id)), Anthropic()
    )

    assert "Art. 50(1)" in [c.ref for c in result.citations]
    assert not result.refused
