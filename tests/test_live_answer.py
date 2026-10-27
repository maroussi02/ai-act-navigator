"""PAID SMOKE TEST: one real question through the full CLI pipeline (about $0.01).

Never runs in CI. Run it by hand:  uv run pytest --run-paid -m paid
Needs ANTHROPIC_API_KEY in .env and a built index (uv run ai-act index).
"""

import pytest

from ai_act_navigator.cli import ask_question, load_api_key

pytestmark = pytest.mark.paid


def test_chatbot_question_cites_article_50_1():
    load_api_key()
    result = ask_question(
        "Does a chatbot have to tell users it is an AI?",
        k=5,
        model="claude-sonnet-5-5",
        effort="medium",
    )
    assert "Art. 50(1)" in [c.ref for c in result.citations]
    assert not result.refused
