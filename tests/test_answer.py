"""UNIT TESTS for answer.py. No API calls, no cost.

A FakeClient stands in for the Anthropic client. Its responses are built with the SDK's
own BetaMessage type, so they have exactly the shape the real API returns.
"""

from typing import Any

import pytest
from anthropic.types.beta import BetaMessage

from ai_act_navigator.answer import DECLINED, AnswerError, build_sources, cost_usd, to_answer
from ai_act_navigator.answer import answer as ask
from ai_act_navigator.models import Chunk, Hit

URL_50 = "https://eur-lex.example/#art_50"
URL_132 = "https://eur-lex.example/#rct_132"
HITS = [
    Hit(
        Chunk("art-50-1", "article", "Art. 50(1)", "Transparency", "1. Providers shall…", URL_50),
        0.8,
    ),
    Hit(Chunk("rct-132-s1", "recital", "Recital 132", "", "(132) Certain AI…", URL_132), 0.7),
    # Second part of the same long recital: same citation, different chunk.
    Hit(Chunk("rct-132-s2", "recital", "Recital 132", "", "…more of it.", URL_132), 0.6),
]


def citation(index: int, text: str = "quoted") -> dict[str, Any]:
    """A citation in the exact shape the API returns, pointing at HITS[index]."""
    return {
        "type": "search_result_location",
        "cited_text": text,
        "search_result_index": index,
        "start_block_index": 0,
        "end_block_index": 1,
        "source": "ignored",
        "title": "ignored",
    }


def message(content: list[dict], stop_reason: str = "end_turn", **extra: Any) -> BetaMessage:
    """A response from Claude, validated by the SDK's own type."""
    return BetaMessage.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5-5",
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "content": content,
            "usage": {"input_tokens": 3000, "output_tokens": 500},
            **extra,
        }
    )


class FakeClient:
    """Records the request and returns a fixed response, like client.beta.messages.create."""

    def __init__(self, response: BetaMessage) -> None:
        self.response = response
        self.request: dict[str, Any] = {}
        self.beta = self
        self.messages = self

    def create(self, **kwargs: Any) -> BetaMessage:
        self.request = kwargs
        return self.response


def test_each_hit_becomes_a_citable_search_result():
    sources = build_sources(HITS)
    assert sources[0] == {
        "type": "search_result",
        "source": URL_50,
        "title": "Art. 50(1) — Transparency",
        "content": [{"type": "text", "text": "1. Providers shall…"}],
        "citations": {"enabled": True},
    }
    assert sources[1]["title"] == "Recital 132"  # recitals have no title


def test_request_sends_sources_then_question_with_the_right_settings():
    client = FakeClient(message([{"type": "text", "text": "Yes."}]))
    ask("Must a chatbot say it's AI?", HITS, client)

    content = client.request["messages"][0]["content"]
    assert [block["type"] for block in content] == ["search_result"] * 3 + ["text"]
    assert content[-1]["text"] == "Must a chatbot say it's AI?"
    assert client.request["fallbacks"] == "default"
    assert client.request["betas"] == ["server-side-fallback-2026-07-01"]
    assert client.request["output_config"] == {"effort": "medium"}


def test_effort_can_be_changed_per_call():
    client = FakeClient(message([{"type": "text", "text": "Yes."}]))
    ask("q", HITS, client, effort="low")
    assert client.request["output_config"] == {"effort": "low"}


def test_citations_are_numbered_and_point_to_the_right_chunk():
    response = message(
        [
            {"type": "text", "text": "Yes, users must be told.", "citations": [citation(0, "c0")]},
            {"type": "text", "text": " Vulnerable groups matter.", "citations": [citation(1)]},
            {"type": "text", "text": " Unless obvious.", "citations": [citation(0, "c0b")]},
        ]
    )
    result = to_answer(response, HITS)

    assert result.text == (
        "Yes, users must be told. [1] Vulnerable groups matter. [2] Unless obvious. [1]"
    )
    assert [(c.number, c.ref, c.url) for c in result.citations] == [
        (1, "Art. 50(1)", URL_50),
        (2, "Recital 132", URL_132),
    ]
    assert result.citations[0].quotes == ("c0", "c0b")  # every quote is kept


def test_same_source_twice_in_one_piece_gets_one_marker():
    response = message(
        [{"type": "text", "text": "Yes.", "citations": [citation(0, "a"), citation(0, "b")]}]
    )
    result = to_answer(response, HITS)
    assert result.text == "Yes. [1]"
    assert result.citations[0].quotes == ("a", "b")


def test_parts_of_the_same_recital_share_one_source():
    response = message(
        [{"type": "text", "text": "Explained.", "citations": [citation(1, "x"), citation(2, "y")]}]
    )
    result = to_answer(response, HITS)
    assert result.text == "Explained. [1]"
    assert [(c.ref, c.quotes) for c in result.citations] == [("Recital 132", ("x", "y"))]


def test_answer_without_sources_has_no_citations():
    result = to_answer(message([{"type": "text", "text": "The excerpts don't answer this."}]), HITS)
    assert result.citations == []
    assert result.refused is False


def test_thinking_blocks_are_not_part_of_the_answer():
    response = message(
        [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": "Yes."}]
    )
    assert to_answer(response, HITS).text == "Yes."


def test_refusal_is_reported_not_hidden():
    result = to_answer(message([], stop_reason="refusal"), HITS)
    assert result.refused is True
    assert result.text == DECLINED


def test_partial_answer_before_a_refusal_is_thrown_away():
    # A safety check can stop Claude mid-answer. The unfinished text must not be shown.
    response = message(
        [{"type": "text", "text": "Here is how to", "citations": [citation(0)]}],
        stop_reason="refusal",
        stop_details={"type": "refusal", "category": "bio", "explanation": None},
    )
    result = to_answer(response, HITS)
    assert result.text == f"{DECLINED} (category: bio)"
    assert result.citations == []


def test_cut_off_answer_raises():
    with pytest.raises(AnswerError):
        to_answer(
            message([{"type": "text", "text": "Half an ans"}], stop_reason="max_tokens"), HITS
        )


def test_cost_is_computed_from_tokens():
    result = to_answer(message([{"type": "text", "text": "Yes."}]), HITS)
    assert result.cost_usd == pytest.approx((3000 * 2.00 + 500 * 10.00) / 1_000_000)
    assert cost_usd("some-unknown-model", 1000, 1000) is None


def test_fallback_model_has_a_price():
    assert cost_usd("claude-sonnet-5", 1_000_000, 0) == pytest.approx(2.0)


def test_cached_input_is_priced_differently():
    # Reading 1M tokens from the cache costs 10% of the input price; writing costs 125%.
    assert cost_usd("claude-sonnet-5-5", 0, 0, cache_read_tokens=1_000_000) == pytest.approx(0.2)
    assert cost_usd("claude-sonnet-5-5", 0, 0, cache_write_tokens=1_000_000) == pytest.approx(2.5)
