"""Answer a question with Claude, citing the exact parts of the AI Act it used.

How citations work: each retrieved chunk is sent as a "search result" block (Claude API
feature built for RAG). With citations enabled, Claude's answer comes back in pieces,
and every piece that uses a chunk carries a citation pointing to that chunk's position in
our list plus the exact quoted text. So a citation can only ever point to a chunk we sent;
Claude can't invent "Art. 99(12)".

Run it:  uv run python -m ai_act_navigator.answer "Does a chatbot have to tell users it is an AI?"
"""

import os
import sys
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic

from ai_act_navigator.models import Hit
from ai_act_navigator.prompts import SYSTEM_PROMPT

# One setting to switch models. AN-002 compares Haiku, Sonnet and Opus on the golden set.
MODEL = "claude-sonnet-5-5"

# USD per 1 million tokens (input, output). Anthropic list prices, checked 2026-10-26.
# Used only to show what a question costs; update when prices change.
PRICES_PER_MILLION = {
    "claude-haiku-5-5": (0.10, 0.50),  # for prompts up to 100K tokens
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}


class AnswerError(Exception):
    """Claude's response could not be turned into an answer (e.g. it was cut off)."""


@dataclass(frozen=True)
class Citation:
    """A source used in the answer, shown as [number] in the text."""

    number: int
    ref: str  # e.g. "Art. 50(1)"
    url: str
    cited_text: str  # the exact text Claude relied on, quoted from our chunk


@dataclass(frozen=True)
class Answer:
    text: str  # the answer, with [1], [2], ... markers where sources were used
    citations: list[Citation]
    model: str  # the model that actually answered (can differ if a fallback ran)
    input_tokens: int
    output_tokens: int
    # At list prices, priced as `model`. Approximate if a fallback ran: the declined first
    # attempt is billed separately (see usage.iterations). None for an unknown model.
    cost_usd: float | None
    refused: bool = False  # True if Claude declined to answer for safety reasons


def build_sources(hits: list[Hit]) -> list[dict[str, Any]]:
    """One "search_result" content block per retrieved chunk, in ranking order.

    The order matters: Claude's citations point back by position ("search result 0").
    """
    return [
        {
            "type": "search_result",
            "source": hit.chunk.url,
            "title": f"{hit.chunk.ref} — {hit.chunk.title}" if hit.chunk.title else hit.chunk.ref,
            "content": [{"type": "text", "text": hit.chunk.text}],
            "citations": {"enabled": True},  # must be on for all results or none
        }
        for hit in hits
    ]


def answer(question: str, hits: list[Hit], client: Anthropic, model: str = MODEL) -> Answer:
    """Ask Claude the question, giving it only the retrieved chunks as sources."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [*build_sources(hits), {"type": "text", "text": question}],
            }
        ],
        # How much the model thinks before answering. "medium" is a balanced start;
        # AN-002 will measure whether "low" is good enough (cheaper and faster).
        output_config={"effort": "medium"},
        # If a safety check declines the request, Anthropic retries on a suitable model
        # inside the same call instead of failing. Recommended default for this model.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    return to_answer(response, hits)


def to_answer(response: Any, hits: list[Hit]) -> Answer:
    """Turn Claude's response into an Answer with numbered citations.

    Kept separate from answer() so tests can feed it a recorded response, no API call.
    """
    # Always check why Claude stopped before reading the content.
    if response.stop_reason == "max_tokens":
        raise AnswerError("The answer was cut off (max_tokens reached).")
    refused = response.stop_reason == "refusal"

    numbers: dict[int, int] = {}  # search result position -> [number] shown to the user
    citations: list[Citation] = []
    parts: list[str] = []
    for block in response.content:
        if block.type != "text":
            continue  # skip thinking and other non-answer blocks
        parts.append(block.text)
        for cite in block.citations or []:
            if cite.type != "search_result_location":
                continue
            if cite.search_result_index not in numbers:
                numbers[cite.search_result_index] = len(numbers) + 1
                hit = hits[cite.search_result_index]
                citations.append(
                    Citation(
                        number=numbers[cite.search_result_index],
                        ref=hit.chunk.ref,
                        url=hit.chunk.url,  # our own link, never a URL taken from model text
                        cited_text=cite.cited_text,
                    )
                )
            parts.append(f" [{numbers[cite.search_result_index]}]")

    text = "".join(parts).strip()
    if refused and not text:
        text = "Claude declined to answer this question."

    usage = response.usage
    input_tokens = (
        usage.input_tokens
        + (usage.cache_creation_input_tokens or 0)
        + (usage.cache_read_input_tokens or 0)
    )
    return Answer(
        text=text,
        citations=citations,
        model=response.model,
        input_tokens=input_tokens,
        output_tokens=usage.output_tokens,
        cost_usd=cost_usd(response.model, input_tokens, usage.output_tokens),
        refused=refused,
    )


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """What a request cost at list prices, or None for a model we have no price for."""
    if model not in PRICES_PER_MILLION:
        return None
    input_price, output_price = PRICES_PER_MILLION[model]
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def main() -> None:
    """Answer the question given on the command line and print the sources and cost."""
    from dotenv import load_dotenv

    from ai_act_navigator.embed import Embedder  # imported here: loading the model is slow
    from ai_act_navigator.paths import PROJECT_ROOT
    from ai_act_navigator.search import search
    from ai_act_navigator.store import VectorStore

    if len(sys.argv) < 2:
        sys.exit('Usage: uv run python -m ai_act_navigator.answer "your question"')
    question = " ".join(sys.argv[1:])

    # Reads ANTHROPIC_API_KEY from the git-ignored .env file into the environment, where
    # the Anthropic client finds it. The key is never printed or logged.
    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add your key.")

    embedder = Embedder()
    hits = search(question, embedder, VectorStore(embedder.model_id))
    result = answer(question, hits, Anthropic())

    print(result.text)
    print("\nSources:" if result.citations else "\nSources: none")
    for c in result.citations:
        print(f"  [{c.number}] {c.ref}  {c.url}")
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "unknown"
    print(
        f"\n{result.model} · {result.input_tokens} tokens in, "
        f"{result.output_tokens} out · cost {cost}"
    )


# Runs main() only when started as a program, not when imported by tests.
if __name__ == "__main__":
    main()
