"""Score the assistant on the golden set (AN-002): 20 questions, one score, one results file.

    uv run ai-act eval          # 20 Claude calls, about $0.20

No LLM judge (that comes in AN-004): every check is plain code over the answer's citations,
at article level ("Art. 50(1)" counts as "Art. 50").
- exact_ref and plain questions pass if the answer cites one of the expected articles.
- out_of_scope questions pass if the answer cites nothing, i.e. it said the excerpts don't
  answer the question. (Answer.refused is something else: a safety decline. It never passes.)

Two extra checks explain *why* a question failed, without changing the score:
- retrieved: was an expected article among the chunks search sent to Claude? If not, it's
  a search miss (AN-003's job), not an answer problem.
- mentioned but not cited: articles the answer text names without a [number].
  "Retrieved but uncited" = the article was sent but not used as a citation.
  "Outside knowledge" = the article wasn't sent at all, so the model knew it from elsewhere.
"""

import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_act_navigator.answer import Answer, AnswerError
from ai_act_navigator.golden import MIN_PER_TYPE, GoldenQuestion, article_of
from ai_act_navigator.models import Hit
from ai_act_navigator.paths import PROJECT_ROOT

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

# The question in, the chunks search found and Claude's answer out. The CLI passes the real
# pipeline; tests pass a fake.
Ask = Callable[[str], tuple[list[Hit], Answer]]

# "Art. 50", "Article 6(1)", "Annex III" in the answer text.
_MENTION = re.compile(r"\b(?:Art\.|Articles?)\s+(\d+)|\bAnnex\s+([IVXL]+)\b")
# What follows a mention of *another* law, e.g. "Article 4(3) of Directive (EU) 2019/790".
_OTHER_LAW = re.compile(r"(?:\(\w+\))*(?:,? points? \(\w+\))?\s+of\s+(?:Directive|Regulation)")


@dataclass(frozen=True)
class QuestionResult:
    id: str
    type: str
    question: str
    passed: bool
    expected: tuple[str, ...]  # article-level, from the golden set
    cited: tuple[str, ...]  # exact refs the answer cited, e.g. ("Art. 50(1)", "Recital 132")
    retrieved: bool | None  # expected article among the chunks sent; None for out_of_scope
    sent: tuple[str, ...]  # refs of all chunks sent to Claude, best first
    retrieved_uncited: tuple[str, ...]  # named in the text, sent, but not cited
    outside_knowledge: tuple[str, ...]  # named in the text, never sent
    refused: bool  # safety decline (Answer.refused)
    answer: str
    cost_usd: float | None
    error: str = ""  # set if no answer came back (e.g. cut off); counts as a fail


def mentioned_articles(text: str) -> set[str]:
    """Articles and annexes of the AI Act that the text names, e.g. {"Art. 50", "Annex III"}.

    Plain pattern matching, so it can miss unusual wording ("Articles 99 and 100" only
    finds Art. 99). Mentions of other laws ("Article 4(3) of Directive ...") are skipped.
    """
    found = set()
    for match in _MENTION.finditer(text):
        if _OTHER_LAW.match(text, match.end()):
            continue
        article, annex = match.groups()
        found.add(f"Art. {article}" if article else f"Annex {annex}")
    return found


def score(question: GoldenQuestion, hits: list[Hit], answer: Answer) -> QuestionResult:
    """Check one answer against its golden question. No API calls."""
    cited = {article_of(c.ref) for c in answer.citations}
    sent = {article_of(hit.chunk.ref) for hit in hits}
    expected = set(question.sources)
    uncited = mentioned_articles(answer.text) - cited

    if question.type == "out_of_scope":
        passed = not answer.citations and not answer.refused
        retrieved = None
    else:
        passed = bool(cited & expected)
        retrieved = bool(sent & expected)

    return QuestionResult(
        id=question.id,
        type=question.type,
        question=question.question,
        passed=passed,
        expected=question.sources,
        cited=tuple(c.ref for c in answer.citations),
        retrieved=retrieved,
        sent=tuple(hit.chunk.ref for hit in hits),
        retrieved_uncited=tuple(sorted(uncited & sent)),
        outside_knowledge=tuple(sorted(uncited - sent)),
        refused=answer.refused,
        answer=answer.text,
        cost_usd=answer.cost_usd,
    )


def evaluate_one(question: GoldenQuestion, ask: Ask) -> QuestionResult:
    """Ask one golden question and score the answer. A cut-off answer is a fail, not a crash,
    so one bad question doesn't throw away the 19 already paid for."""
    try:
        hits, answer = ask(question.question)
    except AnswerError as error:
        return QuestionResult(
            id=question.id,
            type=question.type,
            question=question.question,
            passed=False,
            expected=question.sources,
            cited=(),
            retrieved=None,
            sent=(),
            retrieved_uncited=(),
            outside_knowledge=(),
            refused=False,
            answer="",
            cost_usd=None,
            error=str(error),
        )
    return score(question, hits, answer)


def summarize(results: list[QuestionResult]) -> dict[str, Any]:
    """The numbers for the README: overall score, score per type, and why things failed."""
    in_scope = [r for r in results if r.type != "out_of_scope"]
    costs = [r.cost_usd for r in results if r.cost_usd is not None]
    return {
        "passed": sum(r.passed for r in results),
        "total": len(results),
        "by_type": {
            t: {
                "passed": sum(r.passed for r in results if r.type == t),
                "total": sum(r.type == t for r in results),
            }
            for t in MIN_PER_TYPE
        },
        "retrieved": {"found": sum(bool(r.retrieved) for r in in_scope), "total": len(in_scope)},
        "wrongly_declined": sum(not r.cited and not r.error for r in in_scope),
        "with_retrieved_uncited": sum(bool(r.retrieved_uncited) for r in results),
        "with_outside_knowledge": sum(bool(r.outside_knowledge) for r in results),
        "refused": sum(r.refused for r in results),
        "errors": sum(bool(r.error) for r in results),
        "cost_usd": round(sum(costs), 4),  # approximate if a model has no known price
    }


def save_results(
    results: list[QuestionResult], settings: dict[str, Any], path: Path
) -> dict[str, Any]:
    """Write settings, summary and every answer to a JSON file. Returns the summary."""
    summary = summarize(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings, "summary": summary, "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
