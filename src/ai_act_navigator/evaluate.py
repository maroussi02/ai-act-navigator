"""Score the assistant on the golden set (AN-002): 20 questions, one score, one results file.

    uv run ai-act eval          # 20 Claude calls, about $0.20

No LLM judge (that comes in AN-004): every check is plain code over the answer, at article
level ("Art. 50(1)" counts as "Art. 50").
- exact_ref and plain questions pass if the answer cites one of the expected articles and
  doesn't decline.
- out_of_scope questions pass if the answer declines, i.e. says the excerpts don't answer
  the question. Citing related context while declining is fine (e.g. "The excerpts don't
  answer this. They cover only AI Act fines [1]").
  (Answer.refused is something else: a safety decline. It never passes.)

A decline is recognised by its opening paragraph, which the v1 prompt makes consistent:
"The excerpts don't answer this", "The excerpts don't include Article 4, so I can't say...".

Extra checks explain *why* a question failed, without changing the score:
- retrieved: was an expected article among the chunks search sent to Claude? If not, it's
  a search miss (AN-003's job), not an answer problem.
- wrongly declined: the answer declined although search did send an expected article.
- articles the answer names without a [number], minus the ones the question names:
  "Retrieved but uncited" = the article was sent but not used as a citation.
  "Outside knowledge" = the article is neither sent nor named in any sent chunk's text,
  so the model knew it from elsewhere. Limit: a claim about what a named article *says*
  isn't checked (e.g. hand check v1, Q9: Art. 113 is named in Art. 111(2), so "Article 113
  would set it" isn't flagged). That needs the LLM judge in AN-004.
"""

import hashlib
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anthropic

from ai_act_navigator.answer import Answer, AnswerError
from ai_act_navigator.golden import MIN_PER_TYPE, GoldenQuestion
from ai_act_navigator.models import Hit
from ai_act_navigator.paths import PROJECT_ROOT
from ai_act_navigator.refs import article_of, mentioned_articles

RESULTS_DIR = PROJECT_ROOT / "evals" / "results"

# The question in, the chunks search found and Claude's answer out. The CLI passes the real
# pipeline; tests pass a fake.
Ask = Callable[[str], tuple[list[Hit], Answer]]

# A decline in the opening paragraph: "excerpts" + "don't"/"do not" + "answer" or "so I can't".
_DECLINE = re.compile(
    r"\bexcerpts\b[^\n]*?\b(?:don't|do not|doesn't|does not)\b[^\n]*?"
    r"(?:\banswer|\bso I (?:can't|cannot))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QuestionResult:
    id: str
    type: str
    question: str
    passed: bool
    expected: tuple[str, ...]  # article-level, from the golden set
    cited: tuple[str, ...]  # exact refs the answer cited, e.g. ("Art. 50(1)", "Recital 132")
    declined: bool  # the answer said the excerpts don't answer the question
    retrieved: bool | None  # expected article among the chunks sent; None for out_of_scope
    sent: tuple[str, ...]  # refs of all chunks sent to Claude, best first
    retrieved_uncited: tuple[str, ...]  # named in the answer, sent, but not cited
    outside_knowledge: tuple[str, ...]  # named in the answer, not in the question or excerpts
    refused: bool  # safety decline (Answer.refused)
    answer: str
    model: str  # the model that actually answered (a fallback can differ from the setting)
    cost_usd: float | None
    error: str = ""  # set if no answer came back (cut off, API error); counts as a fail


def is_decline(text: str) -> bool:
    """True if the answer's opening paragraph says the excerpts don't answer the question."""
    return bool(_DECLINE.search(text.strip().split("\n", 1)[0]))


def found_expected(question: GoldenQuestion, hits: list[Hit]) -> bool:
    """True if search sent Claude at least one chunk of an expected article."""
    return bool({article_of(hit.chunk.ref) for hit in hits} & set(question.sources))


def score(question: GoldenQuestion, hits: list[Hit], answer: Answer) -> QuestionResult:
    """Check one answer against its golden question. No API calls."""
    cited = {article_of(c.ref) for c in answer.citations}
    sent = {article_of(hit.chunk.ref) for hit in hits}
    in_excerpts = set().union(*(mentioned_articles(hit.chunk.text) for hit in hits))
    expected = set(question.sources)
    declined = is_decline(answer.text)
    # Naming the article the question asks about ("The excerpts don't include Article 4")
    # is neither a missed citation nor outside knowledge.
    uncited = mentioned_articles(answer.text) - cited - mentioned_articles(question.question)

    if question.type == "out_of_scope":
        passed = declined and not answer.refused
        retrieved = None
    else:
        passed = bool(cited & expected) and not declined
        retrieved = found_expected(question, hits)

    return QuestionResult(
        id=question.id,
        type=question.type,
        question=question.question,
        passed=passed,
        expected=question.sources,
        cited=tuple(c.ref for c in answer.citations),
        declined=declined,
        retrieved=retrieved,
        sent=tuple(hit.chunk.ref for hit in hits),
        retrieved_uncited=tuple(sorted(uncited & sent)),
        outside_knowledge=tuple(sorted(uncited - sent - in_excerpts)),
        refused=answer.refused,
        answer=answer.text,
        model=answer.model,
        cost_usd=answer.cost_usd,
    )


def evaluate_one(question: GoldenQuestion, ask: Ask) -> QuestionResult:
    """Ask one golden question and score the answer.

    A cut-off answer or a server error (500, or 529 "overloaded", which the SDK raises as a
    separate class) becomes a failed result instead of a crash, so one bad question
    doesn't stop the run. Problems that would hit every question (bad key, rate limit, no
    internet) still stop it; the CLI then saves the answers collected so far.
    """
    try:
        hits, answer = ask(question.question)
    except (AnswerError, anthropic.InternalServerError, anthropic.OverloadedError) as error:
        return QuestionResult(
            id=question.id,
            type=question.type,
            question=question.question,
            passed=False,
            expected=question.sources,
            cited=(),
            declined=False,
            retrieved=None,
            sent=(),
            retrieved_uncited=(),
            outside_knowledge=(),
            refused=False,
            answer="",
            model="",
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
        "wrongly_declined": sum(r.declined and bool(r.retrieved) for r in in_scope),
        "with_retrieved_uncited": sum(bool(r.retrieved_uncited) for r in results),
        "with_outside_knowledge": sum(bool(r.outside_knowledge) for r in results),
        "refused": sum(r.refused for r in results),
        "errors": sum(bool(r.error) for r in results),
        "cost_usd": round(sum(costs), 4),  # approximate if a model has no known price
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "missing"


def git_commit() -> str:
    """The commit the code ran from, with "+changes" if there were uncommitted edits.

    Lets anyone check out the exact code behind a score. "unknown" outside a git checkout.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True
        ).stdout.strip()
        changes = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:  # git not installed
        return "unknown"
    return f"{commit}+changes" if commit and changes else commit or "unknown"


def save_results(
    results: list[QuestionResult], settings: dict[str, Any], path: Path
) -> dict[str, Any]:
    """Write settings, summary and every answer to a JSON file. Returns the summary."""
    summary = summarize(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {"settings": settings, "summary": summary, "results": [asdict(r) for r in results]}
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary
