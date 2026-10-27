"""The golden set: 20 questions with known correct answers, checked by hand (AN-002).

It is a fixed exam. Every change to search, prompts or models is scored on the same 20
questions, so "better" is a number instead of a guess. The questions live in
evals/golden-v1.json; this file loads them and refuses a file that breaks the rules
below, so a typo can't silently change the score.

Three question types:
- exact_ref:     names the article ("What does Article 4 require?")
- plain:         everyday wording ("Does our chatbot have to say it's AI?")
- out_of_scope:  the AI Act doesn't answer it ("What is the VAT rate in Germany?")
"""

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from ai_act_navigator.paths import PROJECT_ROOT

GOLDEN_PATH = PROJECT_ROOT / "evals" / "golden-v1.json"

QuestionType = Literal["exact_ref", "plain", "out_of_scope"]

# The mix from AN-002, acceptance criterion 2: at least this many of each type.
MIN_PER_TYPE: dict[QuestionType, int] = {"exact_ref": 5, "plain": 10, "out_of_scope": 5}

# Expected sources are whole articles or annexes. Recitals explain the law but aren't
# rules, so they don't count as "the right article".
_SOURCE = re.compile(r"^(Art\. \d+|Annex [IVXL]+)$")


class GoldenSetError(Exception):
    """The golden set file breaks one of its rules (wrong type, bad source, duplicate id)."""


@dataclass(frozen=True)
class GoldenQuestion:
    id: str  # e.g. "P1"; stays the same so results can be compared between runs
    type: QuestionType
    question: str
    answer: str  # the correct answer in short, for people reading the results
    sources: tuple[str, ...]  # article-level refs, e.g. ("Art. 50",); empty for out_of_scope
    checked: bool  # True once answer and sources were checked against the official text
    origin: str = ""  # where the question came from, e.g. "hand check v1, Q1"
    note: str = ""


def load_golden(path: Path = GOLDEN_PATH) -> list[GoldenQuestion]:
    """Read and check the golden set. Raises GoldenSetError if a rule is broken."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GoldenSetError(f"{path.name} is not valid JSON: {error}") from error
    questions = [_to_question(entry) for entry in data["questions"]]

    ids = [q.id for q in questions]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise GoldenSetError(f"Duplicate ids: {', '.join(duplicates)}")
    return questions


def _to_question(entry: dict) -> GoldenQuestion:
    try:
        question = GoldenQuestion(**{**entry, "sources": tuple(entry["sources"])})
    except (TypeError, KeyError) as error:  # missing or unknown field
        raise GoldenSetError(f"{entry.get('id', '?')}: {error}") from error

    # JSON has no types of its own: "checked": "no" would otherwise count as checked.
    if not isinstance(question.checked, bool):
        raise GoldenSetError(f"{question.id}: checked must be true or false")
    if not question.question.strip():
        raise GoldenSetError(f"{question.id}: question is empty")
    if question.type not in get_args(QuestionType):
        raise GoldenSetError(f"{question.id}: unknown type {question.type!r}")
    if question.type == "out_of_scope" and question.sources:
        raise GoldenSetError(f"{question.id}: out_of_scope questions have no sources")
    if question.type != "out_of_scope" and not question.sources:
        raise GoldenSetError(f"{question.id}: needs at least one source")
    for source in question.sources:
        if not _SOURCE.match(source):
            raise GoldenSetError(
                f"{question.id}: source {source!r} must be a whole article or annex, "
                'e.g. "Art. 50" or "Annex III"'
            )
    return question


def check_mix(questions: list[GoldenQuestion]) -> None:
    """Raise GoldenSetError unless there are at least MIN_PER_TYPE questions of each type.

    Checked before an eval run, not in load_golden, so tests can load small example files.
    """
    counts = Counter(q.type for q in questions)
    short = [f"{t} {counts[t]}/{n}" for t, n in MIN_PER_TYPE.items() if counts[t] < n]
    if short:
        raise GoldenSetError(f"Too few questions of a type: {', '.join(short)}")
