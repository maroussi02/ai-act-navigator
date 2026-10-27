"""UNIT TESTS for the golden set (evals/golden-v1.json) and its loader.

test_every_question_checked_by_hand fails until all 20 answers were checked against the
official text (AN-002, acceptance criterion 1). Its message lists the ones still to do.
"""

import json
from collections import Counter

import pytest

from ai_act_navigator.chunk import parse
from ai_act_navigator.download import DEFAULT_PATH
from ai_act_navigator.golden import (
    MIN_PER_TYPE,
    GoldenSetError,
    article_of,
    load_golden,
)


def write_golden(tmp_path, *questions):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps({"version": "test", "questions": list(questions)}))
    return path


def entry(**changes):
    base = {
        "id": "P1",
        "type": "plain",
        "question": "Does a chatbot have to say it is an AI?",
        "answer": "Yes.",
        "sources": ["Art. 50"],
        "checked": True,
    }
    return {**base, **changes}


def test_golden_set_has_20_questions_in_the_right_mix():
    questions = load_golden()
    counts = Counter(q.type for q in questions)

    assert len(questions) == 20
    for question_type, minimum in MIN_PER_TYPE.items():
        assert counts[question_type] >= minimum, question_type


def test_every_question_checked_by_hand():
    unchecked = [q.id for q in load_golden() if not q.checked]
    assert not unchecked, f"Not checked by hand yet: {', '.join(unchecked)}"


@pytest.mark.parametrize(
    ("ref", "article"),
    [
        ("Art. 50(1)", "Art. 50"),
        ("Art. 5(1), point (f)", "Art. 5"),
        ("Art. 3, point (3)", "Art. 3"),
        ("Art. 113", "Art. 113"),
        ("Annex III, point 1", "Annex III"),
        ("Recital 132", "Recital 132"),
    ],
)
def test_article_of(ref, article):
    assert article_of(ref) == article


def test_article_of_rejects_text_that_is_no_ref():
    with pytest.raises(ValueError):
        article_of("Article 50")


def test_loads_a_valid_file(tmp_path):
    path = write_golden(tmp_path, entry(), entry(id="O1", type="out_of_scope", sources=[]))

    first, second = load_golden(path)

    assert first.sources == ("Art. 50",)
    assert second.sources == ()


@pytest.mark.parametrize(
    ("bad", "message"),
    [
        (entry(type="trick"), "unknown type"),
        (entry(type="out_of_scope"), "have no sources"),
        (entry(sources=[]), "at least one source"),
        (entry(sources=["Art. 50(1)"]), "whole article"),  # paragraph, not article
        (entry(sources=["Recital 132"]), "whole article"),
        (entry(sources=["Article 50"]), "whole article"),
        (entry(extra="x"), "extra"),  # unknown field, e.g. a typo
    ],
)
def test_rejects_a_broken_entry(tmp_path, bad, message):
    with pytest.raises(GoldenSetError, match=message):
        load_golden(write_golden(tmp_path, bad))


def test_rejects_duplicate_ids(tmp_path):
    with pytest.raises(GoldenSetError, match="Duplicate ids: P1"):
        load_golden(write_golden(tmp_path, entry(), entry()))


@pytest.mark.integration
@pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="full text not downloaded")
def test_every_expected_source_exists_in_the_law():
    """Catches a typo like "Art. 133", which no answer could ever cite."""
    articles = {article_of(chunk.ref) for chunk in parse(DEFAULT_PATH.read_text("utf-8"))}
    missing = {s for q in load_golden() for s in q.sources} - articles
    assert not missing
