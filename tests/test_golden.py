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
    check_mix,
    load_golden,
)
from ai_act_navigator.refs import article_of


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
        (entry(checked="no"), "true or false"),  # a string would count as checked
        (entry(question="  "), "question is empty"),
    ],
)
def test_rejects_a_broken_entry(tmp_path, bad, message):
    with pytest.raises(GoldenSetError, match=message):
        load_golden(write_golden(tmp_path, bad))


def test_rejects_a_file_that_is_not_json(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text('{"questions": [')
    with pytest.raises(GoldenSetError, match="not valid JSON"):
        load_golden(path)


def test_check_mix_names_the_type_that_is_short(tmp_path):
    with pytest.raises(GoldenSetError, match="exact_ref 0/5"):
        check_mix(load_golden(write_golden(tmp_path, entry())))


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
