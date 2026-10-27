"""UNIT TESTS for evaluate.py: scoring answers against the golden set. No API calls, no cost."""

import json

import pytest

from ai_act_navigator.answer import Answer, AnswerError, Citation
from ai_act_navigator.evaluate import (
    evaluate_one,
    mentioned_articles,
    save_results,
    score,
    summarize,
)
from ai_act_navigator.golden import GoldenQuestion
from ai_act_navigator.models import Chunk, Hit

CHATBOT = GoldenQuestion("P1", "plain", "Must a chatbot say it is AI?", "Yes.", ("Art. 50",), True)
VAT = GoldenQuestion("O1", "out_of_scope", "What is the VAT rate in Germany?", "No.", (), True)


def hit(ref: str) -> Hit:
    return Hit(Chunk(ref.lower(), "article", ref, "", "text", "u"), 0.5)


def reply(text: str = "Yes. [1]", *refs: str, refused: bool = False) -> Answer:
    citations = [Citation(n, ref, "u", ("q",)) for n, ref in enumerate(refs, start=1)]
    return Answer(text, citations, "claude-sonnet-5-5", 3000, 500, 0.01, refused=refused)


def test_citing_a_paragraph_of_the_expected_article_passes():
    result = score(CHATBOT, [hit("Art. 50(1)")], reply("Yes. [1]", "Art. 50(1)"))
    assert result.passed
    assert result.retrieved
    assert result.cited == ("Art. 50(1)",)


def test_citing_only_a_recital_fails():
    result = score(CHATBOT, [hit("Recital 132")], reply("Yes. [1]", "Recital 132"))
    assert not result.passed


def test_a_search_miss_shows_up_as_not_retrieved():
    result = score(CHATBOT, [hit("Art. 52(1)")], reply("No idea.", "Art. 52(1)"))
    assert not result.passed
    assert result.retrieved is False


def test_out_of_scope_passes_when_nothing_is_cited():
    result = score(VAT, [hit("Art. 99(3)")], reply("The excerpts don't answer this."))
    assert result.passed
    assert result.retrieved is None  # nothing to retrieve


def test_out_of_scope_fails_when_something_is_cited():
    assert not score(VAT, [hit("Art. 99(3)")], reply("Fines... [1]", "Art. 99(3)")).passed


def test_a_safety_decline_never_passes():
    assert not score(VAT, [], reply("Claude declined.", refused=True)).passed


def test_named_but_uncited_articles_are_split_by_whether_they_were_sent():
    text = "Yes. [1] See also Art. 50(2) and Article 113."
    result = score(CHATBOT, [hit("Art. 50(1)"), hit("Art. 50(2)")], reply(text, "Art. 50(1)"))
    # Art. 50 is cited (via 50(1)), so naming 50(2) is fine at article level.
    assert result.retrieved_uncited == ()
    assert result.outside_knowledge == ("Art. 113",)


def test_retrieved_but_uncited():
    text = "Yes. [1] Fines are in Article 99."
    result = score(CHATBOT, [hit("Art. 50(1)"), hit("Art. 99(3)")], reply(text, "Art. 50(1)"))
    assert result.retrieved_uncited == ("Art. 99",)
    assert result.outside_knowledge == ()


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Art. 50(1) and Article 6(1), point (a)", {"Art. 50", "Art. 6"}),
        ("listed in Annex III and Annex I", {"Annex III", "Annex I"}),
        ("Articles 99 and 100", {"Art. 99"}),  # known limit: only the first number
        ("Article 4(3) of Directive (EU) 2019/790", set()),  # another law
        ("Article 2 of Regulation (EU) 2016/679", set()),
        ("No article here. Recital 132.", set()),
    ],
)
def test_mentioned_articles(text, found):
    assert mentioned_articles(text) == found


def test_a_cut_off_answer_is_recorded_as_a_fail():
    def ask(question):
        raise AnswerError("The answer was cut off (max_tokens reached).")

    result = evaluate_one(CHATBOT, ask)
    assert not result.passed
    assert result.error == "The answer was cut off (max_tokens reached)."


def test_summary_counts():
    results = [
        score(CHATBOT, [hit("Art. 50(1)")], reply("Yes. [1]", "Art. 50(1)")),  # pass
        score(CHATBOT, [hit("Art. 52(1)")], reply("Can't tell.")),  # declined, search miss
        score(VAT, [], reply("Not covered.")),  # pass
    ]
    summary = summarize(results)

    assert summary["passed"] == 2
    assert summary["total"] == 3
    assert summary["by_type"]["plain"] == {"passed": 1, "total": 2}
    assert summary["by_type"]["out_of_scope"] == {"passed": 1, "total": 1}
    assert summary["retrieved"] == {"found": 1, "total": 2}
    assert summary["wrongly_declined"] == 1
    assert summary["cost_usd"] == 0.03


def test_results_file_has_settings_summary_and_every_answer(tmp_path):
    results = [score(CHATBOT, [hit("Art. 50(1)")], reply("Yes. [1]", "Art. 50(1)"))]
    path = tmp_path / "results" / "run.json"

    save_results(results, {"model": "claude-sonnet-5-5"}, path)

    report = json.loads(path.read_text())
    assert report["settings"] == {"model": "claude-sonnet-5-5"}
    assert report["summary"]["passed"] == 1
    assert report["results"][0]["answer"] == "Yes. [1]"
    assert report["results"][0]["cited"] == ["Art. 50(1)"]
