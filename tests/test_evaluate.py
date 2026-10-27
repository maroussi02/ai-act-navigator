"""UNIT TESTS for evaluate.py: scoring answers against the golden set. No API calls, no cost."""

import json

import anthropic
import httpx
import pytest

from ai_act_navigator.answer import Answer, AnswerError, Citation
from ai_act_navigator.evaluate import (
    evaluate_one,
    is_decline,
    save_results,
    score,
    summarize,
)
from ai_act_navigator.golden import GoldenQuestion
from ai_act_navigator.models import Chunk, Hit

CHATBOT = GoldenQuestion("P1", "plain", "Must a chatbot say it is AI?", "Yes.", ("Art. 50",), True)
VAT = GoldenQuestion("O1", "out_of_scope", "What is the VAT rate in Germany?", "No.", (), True)


def hit(ref: str, text: str = "text") -> Hit:
    return Hit(Chunk(ref.lower(), "article", ref, "", text, "u"), 0.5)


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


def test_out_of_scope_passes_when_it_declines_and_cites_context():
    text = "The excerpts don't answer this. They cover only AI Act fines. [1]"
    result = score(VAT, [hit("Art. 99(3)")], reply(text, "Art. 99(3)"))
    assert result.passed
    assert result.declined


def test_out_of_scope_fails_when_it_answers():
    assert not score(VAT, [hit("Art. 99(3)")], reply("Fines are... [1]", "Art. 99(3)")).passed
    assert not score(VAT, [], reply("Germany's VAT rate is 19%.")).passed  # from memory


def test_in_scope_fails_when_it_declines_even_if_it_cites_the_article():
    text = "The excerpts don't answer this, so I can't say. Related: [1]"
    result = score(CHATBOT, [hit("Art. 50(1)")], reply(text, "Art. 50(1)"))
    assert not result.passed
    assert result.declined


# Opening lines of real answers from the first baseline run (2026-10-27).
@pytest.mark.parametrize(
    ("text", "declined"),
    [
        ("The excerpts don't answer this. They cover only administrative fines", True),
        ("The excerpts you provided don't answer this question. They cover", True),
        ("The excerpts don't include Article 4 of the AI Act, so I can't say what it", True),
        ("The excerpts don't give a general start date, so I can't state one from them.", True),
        (
            "The excerpts cover only parts of Article 26 (paragraphs 1, 2), so this is partial",
            False,
        ),
        ("The excerpts cover five practices. Point (c) is not included, so the list", False),
        ("Yes, in general. Providers must ensure that AI systems", False),
        ("Yes. [1]\nThe excerpts don't answer the second part.", False),  # only the opening
    ],
)
def test_is_decline(text, declined):
    assert is_decline(text) == declined


def test_a_safety_decline_never_passes():
    assert not score(VAT, [], reply("Claude declined.", refused=True)).passed


def test_named_but_uncited_articles_are_split_by_whether_they_were_sent():
    text = "Yes. [1] See also Art. 50(2) and Article 113."
    result = score(CHATBOT, [hit("Art. 50(1)"), hit("Art. 50(2)")], reply(text, "Art. 50(1)"))
    # Art. 50 is cited (via 50(1)), so naming 50(2) is fine at article level.
    assert result.retrieved_uncited == ()
    assert result.outside_knowledge == ("Art. 113",)


def test_articles_in_the_question_or_the_excerpts_are_not_outside_knowledge():
    question = GoldenQuestion(
        "E1", "exact_ref", "What does Article 4 require?", "a", ("Art. 4",), True
    )
    sent = [hit("Art. 26(5)", "inform the provider... pursuant to Article 72 and Article 73")]
    text = "The excerpts don't include Article 4, so I can't say. Art. 26(5) mentions Article 72."
    result = score(question, sent, reply(text))
    assert result.outside_knowledge == ()  # Art. 4: in the question; Art. 72: in Art. 26(5)
    assert result.retrieved_uncited == ("Art. 26",)  # sent and named, but no [number]


def test_retrieved_but_uncited():
    text = "Yes. [1] Fines are in Article 99."
    result = score(CHATBOT, [hit("Art. 50(1)"), hit("Art. 99(3)")], reply(text, "Art. 50(1)"))
    assert result.retrieved_uncited == ("Art. 99",)
    assert result.outside_knowledge == ()


def test_a_cut_off_answer_is_recorded_as_a_fail():
    def ask(question):
        raise AnswerError("The answer was cut off (max_tokens reached).")

    result = evaluate_one(CHATBOT, ask)
    assert not result.passed
    assert result.error == "The answer was cut off (max_tokens reached)."


@pytest.mark.parametrize(
    ("cls", "status"), [(anthropic.OverloadedError, 529), (anthropic.InternalServerError, 500)]
)
def test_a_server_error_is_recorded_as_a_fail(cls, status):
    def ask(question):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise cls("busy", response=httpx.Response(status, request=request), body=None)

    result = evaluate_one(CHATBOT, ask)
    assert not result.passed
    assert "busy" in result.error


def test_a_bad_api_key_still_stops_the_run():
    def ask(question):
        request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        raise anthropic.AuthenticationError(
            "bad key", response=httpx.Response(401, request=request), body=None
        )

    with pytest.raises(anthropic.AuthenticationError):
        evaluate_one(CHATBOT, ask)


def test_summary_counts():
    results = [
        score(CHATBOT, [hit("Art. 50(1)")], reply("Yes. [1]", "Art. 50(1)")),  # pass
        # declined after a search miss: correct, so not "wrongly declined"
        score(CHATBOT, [hit("Art. 52(1)")], reply("The excerpts don't answer this.")),
        # declined although search found the article: wrongly declined
        score(CHATBOT, [hit("Art. 50(1)")], reply("The excerpts don't answer this.")),
        score(VAT, [], reply("The excerpts don't answer this.")),  # pass
    ]
    summary = summarize(results)

    assert summary["passed"] == 2
    assert summary["total"] == 4
    assert summary["by_type"]["plain"] == {"passed": 1, "total": 3}
    assert summary["by_type"]["out_of_scope"] == {"passed": 1, "total": 1}
    assert summary["retrieved"] == {"found": 2, "total": 3}
    assert summary["wrongly_declined"] == 1
    assert summary["cost_usd"] == 0.04


def test_results_file_has_settings_summary_and_every_answer(tmp_path):
    results = [score(CHATBOT, [hit("Art. 50(1)")], reply("Yes. [1]", "Art. 50(1)"))]
    path = tmp_path / "results" / "run.json"

    save_results(results, {"model": "claude-sonnet-5-5"}, path)

    report = json.loads(path.read_text())
    assert report["settings"] == {"model": "claude-sonnet-5-5"}
    assert report["summary"]["passed"] == 1
    assert report["results"][0]["answer"] == "Yes. [1]"
    assert report["results"][0]["cited"] == ["Art. 50(1)"]
    assert report["results"][0]["model"] == "claude-sonnet-5-5"
