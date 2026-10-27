"""UNIT TESTS for the `ai-act` command-line tool. No model, no API calls, no cost.

`monkeypatch` swaps the slow or paid parts (the pipeline, the key check) for fakes, and
`capsys` captures what the command prints, so we can check the exact output.
"""

import dataclasses
import json

import anthropic
import httpx
import pytest

from ai_act_navigator import cli, index
from ai_act_navigator.answer import Answer, AnswerError, Citation
from ai_act_navigator.download import IntegrityError
from ai_act_navigator.embed import ModelNotAvailableError
from ai_act_navigator.golden import GoldenQuestion, GoldenSetError, load_golden
from ai_act_navigator.models import Chunk, Hit
from ai_act_navigator.prompts import PROMPT_SHA256, PROMPT_VERSION
from ai_act_navigator.store import IndexMismatchError

ANSWER = Answer(
    text="Yes. [1]",
    citations=[Citation(1, "Art. 50(1)", "https://eur-lex.example/#art_50", ("quoted a", "b"))],
    model="claude-sonnet-5-5",
    input_tokens=3000,
    output_tokens=500,
    cost_usd=0.011,
)


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Replace the key check and the pipeline; record what the pipeline was called with."""
    calls = {}

    def fake_ask(question, k, search_mode, model, effort):
        calls.update(question=question, k=k, search_mode=search_mode, model=model, effort=effort)
        return ANSWER

    monkeypatch.setattr(cli, "load_api_key", lambda: None)
    monkeypatch.setattr(cli, "ask_question", fake_ask)
    return calls


def test_ask_prints_answer_sources_and_cost(fake_pipeline, capsys):
    assert cli.main(["ask", "Must", "a", "chatbot", "say", "so?"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Yes. [1]\n\nSources:\n  [1] Art. 50(1)  https://eur-lex.example/#art_50")
    assert "claude-sonnet-5-5 · 3000 tokens in, 500 out · cost $0.0110" in out
    assert "quoted a" not in out  # quotes only with --quotes
    assert fake_pipeline["question"] == "Must a chatbot say so?"  # words joined


def test_options_are_passed_to_the_pipeline(fake_pipeline):
    cli.main(
        ["ask", "q", "--k", "8", "--search", "vector", "--model", "claude-haiku-5-5"]
        + ["--effort", "low"]
    )
    assert fake_pipeline == {
        "question": "q",
        "k": 8,
        "search_mode": "vector",
        "model": "claude-haiku-5-5",
        "effort": "low",
    }


def test_search_is_hybrid_by_default(fake_pipeline):
    cli.main(["ask", "q"])
    assert fake_pipeline["search_mode"] == "hybrid"


def test_quotes_option_shows_every_quote(fake_pipeline, capsys):
    cli.main(["ask", "q", "--quotes"])
    out = capsys.readouterr().out
    assert '      "quoted a"\n      "b"' in out


@pytest.mark.parametrize("bad_k", ["0", "-1", "two", "51"])
def test_k_must_be_between_1_and_50(bad_k):
    with pytest.raises(SystemExit) as exit_info:  # argparse prints usage and exits
        cli.main(["ask", "q", "--k", bad_k])
    assert exit_info.value.code == 2  # argparse's code for wrong arguments


def test_k_of_50_is_allowed(fake_pipeline):
    cli.main(["ask", "q", "--k", "50"])
    assert fake_pipeline["k"] == 50


def test_empty_question_is_a_clear_error(capsys):
    assert cli.main(["ask", "  "]) == 1
    assert capsys.readouterr().err.startswith("Error: Please ask a question")


def test_missing_api_key_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", cli.PROJECT_ROOT / "does-not-exist")  # no .env
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["ask", "q"]) == 1
    assert "ANTHROPIC_API_KEY is missing" in capsys.readouterr().err


def api_error(cls, status):
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls("boom", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (IndexMismatchError("The index is empty."), "The index is empty."),
        (AnswerError("The answer was cut off."), "The answer was cut off."),
        (ModelNotAvailableError("Run: uv run ai-act index"), "Run: uv run ai-act index"),
        (api_error(anthropic.AuthenticationError, 401), "API key was rejected"),
        (api_error(anthropic.RateLimitError, 429), "Rate limit or spending limit"),
        (api_error(anthropic.InternalServerError, 500), "returned an error (500)"),
        (
            anthropic.APIConnectionError(request=httpx.Request("POST", "https://x")),
            "Could not reach the Anthropic API",
        ),
    ],
)
def test_known_problems_print_one_line_and_exit_1(monkeypatch, capsys, error, message):
    def broken(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "load_api_key", lambda: None)
    monkeypatch.setattr(cli, "ask_question", broken)
    assert cli.main(["ask", "q"]) == 1
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_unexpected_errors_are_not_hidden(monkeypatch):
    def bug(*args, **kwargs):
        raise ZeroDivisionError

    monkeypatch.setattr(cli, "load_api_key", lambda: None)
    monkeypatch.setattr(cli, "ask_question", bug)
    with pytest.raises(ZeroDivisionError):  # real bugs still show their traceback
        cli.main(["ask", "q"])


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (httpx.ConnectError("no route to host"), "Could not download the AI Act"),
        (IntegrityError("Expected SHA-256 aaa, got bbb."), "isn't the expected file"),
        (ModelNotAvailableError("Could not download the embedding model"), "embedding model"),
    ],
)
def test_index_problems_print_one_line_and_exit_1(monkeypatch, capsys, error, message):
    def broken():
        raise error

    monkeypatch.setattr(index, "main", broken)  # run_index imports index.main when it runs
    assert cli.main(["index"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: ") and message in err
    assert "Traceback" not in err


def fake_asker(k, search_mode, model, effort):
    """Answers every golden question by citing Art. 50(1), the chunk it 'found'."""
    hits = [Hit(Chunk("art-50-1", "article", "Art. 50(1)", "", "text", "u"), 0.5)]
    return lambda question: (hits, ANSWER)


@pytest.fixture
def fake_eval(monkeypatch):
    """The real golden questions, all marked checked, answered by fake_asker. No cost."""
    questions = [dataclasses.replace(q, checked=True) for q in load_golden()]
    monkeypatch.setattr(cli, "load_golden", lambda: questions)
    monkeypatch.setattr(cli, "load_api_key", lambda: None)
    monkeypatch.setattr(cli, "make_asker", fake_asker)


def test_eval_scores_every_question_and_saves_the_results(fake_eval, capsys, tmp_path):
    out = tmp_path / "run.json"

    assert cli.main(["eval", "--out", str(out), "--effort", "low"]) == 0

    printed = capsys.readouterr().out
    report = json.loads(out.read_text())
    assert len(report["results"]) == 20
    assert "P1   pass  cited Art. 50(1)" in printed  # P1 expects Art. 50
    assert "O1   FAIL  cited Art. 50(1)" in printed  # out of scope, but answered
    assert f"Score: {report['summary']['passed']}/20" in printed


def test_results_file_records_what_is_needed_to_rerun_the_score(fake_eval, tmp_path):
    out = tmp_path / "run.json"
    cli.main(["eval", "--out", str(out), "--effort", "low"])

    settings = json.loads(out.read_text())["settings"]
    assert settings["effort"] == "low"
    assert settings["search"] == "hybrid"
    assert settings["prompt_version"] == PROMPT_VERSION
    assert settings["prompt_sha256"] == PROMPT_SHA256
    assert settings["embedding_model"].startswith("BAAI/bge-small-en-v1.5@")
    assert len(settings["golden_set_sha256"]) == 64
    assert settings["git_commit"]  # a hash, "...+changes" or "unknown"
    assert "incomplete" not in settings


def test_eval_saves_partial_results_when_the_api_fails(fake_eval, monkeypatch, capsys, tmp_path):
    def asker(k, search_mode, model, effort):
        answered = []

        def ask(question):
            if len(answered) == 3:
                raise api_error(anthropic.RateLimitError, 429)
            answered.append(question)
            return fake_asker(k, search_mode, model, effort)(question)

        return ask

    monkeypatch.setattr(cli, "make_asker", asker)
    out = tmp_path / "run.json"

    assert cli.main(["eval", "--out", str(out)]) == 1

    report = json.loads(out.read_text())
    assert len(report["results"]) == 3
    assert report["settings"]["incomplete"] is True
    err = capsys.readouterr().err
    assert "Stopped after 3 of 20 questions" in err
    assert "Error: Rate limit" in err


def test_eval_never_overwrites_a_results_file(fake_eval, capsys, tmp_path):
    out = tmp_path / "baseline.json"
    out.write_text("{}")
    assert cli.main(["eval", "--out", str(out)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert out.read_text() == "{}"


def test_eval_refuses_an_unchecked_golden_set(monkeypatch, capsys):
    unchecked = GoldenQuestion("P1", "plain", "q", "a", ("Art. 50",), checked=False)
    monkeypatch.setattr(cli, "load_golden", lambda: [unchecked])
    assert cli.main(["eval"]) == 1
    assert "not checked by hand yet: P1" in capsys.readouterr().err


def test_a_broken_golden_set_is_one_error_line(monkeypatch, capsys):
    def broken():
        raise GoldenSetError("golden-v1.json is not valid JSON: Expecting value")

    monkeypatch.setattr(cli, "load_golden", broken)
    assert cli.main(["eval"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("Error: golden-v1.json is not valid JSON")
    assert "Traceback" not in err


class FakeSearcher:
    """Finds Art. 50(1) for every question and records the settings it was called with."""

    def __init__(self):
        self.calls = []

    def search(self, question, k, mode):
        self.calls.append((k, mode))
        return [Hit(Chunk("art-50-1", "article", "Art. 50(1)", "", "text", "u"), 0.5)]


def test_search_only_checks_search_without_claude_or_api_key(fake_eval, monkeypatch, capsys):
    searcher = FakeSearcher()
    monkeypatch.setattr(cli, "make_searcher", lambda: searcher)

    def no_key():
        raise AssertionError("--search-only must not need the API key")

    monkeypatch.setattr(cli, "load_api_key", no_key)
    monkeypatch.setattr(cli, "make_asker", no_key)  # and must not build the Claude pipeline

    assert cli.main(["eval", "--search-only", "--search", "vector", "--k", "3"]) == 0

    out = capsys.readouterr().out
    assert len(searcher.calls) == 15  # the 15 in-scope golden questions, none out of scope
    assert set(searcher.calls) == {(3, "vector")}
    assert "P1   found   sent Art. 50(1)" in out  # P1 expects Art. 50
    assert "E1   MISSED  sent Art. 50(1)" in out  # E1 expects Art. 4
    assert "Search found the expected article: 2/15" in out  # P1 and P5 expect Art. 50


def test_search_only_refuses_out(fake_eval, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(cli, "make_searcher", FakeSearcher)
    assert cli.main(["eval", "--search-only", "--out", str(tmp_path / "x.json")]) == 1
    assert "--search-only writes no results file" in capsys.readouterr().err
