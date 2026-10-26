"""UNIT TESTS for the `ai-act` command-line tool. No model, no API calls, no cost.

`monkeypatch` swaps the slow or paid parts (the pipeline, the key check) for fakes, and
`capsys` captures what the command prints, so we can check the exact output.
"""

import anthropic
import httpx
import pytest

from ai_act_navigator import cli
from ai_act_navigator.answer import Answer, AnswerError, Citation
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

    def fake_ask(question, k, model, effort):
        calls.update(question=question, k=k, model=model, effort=effort)
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
    cli.main(["ask", "q", "--k", "8", "--model", "claude-haiku-5-5", "--effort", "low"])
    assert fake_pipeline == {"question": "q", "k": 8, "model": "claude-haiku-5-5", "effort": "low"}


def test_quotes_option_shows_every_quote(fake_pipeline, capsys):
    cli.main(["ask", "q", "--quotes"])
    out = capsys.readouterr().out
    assert '      "quoted a"\n      "b"' in out


@pytest.mark.parametrize("bad_k", ["0", "-1", "two"])
def test_k_must_be_a_positive_number(bad_k, capsys):
    with pytest.raises(SystemExit):  # argparse prints usage and exits
        cli.main(["ask", "q", "--k", bad_k])


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
