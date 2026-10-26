"""The `ai-act` command-line tool: the way people use AI Act Navigator.

    uv run ai-act index                      # build the search index (once, ~20 s)
    uv run ai-act ask "Does a chatbot have to tell users it is an AI?"
    uv run ai-act ask --quotes "..."         # also show the exact quoted passages
    uv run ai-act --help                     # all commands and options

Built with argparse, which comes with Python. It turns the words typed after `ai-act` into
named settings (args.question, args.k, ...) and writes the --help text for us.

Errors the user can fix (no key, empty index, no internet, ...) print one clear line and
exit with code 1. Anything unexpected still shows the full traceback, so real bugs stay
visible instead of being hidden behind a vague message.
"""

import argparse
import os
import sys
from collections.abc import Sequence

import anthropic

from ai_act_navigator.answer import EFFORT, MODEL, Answer, AnswerError
from ai_act_navigator.paths import PROJECT_ROOT
from ai_act_navigator.search import DEFAULT_K
from ai_act_navigator.store import IndexMismatchError


class UserError(Exception):
    """A problem the user can fix; shown as a single line, without a traceback."""


def build_parser() -> argparse.ArgumentParser:
    """Describe the commands and options. argparse uses this for parsing and for --help."""
    parser = argparse.ArgumentParser(
        prog="ai-act",
        description="Ask questions about the EU AI Act and get answers that cite the exact "
        "article.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    ask = commands.add_parser("ask", help="answer a question with cited sources")
    ask.add_argument("question", nargs="*", help='your question, e.g. "What is banned?"')
    ask.add_argument(
        "--k",
        type=positive_int,
        default=DEFAULT_K,
        help=f"how many chunks of the law to give Claude (default {DEFAULT_K})",
    )
    ask.add_argument("--model", default=MODEL, help=f"Claude model (default {MODEL})")
    ask.add_argument(
        "--effort",
        default=EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=f"how much Claude thinks first (default {EFFORT})",
    )
    ask.add_argument(
        "--quotes",
        action="store_true",
        help="show the exact passages Claude quoted under each source",
    )
    ask.set_defaults(handler=run_ask)

    index = commands.add_parser("index", help="download the law and build the search index")
    index.set_defaults(handler=run_index)
    return parser


def positive_int(value: str) -> int:
    """argparse type for --k: a whole number of at least 1."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be 1 or more")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return the exit code: 0 = success, 1 = error.

    `argv` lets tests pass arguments directly; normally it comes from the terminal.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (UserError, IndexMismatchError, AnswerError) as error:
        return fail(str(error))
    except anthropic.AuthenticationError:
        return fail("The Anthropic API key was rejected. Check ANTHROPIC_API_KEY in .env.")
    except anthropic.RateLimitError:
        return fail(
            "Rate limit or spending limit reached. Wait a minute, or check your limits in "
            "the Anthropic Console."
        )
    except anthropic.APIConnectionError:
        return fail("Could not reach the Anthropic API. Check your internet connection.")
    except anthropic.APIStatusError as error:
        return fail(f"The Anthropic API returned an error ({error.status_code}): {error.message}")


def fail(message: str) -> int:
    print(f"Error: {message}", file=sys.stderr)  # errors go to stderr, answers to stdout
    return 1


def run_ask(args: argparse.Namespace) -> int:
    question = " ".join(args.question).strip()
    if not question:
        raise UserError('Please ask a question, e.g.: uv run ai-act ask "What is banned?"')

    load_api_key()
    result = ask_question(question, k=args.k, model=args.model, effort=args.effort)
    print(format_answer(result, show_quotes=args.quotes))
    return 0


def load_api_key() -> None:
    """Read ANTHROPIC_API_KEY from the git-ignored .env file. The key is never printed."""
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise UserError("ANTHROPIC_API_KEY is missing. Copy .env.example to .env and add your key.")


def ask_question(question: str, k: int, model: str, effort: str) -> Answer:
    """The full pipeline: find the closest chunks, then let Claude answer from them.

    Imports are inside the function because loading the embedding model and its libraries
    takes a few seconds; `ai-act --help` shouldn't have to wait for that.
    """
    from ai_act_navigator.answer import answer
    from ai_act_navigator.embed import Embedder
    from ai_act_navigator.search import search
    from ai_act_navigator.store import VectorStore

    embedder = Embedder()
    hits = search(question, embedder, VectorStore(embedder.model_id), k=k)
    return answer(question, hits, anthropic.Anthropic(), model=model, effort=effort)


def format_answer(result: Answer, show_quotes: bool = False) -> str:
    """The answer as terminal text: answer, numbered sources, then model, tokens and cost."""
    lines = [result.text, ""]
    lines.append("Sources:" if result.citations else "Sources: none")
    for c in result.citations:
        lines.append(f"  [{c.number}] {c.ref}  {c.url}")
        if show_quotes:
            lines.extend(f'      "{quote}"' for quote in c.quotes)
    cost = f"${result.cost_usd:.4f}" if result.cost_usd is not None else "unknown"
    lines += [
        "",
        f"{result.model} · {result.input_tokens} tokens in, {result.output_tokens} out "
        f"· cost {cost}",
    ]
    return "\n".join(lines)


def run_index(args: argparse.Namespace) -> int:
    from ai_act_navigator.index import main as build_index

    build_index()
    return 0


def run() -> None:
    """Entry point for the `ai-act` command (see [project.scripts] in pyproject.toml)."""
    sys.exit(main())


if __name__ == "__main__":
    run()
