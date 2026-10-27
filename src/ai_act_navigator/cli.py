"""The `ai-act` command-line tool: the way people use AI Act Navigator.

    uv run ai-act index                      # build the search index (once, ~20 s)
    uv run ai-act ask "Does a chatbot have to tell users it is an AI?"
    uv run ai-act ask --quotes "..."         # also show the exact quoted passages
    uv run ai-act eval                       # score the golden set (20 questions, ~$0.20)
    uv run ai-act --help                     # all commands and options

Built with argparse, which comes with Python. It turns the words typed after `ai-act` into
named settings (args.question, args.k, ...) and writes the --help text for us.

Exit codes: 0 = success, 1 = a problem the user can fix (no key, empty index, no internet,
...), shown as one line starting with "Error:", 2 = wrong arguments (argparse prints the
usage). Anything unexpected still shows the full traceback, so real bugs stay visible
instead of being hidden behind a vague message.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import httpx

from ai_act_navigator.answer import EFFORT, MODEL, Answer, AnswerError
from ai_act_navigator.chunk import OUTPUT_PATH as CHUNKS_PATH
from ai_act_navigator.download import IntegrityError
from ai_act_navigator.embed import MODEL_NAME as EMBED_MODEL
from ai_act_navigator.embed import MODEL_REVISION as EMBED_REVISION
from ai_act_navigator.embed import ModelNotAvailableError
from ai_act_navigator.evaluate import (
    RESULTS_DIR,
    Ask,
    QuestionResult,
    evaluate_one,
    file_sha256,
    git_commit,
    save_results,
)
from ai_act_navigator.golden import GOLDEN_PATH, GoldenSetError, check_mix, load_golden
from ai_act_navigator.models import Hit
from ai_act_navigator.paths import PROJECT_ROOT
from ai_act_navigator.prompts import PROMPT_VERSION
from ai_act_navigator.search import DEFAULT_K
from ai_act_navigator.store import IndexMismatchError

# Upper limit for --k. 50 chunks are roughly 8,000 to 22,000 tokens, about $0.02 to $0.05
# of input per question with Sonnet. Without a limit, a typo like --k 900 would send almost
# the whole law (~140,000 tokens, ~$0.30) with every question.
MAX_K = 50


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
    add_pipeline_options(ask)
    ask.add_argument(
        "--quotes",
        action="store_true",
        help="show the exact passages Claude quoted under each source",
    )
    ask.set_defaults(handler=run_ask)

    evaluate = commands.add_parser(
        "eval", help="score the answers on the golden set (20 questions, about $0.20)"
    )
    add_pipeline_options(evaluate)
    evaluate.add_argument(
        "--out",
        type=Path,
        help="results file (default: a new file in evals/results/)",
    )
    evaluate.set_defaults(handler=run_eval)

    index = commands.add_parser("index", help="download the law and build the search index")
    index.set_defaults(handler=run_index)
    return parser


def add_pipeline_options(command: argparse.ArgumentParser) -> None:
    """The settings `ask` and `eval` share, so eval scores exactly what ask would answer."""
    command.add_argument(
        "--k",
        type=chunk_count,
        default=DEFAULT_K,
        help=f"how many chunks of the law to give Claude, 1 to {MAX_K} (default {DEFAULT_K})",
    )
    command.add_argument("--model", default=MODEL, help=f"Claude model (default {MODEL})")
    command.add_argument(
        "--effort",
        default=EFFORT,
        choices=["low", "medium", "high", "xhigh", "max"],
        help=f"how much Claude thinks first (default {EFFORT})",
    )


def chunk_count(value: str) -> int:
    """argparse type for --k: a whole number from 1 to MAX_K."""
    number = int(value)
    if not 1 <= number <= MAX_K:
        raise argparse.ArgumentTypeError(f"must be between 1 and {MAX_K}")
    return number


def main(argv: Sequence[str] | None = None) -> int:
    """Run one command and return the exit code: 0 = success, 1 = error.

    (Wrong arguments never get here: argparse prints the usage and exits with code 2.)
    `argv` lets tests pass arguments directly; normally it comes from the terminal.
    """
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except (
        UserError,
        IndexMismatchError,
        AnswerError,
        ModelNotAvailableError,
        GoldenSetError,
    ) as error:
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
    """The full pipeline: find the closest chunks, then let Claude answer from them."""
    _, result = make_asker(k, model, effort)(question)
    return result


def make_asker(k: int, model: str, effort: str) -> Ask:
    """Load the pipeline once and return a function that answers one question with it.

    `eval` asks 20 questions; loading the embedding model once instead of 20 times saves
    about a minute. Imports are inside the function because loading the model and its
    libraries takes a few seconds; `ai-act --help` shouldn't have to wait for that.
    """
    from ai_act_navigator.answer import answer
    from ai_act_navigator.embed import Embedder
    from ai_act_navigator.search import search
    from ai_act_navigator.store import VectorStore

    # allow_download=False: asking never contacts Hugging Face. If the model is missing,
    # the error says to run `ai-act index`, which downloads it.
    embedder = Embedder(allow_download=False)
    store = VectorStore(embedder.model_id)
    client = anthropic.Anthropic()

    def ask(question: str) -> tuple[list[Hit], Answer]:
        hits = search(question, embedder, store, k=k)
        return hits, answer(question, hits, client, model=model, effort=effort)

    return ask


def run_eval(args: argparse.Namespace) -> int:
    questions = load_golden()
    unchecked = [q.id for q in questions if not q.checked]
    if unchecked:  # a score on unchecked answers would be a guess, not a baseline
        raise UserError(f"Golden set not checked by hand yet: {', '.join(unchecked)}")
    check_mix(questions)

    started = datetime.now(UTC)
    path = args.out or RESULTS_DIR / (
        f"{started:%Y-%m-%d-%H%M%S}-{args.model}-{args.effort}-k{args.k}.json"
    )
    if path.exists():  # e.g. --out pointing at the committed baseline
        raise UserError(f"{path} already exists. Choose another --out or delete it first.")

    load_api_key()
    ask = make_asker(args.k, args.model, args.effort)
    # Everything needed to rerun this exact score later: code, data, models, settings.
    settings = {
        "date": started.isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "golden_set": GOLDEN_PATH.name,
        "golden_set_sha256": file_sha256(GOLDEN_PATH),
        "chunks_sha256": file_sha256(CHUNKS_PATH),
        "embedding_model": f"{EMBED_MODEL}@{EMBED_REVISION}",
        "model": args.model,
        "effort": args.effort,
        "k": args.k,
        "prompt_version": PROMPT_VERSION,
    }
    print(f"Asking {len(questions)} questions ({args.model}, effort {args.effort}, k {args.k})")
    results = []
    try:
        for question in questions:
            result = evaluate_one(question, ask)
            print(format_result_line(result))
            results.append(result)
    except BaseException:  # API error, Ctrl+C, ...: keep what was already paid for
        if results:
            save_results(results, {**settings, "incomplete": True}, path)
            print(
                f"Stopped after {len(results)} of {len(questions)} questions. "
                f"Partial results saved to {path}",
                file=sys.stderr,
            )
        raise

    summary = save_results(results, settings, path)
    print(format_summary(summary, path))
    return 0


def format_result_line(result: QuestionResult) -> str:
    """One line per question while eval runs, e.g. "P1   pass  cited Art. 50(1)"."""
    verdict = "pass" if result.passed else "FAIL"
    if result.error:
        detail = f"error: {result.error}"
    elif result.refused:
        detail = "safety decline"
    elif result.declined:
        detail = "declined" + (f", cited {', '.join(result.cited)}" if result.cited else "")
    elif result.cited:
        detail = f"cited {', '.join(result.cited)}"
    else:
        detail = "cited nothing"
    if result.retrieved is False:
        detail += f" · search missed {', '.join(result.expected)}"
    return f"{result.id:<4} {verdict}  {detail}"


def format_summary(summary: dict, path: Path) -> str:
    types = ", ".join(f"{t} {s['passed']}/{s['total']}" for t, s in summary["by_type"].items())
    found = summary["retrieved"]
    return "\n".join(
        [
            "",
            f"Score: {summary['passed']}/{summary['total']} ({types})",
            f"Search found the expected article: {found['found']}/{found['total']}",
            f"Wrongly declined: {summary['wrongly_declined']} · "
            f"retrieved but uncited: {summary['with_retrieved_uncited']} · "
            f"outside knowledge: {summary['with_outside_knowledge']} · "
            f"errors: {summary['errors']}",
            f"Cost: ${summary['cost_usd']:.4f} · saved to {path}",
        ]
    )


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

    try:
        build_index()
    except httpx.HTTPError as error:  # no internet, timeout, or the server answered 404, 503, ...
        raise UserError(
            f"Could not download the AI Act from the EU Publications Office ({error}). "
            "Check your internet connection."
        ) from error
    except IntegrityError as error:
        raise UserError(f"The downloaded AI Act text isn't the expected file. {error}") from error
    return 0


def run() -> None:
    """Entry point for the `ai-act` command (see [project.scripts] in pyproject.toml)."""
    sys.exit(main())


if __name__ == "__main__":
    run()
