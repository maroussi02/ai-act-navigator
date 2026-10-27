# AI Act Navigator: notes for Claude

A RAG over the EU AI Act that answers questions and cites the exact article. Plain Python
on purpose (no LangChain yet), so every step stays readable. Tickets are GitHub issues
(AN-001 to AN-004); README.md explains the project for humans.

## Commands
```bash
uv sync                                  # install (uv, Python 3.12)
uv run ai-act index                      # download the law, chunk, embed (~20 s)
uv run ai-act ask "question"             # one answer (~$0.01, real API call)
uv run ai-act eval                       # score the 20-question golden set (~$0.20)
uv run pytest -m "not integration"       # fast unit tests
uv run pytest                            # all free tests (integration loads the real model)
uv run pytest --run-paid -m paid         # real Claude call; only by hand, never in CI
uv run ruff check . && uv run ruff format .
```

## Layout
- `src/ai_act_navigator/`: one module per pipeline step: `download` → `chunk` → `embed` /
  `store` / `index` → `search` → `answer`; `cli.py` is the `ai-act` command; `golden.py` +
  `evaluate.py` run the golden set (`evals/golden-v1.json`).
- `tests/`: each file starts with its type (UNIT / INTEGRATION TESTS ...).
- `docs/hand-check-v1.md`: the manual check of 10 answers (owner's verdicts).
- `REVIEW.md`: what counts as a real bug in this repo; give it to every reviewer.
- `.github/pull_request_template.md`: every PR answers Testing, Risk and How to revert.

## Conventions
- **Comments:** every module and function gets a plain-language docstring saying what it does
  and why. Verify every factual claim before writing it.
- **Tests:** every PR adds at least one automated test. Use the pyramid: unit tests with fakes
  (FakeClient, FakeEmbedder) first, a few `integration` tests on real data, one smoke test per
  command in CI. Anything that costs money is marked `paid` and needs `--run-paid`.
- **Errors:** problems the user can fix become one `Error:` line (exit 1) in `cli.py`.
  Unexpected errors keep their traceback.
- **Claude API:** model and effort are configurable (`answer.py`); chunks are sent as
  `search_result` blocks with citations, so sources can only be chunks we sent.

## Security
- Never commit, read or print `.env` (holds `ANTHROPIC_API_KEY`). Pre-commit runs gitleaks.
- Pin versions: the law's download by SHA-256, the embedding model by git revision, GitHub
  Actions by commit hash. Keep `trust_remote_code=False`.
- CI token stays read-only. Prompts treat retrieved text as source material, never as
  instructions.
