# AI Act Navigator

Ask questions about the EU AI Act and get answers that cite the exact article, with a test set and quality gate that prove the answers are right.

> Work in progress. Tickets: [GitHub issues](https://github.com/maroussi02/ai-act-navigator/issues).

## Source text

The official English text of [Regulation (EU) 2024/1689](https://eur-lex.europa.eu/eli/reg/2024/1689/oj) (AI Act), as published in the Official Journal on 12 July 2024. It is downloaded from the EU Publications Office.

## Setup

1. Install [uv](https://docs.astral.sh/uv/) (it installs the right Python version for you).
2. Install dependencies and the secret-scan hook:
   ```bash
   uv sync
   uv run pre-commit install
   ```
3. Copy `.env.example` to `.env` and add your Anthropic API key. `.env` is git-ignored, so never commit it.

## Usage

Split the AI Act into citable chunks:

```bash
uv run python -m ai_act_navigator.chunk
# Wrote 821 chunks to data/chunks.jsonl: {'recital': 180, 'article': 600, 'annex': 41}
```

Each line of `data/chunks.jsonl` is one citable piece, e.g. `Art. 50(1)`, `Art. 3, point (1)`, `Recital 12` or `Annex III, point 1`.

## Development

```bash
uv run pytest          # tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

CI runs the secret scan, lint, format check and tests on every pull request.
