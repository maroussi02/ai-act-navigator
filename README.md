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
# Wrote 954 chunks to …/data/chunks.jsonl: {'recital': 218, 'article': 686, 'annex': 50}
```

Each line of `data/chunks.jsonl` is one citable piece, e.g. `Art. 50(1)`, `Art. 5(1), point (h)`, `Art. 3, point (1)`, `Recital 12` or `Annex III, point 1`. Every chunk is short enough (max 1,800 characters) for the embedding model to read it completely.

Build the search index (embeds every chunk and stores it in a local Chroma database):

```bash
uv run python -m ai_act_navigator.index
# Indexed 954 chunks into …/data/chroma (0 truncated at 512 tokens) in 20s
```

An **embedding** is a list of numbers that captures what a text means, so texts with similar meaning end up close together. The model ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)) runs locally. It's downloaded once from Hugging Face (about 130 MB), and no text leaves your machine. Chroma is a library that stores the vectors as files in `data/chroma/`, with no account or server.

## Development

```bash
uv run pytest          # all tests
uv run pytest -m "not integration"   # only the fast unit tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

CI runs the secret scan, lint, format check, tests and a smoke test on every pull request.

### Tests

| Type | What it checks | Where |
|---|---|---|
| Unit | One function alone, small or fake input, milliseconds | `test_chunk.py`, `test_download.py`, `test_index.py`, `test_load_chunks.py`, `test_paths.py` |
| Integration | The parts together on real data or the real model | `test_full_text.py` (all parts found, no text lost, nothing cut off), `test_embed.py` |
| Smoke | The real commands run and produce output | CI steps "Smoke test" (chunk and index) |
