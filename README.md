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
3. Copy `.env.example` to `.env` and add your Anthropic API key (only needed to ask questions). `.env` is git-ignored, so never commit it.

## Quickstart

```bash
uv run ai-act index                    # download the law and build the search index (~20 s, once)
uv run ai-act ask "Does a chatbot have to tell users it is an AI?"
uv run ai-act eval                     # score the 20-question golden set (~$0.20)
```

| Option | What it does | Default |
|---|---|---|
| `--quotes` | Show the exact passages Claude quoted under each source | off |
| `--k N` | How many chunks of the law Claude gets (1 to 50) | 5 |
| `--model` | Claude model | `claude-sonnet-5-5` |
| `--effort` | How much Claude thinks first: `low`, `medium`, `high`, `xhigh`, `max` | `medium` |

`uv run ai-act --help` lists everything. Problems you can fix yourself (missing key, empty index, model not downloaded, no internet, rate limit) print one line starting with `Error:` and exit with code 1. Wrong options print the usage and exit with code 2.

## Examples

Three real answers from the [hand check](docs/hand-check-v1.md), shortened for the README: formatting removed, lines re-wrapped, and some parts cut (marked "…"). The full answers are in the hand check.

**1. Does a chatbot have to tell users it is an AI?**

```text
Yes, in general. Providers must ensure that AI systems intended to interact directly with natural
persons are designed and developed so that the people concerned are informed they are interacting
with an AI system. [1] A chatbot falls squarely within that description.

Exceptions and conditions:
- Obvious from context: The duty doesn't apply if it is obvious from the point of view of a
  reasonably well-informed, observant and circumspect person … [1]
- Law enforcement: … unless those systems are available to the public to report a criminal
  offence. [1]
…
Sources:
  [1] Art. 50(1)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_50
  [2] Recital 132  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#rct_132

claude-sonnet-5-5 · 3246 tokens in, 547 out · cost $0.0120
```

**2. What are the maximum fines for breaking the prohibitions?**

```text
Maximum fines depend on who breaks the Article 5 prohibitions:

- Private operators (e.g. companies): … administrative fines of up to EUR 35,000,000 or, if the
  offender is an undertaking, up to 7% of its total worldwide annual turnover for the preceding
  financial year, whichever is higher. [1]
- EU institutions, bodies, offices and agencies: … administrative fines of up to EUR 1,500,000. [2]
…
Sources:
  [1] Art. 99(3)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_99
  [2] Art. 100(2)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_100
  [3] Art. 99(4)  https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689#art_99

claude-sonnet-5-5 · 2432 tokens in, 342 out · cost $0.0083
```

**3. What is the VAT rate in Germany?** (not in the AI Act)

```text
The excerpts don't answer this. They cover only administrative fines under the EU AI Act
(Articles 99 and 100), and nothing in them mentions VAT rates in Germany or any other tax matter.
…
Sources: none

claude-sonnet-5-5 · 2426 tokens in, 96 out · cost $0.0058
```

## How it works

1. **Download** (`download.py`): the official text over HTTPS, checked against a fixed SHA-256 fingerprint.
2. **Chunk** (`chunk.py`): split along the law's own structure into 954 citable pieces, e.g. `Art. 50(1)`, `Art. 5(1), point (h)`, `Recital 12`, `Annex III, point 1`. Each is short enough (max 1,800 characters) for the embedding model to read in full.
3. **Embed and store** (`embed.py`, `store.py`): an **embedding** is a list of numbers that captures what a text means, so texts with similar meaning end up close together. The model ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)) runs locally, so no text leaves your machine. `ai-act index` downloads it once from Hugging Face (about 130 MB); `ai-act ask` then loads it in full offline mode, with no requests to Hugging Face. The vectors are saved in Chroma, a library that stores them as files in `data/chroma/`, with no account or server.
4. **Search** (`search.py`): the question is embedded the same way, and the 5 closest chunks are picked.
5. **Answer** (`answer.py`): the chunks go to Claude as *search result* blocks with citations enabled. Every numbered source in the answer is one of those chunks, with the exact quoted passage. (The prose could still name another article; the prompt forbids it, and AN-002 measures it.) If the chunks don't answer the question, Claude says so instead of guessing.

A question costs about $0.01 at list prices.

## Quality so far

[Hand check v1](docs/hand-check-v1.md): 10 real questions, each source checked against what the answer claims. **Result: 10 of 10 answers supported by their citations.** Known limits, to be fixed in the next tickets:

- **Search misses (AN-003):** "Is emotion recognition at work allowed?" and "When do the high-risk rules apply?" don't retrieve the key article (Art. 5(1)(f), Art. 113). The answers say so instead of guessing.
- **Long lists are incomplete:** only 5 chunks are sent, so "Which practices are prohibited?" gets 5 of the 8. Fixed by searching wider and reranking (AN-003).
- **Gaps and additions:** some answers leave out an exception or a rule that wasn't retrieved (e.g. the SME fines rule, Art. 99(6)), or mention an article they didn't cite (Art. 50(2), Art. 113). AN-002 tracks these separately.
- **Measured, not guessed:** a 20-question golden set with an automatic score comes next (AN-002).

## Development

```bash
uv run pytest          # all tests
uv run pytest -m "not integration"   # only the fast unit tests
uv run pytest --run-paid -m paid     # one real Claude call (~$0.01); skipped unless --run-paid
uv run ruff check .    # lint
uv run ruff format .   # format
```

CI runs the secret scan, lint, format check, tests and smoke tests on every pull request.

### Tests

| Type | What it checks | Where |
|---|---|---|
| Unit | One function alone, small or fake input, milliseconds | `test_chunk.py`, `test_download.py`, `test_load_model.py`, `test_index.py`, `test_store.py`, `test_search.py`, `test_answer.py` (fake Claude client), `test_cli.py`, `test_paid_guard.py`, `test_load_chunks.py`, `test_paths.py`, `test_golden.py` (also fails until every golden-set question is checked by hand), `test_evaluate.py` |
| Integration | The parts together on real data or the real model | `test_full_text.py` (all parts found, no text lost, nothing cut off), `test_embed.py` |
| Smoke | The real commands run and produce output | CI steps "Smoke test" (`ai-act --help`, `ai-act index`) |
| Paid smoke | One real question through the full pipeline cites Art. 50(1) | `test_live_answer.py`, run by hand with `--run-paid -m paid` |
