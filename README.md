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
uv run ai-act eval --search-only       # only check search, no Claude calls (free)
```

| Option | What it does | Default |
|---|---|---|
| `--quotes` | Show the exact passages Claude quoted under each source | off |
| `--k N` | How many chunks of the law Claude gets (1 to 50) | 5 |
| `--search` | `rerank` (hybrid + reranker), `hybrid` (meaning + keywords + named articles) or `vector` (meaning only, as in v1) | `rerank` |
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
3. **Embed and store** (`embed.py`, `store.py`): an **embedding** is a list of numbers that captures what a text means, so texts with similar meaning end up close together. The model ([BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)) runs locally, so no text leaves your machine. `ai-act index` downloads it once from Hugging Face (about 130 MB); `ai-act ask` then loads it in full offline mode, with no requests to Hugging Face. The vectors are saved in Chroma, a library that stores them as files in `data/chroma/`, with no account or server. Search compares the question with every stored vector (exact, about 8 ms) instead of using Chroma's approximate index, which sometimes skipped a chunk; so the same question always gets the same chunks.
4. **Search** (`search.py`, `keyword.py`): three ranked lists of candidates, merged into one:
   - **by meaning**: the question is embedded the same way, and the closest chunks come first;
   - **by keywords** (BM25, the classic search-engine formula): chunks sharing the question's words, rare words counting most. "Article 4", "Art. 4" and "article 4" all count as one word, so a reference matches however it is typed;
   - **named articles**: if the question names an article or annex ("What does Article 4 require?"), all its chunks are candidates.

   The lists are merged with *Reciprocal Rank Fusion*: each chunk scores 1 / (60 + its rank) in every list it is in, and the scores add up.
5. **Rerank** (`rerank.py`): a *reranker* (cross-encoder) reads the question together with each of the 30 best candidates and scores how well that chunk answers it. That's slower than search but more precise, so it puts the key paragraph first. The model ([cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), Apache-2.0, about 90 MB) runs locally and is pinned like the embedding model; `ai-act index` downloads it once and `ai-act ask` loads it offline. It takes about 0.15 s per question on an Apple M5. A reranker doesn't know that "Article 4" means the chunk cited as "Art. 4", so an article the question names always keeps one of the 5 places. The 5 best chunks go to Claude.
6. **Answer** (`answer.py`): the chunks go to Claude as *search result* blocks with citations enabled. Every numbered source in the answer is one of those chunks, with the exact quoted passage. (The prose could still name another article; the prompt forbids it, and AN-002 measures it.) If the chunks don't answer the question, Claude says so instead of guessing.
   The instructions Claude gets live in one versioned file, [`prompts.toml`](src/ai_act_navigator/prompts.toml). From now on, every eval result records the prompt version and a fingerprint (SHA-256) of its exact text, and a test fails if the text changes without a new version. (The v1 baseline predates the fingerprint; v1's is listed in `tests/test_prompts.py`.)

A question costs about $0.01 at list prices.

## Quality so far

### Golden set v1 baseline (AN-002)

[20 questions](evals/golden-v1.json) with known correct answers and source articles, each checked by hand against the official text. `uv run ai-act eval` asks all 20 and scores them in code, no LLM judge:

- An answer to a question the AI Act covers passes if it cites one of the expected articles. Article level is enough: `Art. 50(1)` counts for `Art. 50`.
- An answer to a question the AI Act doesn't cover passes if it says the excerpts don't answer it.

**v1 score: 14 of 20** (Sonnet 5.5, effort medium, k = 5, prompt v1, run on 2026-10-27, cost $0.20, [results file](evals/results/2026-10-27-171944-claude-sonnet-5-5-medium-k5.json)):

| Question type | Passed |
|---|---|
| Exact reference ("What does Article 4 require?") | 1 of 5 |
| Plain language ("Does a chatbot have to tell users it is an AI?") | 8 of 10 |
| Not in the AI Act ("What is the VAT rate in Germany?") | 5 of 5 |

**All 6 failures are search misses:** the expected article was never among the 5 chunks sent to Claude (search found it for 9 of 15 questions). 5 of the 6 answers say the excerpts don't cover the question; the sixth (emotion recognition at work) answers from related passages instead. No answer declined when the article was there, and none named an article from outside the excerpts. Search is the bottleneck, which is what AN-003 fixes.

Limits of the automatic score: it checks *which* articles are cited, not whether each claim is right (that is the hand check below, and the LLM judge in AN-004). A decline is recognised by its opening sentence ("The excerpts don't answer this…").

### Score after each change (AN-003)

Same 20 questions, same model (Sonnet 5.5, effort medium, k = 5), prompt v1. "Search found" counts the 15 questions the AI Act answers.

| Change | Score | Exact reference | Plain language | Not in the AI Act | Search found | Results |
|---|---|---|---|---|---|---|
| v1: meaning search | 14/20 | 1/5 | 8/10 | 5/5 | 9/15 | [file](evals/results/2026-10-27-171944-claude-sonnet-5-5-medium-k5.json) |
| Combined search (AN-003b) | 15/20 | 5/5 | 5/10 | 5/5 | 14/15 | [file](evals/results/2026-10-27-194440-claude-sonnet-5-5-medium-k5-hybrid.json) |
| Reranker (AN-003c) | 18/20 | 5/5 | 8/10 | 5/5 | 14/15 | [file](evals/results/2026-10-27-204856-claude-sonnet-5-5-medium-k5-rerank.json) |
| Exact search + refusal check (AN-003d) | **18/20** | **5/5** | **8/10** | **5/5** | **14/15** | [file](evals/results/2026-10-27-210215-claude-sonnet-5-5-medium-k5-rerank.json) |

**Combined search:** every question that names an article now finds and cites it (1/5 → 5/5). Emotion recognition at work (P8) now passes too. But 4 plain-language questions that passed in v1 now fail (P1, P3, P6, P7). All 4 decline although their article was sent: search found the right *article* but not its key *paragraph*. Example: "Does a chatbot have to tell users it is an AI?" gets Art. 50(4) (deepfakes) instead of Art. 50(1) (chatbots), and the answer says the excerpts don't include the rule. Merging rewards chunks that several lists rank fairly high, which can push out meaning search's single best hit: those paragraphs ended up at ranks 6 to 17. Putting the best of the candidates first is the reranker's job (next step). The remaining search miss is P9 ("When do the high-risk rules start to apply?"): nothing in its wording points to Art. 113.

**Reranker:** P3, P6 and P7 pass again, because their key paragraph (Annex III, Art. 53(1), Art. 3, point (3)) is back in the top 5, and the fines question now cites Art. 99(3), the EUR 35 million rule. Two failures remain. P1 (chatbots): the reranker still prefers Art. 50(4) (deepfakes) over Art. 50(1), and the answer declines. P9: the search miss above. The model was picked on these same questions, by a rule fixed before comparing: the small model unless the large one ([BAAI/bge-reranker-base](https://huggingface.co/BAAI/bge-reranker-base), 1.1 GB) brought back at least 2 more key paragraphs; both brought back 3 of 4.

**Exact search:** Chroma's approximate index sometimes skipped a chunk, so the same question could get different chunks from run to run (Recital 119 for the chatbot question was missing in 2 of 4 runs). Search is now exact and gives the same result every run. The scores didn't change.

**Refusal ("I don't know"):** questions the AI Act doesn't answer were declined 5 of 5 times in all four runs in the table above, and an answer whose key article wasn't found says so instead of guessing (P9: "The excerpts don't state the general application date for high-risk AI systems, so I can't give you that date from them"). So no extra refusal step was added. A cutoff on the reranker's score (skip Claude when no chunk scores high enough) was tested and rejected: the best scores of real and out-of-scope questions overlap. "Do deepfakes have to be labelled?" (answered correctly) scores −6.37, while three of the five out-of-scope questions score between 0.80 and 3.43, so any cutoff that catches them would also refuse real questions.

The search columns are free to check: `uv run ai-act eval --search-only` (add `--search vector` for v1). Every step is measured on the same 20 questions, so a better score can partly reflect fitting them. To limit that, the search settings are standard values (BM25 k1 = 1.5, b = 0.75; fusion constant 60), not tuned on these questions. Three choices did look at them: the reranker model (see above), how many candidates it rescores (30, because the key paragraphs were at ranks 6 to 17), and keeping named articles in the top 5 (the reranker dropped Art. 4 and Art. 113). A held-out set of new questions would show how much of the gain is real.

### Hand check v1

[Hand check v1](docs/hand-check-v1.md): 10 real questions, each source checked against what the answer claims. **Result: 10 of 10 answers supported by their citations.** Known limits, to be fixed in the next tickets:

- **Search misses (AN-003):** "Is emotion recognition at work allowed?" and "When do the high-risk rules apply?" don't retrieve the key article (Art. 5(1)(f), Art. 113). The answers say so instead of guessing.
- **Long lists are incomplete:** only 5 chunks are sent, so "Which practices are prohibited?" gets 5 of the 8. Fixed by searching wider and reranking (AN-003).
- **Gaps and additions:** some answers leave out an exception or a rule that wasn't retrieved (e.g. the SME fines rule, Art. 99(6)), or mention an article they didn't cite (Art. 50(2), Art. 113). `ai-act eval` counts mentions of articles that were sent but not cited, and of articles that weren't in the excerpts at all.

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
| Unit | One function alone, small or fake input, milliseconds | `test_chunk.py`, `test_download.py`, `test_load_model.py`, `test_index.py`, `test_store.py`, `test_search.py`, `test_answer.py` (fake Claude client), `test_cli.py`, `test_paid_guard.py`, `test_load_chunks.py`, `test_paths.py`, `test_golden.py` (also fails until every golden-set question is checked by hand), `test_evaluate.py`, `test_prompts.py` (also fails if the prompt text changes without a new version), `test_refs.py`, `test_keyword.py`, `test_rerank.py` (fake model) |
| Integration | The parts together on real data or the real model | `test_full_text.py` (all parts found, no text lost, nothing cut off), `test_embed.py`, `test_named_articles.py` (a question naming an article finds it, even with meaning search switched off, before and after reranking), `test_rerank_real.py` (the real reranker puts the key paragraph first) |
| Smoke | The real commands run and produce output | CI steps "Smoke test" (`ai-act --help`, `ai-act index`) |
| Paid smoke | One real question through the full pipeline cites Art. 50(1) | `test_live_answer.py`, run by hand with `--run-paid -m paid` |
