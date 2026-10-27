# Review guide

What a reviewer (human or AI) should look for in this repo. `CLAUDE.md` holds the general
rules; this file says what "broken" means here, so a review spends its time on real bugs.

## What failure looks like here

- **It cites wrongly.** An answer points to the wrong article, or a source that was not one
  of the chunks we sent. Sources must only come from the `search_result` blocks in
  `answer.py`.
- **It obeys the law text.** Retrieved text is source material, never instructions. Flag
  any change that lets chunk text steer the prompt or the program.
- **It loosens a pin.** The law's SHA-256 (`download.py`), the embedding model's git
  revision (`embed.py`), or a GitHub Action's commit hash is removed, changed without a
  reason, or replaced by a tag. `trust_remote_code` must stay `False`.
- **It costs money by accident.** A test that calls the real API without the `paid` marker,
  or CI running `--run-paid`.
- **It widens CI rights.** The workflow token gets write access, or a secret reaches a step
  that does not need it.
- **The tests were bent to fit.** Assertions changed to match new behaviour instead of the
  code being fixed. Review test diffs first.
- **It lies.** A comment, docstring or README sentence that is not true of the code or the
  law.

## Skip

- `uv.lock` (generated), `data/` (downloaded and generated), `evals/results/` (eval output).

## How to report

- Every finding needs `file:line` and evidence (a command you ran, or the line that shows
  it). No guesses from a name.
- At most 3 nits (style, naming, wording). Bugs first.
- Start the summary with the number of findings. If there is nothing to fix, say so plainly.
