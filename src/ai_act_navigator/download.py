"""Download the official AI Act text (Regulation (EU) 2024/1689).

The EUR-Lex website blocks scripted downloads, so we fetch the same official text
from the EU Publications Office, which serves it as clean XHTML.
"""

from pathlib import Path

import httpx

from ai_act_navigator.paths import DATA_DIR

SOURCE_URL = "http://publications.europa.eu/resource/celex/32024R1689"
DEFAULT_PATH = DATA_DIR / "raw" / "ai_act.html"  # can always be downloaded again
TIMEOUT_SECONDS = 60  # give up instead of hanging forever if the server doesn't answer


def download(dest: Path = DEFAULT_PATH) -> Path:
    """Save the English XHTML text to `dest` and return its path.

    Skips the download if the file already exists, so running the pipeline again is fast
    and doesn't hit the EU server every time. Delete the file to force a fresh download.
    """
    if dest.exists():
        return dest

    response = httpx.get(
        SOURCE_URL,
        # Ask for the XHTML version in English (the server has many formats and languages).
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        follow_redirects=True,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()  # stop with a clear error on 404, 503, ... instead of saving it

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


# Runs only when started as a program: uv run python -m ai_act_navigator.download
if __name__ == "__main__":
    print(f"Saved to {download()}")
