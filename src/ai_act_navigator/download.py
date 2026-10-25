"""Download the official AI Act text (Regulation (EU) 2024/1689).

The EUR-Lex website blocks scripted downloads, so we fetch the same official text
from the EU Publications Office, which serves it as clean XHTML.
"""

from pathlib import Path

import httpx

SOURCE_URL = "http://publications.europa.eu/resource/celex/32024R1689"
DEFAULT_PATH = Path("data/raw/ai_act.html")
TIMEOUT_SECONDS = 60


def download(dest: Path = DEFAULT_PATH) -> Path:
    """Save the English XHTML text to `dest`. Skips the download if the file exists."""
    if dest.exists():
        return dest

    response = httpx.get(
        SOURCE_URL,
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        follow_redirects=True,
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    return dest


if __name__ == "__main__":
    print(f"Saved to {download()}")
