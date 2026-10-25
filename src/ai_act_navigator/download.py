"""Download the official AI Act text (Regulation (EU) 2024/1689) and prove it's genuine.

The EUR-Lex website blocks scripted downloads, so we fetch the same official text
from the EU Publications Office, which serves it as clean XHTML.

Two protections, because this text later goes into the LLM's prompts:
- HTTPS: nobody on the network can read or change the file on its way to us.
- SHA-256 check: the published law never changes, so its "fingerprint" is fixed. Any file
  with a different fingerprint (tampered, half-downloaded, wrong version) is rejected.
"""

import hashlib
from pathlib import Path

import httpx

from ai_act_navigator.paths import DATA_DIR

# Direct HTTPS link to the English XHTML file. (The shorter ".../celex/32024R1689" link
# redirects over plain http://, so we skip it.)
SOURCE_URL = (
    "https://publications.europa.eu/resource/cellar/"
    "dc8116a1-3fe6-11ef-865a-01aa75ed71a1.0006.03/DOC_1"
)
# Fingerprint of the official file (Official Journal version of 12 July 2024).
SOURCE_SHA256 = "8f0b656302f9864cc87e040c371f209a9d65ae1a6cecc25ca5eb737e872d721a"

DEFAULT_PATH = DATA_DIR / "raw" / "ai_act.html"  # can always be downloaded again
TIMEOUT_SECONDS = 60  # give up instead of hanging forever if the server doesn't answer


class IntegrityError(Exception):
    """The downloaded file is not the official text we expect."""


def sha256_of(data: bytes) -> str:
    """The SHA-256 fingerprint of some bytes, as 64 hex characters."""
    return hashlib.sha256(data).hexdigest()


def download(
    dest: Path = DEFAULT_PATH, url: str = SOURCE_URL, expected_sha256: str = SOURCE_SHA256
) -> Path:
    """Save the official text to `dest` and return its path.

    Skips the download if `dest` already holds the right file, so running the pipeline
    again is fast and doesn't hit the EU server every time. A file with the wrong
    fingerprint is replaced. (`url` and `expected_sha256` can be changed for tests.)
    """
    if dest.exists() and sha256_of(dest.read_bytes()) == expected_sha256:
        return dest

    response = httpx.get(
        url,
        # Ask for the XHTML version in English (the server has many formats and languages).
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        follow_redirects=False,  # a redirect could lead to plain http://, so we don't follow
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()  # stop with a clear error on 404, 503, ... instead of saving it

    actual = sha256_of(response.content)
    if actual != expected_sha256:
        raise IntegrityError(f"Expected SHA-256 {expected_sha256}, got {actual}. Not saved.")

    # Write to a temporary file first, then rename: a crash mid-write can never leave a
    # half-written file behind at `dest`.
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    tmp.write_bytes(response.content)
    tmp.replace(dest)
    return dest


# Runs only when started as a program: uv run python -m ai_act_navigator.download
if __name__ == "__main__":
    print(f"Saved to {download()}")
