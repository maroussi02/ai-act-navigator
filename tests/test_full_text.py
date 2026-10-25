"""INTEGRATION TESTS: parse() on the full, real AI Act (1.2 MB).

They catch problems the small sample in the unit tests can't, e.g. an article with an
unusual structure. Slower (about a second) and need the downloaded file. Skipped unless
it was downloaded first (CI downloads it before running tests):

uv run python -m ai_act_navigator.download
"""

import re
from collections import Counter

import pytest
from bs4 import BeautifulSoup

from ai_act_navigator.chunk import parse
from ai_act_navigator.download import DEFAULT_PATH

pytestmark = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="full text not downloaded")


def test_all_parts_of_the_law_are_found():
    chunks = parse(DEFAULT_PATH.read_text(encoding="utf-8"))
    # A chunk id starts with its parent, e.g. "art-50-1" belongs to article 50.
    parents = {(c.type, c.id.split("-")[1]) for c in chunks}
    assert Counter(kind for kind, _ in parents) == {"article": 113, "recital": 180, "annex": 13}


def test_no_text_is_lost():
    """Every paragraph of every article and annex must end up in some chunk."""
    html = DEFAULT_PATH.read_text(encoding="utf-8")
    chunks = parse(html)
    all_chunk_text = " ".join(f"{c.title} {c.text}" for c in chunks)

    soup = BeautifulSoup(html, "lxml-xml")
    heading = re.compile(r"(Article \d+|ANNEX [IVXL]+)")  # only used as labels, not as text
    missing = [
        text
        for part in soup.find_all("div", id=re.compile(r"^(art_\d+|anx_[IVXL]+)$"))
        for p in part.find_all("p")
        if (text := re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip())
        and not heading.fullmatch(text)
        and text not in all_chunk_text
    ]
    assert missing == []
