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

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="full text not downloaded"),
]


def test_all_parts_of_the_law_are_found():
    chunks = parse(DEFAULT_PATH.read_text(encoding="utf-8"))
    # A chunk id starts with its parent, e.g. "art-50-1" belongs to article 50.
    parents = {(c.type, c.id.split("-")[1]) for c in chunks}
    assert Counter(kind for kind, _ in parents) == {"article": 113, "recital": 180, "annex": 13}


def test_no_text_is_lost():
    """Every paragraph of every article, annex and recital must end up in some chunk."""
    html = DEFAULT_PATH.read_text(encoding="utf-8")
    chunks = parse(html)
    # Texts first, in order: a paragraph split into sentence groups ("-s1", "-s2") sits in
    # consecutive chunks, so joining the texts with spaces puts it back together.
    all_chunk_text = " ".join(c.text for c in chunks) + " " + " ".join(c.title for c in chunks)

    soup = BeautifulSoup(html, "lxml-xml")
    heading = re.compile(r"(Article \d+|ANNEX [IVXL]+)")  # only used as labels, not as text
    missing = [
        text
        for part in soup.find_all("div", id=re.compile(r"^(art_\d+|anx_[IVXL]+|rct_\d+)$"))
        for p in part.find_all("p")
        if (text := re.sub(r"\s+", " ", p.get_text(" ", strip=True)).strip())
        and not heading.fullmatch(text)
        and text not in all_chunk_text
    ]
    assert missing == []


def test_no_chunk_is_cut_off_by_the_embedding_model():
    """Every chunk must fit the model's 512-token limit, or its end is never searchable."""
    from ai_act_navigator.embed import Embedder
    from ai_act_navigator.index import text_to_embed

    chunks = parse(DEFAULT_PATH.read_text(encoding="utf-8"))
    assert Embedder().count_truncated([text_to_embed(c) for c in chunks]) == 0
