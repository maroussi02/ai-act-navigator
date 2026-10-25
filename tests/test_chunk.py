"""UNIT TESTS for chunk.py.

They run parse() on a small excerpt of the real law (tests/fixtures/sample.html):
Recital 1, Articles 3, 4 and 50, Annexes III and VI. No internet, runs in milliseconds.
If one fails, the test name tells you which chunking rule broke.
"""

from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from ai_act_navigator.chunk import MAX_CHARS, _label, parse
from ai_act_navigator.models import Chunk

FIXTURE = Path(__file__).parent / "fixtures" / "sample.html"


# A pytest "fixture" prepares data for tests. Any test with a `chunks` argument receives
# this list. scope="module" = parse the file once and share it across all tests in this file.
@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    return parse(FIXTURE.read_text(encoding="utf-8"))


def by_ref(chunks: list[Chunk], ref: str) -> list[Chunk]:
    """Helper: all chunks with this citation, e.g. "Art. 50(1)"."""
    return [c for c in chunks if c.ref == ref]


def test_article_paragraph_is_its_own_chunk(chunks):
    matches = by_ref(chunks, "Art. 50(1)")
    assert len(matches) == 1
    chunk = matches[0]
    assert chunk.text.startswith("1. Providers shall ensure")
    assert (
        chunk.title == "Transparency obligations for providers and deployers of certain AI systems"
    )
    assert chunk.url.endswith("#art_50")


def test_paragraphs_do_not_leak_into_each_other(chunks):
    first = by_ref(chunks, "Art. 50(1)")[0]
    assert "2. Providers of AI systems" not in first.text


def test_short_article_without_paragraphs_stays_whole(chunks):
    assert len(by_ref(chunks, "Art. 4")) == 1


def test_long_article_is_split_by_point_with_intro(chunks):
    definition = by_ref(chunks, "Art. 3, point (1)")[0]
    # The intro is repeated so each definition makes sense on its own.
    assert definition.text.startswith("For the purposes of this Regulation")
    assert "‘AI system’ means" in definition.text


def test_recital(chunks):
    recital = by_ref(chunks, "Recital 1")[0]
    assert recital.type == "recital"
    assert recital.text.startswith("(1) The purpose of this Regulation")


def test_long_annex_is_split_by_point(chunks):
    biometrics = by_ref(chunks, "Annex III, point 1")[0]
    assert biometrics.title == "High-risk AI systems referred to in Article 6(2)"
    assert "Biometrics" in biometrics.text
    assert len(by_ref(chunks, "Annex VI")) == 1  # short annex stays whole


def test_every_chunk_has_text_and_a_unique_id(chunks):
    assert all(c.text for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_split_parts_respect_size_limit_roughly(chunks):
    # Parts can exceed MAX_CHARS only when a single point is that long on its own.
    split_parts = [c for c in chunks if "point" in c.ref]
    assert max(len(c.text) for c in split_parts) < 2 * MAX_CHARS


# parametrize = run the same test once per (marker, expected) pair below.
@pytest.mark.parametrize(
    ("marker", "expected"),
    [("(a)", "point (a)"), ("(12)", "point (12)"), ("3.", "point 3")],
)
def test_point_labels(marker, expected):
    table = BeautifulSoup(f"<table><tr><td><p>{marker}</p></td></tr></table>", "lxml-xml").table
    assert _label(table) == expected
