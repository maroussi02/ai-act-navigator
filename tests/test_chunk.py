"""UNIT TESTS for chunk.py.

They run parse() on a small excerpt of the real law (tests/fixtures/sample.html):
Recitals 1 and 53, Articles 3, 4, 5 and 50, Annexes III and VI. No internet, runs in milliseconds.
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


def test_no_chunk_is_longer_than_the_limit(chunks):
    assert max(len(c.text) for c in chunks) <= MAX_CHARS


def test_long_paragraph_is_split_by_point(chunks):
    # Art. 5(1) lists the banned AI practices (a)-(h). As one chunk, the model would only
    # read the first 512 tokens and never "see" points (g) and (h).
    refs = [c.ref for c in chunks if c.ref.startswith("Art. 5(1)")]
    assert refs == [f"Art. 5(1), point ({letter})" for letter in "abcdefgh"]
    point_h = by_ref(chunks, "Art. 5(1), point (h)")[0]
    assert point_h.text.startswith("1. The following AI practices shall be prohibited:")
    assert "‘real-time’ remote biometric identification" in point_h.text


def test_long_text_without_points_is_split_by_sentences(chunks):
    parts = by_ref(chunks, "Recital 53")  # about 4,400 characters, no points to split by
    assert len(parts) > 1
    assert [p.id for p in parts] == [f"rct-53-s{i}" for i in range(1, len(parts) + 1)]
    assert all(p.text.endswith((".", ";")) for p in parts)  # cut between sentences only


# parametrize = run the same test once per (marker, expected) pair below.
@pytest.mark.parametrize(
    ("marker", "expected"),
    [("(a)", "point (a)"), ("(12)", "point (12)"), ("3.", "point 3")],
)
def test_point_labels(marker, expected):
    table = BeautifulSoup(f"<table><tr><td><p>{marker}</p></td></tr></table>", "lxml-xml").table
    assert _label(table) == expected
