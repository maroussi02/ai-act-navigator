"""Split the official AI Act text into citable chunks.

Chunking follows the law's own structure instead of a fixed token size:
- article with numbered paragraphs -> one chunk per paragraph ("Art. 50(1)")
- long article without paragraphs  -> one chunk per point ("Art. 3, point (1)")
- recital                          -> one chunk each ("Recital 1")
- long annex                       -> one chunk per section, or per point if it has no sections
- anything short                   -> one chunk for the whole article or annex
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ai_act_navigator.download import download
from ai_act_navigator.models import Chunk

EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
OUTPUT_PATH = Path("data/chunks.jsonl")

# About 750 tokens. Articles and annexes longer than this are split into smaller parts.
MAX_CHARS = 3000

ARTICLE_ID = re.compile(r"^art_(\d+)$")
PARAGRAPH_ID = re.compile(r"^\d{3}\.(\d{3})$")  # "050.001" = Article 50, paragraph 1
RECITAL_ID = re.compile(r"^rct_(\d+)$")
ANNEX_ID = re.compile(r"^anx_([IVXL]+)$")


def parse(html: str) -> list[Chunk]:
    """Turn the official XHTML text into chunks, in reading order."""
    soup = BeautifulSoup(html, "lxml-xml")
    return [*_recitals(soup), *_articles(soup), *_annexes(soup)]


def _recitals(soup: BeautifulSoup) -> list[Chunk]:
    chunks = []
    for div in soup.find_all("div", id=RECITAL_ID):
        number = RECITAL_ID.match(div["id"]).group(1)
        chunks.append(
            Chunk(
                id=f"rct-{number}",
                type="recital",
                ref=f"Recital {number}",
                title="",
                text=_text(div),
                url=_url(div["id"]),
            )
        )
    return chunks


def _articles(soup: BeautifulSoup) -> list[Chunk]:
    chunks = []
    for div in soup.find_all("div", id=ARTICLE_ID):
        number = ARTICLE_ID.match(div["id"]).group(1)
        title = _text(div.find("p", class_="oj-sti-art"))
        url = _url(div["id"])

        paragraphs = div.find_all("div", id=PARAGRAPH_ID, recursive=False)
        if paragraphs:
            for paragraph in paragraphs:
                para = int(PARAGRAPH_ID.match(paragraph["id"]).group(1))
                chunks.append(
                    Chunk(
                        id=f"art-{number}-{para}",
                        type="article",
                        ref=f"Art. {number}({para})",
                        title=title,
                        text=_text(paragraph),
                        url=url,
                    )
                )
            continue

        body = [
            el
            for el in div.find_all(recursive=False)
            if not (_has_class(el, "oj-ti-art") or _has_class(el, "eli-title"))
        ]
        parts = _split_long(body, is_boundary=lambda el: el.name == "table")
        if not parts:
            chunks.append(
                Chunk(f"art-{number}", "article", f"Art. {number}", title, _text_of(body), url)
            )
        for i, (label, text) in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    id=f"art-{number}-p{i}",
                    type="article",
                    ref=f"Art. {number}, point {label}",
                    title=title,
                    text=text,
                    url=url,
                )
            )
    return chunks


def _annexes(soup: BeautifulSoup) -> list[Chunk]:
    chunks = []
    for div in soup.find_all("div", id=ANNEX_ID):
        numeral = ANNEX_ID.match(div["id"]).group(1)
        headings = div.find_all("p", class_="oj-doc-ti", recursive=False)
        # First heading is "ANNEX III", the second (if any) is the annex's title.
        title = _text(headings[1]) if len(headings) > 1 else ""
        url = _url(div["id"])
        body = [el for el in div.find_all(recursive=False) if not _has_class(el, "oj-doc-ti")]

        has_sections = any(_is_section_heading(el) for el in body)
        boundary = _is_section_heading if has_sections else (lambda el: el.name == "table")
        parts = _split_long(body, is_boundary=boundary)
        if not parts:
            chunks.append(
                Chunk(
                    f"anx-{numeral.lower()}",
                    "annex",
                    f"Annex {numeral}",
                    title,
                    _text_of(body),
                    url,
                )
            )
        for i, (label, text) in enumerate(parts, start=1):
            chunks.append(
                Chunk(
                    id=f"anx-{numeral.lower()}-p{i}",
                    type="annex",
                    ref=f"Annex {numeral}, {label}",
                    title=title,
                    text=text,
                    url=url,
                )
            )
    return chunks


def _split_long(body: list[Tag], is_boundary: Callable[[Tag], bool]) -> list[tuple[str, str]]:
    """Split a long body into (label, text) parts. Returns [] if it's short enough to keep whole.

    Elements before the first boundary are an intro (e.g. "the following definitions apply:")
    and are repeated at the start of every part, so each part makes sense on its own.
    Consecutive section headings are merged into one part.
    """
    if len(_text_of(body)) <= MAX_CHARS or not any(is_boundary(el) for el in body):
        return []

    intro: list[Tag] = []
    groups: list[list[Tag]] = []
    for el in body:
        starts_new = is_boundary(el) and (
            not groups or any(not _is_section_heading(e) for e in groups[-1])
        )
        if starts_new:
            groups.append([el])
        elif groups:
            groups[-1].append(el)
        else:
            intro.append(el)

    return [(_label(group[0]), _text_of(intro + group)) for group in groups]


def _label(el: Tag) -> str:
    """Citation label for a part: "(1)", "point 3" or "Section A"."""
    if el.name == "table":
        marker = _text(el.find("td"))  # first cell holds the point marker, e.g. "(a)" or "3."
        return (
            f"({marker.strip('().')})" if marker.startswith("(") else f"point {marker.strip('.')}"
        )

    heading = _text(el)
    if section := re.match(r"Section \w+", heading):
        return section.group(0)
    if number := re.match(r"(\d+)\.", heading):
        return f"point {number.group(1)}"
    return heading[:40]


def _is_section_heading(el: Tag) -> bool:
    return _has_class(el, "oj-ti-grseq-1")


def _has_class(el: Tag, name: str) -> bool:
    # In XML mode BeautifulSoup keeps "class" as a plain string, not a list.
    return name in (el.get("class") or "").split()


def _text(el: Tag | None) -> str:
    """Visible text with all whitespace (incl. non-breaking spaces) collapsed to single spaces."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def _text_of(elements: list[Tag]) -> str:
    return " ".join(t for t in (_text(el) for el in elements) if t)


def _url(anchor: str) -> str:
    return f"{EURLEX_URL}#{anchor}"


def main() -> None:
    html = download().read_text(encoding="utf-8")
    chunks = parse(html)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")

    counts = Counter(chunk.type for chunk in chunks)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}: {dict(counts)}")


if __name__ == "__main__":
    main()
