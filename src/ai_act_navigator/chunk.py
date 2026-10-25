"""Split the official AI Act text into citable chunks.

Why chunks? Later we search the law for the pieces that answer a question and send only
those to the LLM. A chunk must be small enough to be precise, but big enough to make
sense on its own, and it must know exactly where it came from so we can cite it.

Chunking follows the law's own structure instead of a fixed token size:
- article with numbered paragraphs -> one chunk per paragraph ("Art. 50(1)")
- long article without paragraphs  -> one chunk per point ("Art. 3, point (1)")
- recital                          -> one chunk each ("Recital 1")
- long annex                       -> one chunk per section, or per point if it has no sections
- anything short                   -> one chunk for the whole article or annex

How to read the HTML: every part of the law sits in a <div> with a predictable id,
e.g. <div id="art_50"> for Article 50 or <div id="050.001"> for its paragraph 1.
We find those divs by their id and turn their text into Chunk objects.

Run it:  uv run python -m ai_act_navigator.chunk   (writes data/chunks.jsonl)
"""

import json
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from ai_act_navigator.download import download
from ai_act_navigator.models import Chunk, ChunkType
from ai_act_navigator.paths import DATA_DIR

# Base link to the official text. Adding "#art_50" jumps straight to Article 50.
EURLEX_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/HTML/?uri=OJ:L_202401689"
OUTPUT_PATH = DATA_DIR / "chunks.jsonl"

# About 750 tokens. Articles and annexes longer than this are split into smaller parts.
MAX_CHARS = 3000

# Patterns for the div ids. The part in (...) is "captured" so we can read the number out.
ARTICLE_ID = re.compile(r"^art_(\d+)$")  # "art_50"  -> 50
PARAGRAPH_ID = re.compile(r"^\d{3}\.(\d{3})$")  # "050.001" -> paragraph 001 of Article 50
RECITAL_ID = re.compile(r"^rct_(\d+)$")  # "rct_12"  -> 12
ANNEX_ID = re.compile(r"^anx_([IVXL]+)$")  # "anx_III" -> III (roman numeral)


def parse(html: str) -> list[Chunk]:
    """Turn the official XHTML text into chunks, in reading order.

    This is the only public function: HTML text in, list of chunks out. It reads no files
    and calls no network ("pure function"), which makes it easy to test.
    """
    # "lxml-xml" because the official file is XHTML (strict XML), not loose web HTML.
    soup = BeautifulSoup(html, "lxml-xml")
    return [*_recitals(soup), *_articles(soup), *_annexes(soup)]


def _recitals(soup: BeautifulSoup) -> list[Chunk]:
    """One chunk per recital (the numbered "why" paragraphs at the start of the law).

    Example: <div id="rct_1">(1) The purpose of this Regulation...</div>
          -> Chunk(id="rct-1", ref="Recital 1", text="(1) The purpose of this Regulation...")
    """
    chunks = []
    for div in soup.find_all("div", id=RECITAL_ID):
        number = RECITAL_ID.match(div["id"]).group(1)
        chunks.append(
            Chunk(
                id=f"rct-{number}",
                type="recital",
                ref=f"Recital {number}",
                title="",  # recitals have no title
                text=_text(div),
                url=_url(div["id"]),
            )
        )
    return chunks


def _articles(soup: BeautifulSoup) -> list[Chunk]:
    """Chunks for all 113 articles.

    Most articles have numbered paragraphs -> one chunk per paragraph ("Art. 50(1)").
    The rest are kept whole, or split by point if they're long (Art. 3 has 68 definitions).
    """
    chunks = []
    for div in soup.find_all("div", id=ARTICLE_ID):
        number = ARTICLE_ID.match(div["id"]).group(1)
        title = _text(div.find("p", class_="oj-sti-art"))  # e.g. "Transparency obligations..."
        url = _url(div["id"])

        # recursive=False: only look at direct children, not paragraphs nested deeper.
        paragraphs = div.find_all("div", id=PARAGRAPH_ID, recursive=False)
        if paragraphs:
            for paragraph in paragraphs:
                para = int(PARAGRAPH_ID.match(paragraph["id"]).group(1))  # "001" -> 1
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
            continue  # this article is done, go to the next one

        # No numbered paragraphs: take everything except the "Article 3" heading and the title.
        body = [
            el
            for el in div.find_all(recursive=False)
            if not (_has_class(el, "oj-ti-art") or _has_class(el, "eli-title"))
        ]
        chunks += _whole_or_split(
            id_prefix=f"art-{number}",
            chunk_type="article",
            ref=f"Art. {number}",
            title=title,
            url=url,
            body=body,
            is_boundary=_is_table,  # in articles, each point "(a)", "(1)" is its own <table>
        )
    return chunks


def _annexes(soup: BeautifulSoup) -> list[Chunk]:
    """Chunks for all 13 annexes (the lists and details at the end of the law).

    Short annexes stay whole. Long ones are split by section ("Annex VIII, Section A")
    if they have sections, otherwise by point ("Annex III, point 1").
    """
    chunks = []
    for div in soup.find_all("div", id=ANNEX_ID):
        numeral = ANNEX_ID.match(div["id"]).group(1)
        headings = div.find_all("p", class_="oj-doc-ti", recursive=False)
        # First heading is "ANNEX III", the second (if any) is the annex's title.
        title = _text(headings[1]) if len(headings) > 1 else ""
        url = _url(div["id"])
        body = [el for el in div.find_all(recursive=False) if not _has_class(el, "oj-doc-ti")]

        has_sections = any(_is_section_heading(el) for el in body)
        chunks += _whole_or_split(
            id_prefix=f"anx-{numeral.lower()}",
            chunk_type="annex",
            ref=f"Annex {numeral}",
            title=title,
            url=url,
            body=body,
            is_boundary=_is_section_heading if has_sections else _is_table,
        )
    return chunks


def _whole_or_split(
    *,  # the "*" forces callers to name every argument, so calls are easy to read
    id_prefix: str,
    chunk_type: ChunkType,
    ref: str,
    title: str,
    url: str,
    body: list[Tag],
    is_boundary: Callable[[Tag], bool],
) -> list[Chunk]:
    """One chunk for a short body, or one chunk per part for a long one.

    Shared by articles and annexes so the "keep whole or split" logic lives in one place.
    Example: ref="Art. 3" + parts [("point (1)", ...), ("point (2)", ...)]
          -> chunks "Art. 3, point (1)", "Art. 3, point (2)", ...
    """
    parts = _split_long(body, is_boundary)
    if not parts:
        return [
            Chunk(id=id_prefix, type=chunk_type, ref=ref, title=title, text=_text_of(body), url=url)
        ]
    return [
        Chunk(
            id=f"{id_prefix}-p{i}",  # "p" for part: art-3-p1, art-3-p2, ...
            type=chunk_type,
            ref=f"{ref}, {label}",
            title=title,
            text=text,
            url=url,
        )
        for i, (label, text) in enumerate(parts, start=1)
    ]


def _split_long(body: list[Tag], is_boundary: Callable[[Tag], bool]) -> list[tuple[str, str]]:
    """Split a long body into (label, text) parts. Returns [] if it's short enough to keep whole.

    `is_boundary` decides where a new part starts (a point's <table>, or a section heading).
    Passing it in as a function lets articles and annexes reuse this same code.

    Elements before the first boundary are an intro (e.g. "the following definitions apply:")
    and are repeated at the start of every part, so each part makes sense on its own.
    Consecutive section headings ("Section 1" directly followed by its subtitle) are merged
    into one part instead of creating an empty part.
    """
    if len(_text_of(body)) <= MAX_CHARS or not any(is_boundary(el) for el in body):
        return []

    intro: list[Tag] = []
    groups: list[list[Tag]] = []  # each group = the elements of one part
    for el in body:
        # Start a new part at a boundary, unless the current part is only headings so far.
        starts_new = is_boundary(el) and (
            not groups or any(not _is_section_heading(e) for e in groups[-1])
        )
        if starts_new:
            groups.append([el])
        elif groups:
            groups[-1].append(el)  # belongs to the current part
        else:
            intro.append(el)  # before the first boundary -> intro

    return [(_label(group[0]), _text_of(intro + group)) for group in groups]


def _label(el: Tag) -> str:
    """Citation label for the element that starts a part.

    <table> with "(a)" -> "point (a)",  <table> with "3." -> "point 3",
    heading "Section A — ..." -> "Section A",  heading "4. Control of..." -> "point 4".
    """
    if _is_table(el):
        marker = _text(el.find("td"))  # first cell holds the point marker, e.g. "(a)" or "3."
        return f"point {marker.rstrip('.')}"

    heading = _text(el)
    if section := re.match(r"Section \w+", heading):
        return section.group(0)
    if number := re.match(r"(\d+)\.", heading):
        return f"point {number.group(1)}"
    return heading[:40]  # fallback: first 40 characters of the heading


def _is_table(el: Tag) -> bool:
    """True if the element is a <table>. The official text puts each list point in one."""
    return el.name == "table"


def _is_section_heading(el: Tag) -> bool:
    """True for section headings inside annexes, e.g. "Section A — Information to be..."."""
    return _has_class(el, "oj-ti-grseq-1")


def _has_class(el: Tag, name: str) -> bool:
    """True if the element has the CSS class `name`."""
    # In XML mode BeautifulSoup keeps "class" as a plain string, not a list.
    return name in (el.get("class") or "").split()


def _text(el: Tag | None) -> str:
    """Visible text with all whitespace (incl. non-breaking spaces) collapsed to single spaces.

    Example: "1.\\xa0\\xa0 Providers  shall" -> "1. Providers shall"
    """
    if el is None:
        return ""
    return re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()


def _text_of(elements: list[Tag]) -> str:
    """Text of several elements joined with spaces, skipping empty ones."""
    return " ".join(t for t in (_text(el) for el in elements) if t)


def _url(anchor: str) -> str:
    """Link that jumps to one part of the official text, e.g. "...#art_50"."""
    return f"{EURLEX_URL}#{anchor}"


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    """Save chunks as JSONL: one JSON object per line. Easy to read line by line and to grep."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            # ensure_ascii=False keeps characters like ‘ ’ readable instead of ‘.
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path: Path) -> list[Chunk]:
    """Read chunks back from a JSONL file written by save_chunks.

    Chunk(**data) turns {"id": ..., "ref": ...} into Chunk(id=..., ref=...).
    """
    with path.open(encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f if line.strip()]


def main() -> None:
    """Download the law (if needed), chunk it, and save the chunks to OUTPUT_PATH."""
    html = download().read_text(encoding="utf-8")
    chunks = parse(html)
    save_chunks(chunks, OUTPUT_PATH)

    counts = Counter(chunk.type for chunk in chunks)
    print(f"Wrote {len(chunks)} chunks to {OUTPUT_PATH}: {dict(counts)}")


# Runs main() only when started as a program (python -m ...), not when imported by tests.
if __name__ == "__main__":
    main()
