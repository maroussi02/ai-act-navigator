"""Find references to articles and annexes of the AI Act, e.g. "Article 50" or "Annex III".

Shared by search (a question that names an article should find it), the eval (which
articles does an answer name?) and the golden set (which article does a chunk belong to?).
Kept in its own module so search doesn't depend on eval code.
"""

import re

# A mention in free text: "Art. 50", "Art 50", "article 6(1)", "Articles 99", "annex iii".
# Case-insensitive, because people type "article 4" as often as "Article 4".
MENTION = re.compile(r"\b(?:Art\.?|Articles?)\s+(\d+)|\bAnnex\s+([IVXL]+)\b", re.IGNORECASE)
# What follows a mention of *another* law, e.g. "Article 4(3) of Directive (EU) 2019/790".
_OTHER_LAW = re.compile(
    r"(?:\(\w+\))*(?:,? points? \(\w+\))?\s+of\s+(?:Directive|Regulation)", re.IGNORECASE
)
# The part of a chunk ref that names the article or annex, e.g. "Art. 5" in
# "Art. 5(1), point (f)", "Annex III" in "Annex III, point 1". Chunk refs are written by
# chunk.py in one fixed format, so this one is case-sensitive.
_ARTICLE = re.compile(r"^(Art\. \d+|Annex [IVXL]+|Recital \d+)(?![\dIVXL])")


def article_of(ref: str) -> str:
    """The article (or annex, or recital) a chunk ref belongs to.

    Example: "Art. 5(1), point (f)" -> "Art. 5", "Annex III, point 1" -> "Annex III"
    """
    match = _ARTICLE.match(ref)
    if not match:
        raise ValueError(f"Not a chunk ref: {ref!r}")
    return match.group(1)


def mentioned_articles(text: str) -> set[str]:
    """Articles and annexes of the AI Act that the text names, e.g. {"Art. 50", "Annex III"}.

    The result uses the chunk refs' spelling, so "article 4" and "Art. 4" both give "Art. 4".
    Plain pattern matching, so it can miss unusual wording ("Articles 99 and 100" only
    finds Art. 99). Mentions of other laws ("Article 4(3) of Directive ...") are skipped.
    """
    found = set()
    for match in MENTION.finditer(text):
        if _OTHER_LAW.match(text, match.end()):
            continue
        article, annex = match.groups()
        found.add(f"Art. {article}" if article else f"Annex {annex.upper()}")
    return found
