"""Data shapes shared across the pipeline."""

from dataclasses import dataclass
from typing import Literal

# Literal = only these exact strings are allowed (a type checker would flag "articel").
ChunkType = Literal["article", "recital", "annex"]


@dataclass(frozen=True)
class Chunk:
    """One citable piece of the AI Act, e.g. a single article paragraph.

    A dataclass is a class that just holds data: Python writes __init__, ==, and the
    printout for us. Frozen = read-only, so a chunk can't be changed by accident later.
    """

    id: str  # stable, unique key, e.g. "art-50-1"
    type: ChunkType
    ref: str  # human-readable citation, e.g. "Art. 50(1)"
    title: str  # article or annex title, empty for recitals
    text: str
    url: str  # link to the exact spot in the official text


@dataclass(frozen=True)
class Hit:
    """One search result: a chunk and how well it matches the question.

    Higher score = better match. What the number means depends on the search:
    - meaning only (vector): cosine similarity, 1.0 = same meaning, 0.0 = unrelated;
    - hybrid: the fused rank score from search.fuse, only useful for ordering.
    """

    chunk: Chunk
    score: float
