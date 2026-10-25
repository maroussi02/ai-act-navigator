"""Data shapes shared across the pipeline."""

from dataclasses import dataclass
from typing import Literal

ChunkType = Literal["article", "recital", "annex"]


@dataclass(frozen=True)
class Chunk:
    """One citable piece of the AI Act, e.g. a single article paragraph.

    Frozen so a chunk can't be changed by accident after parsing.
    """

    id: str  # stable, unique key, e.g. "art-50-1"
    type: ChunkType
    ref: str  # human-readable citation, e.g. "Art. 50(1)"
    title: str  # article or annex title, empty for recitals
    text: str
    url: str  # link to the exact spot in the official text
