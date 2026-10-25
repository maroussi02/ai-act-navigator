"""Checks on the full official text. Skipped unless it was downloaded first:

uv run python -m ai_act_navigator.download
"""

from collections import Counter

import pytest

from ai_act_navigator.chunk import parse
from ai_act_navigator.download import DEFAULT_PATH

pytestmark = pytest.mark.skipif(not DEFAULT_PATH.exists(), reason="full text not downloaded")


def test_all_parts_of_the_law_are_found():
    chunks = parse(DEFAULT_PATH.read_text(encoding="utf-8"))
    # A chunk id starts with its parent, e.g. "art-50-1" belongs to article 50.
    parents = {(c.type, c.id.split("-")[1]) for c in chunks}
    assert Counter(kind for kind, _ in parents) == {"article": 113, "recital": 180, "annex": 13}
