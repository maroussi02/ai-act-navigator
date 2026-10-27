"""UNIT TESTS for keyword.py: the tokenizer and BM25 ranking, on tiny hand-made chunks."""

import pytest

from ai_act_navigator.keyword import KeywordIndex, tokenize
from ai_act_navigator.models import Chunk

URL = "https://example.org"


def chunk(ref: str, text: str) -> Chunk:
    return Chunk(ref.lower(), "article", ref, "", text, URL)


@pytest.mark.parametrize("spelling", ["Article 4", "article 4", "Art. 4", "Art 4", "ARTICLES 4"])
def test_every_spelling_of_a_reference_becomes_one_word(spelling):
    assert tokenize(f"What does {spelling} require?") == ["what", "does", "art_4", "require"]


def test_annex_references_become_one_word():
    assert tokenize("Annex III, point 1") == ["annex_iii", "point", "1"]


def test_a_rare_word_beats_a_common_one():
    index = KeywordIndex(
        [
            chunk("Art. 1", "Providers shall act."),
            chunk("Art. 2", "Providers shall use biometric data carefully."),
            chunk("Art. 3", "Providers shall report."),
        ]
    )
    # "providers" and "shall" are in every chunk; only "biometric" tells them apart.
    assert [c.ref for c in index.search("biometric providers")][0] == "Art. 2"


def test_a_named_article_ranks_its_own_chunk_first():
    index = KeywordIndex(
        [
            chunk("Art. 5", "Prohibited practices, see paragraph 4."),
            chunk("Art. 4", "Providers shall ensure AI literacy."),
        ]
    )
    # Without the art_4 token, "4" would also match "paragraph 4" in Art. 5.
    assert [c.ref for c in index.search("What does Article 4 require?")] == ["Art. 4"]


def test_chunks_without_a_shared_word_are_left_out():
    index = KeywordIndex([chunk("Art. 1", "Providers."), chunk("Art. 2", "Deployers.")])
    assert [c.ref for c in index.search("deployers", n=5)] == ["Art. 2"]


@pytest.mark.parametrize("question", ["", "   ", "VAT Germany"])
def test_no_known_words_give_no_results(question):
    assert KeywordIndex([chunk("Art. 1", "Providers.")]).search(question) == []


def test_an_empty_index_returns_nothing():
    assert KeywordIndex([]).search("anything") == []
