"""UNIT TESTS for refs.py: finding article and annex references in text."""

import pytest

from ai_act_navigator.refs import article_of, mentioned_articles


@pytest.mark.parametrize(
    ("ref", "article"),
    [
        ("Art. 50(1)", "Art. 50"),
        ("Art. 5(1), point (f)", "Art. 5"),
        ("Art. 3, point (3)", "Art. 3"),
        ("Art. 113", "Art. 113"),
        ("Annex III, point 1", "Annex III"),
        ("Recital 132", "Recital 132"),
    ],
)
def test_article_of(ref, article):
    assert article_of(ref) == article


def test_article_of_rejects_text_that_is_no_ref():
    with pytest.raises(ValueError):
        article_of("Article 50")


@pytest.mark.parametrize(
    ("text", "found"),
    [
        ("Art. 50(1) and Article 6(1), point (a)", {"Art. 50", "Art. 6"}),
        ("listed in Annex III and Annex I", {"Annex III", "Annex I"}),
        ("Articles 99 and 100", {"Art. 99"}),  # known limit: only the first number
        ("Article 4(3) of Directive (EU) 2019/790", set()),  # another law
        ("Article 2 of Regulation (EU) 2016/679", set()),
        ("No article here. Recital 132.", set()),
        # How people type questions: lowercase, no dot after "Art".
        ("what does article 4 require?", {"Art. 4"}),
        ("Is Art 50 about chatbots?", {"Art. 50"}),
        ("what is in annex iii?", {"Annex III"}),
        ("article 4(3) of directive (EU) 2019/790", set()),
    ],
)
def test_mentioned_articles(text, found):
    assert mentioned_articles(text) == found
