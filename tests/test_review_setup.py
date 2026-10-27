"""UNIT TESTS: the PR template and the review guide exist and keep their key sections.

These files steer every PR and every review. The tests guard against a section being
deleted by accident; they don't check the wording.
"""

from ai_act_navigator.paths import PROJECT_ROOT

TEMPLATE = PROJECT_ROOT / ".github" / "pull_request_template.md"
REVIEW_GUIDE = PROJECT_ROOT / "REVIEW.md"


def test_pr_template_asks_for_testing_risk_and_revert():
    text = TEMPLATE.read_text(encoding="utf-8")
    for heading in ("## Testing", "## Risk", "## How to revert"):
        assert heading in text, f"PR template is missing {heading!r}"


def test_review_guide_has_failure_skip_and_report_sections():
    text = REVIEW_GUIDE.read_text(encoding="utf-8")
    for heading in ("## What failure looks like here", "## Skip", "## How to report"):
        assert heading in text, f"REVIEW.md is missing {heading!r}"
