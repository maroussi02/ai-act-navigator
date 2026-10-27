"""UNIT TESTS for prompts.py: the versioned prompt file loads, and can't change silently."""

import re

import pytest

from ai_act_navigator.prompts import (
    PROMPT_SHA256,
    PROMPT_VERSION,
    PromptFileError,
    load_prompts,
)

# The fingerprint (SHA-256) of every prompt version's exact text. When you change the text in
# prompts.toml, bump its version and add the new fingerprint here. Never edit an old entry:
# old eval results point to these.
KNOWN_PROMPTS = {
    "v1": "69c2ad1ad8854ee843412e195121a39047899a54228da2511aad5d5f02ba25d8",
}


def test_prompt_text_matches_its_version():
    assert PROMPT_VERSION in KNOWN_PROMPTS, (
        f"New prompt version {PROMPT_VERSION!r}: add its fingerprint {PROMPT_SHA256} "
        "to KNOWN_PROMPTS"
    )
    assert KNOWN_PROMPTS[PROMPT_VERSION] == PROMPT_SHA256, (
        f"The text of prompt {PROMPT_VERSION} changed. Bump `version` in prompts.toml "
        "and add the new fingerprint to KNOWN_PROMPTS."
    )


def test_every_version_has_its_own_fingerprint():
    # Two versions with the same text would make "v1 vs v2" scores look different for no reason.
    assert len(set(KNOWN_PROMPTS.values())) == len(KNOWN_PROMPTS)


def test_a_valid_file_loads_version_text_and_fingerprint(tmp_path):
    path = tmp_path / "prompts.toml"
    path.write_text('version = "v9"\nsystem = """\nBe brief.\n"""\n', encoding="utf-8")

    prompts = load_prompts(path)

    assert prompts.version == "v9"
    assert prompts.system == "Be brief.\n"  # TOML drops the newline right after """
    assert len(prompts.sha256) == 64


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ('version = "v1"\n', "'system' must be a non-empty text"),
        ('version = "v1"\nsystem = "  "\n', "'system' must be a non-empty text"),
        ('system = "Be brief."\nversion = 1\n', "'version' must be a non-empty text"),
        ('version = "v1"\nsystem = "x"\nsytem = "y"\n', "unknown field(s): sytem"),
        ('version = "v1\n', "not valid TOML"),
    ],
)
def test_a_broken_file_is_a_clear_error(tmp_path, content, message):
    path = tmp_path / "prompts.toml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(PromptFileError, match=re.escape(message)):  # "(s)" is plain text
        load_prompts(path)
