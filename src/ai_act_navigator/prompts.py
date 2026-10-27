"""Load the instructions for Claude from prompts.toml, the one versioned prompt file.

Prompts live in their own file, not in the code, because changing a prompt changes the
product's behavior as much as changing code. So each prompt text gets a version ("v1") and a
fingerprint (SHA-256 of the exact text). Every eval result records both: two scores with the
same fingerprint were made with the same instructions, letter for letter.

tests/test_prompts.py lists the fingerprint of every version and fails if the text changes
while the version stays the same, so a prompt can't change silently.
"""

import hashlib
import tomllib
from dataclasses import dataclass
from pathlib import Path

# Next to this file inside the package, so it ships with the code.
PROMPTS_PATH = Path(__file__).parent / "prompts.toml"

_FIELDS = {"version", "system"}


class PromptFileError(Exception):
    """prompts.toml is broken: not valid TOML, or a field is missing, empty or unknown."""


@dataclass(frozen=True)
class Prompts:
    version: str  # e.g. "v1"; bump it whenever the text changes
    system: str  # the system prompt sent with every question
    sha256: str  # fingerprint of `system`: changes if even one character changes


def load_prompts(path: Path = PROMPTS_PATH) -> Prompts:
    """Read and check the prompt file. Raises PromptFileError if it breaks a rule."""
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise PromptFileError(f"{path.name} is not valid TOML: {error}") from error

    # An unknown field is most likely a typo (e.g. "sytem"). Without this check it would be
    # silently ignored.
    unknown = sorted(data.keys() - _FIELDS)
    if unknown:
        raise PromptFileError(f"{path.name}: unknown field(s): {', '.join(unknown)}")
    for field in sorted(_FIELDS):
        value = data.get(field)
        if not isinstance(value, str) or not value.strip():
            raise PromptFileError(f"{path.name}: {field!r} must be a non-empty text")

    system = data["system"]
    return Prompts(
        version=data["version"],
        system=system,
        sha256=hashlib.sha256(system.encode("utf-8")).hexdigest(),
    )


# Loaded once when the module is imported. The rest of the code uses these names.
_PROMPTS = load_prompts()
PROMPT_VERSION = _PROMPTS.version
SYSTEM_PROMPT = _PROMPTS.system
PROMPT_SHA256 = _PROMPTS.sha256
