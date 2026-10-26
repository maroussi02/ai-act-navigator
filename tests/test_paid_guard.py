"""UNIT TEST for the paid-test guard in conftest.py: no command runs paid tests by accident.

Starts a separate pytest run, the way you would in the terminal, and checks the paid test
is skipped even with a `-m` filter that would otherwise select it.
"""

import subprocess
import sys

from ai_act_navigator.paths import PROJECT_ROOT


def test_paid_test_is_skipped_without_the_flag():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_live_answer.py",
            "-m",
            "not integration",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert "1 skipped" in result.stdout, result.stdout
