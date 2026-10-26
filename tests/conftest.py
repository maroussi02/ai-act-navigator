"""Shared pytest setup. pytest loads this file automatically before running tests.

Paid tests (marked `paid`) call the real Claude API and cost money. They are skipped unless
you pass --run-paid on purpose:

    uv run pytest --run-paid -m paid

A flag instead of a `-m` filter, because any other `-m` (like -m "not integration") would
replace the filter and silently run paid tests.
"""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--run-paid", action="store_true", help="run tests that call the paid API")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-paid"):
        return
    skip = pytest.mark.skip(reason="costs money; run with --run-paid")
    for item in items:
        if "paid" in item.keywords:
            item.add_marker(skip)
