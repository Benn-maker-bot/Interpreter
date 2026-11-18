"""Shared set-up for every test (pytest loads this file automatically).

Importing interpreter.py has side effects: it sets up logging to
interpreter.log.jsonl and records an "app_module_loaded" event. Tests must not
fill the real logs, so this file imports the module once and then removes all
of loguru's outputs ("sinks"). One "app_module_loaded" line per test run still
reaches the log, because it is written during the import itself.
"""
import pytest
from loguru import logger

import interpreter

# Stop the app's logging while the tests run (console and both .jsonl files).
logger.remove()


@pytest.fixture
def interp():
    """Give a test the interpreter module, e.g. `def test_x(interp): ...`.

    A fixture is a function pytest runs before a test and passes in as an
    argument. Here it just returns the already-imported module, but fixtures
    can also build objects, create temporary files and clean up afterwards.
    """
    return interpreter
