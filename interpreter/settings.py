"""The user's saved choices (interpreter.settings.json).

Unlike config.py, which holds values fixed in the code, these are the
choices made in the window (devices, languages, model ...), remembered
from one run to the next.
"""

import json
from typing import Any

from .config import SETTINGS_PATH
from .logging_setup import log_exception


# ─────────────────────────────────────────────────────────────────────────────
# Persisted user settings
#
# The user's choices are saved in interpreter.settings.json so they are
# remembered the next time the app starts.
# ─────────────────────────────────────────────────────────────────────────────

def load_settings() -> dict[str, Any]:
    """Read interpreter.settings.json and return it as a dict ({} if missing or broken)."""
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as e:
        log_exception("settings_load_failed", e, path=str(SETTINGS_PATH))
        return {}


def save_settings(**changes: Any) -> None:
    """Update some settings (e.g. save_settings(model="…")) and write the file."""
    settings = load_settings()
    settings.update(changes)
    try:
        SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
    except Exception as e:
        log_exception("settings_save_failed", e, path=str(SETTINGS_PATH))
