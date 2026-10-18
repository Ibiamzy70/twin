"""
Central session-state initialization.

Streamlit reruns the whole script on every interaction, so state must be created
exactly once and never re-assigned by individual views. All default values live
here; views only read and update ``st.session_state``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Dict

import streamlit as st

# ---------------------------------------------------------------------------
# Keys - import these instead of repeating string literals in views/services.
# ---------------------------------------------------------------------------
MESSAGES = "messages"
TASKS = "tasks"
CHAT_COUNT = "chat_count"
USER_PROFILE = "user_profile"
SETTINGS = "settings"

DEFAULT_DISPLAY_NAME = "Guest"

# ---------------------------------------------------------------------------
# Defaults - factories, so every session gets its own fresh list/dict objects
# (a shared mutable default would leak data between reruns and sessions).
# ---------------------------------------------------------------------------
_DEFAULT_FACTORIES: Dict[str, Callable[[], Any]] = {
    MESSAGES: list,
    TASKS: list,
    CHAT_COUNT: lambda: 0,
    USER_PROFILE: lambda: {"display_name": DEFAULT_DISPLAY_NAME},
    SETTINGS: lambda: {
        "show_timestamps": True,
        "compact_mode": False,
        "confirm_destructive_actions": True,
    },
}


def initialize_state() -> None:
    """
    Create any missing session-state keys with their defaults.

    Safe to call on every rerun: existing values are never overwritten.
    Call once near the top of ``app.py``, before rendering the sidebar or views.
    """
    for key, factory in _DEFAULT_FACTORIES.items():
        if key not in st.session_state:
            st.session_state[key] = factory()


def reset_state() -> None:
    """
    Explicitly restore every managed key to its default.

    Intended for a deliberate user action (e.g. "Reset workspace" in Settings).
    Never call this implicitly during a normal render.
    """
    for key, factory in _DEFAULT_FACTORIES.items():
        st.session_state[key] = factory()


def get_default(key: str) -> Any:
    """Return a fresh copy of the default value for a managed key."""
    return deepcopy(_DEFAULT_FACTORIES[key]())
