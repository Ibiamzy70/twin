"""
Settings view.

Exposes ``render()`` for the router in ``app.py`` and ``apply_display_preferences()``
so the app shell can honour the display options globally.

Design notes
------------
* Preferences live in ``st.session_state[SETTINGS]`` (a plain dict created by
  ``state.initialize_state``). Streamlit drops a widget's own key when the widget
  is not rendered, so every control writes its value into that dict from an
  ``on_change`` callback and is re-seeded from it on the next visit.
* Destructive actions go through a two-step confirmation, which can be switched off
  with the "Confirm destructive actions" preference.
* All dynamic text rendered through ``unsafe_allow_html`` is escaped.
* Icons are Phosphor SVGs from ``assets/icons``; a missing icon renders nothing.
"""

from __future__ import annotations

import json
import logging
import platform
import re
from dataclasses import dataclass
from datetime import datetime
from html import escape
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from components.app_header import render_header
from components.icons import icon_html
from state import session as session_state
from state.session import (
    CHAT_COUNT,
    DEFAULT_DISPLAY_NAME,
    MESSAGES,
    SETTINGS,
    TASKS,
    USER_PROFILE,
    get_default,
    reset_state,
)

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"
EXPORT_SCHEMA_VERSION = 1
MAX_NAME_LENGTH = 40
CALC_HISTORY_KEY = "calc_history"

# Keys of widgets owned by this view (cleared on workspace reset).
_K_NAME = "settings_display_name"
_K_FORM = "settings_profile_form"
_K_PENDING = "settings_pending_action"
_K_FLASH = "settings_flash"
_K_PREFIX = "settings_pref_"

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")

# ---------------------------------------------------------------------------
# Preference definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preference:
    key: str
    label: str
    help: str


PREFERENCES: tuple[Preference, ...] = (
    Preference(
        "show_timestamps",
        "Show timestamps",
        "Display the time next to chat messages and activity entries.",
    ),
    Preference(
        "compact_mode",
        "Compact layout",
        "Tighter spacing so more content fits on screen.",
    ),
    Preference(
        "confirm_destructive_actions",
        "Confirm destructive actions",
        "Ask for confirmation before deleting or resetting data.",
    ),
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------

SETTINGS_CSS = """
<style>
.st-key-settings-root { animation: settings-in var(--duration-slow, 300ms) ease both; }
@keyframes settings-in { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }

.settings-section {
    display: flex; align-items: center; gap: var(--space-3, 12px);
    margin: var(--space-2, 8px) 0 var(--space-4, 16px);
}
.settings-section__icon {
    display: inline-flex; align-items: center; justify-content: center;
    width: 40px; height: 40px; flex: none;
    border-radius: var(--radius-md, 12px);
    background: var(--color-accent-soft, rgba(138,168,255,.14));
    color: var(--color-accent, #8AA8FF);
}
.settings-section__title { margin: 0; font-size: 1.05rem; font-weight: 650; letter-spacing: -0.01em; }
.settings-section__text { margin: 2px 0 0; font-size: .85rem; opacity: .7; }

.settings-avatar {
    display: flex; align-items: center; gap: var(--space-4, 16px);
    padding: var(--space-4, 16px); margin-bottom: var(--space-4, 16px);
    border: 1px solid var(--color-border, rgba(255,255,255,.08));
    border-radius: var(--radius-lg, 16px);
    background: var(--color-surface, transparent);
    box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,.2));
    transition: box-shadow var(--duration-base, 200ms) ease, transform var(--duration-base, 200ms) ease;
}
.settings-avatar:hover { box-shadow: var(--shadow-md, 0 6px 18px rgba(0,0,0,.25)); transform: translateY(-1px); }
.settings-avatar__badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 56px; height: 56px; flex: none; border-radius: 50%;
    font-weight: 700; font-size: 1.15rem; letter-spacing: .02em;
    background: var(--color-accent, #8AA8FF); color: var(--color-on-accent, #0B1020);
}
.settings-avatar__name { font-weight: 650; font-size: 1.05rem; line-height: 1.2; overflow-wrap: anywhere; }
.settings-avatar__meta { font-size: .82rem; opacity: .65; margin-top: 2px; }

.settings-row-title { font-weight: 600; font-size: .95rem; margin: 0; }
.settings-row-text { font-size: .83rem; opacity: .68; margin: 2px 0 0; }

.settings-stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: var(--space-3, 12px); margin-bottom: var(--space-4, 16px); }
.settings-stat {
    padding: var(--space-4, 16px);
    border: 1px solid var(--color-border, rgba(255,255,255,.08));
    border-radius: var(--radius-lg, 16px);
    background: var(--color-surface, transparent);
    box-shadow: var(--shadow-sm, 0 1px 2px rgba(0,0,0,.2));
    transition: box-shadow var(--duration-base, 200ms) ease, transform var(--duration-base, 200ms) ease;
}
.settings-stat:hover { box-shadow: var(--shadow-md, 0 6px 18px rgba(0,0,0,.25)); transform: translateY(-2px); }
.settings-stat__value { font-size: 1.5rem; font-weight: 700; line-height: 1.1; }
.settings-stat__label { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .65; margin-top: 4px; }

.settings-about { width: 100%; border-collapse: collapse; font-size: .9rem; }
.settings-about th, .settings-about td { text-align: left; padding: 10px 4px; border-bottom: 1px solid var(--color-border, rgba(255,255,255,.08)); }
.settings-about th { font-weight: 500; opacity: .65; width: 40%; }
.settings-about tr:last-child th, .settings-about tr:last-child td { border-bottom: none; }

.st-key-settings-root [data-testid="stVerticalBlockBorderWrapper"] {
    transition: box-shadow var(--duration-base, 200ms) ease, border-color var(--duration-base, 200ms) ease;
}
.st-key-settings-root [data-testid="stVerticalBlockBorderWrapper"]:hover {
    box-shadow: var(--shadow-sm, 0 2px 8px rgba(0,0,0,.18));
}

@media (prefers-reduced-motion: reduce) {
    .st-key-settings-root, .settings-avatar, .settings-stat { animation: none; transition: none; }
}
</style>
"""

COMPACT_CSS = """
<style>
.block-container { padding-top: 1.25rem !important; padding-bottom: 2rem !important; }
[data-testid="stVerticalBlock"] { gap: .6rem !important; }
</style>
"""

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------


def _settings() -> Dict[str, Any]:
    """Return the settings dict, repairing it if it is missing or malformed."""
    current = st.session_state.get(SETTINGS)
    defaults = get_default(SETTINGS)
    if not isinstance(current, dict):
        current = {}
    for key, value in defaults.items():
        current.setdefault(key, value)
    st.session_state[SETTINGS] = current
    return current


def _profile() -> Dict[str, Any]:
    current = st.session_state.get(USER_PROFILE)
    if not isinstance(current, dict):
        current = get_default(USER_PROFILE)
    current.setdefault("display_name", DEFAULT_DISPLAY_NAME)
    st.session_state[USER_PROFILE] = current
    return current


def _as_list(key: str) -> list:
    value = st.session_state.get(key)
    return value if isinstance(value, list) else []


def _flash(kind: str, message: str) -> None:
    st.session_state[_K_FLASH] = (kind, message)


def _show_flash() -> None:
    flash = st.session_state.pop(_K_FLASH, None)
    if not flash:
        return
    kind, message = flash
    {"success": st.success, "error": st.error, "warning": st.warning}.get(
        kind, st.info
    )(message)


def _esc(value: Any) -> str:
    """Escape for HTML and neutralise characters Streamlit's markdown would interpret."""
    text = " ".join(str(value).split())
    return escape(text, quote=True).replace("$", "&#36;")


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"\s+", name.strip()) if p]
    if not parts:
        return "?"
    letters = parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")
    return letters.upper()


def clean_display_name(raw: Optional[str]) -> str:
    """Strip control characters, collapse whitespace and trim to the max length."""
    text = _CONTROL_CHARS.sub("", raw or "")
    return " ".join(text.split())[:MAX_NAME_LENGTH].strip()


# ---------------------------------------------------------------------------
# Public: display preferences (call from app.py before rendering the view)
# ---------------------------------------------------------------------------


def apply_display_preferences() -> None:
    """Inject CSS for display preferences that affect the whole app."""
    if _settings().get("compact_mode"):
        st.markdown(COMPACT_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


def _on_preference_change(pref_key: str) -> None:
    _settings()[pref_key] = bool(st.session_state.get(f"{_K_PREFIX}{pref_key}"))
    # Turning confirmations off must not leave a stale pending prompt around.
    if pref_key == "confirm_destructive_actions":
        st.session_state.pop(_K_PENDING, None)


def _save_profile() -> None:
    name = clean_display_name(st.session_state.get(_K_NAME))
    if not name:
        _flash("error", "Display name cannot be empty.")
        return
    _profile()["display_name"] = name
    st.session_state[_K_NAME] = name  # show the normalised value
    _flash("success", "Profile saved.")


def _reset_preferences() -> None:
    st.session_state[SETTINGS] = get_default(SETTINGS)
    for pref in PREFERENCES:
        st.session_state.pop(f"{_K_PREFIX}{pref.key}", None)
    _flash("success", "Preferences restored to their defaults.")


def _clear_chat() -> None:
    st.session_state[MESSAGES] = get_default(MESSAGES)
    st.session_state[CHAT_COUNT] = get_default(CHAT_COUNT)


def _clear_tasks() -> None:
    st.session_state[TASKS] = get_default(TASKS)


def _clear_calculator_history() -> None:
    st.session_state[CALC_HISTORY_KEY] = []


def _reset_workspace() -> None:
    reset_state()
    st.session_state[CALC_HISTORY_KEY] = []
    st.session_state.pop(_K_NAME, None)
    st.session_state.pop(_K_FORM, None)
    for pref in PREFERENCES:
        st.session_state.pop(f"{_K_PREFIX}{pref.key}", None)


@dataclass(frozen=True)
class DangerAction:
    id: str
    title: str
    description: str
    button: str
    confirm: str
    run: Callable[[], None]
    done: str

    def count(self) -> Optional[int]:
        return _COUNTERS.get(self.id, lambda: None)()


_COUNTERS: Dict[str, Callable[[], Optional[int]]] = {
    "chat": lambda: len(_as_list(MESSAGES)),
    "tasks": lambda: len(_as_list(TASKS)),
    "calc": lambda: len(_as_list(CALC_HISTORY_KEY)),
}

DANGER_ACTIONS: tuple[DangerAction, ...] = (
    DangerAction(
        "chat",
        "Clear chat history",
        "Remove every message from the assistant conversation.",
        "Clear chat",
        "Delete all chat messages? This cannot be undone.",
        _clear_chat,
        "Chat history cleared.",
    ),
    DangerAction(
        "tasks",
        "Delete all tasks",
        "Remove every task, including completed ones.",
        "Delete tasks",
        "Delete all tasks? This cannot be undone.",
        _clear_tasks,
        "All tasks deleted.",
    ),
    DangerAction(
        "calc",
        "Clear calculator history",
        "Remove the saved calculations from the calculator.",
        "Clear history",
        "Delete the calculator history? This cannot be undone.",
        _clear_calculator_history,
        "Calculator history cleared.",
    ),
    DangerAction(
        "workspace",
        "Reset workspace",
        "Restore the profile, preferences, chat, tasks and history to a fresh start.",
        "Reset workspace",
        "Reset the entire workspace? Everything above will be erased.",
        _reset_workspace,
        "Workspace reset.",
    ),
)
_ACTIONS_BY_ID = {a.id: a for a in DANGER_ACTIONS}


def _execute(action_id: str) -> None:
    action = _ACTIONS_BY_ID.get(action_id)
    st.session_state.pop(_K_PENDING, None)
    if action is None:
        return
    try:
        action.run()
    except Exception:  # pragma: no cover - defensive, never crash a callback
        logger.exception("Settings action %r failed", action_id)
        _flash("error", "That action could not be completed. Please try again.")
        return
    _flash("success", action.done)


def _request(action_id: str) -> None:
    """Ask for confirmation, or run immediately if confirmations are disabled."""
    if _settings().get("confirm_destructive_actions", True):
        st.session_state[_K_PENDING] = action_id
    else:
        _execute(action_id)


def _cancel() -> None:
    st.session_state.pop(_K_PENDING, None)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def build_export() -> Dict[str, Any]:
    """Collect everything the workspace stores into a JSON-serialisable dict."""
    return {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "exported_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "profile": _profile(),
        "settings": _settings(),
        "messages": _as_list(MESSAGES),
        "tasks": _as_list(TASKS),
        "calculator_history": _as_list(CALC_HISTORY_KEY),
    }


def _export_bytes() -> bytes:
    return json.dumps(build_export(), indent=2, ensure_ascii=False, default=str).encode(
        "utf-8"
    )


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------


def _section(icon: str, title: str, text: str) -> None:
    st.markdown(
        f'<div class="settings-section">'
        f'<span class="settings-section__icon">{icon_html(icon, 20)}</span>'
        f'<div><p class="settings-section__title">{_esc(title)}</p>'
        f'<p class="settings-section__text">{_esc(text)}</p></div></div>',
        unsafe_allow_html=True,
    )


def _render_profile() -> None:
    _section("user", "Profile", "How the workspace addresses you.")
    name = _esc(_profile()["display_name"])
    st.markdown(
        f'<div class="settings-avatar">'
        f'<span class="settings-avatar__badge">{_esc(_initials(_profile()["display_name"]))}</span>'
        f'<div><div class="settings-avatar__name">{name}</div>'
        f'<div class="settings-avatar__meta">Local profile for this session</div></div></div>',
        unsafe_allow_html=True,
    )
    if _K_NAME not in st.session_state:
        st.session_state[_K_NAME] = _profile()["display_name"]
    with st.form(_K_FORM, border=False):
        st.text_input(
            "Display name",
            key=_K_NAME,
            max_chars=MAX_NAME_LENGTH,
            placeholder="Your name",
            help=f"Up to {MAX_NAME_LENGTH} characters.",
        )
        st.form_submit_button("Save profile", type="primary", on_click=_save_profile)


def _render_preferences() -> None:
    _section("sliders-horizontal", "Preferences", "Changes apply immediately.")
    current = _settings()
    for pref in PREFERENCES:
        with st.container(border=True):
            left, right = st.columns([5, 1], vertical_alignment="center")
            with left:
                st.markdown(
                    f'<p class="settings-row-title">{_esc(pref.label)}</p>'
                    f'<p class="settings-row-text">{_esc(pref.help)}</p>',
                    unsafe_allow_html=True,
                )
            with right:
                st.toggle(
                    pref.label,
                    value=bool(current.get(pref.key)),
                    key=f"{_K_PREFIX}{pref.key}",
                    on_change=_on_preference_change,
                    args=(pref.key,),
                    label_visibility="collapsed",
                )
    st.button("Restore default preferences", on_click=_reset_preferences)


def _render_stats() -> None:
    stats: List[tuple[str, int]] = [
        ("Chat messages", len(_as_list(MESSAGES))),
        ("Tasks", len(_as_list(TASKS))),
        ("Calculations", len(_as_list(CALC_HISTORY_KEY))),
    ]
    cells = "".join(
        f'<div class="settings-stat"><div class="settings-stat__value">{value}</div>'
        f'<div class="settings-stat__label">{_esc(label)}</div></div>'
        for label, value in stats
    )
    st.markdown(
        f'<div class="settings-stat-grid">{cells}</div>', unsafe_allow_html=True
    )


def _render_export() -> None:
    _section("download-simple", "Export", "Download a copy of your data as JSON.")
    _render_stats()
    st.download_button(
        "Download workspace data",
        data=_export_bytes(),
        file_name=f"digital-twin-export-{datetime.now():%Y%m%d-%H%M}.json",
        mime="application/json",
        type="primary",
    )
    st.caption(
        "Data lives in this browser session only. Export it before closing the tab."
    )


def _render_danger_zone() -> None:
    _section("warning", "Danger zone", "These actions cannot be undone.")
    pending = st.session_state.get(_K_PENDING)
    for action in DANGER_ACTIONS:
        count = action.count()
        with st.container(border=True):
            left, right = st.columns([4, 1.6], vertical_alignment="center")
            with left:
                detail = f" ({count} stored)" if count is not None else ""
                st.markdown(
                    f'<p class="settings-row-title">{_esc(action.title)}{_esc(detail)}</p>'
                    f'<p class="settings-row-text">{_esc(action.description)}</p>',
                    unsafe_allow_html=True,
                )
            with right:
                disabled = count == 0
                st.button(
                    action.button,
                    key=f"settings_req_{action.id}",
                    on_click=_request,
                    args=(action.id,),
                    disabled=disabled,
                    use_container_width=True,
                )
            if pending == action.id:
                st.warning(action.confirm)
                yes, no = st.columns(2)
                yes.button(
                    "Yes, continue",
                    key=f"settings_yes_{action.id}",
                    type="primary",
                    on_click=_execute,
                    args=(action.id,),
                    use_container_width=True,
                )
                no.button(
                    "Cancel",
                    key=f"settings_no_{action.id}",
                    on_click=_cancel,
                    use_container_width=True,
                )


def _render_about() -> None:
    _section("info", "About", "Version and environment details.")
    rows = (
        ("Application", "Digital Twin"),
        ("Version", APP_VERSION),
        ("Streamlit", st.__version__),
        ("Python", platform.python_version()),
        ("Session data", "Stored in memory, cleared when the session ends"),
    )
    body = "".join(f"<tr><th>{_esc(k)}</th><td>{_esc(v)}</td></tr>" for k, v in rows)
    st.markdown(f'<table class="settings-about">{body}</table>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the Settings route."""
    # Make sure managed keys exist even if this view is rendered in isolation.
    session_state.initialize_state()
    _settings()
    _profile()

    st.markdown(SETTINGS_CSS, unsafe_allow_html=True)
    apply_display_preferences()

    render_header(
        "Settings",
        "Manage your profile, preferences and stored data.",
        eyebrow="Workspace",
    )

    with st.container(key="settings-root"):
        _show_flash()
        tab_general, tab_data, tab_about = st.tabs(
            ["General", "Data and privacy", "About"]
        )
        with tab_general:
            _render_profile()
            st.write("")
            _render_preferences()
        with tab_data:
            _render_export()
            st.write("")
            _render_danger_zone()
        with tab_about:
            _render_about()
