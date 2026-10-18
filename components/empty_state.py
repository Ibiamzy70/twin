"""
Empty state.

A calm, premium placeholder for views that have no data yet. It keeps the class
names the dashboard already relies on (``empty-state``, ``empty-state__icon``,
``empty-state__title``, ``empty-state__description``), so existing inline markup
keeps working, and adds an optional call-to-action button.

Two entry points:

* ``empty_state_html(...)``    pure function, returns an HTML string you can embed
                                inside another card (the dashboard feed does this).
* ``render_empty_state(...)``  renders the card with Streamlit, optionally with a
                                real ``st.button`` action underneath.

Styling uses only theme tokens from ``styles/theme.css``; no raw colours.
Every piece of text is escaped before it reaches the page.
"""

from __future__ import annotations

from html import escape
from typing import Any, Callable, Optional, Sequence

import streamlit as st

from components.icons import icon_html

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_EMPTY_STATE_CSS = """
.empty-state {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--space-3);
  padding: var(--space-8) var(--space-6);
  text-align: center;
  overflow: hidden;
  background:
    radial-gradient(90% 70% at 50% 0%, var(--color-accent-soft), transparent 70%),
    var(--color-surface);
  border: var(--border-width) dashed var(--color-border-strong);
  border-radius: var(--radius-xl);
  animation: empty-rise var(--duration-slow) var(--ease-standard) both;
}
.empty-state--compact { padding: var(--space-5) var(--space-4); gap: var(--space-2); }

.empty-state__icon {
  position: relative;
  display: grid;
  place-items: center;
  width: 3.75rem;
  height: 3.75rem;
  border-radius: var(--radius-full);
  background: var(--color-accent-soft);
  color: var(--color-accent);
}
.empty-state__icon::after {
  content: "";
  position: absolute;
  inset: -0.45rem;
  border-radius: var(--radius-full);
  border: var(--border-width) solid var(--color-accent-border);
  opacity: 0.6;
  animation: empty-halo 3.6s var(--ease-standard) infinite;
}
.empty-state--compact .empty-state__icon { width: 2.75rem; height: 2.75rem; }

.empty-state__title {
  margin: var(--space-1) 0 0;
  color: var(--color-text);
  font-size: var(--font-size-lg);
  font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-tight);
}
.empty-state__description {
  margin: 0;
  max-width: 34ch;
  color: var(--color-text-secondary);
  font-size: var(--font-size-sm);
  line-height: var(--line-height-normal);
}
.empty-state__hints {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: var(--space-2);
  margin: var(--space-1) 0 0;
  padding: 0;
  list-style: none;
}
.empty-state__hints li {
  padding: var(--space-1) var(--space-3);
  border: var(--border);
  border-radius: var(--radius-full);
  background: var(--color-surface-raised);
  color: var(--color-text-secondary);
  font-size: var(--font-size-xs);
  font-weight: var(--font-weight-medium);
}

/* call-to-action button rendered by render_empty_state() */
[class*="st-key-empty_state_"] .stButton { display: flex; justify-content: center; }
[class*="st-key-empty_state_"] .stButton button {
  min-width: 10rem;
  background: var(--color-accent);
  border: 0;
  border-radius: var(--radius-full);
  color: var(--color-text-on-accent);
  font-weight: var(--font-weight-semibold);
  box-shadow: var(--shadow-sm);
  transition: transform var(--duration-base) var(--ease-standard),
              box-shadow var(--duration-base) var(--ease-standard);
}
[class*="st-key-empty_state_"] .stButton button:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
[class*="st-key-empty_state_"] .stButton button:active { transform: translateY(0); }
[class*="st-key-empty_state_"] .stButton button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 3px; }

@keyframes empty-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@keyframes empty-halo { 0%, 100% { transform: scale(0.94); opacity: 0.25; } 50% { transform: scale(1.06); opacity: 0.7; } }

@media (prefers-reduced-motion: reduce) {
  .empty-state, .empty-state__icon::after { animation: none; }
}
"""

DEFAULT_ICON = "chart-bar"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_icon(name: Optional[str], size: int) -> str:
    """Icon markup, or an empty string if the icon is missing or the lookup fails."""
    if not name:
        return ""
    try:
        return icon_html(name, size=size) or ""
    except (KeyError, ValueError, TypeError):
        return ""


def _clean(text: Any) -> str:
    """Collapse whitespace; escape before rendering."""
    return " ".join(str(text or "").split())


def _styles_once() -> None:
    """Inject the stylesheet once per script run (a cheap no-op on later calls).

    Streamlit creates a fresh ``cursors`` mapping for every script run, so holding a
    reference to it in session state identifies "this run". If Streamlit's internals
    change, the stylesheet is simply injected every time, which is safe, only wasteful.
    """
    marker = "_empty_state_css_run"
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        run_token = getattr(ctx, "cursors", None) if ctx else None
        if run_token is not None:
            if st.session_state.get(marker) is run_token:
                return
            st.session_state[marker] = run_token
    except Exception:  # pragma: no cover - never let styling break the page
        pass
    st.markdown(f"<style>{_EMPTY_STATE_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def empty_state_html(
    title: str,
    description: str = "",
    *,
    icon: Optional[str] = DEFAULT_ICON,
    hints: Optional[Sequence[str]] = None,
    compact: bool = False,
    plain: bool = False,
) -> str:
    """Return the empty-state markup (no styles, no Streamlit calls).

    ``plain=True`` drops the card background and uses a solid border, for use
    inside another card.
    """
    title_text = _clean(title) or "Nothing here yet"
    classes = "empty-state" + (" empty-state--compact" if compact else "")
    style = ' style="border-style:solid;background:transparent"' if plain else ""

    parts = [f'<div class="{classes}" role="status"{style}>']
    icon_markup = _safe_icon(icon, 22 if compact else 26)
    if icon_markup:
        parts.append(
            f'<div class="empty-state__icon" aria-hidden="true">{icon_markup}</div>'
        )
    parts.append(f'<p class="empty-state__title">{escape(title_text)}</p>')

    description_text = _clean(description)
    if description_text:
        parts.append(
            f'<p class="empty-state__description">{escape(description_text)}</p>'
        )

    hint_items = [h for h in (_clean(x) for x in (hints or [])) if h][:4]
    if hint_items:
        items = "".join(f"<li>{escape(h)}</li>" for h in hint_items)
        parts.append(f'<ul class="empty-state__hints">{items}</ul>')

    parts.append("</div>")
    return "".join(parts)


def render_empty_state(
    title: str,
    description: str = "",
    *,
    icon: Optional[str] = DEFAULT_ICON,
    hints: Optional[Sequence[str]] = None,
    action_label: Optional[str] = None,
    on_click: Optional[Callable[..., None]] = None,
    args: Sequence[Any] = (),
    key: Optional[str] = None,
    compact: bool = False,
    plain: bool = False,
) -> bool:
    """Render the empty state. Returns True if the action button was clicked.

    An action button is shown only when ``action_label`` is given. Pass a unique
    ``key`` whenever you render more than one empty state on a page.
    """
    _styles_once()
    markup = empty_state_html(
        title, description, icon=icon, hints=hints, compact=compact, plain=plain
    )

    if not action_label:
        st.markdown(markup, unsafe_allow_html=True)
        return False

    suffix = "".join(c if c.isalnum() or c == "_" else "_" for c in (key or "default"))
    with st.container(key=f"empty_state_{suffix}"):
        st.markdown(markup, unsafe_allow_html=True)
        return bool(
            st.button(
                action_label,
                key=f"empty_state_btn_{suffix}",
                on_click=on_click,
                args=tuple(args),
            )
        )
