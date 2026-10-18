from __future__ import annotations

from html import escape
from typing import Optional

import streamlit as st

_STATUS_TONES = {"neutral", "success", "warning", "error", "info"}


def render_header(
    title: str,
    description: str = "",
    *,
    eyebrow: Optional[str] = None,
    status: Optional[str] = None,
    status_tone: str = "neutral",
) -> None:
    """
    Render the page header.

    Args:
        title:       Page title (required).
        description: Short supporting sentence under the title.
        eyebrow:     Small uppercase label above the title, e.g. "Workspace".
        status:      Optional status text, e.g. "Connected".
        status_tone: One of neutral, success, warning, error, info.
    """
    tone = status_tone if status_tone in _STATUS_TONES else "neutral"

    parts = ['<div class="app-header"><div>']
    if eyebrow:
        parts.append(f'<p class="app-header__eyebrow">{escape(eyebrow)}</p>')
    parts.append(f'<h1 class="app-header__title">{escape(title)}</h1>')
    if description:
        parts.append(f'<p class="app-header__subtitle">{escape(description)}</p>')
    parts.append("</div>")

    if status:
        parts.append(
            '<div class="app-header__actions">'
            f'<span class="status-pill status-pill--{tone}">'
            '<span class="status-pill__dot"></span>'
            f"{escape(status)}"
            "</span></div>"
        )

    parts.append("</div>")
    st.markdown("".join(parts), unsafe_allow_html=True)
