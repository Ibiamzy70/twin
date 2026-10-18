"""
Workspace navigation shell.

All navigation lives here. To add or rename a view, edit ``ROUTES`` only;
the radio, the icons and the route lookup are all derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import streamlit as st

from components.icons import icon_data_uri


@dataclass(frozen=True)
class Route:
    """A navigable view: stable internal id, visible label, Phosphor icon name."""

    id: str
    label: str
    icon: str


ROUTES: Tuple[Route, ...] = (
    Route(id="overview", label="Overview", icon="chart-bar"),
    Route(id="assistant", label="AI Assistant", icon="chat-circle-dots"),
    Route(id="calculator", label="Calculator", icon="calculator"),
    Route(id="tasks", label="Tasks", icon="list-checks"),
    Route(id="settings", label="Settings", icon="gear"),
)

ROUTES_BY_ID: Dict[str, Route] = {route.id: route for route in ROUTES}
DEFAULT_ROUTE_ID: str = ROUTES[0].id

NAV_KEY = "workspace_nav"
# Streamlit (1.39+) adds a ``st-key-<key>`` class to the widget container.
_NAV_SELECTOR = f".st-key-{NAV_KEY} [role='radiogroup']"


def _inject_nav_icons() -> None:
    """Attach each route's icon to its radio label via a CSS mask (theme-aware)."""
    rules = [
        f"{_NAV_SELECTOR} label{{display:flex;align-items:center;gap:var(--space-3);}}"
    ]
    for position, route in enumerate(ROUTES, start=1):
        uri = icon_data_uri(route.icon)
        if uri is None:
            continue  # missing asset: the label simply renders without an icon
        rules.append(
            f"{_NAV_SELECTOR} label:nth-of-type({position})::before{{"
            "content:'';display:block;flex:none;width:1.25rem;height:1.25rem;"
            "background-color:currentColor;"
            f"-webkit-mask:url('{uri}') center/contain no-repeat;"
            f"mask:url('{uri}') center/contain no-repeat;"
            "}"
        )
    rules.append(
        f"{_NAV_SELECTOR} label:has(input:checked)::before{{"
        "background-color:var(--color-accent);}"
    )
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def _render_brand() -> None:
    st.markdown(
        "".join(
            [
                '<div class="sidebar-brand">',
                '<div class="sidebar-brand__mark">D</div>',
                "<div>",
                '<p class="sidebar-brand__name">Digital Twin</p>',
                '<p class="sidebar-brand__tagline">Personal workspace</p>',
                "</div>",
                "</div>",
                '<div class="sidebar-section-label">Workspace</div>',
            ]
        ),
        unsafe_allow_html=True,
    )


def get_route(route_id: str) -> Route:
    """Look up a route by id, falling back to the default route."""
    return ROUTES_BY_ID.get(route_id, ROUTES_BY_ID[DEFAULT_ROUTE_ID])


def render_sidebar() -> str:
    """
    Render the sidebar and return the selected route id (e.g. ``"tasks"``).

    Call once per run, after ``st.set_page_config()`` and the theme is loaded.
    """
    if st.session_state.get(NAV_KEY) not in ROUTES_BY_ID:
        st.session_state[NAV_KEY] = DEFAULT_ROUTE_ID

    with st.sidebar:
        _render_brand()
        _inject_nav_icons()
        route_id = st.radio(
            "Workspace navigation",
            options=[route.id for route in ROUTES],
            format_func=lambda rid: ROUTES_BY_ID[rid].label,
            key=NAV_KEY,
            label_visibility="collapsed",
        )

    return route_id
