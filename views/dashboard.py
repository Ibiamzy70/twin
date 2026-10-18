"""
Overview dashboard.

Structure
---------
  Header                    title + description
  Headline metrics          three native ``st.metric`` tiles (always cover all data)
  Modules                   four square cards, one per view
  Tabs
    Performance             progress bars, derived metric cards, profile / weekly
                            events / completion ring / task list
    Activity                recent conversations and task activity   (filtered)
    Reports                 summary table, preferences, detail table + CSV  (filtered)
  Expanders                 how metrics are calculated, recently completed tasks,
                            assistant usage breakdown, data notes
  Sidebar filters           task status, activity type, date range
                            (apply to Activity and Reports only, never to headline totals)

Navigation
----------
Every card opens a view, exactly like the sidebar: the click sets the sidebar radio's
session-state value (``NAV_KEY``) in a button callback, so the next run renders that view.
Which view each card opens is the ``CARD_ROUTES`` table in
``services/dashboard_service.py``; the ids must be ``components/sidebar.py::ROUTES`` ids.

  Whole card is the click target     navigation cards, headline metrics, derived metrics,
                                     Profile, Completion
  A visible "Open ..." link instead  Task list and the two activity feeds, because those
                                     scroll and a covering click target would block scrolling

Where the numbers come from
---------------------------
All counting lives in ``services/dashboard_service.py`` (no Streamlit there, unit tested).
This module only reads the session, asks the service, and draws the result. Tasks come from
the shared task store, so a task the twin creates is counted here with no sync step.

All colors, spacing, radii and shadows come from ``styles/theme.css`` tokens.
All dynamic text is HTML-escaped before it is rendered. Icons are Phosphor SVGs
(``assets/icons/``); there are no emoji anywhere on this page.
"""

from __future__ import annotations

import csv
import inspect
import io
import logging
from contextlib import contextmanager
from datetime import date, datetime
from functools import lru_cache
from html import escape
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import streamlit as st
from streamlit.errors import StreamlitAPIException

from components.app_header import render_header
from components.empty_state import empty_state_html, render_empty_state
from components.icons import icon_data_uri, icon_html
from components.metric_card import render_metric_card
from components.sidebar import NAV_KEY, ROUTES_BY_ID
from services import dashboard_service as ds
from services.dashboard_service import (
    CARD_ROUTES,
    DashboardData,
    Event,
    Filters,
    ModuleCard,
)
from state import CHAT_COUNT, MESSAGES, SETTINGS, USER_PROFILE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_TASKS_SHOWN = 20
MAX_FEED_ITEMS = 12

STATUS_OPTIONS = ds.STATUS_OPTIONS
KIND_OPTIONS = ds.KIND_OPTIONS

KEY_STATUS = "dash_filter_status"
KEY_KINDS = "dash_filter_kinds"
KEY_RANGE = "dash_filter_range"

# Session keys for this app (single source: state/session.py).
_SERVICE = ds.DashboardService(
    keys=ds.SessionKeys(
        messages=MESSAGES,
        chat_count=CHAT_COUNT,
        profile=USER_PROFILE,
        settings=SETTINGS,
    )
)


# ---------------------------------------------------------------------------
# Clock and data
# ---------------------------------------------------------------------------
def _today() -> date:
    """Today in the visitor's timezone when Streamlit can tell us, else server time.

    A server running in UTC would otherwise flip the weekly chart to tomorrow hours early
    or late for the person using the app.
    """
    try:
        tz_name = st.context.timezone
        if tz_name:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz_name)).date()
    except Exception:  # old Streamlit, unknown zone, or no tzdata installed
        pass
    return date.today()


def _collect_data() -> DashboardData:
    return _SERVICE.load(st.session_state)


# ---------------------------------------------------------------------------
# Filters (sidebar)
# ---------------------------------------------------------------------------
def _reset_filters() -> None:
    """Callback: dropping the widget keys restores every filter to its default."""
    for key in (KEY_STATUS, KEY_KINDS, KEY_RANGE):
        st.session_state.pop(key, None)


def _render_filters(data: DashboardData, today: date) -> Filters:
    start: Optional[date] = None
    end: Optional[date] = None
    bounds = ds.date_bounds(data.dated_dates, today)
    default_range = bounds[0] if bounds else None

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-label">Filters</div>', unsafe_allow_html=True
        )
        statuses = st.multiselect(
            "Task status",
            list(STATUS_OPTIONS),
            default=list(STATUS_OPTIONS),
            key=KEY_STATUS,
        )
        kinds = st.multiselect(
            "Activity type",
            list(KIND_OPTIONS),
            default=list(KIND_OPTIONS),
            key=KEY_KINDS,
        )

        if bounds:
            _, min_value, max_value = bounds
            limits = {"min_value": min_value, "max_value": max_value}
            try:
                chosen = st.date_input(
                    "Date range", value=default_range, key=KEY_RANGE, **limits
                )
            except StreamlitAPIException:
                # A stored selection can fall outside new bounds (for example after
                # the oldest item was removed). Drop it and start from the default.
                st.session_state.pop(KEY_RANGE, None)
                chosen = st.date_input(
                    "Date range", value=default_range, key=KEY_RANGE, **limits
                )
            if isinstance(chosen, (tuple, list)):
                if len(chosen) == 2:
                    start, end = chosen
                elif len(chosen) == 1:
                    start, end = chosen[0], default_range[1]
            elif isinstance(chosen, date):
                start = end = chosen
        else:
            st.caption(
                "The date filter appears once messages or tasks carry timestamps."
            )

        st.button("Reset filters", key="dash_reset_filters", on_click=_reset_filters)
        st.caption(
            "Filters apply to Activity and Reports. Headline totals always include everything."
        )

    narrowed = ds.is_narrowed(statuses, kinds, start, end, default_range)
    return Filters(frozenset(statuses), frozenset(kinds), start, end, narrowed)


# ---------------------------------------------------------------------------
# Streamlit helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _stretch_kwargs(func_name: str) -> Dict[str, Any]:
    """Full-width option for the installed Streamlit: ``width="stretch"`` on current
    releases, ``use_container_width=True`` on older ones."""
    params = inspect.signature(getattr(st, func_name)).parameters
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def _show_dataframe(rows: Sequence[Dict[str, Any]]) -> None:
    """Full-width table that works across Streamlit versions.

    Older releases give ``st.dataframe`` an integer-only ``width``, so a signature
    check is not enough; try the current option and fall back.
    """
    try:
        st.dataframe(rows, hide_index=True, width="stretch")
    except (TypeError, ValueError, StreamlitAPIException):
        try:
            st.dataframe(rows, hide_index=True, use_container_width=True)
        except TypeError:
            st.dataframe(rows, hide_index=True)


# ``delta_arrow`` exists only on newer Streamlit releases.
_SUPPORTS_DELTA_ARROW = "delta_arrow" in inspect.signature(st.metric).parameters


def _navigate(route_id: str) -> None:
    """Button callback: runs before the rerun, so the sidebar radio can be changed."""
    if route_id in ROUTES_BY_ID:
        st.session_state[NAV_KEY] = route_id


def _route(card_id: str) -> str:
    return CARD_ROUTES[card_id]


def _link_button(card_id: str, label: str) -> None:
    route = _route(card_id)
    st.button(
        label,
        key=f"dash-go-{card_id}",
        on_click=_navigate,
        args=(route,),
        disabled=route not in ROUTES_BY_ID,
        **_stretch_kwargs("button"),
    )


@contextmanager
def _linked(card_id: str, label: str) -> Iterator[None]:
    """Make everything drawn inside a click target that opens the card's view.

    The button is stretched over the content by CSS (see ``.st-key-dash-link-*``); its label
    stays in the page for keyboard and screen-reader users.
    """
    with st.container(key=f"dash-link-{card_id}"):
        yield
        _link_button(card_id, label)


def _more_link(card_id: str, label: str) -> None:
    """A visible 'Open ...' link under a card whose content scrolls."""
    with st.container(key=f"dash-more-{card_id}"):
        _link_button(card_id, label)


def _inject_component_styles() -> None:
    """Make sure the shared components' stylesheets are present even when only their
    ``*_html`` helpers are used."""
    try:
        from components.empty_state import _styles_once as empty_styles
        from components.metric_card import _styles_once as metric_styles
    except ImportError:
        return
    empty_styles()
    metric_styles()


_KIND_EMPTY = {
    "Messages": (
        "No messages yet",
        "Start a conversation in the AI Assistant.",
        "chat-circle-dots",
    ),
    "Tasks": ("No tasks yet", "Add one from the Tasks view.", "list-checks"),
}


def _no_results(
    kind: Optional[str], all_events: Sequence[Event], filters: Filters, key: str
) -> None:
    """Empty state for a list with nothing to show.

    The filter message (with a reset button) appears only when items of this kind
    exist and the active filters hid them. Otherwise it says nothing exists yet.
    ``kind=None`` covers every kind (used by the Reports table).
    """
    exists = any(kind is None or event.kind == kind for event in all_events)
    if exists and filters.narrowed:
        render_empty_state(
            "No items match your filters",
            "Widen the date range or select more types and statuses.",
            icon="chart-bar",
            action_label="Reset filters",
            on_click=_reset_filters,
            key=key,
            compact=True,
        )
        return

    title, text, icon = _KIND_EMPTY.get(
        kind,
        (
            "No activity yet",
            "Messages and tasks appear here as you use the app.",
            "chart-bar",
        ),
    )
    render_empty_state(title, text, icon=icon, key=key, compact=True)


# ---------------------------------------------------------------------------
# HTML builders
# ---------------------------------------------------------------------------
def _esc(value: Any) -> str:
    """Escape for inline HTML; also neutralizes markdown math and blank lines."""
    return escape(" ".join(str(value).split())).replace("$", "&#36;")


def _html(*parts: str) -> str:
    return "".join(parts)


def _section(title: str) -> None:
    st.markdown(
        f'<div class="dash-section">{_esc(title)}</div>', unsafe_allow_html=True
    )


def _bars_html(d: DashboardData) -> str:
    bars = [("Task completion", d.completion_pct), ("Open workload", d.open_pct)]
    inner = "".join(
        _html(
            '<div class="dash-bar">',
            f'<div class="dash-bar__head"><span>{_esc(label)}</span><strong>{pct}%</strong></div>',
            f'<div class="dash-bar__track"><div class="dash-bar__fill" style="--pct:{pct}"></div></div>',
            "</div>",
        )
        for label, pct in bars
    )
    return f'<div class="dash-bars">{inner}</div>'


def _module_html(card: ModuleCard, index: int) -> str:
    return _html(
        f'<div class="dash-card dash-module" style="--i:{index}">',
        '<div class="dash-card__head">',
        f'<div class="dash-module__icon">{icon_html(card.icon, 24)}</div>',
        f'<div class="dash-module__go">{icon_html("arrow-up-right", 16)}</div>',
        "</div>",
        '<div class="dash-module__body">',
        f'<div class="dash-card__title">{_esc(card.title)}</div>',
        f'<div class="dash-card__text">{_esc(card.description)}</div>',
        f'<span class="dash-chip">{_esc(card.metric)}</span>',
        "</div>",
        "</div>",
    )


def _profile_html(d: DashboardData, index: int) -> str:
    return _html(
        f'<div class="dash-card" style="--i:{index}">',
        '<div class="dash-card__head">',
        '<div class="dash-card__title">Profile</div>',
        '<span class="status-pill status-pill--success"><span class="status-pill__dot"></span>Active</span>',
        "</div>",
        '<div class="dash-profile">',
        f'<div class="dash-avatar">{_esc(d.initials)}</div>',
        f'<div class="dash-profile__name">{_esc(d.name)}</div>',
        f'<div class="dash-card__text">{_esc(d.role)}</div>',
        "</div>",
        '<div class="dash-mini">',
        f'<div class="dash-mini__item"><strong>{len(d.messages)}</strong><span>Messages</span></div>',
        f'<div class="dash-mini__item"><strong>{d.total_tasks}</strong><span>Tasks</span></div>',
        "</div>",
        "</div>",
    )


def _activity_html(
    d: DashboardData, week: List[Tuple[str, int, bool]], index: int
) -> str:
    total = sum(count for _, count, _ in week)
    head = _html(
        f'<div class="dash-card" style="--i:{index}">',
        '<div class="dash-card__head"><div class="dash-card__title">Weekly events</div></div>',
    )
    if total == 0:
        if d.dated_dates:
            title, text = (
                "A quiet week",
                "No dated messages or tasks in the last 7 days.",
            )
        else:
            title, text = (
                "No dated activity yet",
                "Messages and tasks with timestamps will appear here.",
            )
        return (
            head
            + empty_state_html(title, text, icon="chart-bar", compact=True, plain=True)
            + "</div>"
        )

    peak = max(count for _, count, _ in week)
    cols = []
    for position, (label, count, is_today) in enumerate(week):
        height = round(count / peak * 100)
        today_class = " is-today" if is_today else ""
        cols.append(
            _html(
                f'<div class="dash-chart__col{today_class}" title="{count}">',
                '<div class="dash-chart__track">',
                f'<div class="dash-chart__bar" style="--h:{height};--i:{position}"></div>',
                "</div>",
                f'<div class="dash-chart__day">{_esc(label)}</div>',
                "</div>",
            )
        )
    return _html(
        head,
        f'<div class="dash-figure__value">{total}</div>',
        '<div class="dash-card__text">Dated messages and tasks, last 7 days</div>',
        f'<div class="dash-chart">{"".join(cols)}</div>',
        "</div>",
    )


def _completion_html(d: DashboardData, index: int) -> str:
    footer = (
        f"{d.done_tasks} of {d.total_tasks} tasks done"
        if d.total_tasks
        else "No tasks yet"
    )
    return _html(
        f'<div class="dash-card" style="--i:{index}">',
        '<div class="dash-card__head"><div class="dash-card__title">Completion</div></div>',
        f'<div class="dash-ring" style="--pct:{d.completion_pct}">',
        f'<span class="dash-ring__value">{d.completion_pct}%</span>',
        "</div>",
        f'<div class="dash-card__text dash-card__text--center">{_esc(footer)}</div>',
        "</div>",
    )


def _tasks_html(d: DashboardData, index: int) -> str:
    rows = sorted(d.tasks, key=lambda t: t.done)[:MAX_TASKS_SHOWN]  # open first
    if rows:
        items = []
        for task in rows:
            done_class = " is-done" if task.done else ""
            mark = icon_html("check", 12) if task.done else ""
            items.append(
                _html(
                    f'<li class="dash-task{done_class}">',
                    f'<span class="dash-task__mark">{mark}</span>',
                    f'<span class="dash-task__text">{_esc(task.label)}</span>',
                    "</li>",
                )
            )
        body = f'<ul class="dash-list">{"".join(items)}</ul>'
    else:
        body = empty_state_html(
            "No tasks yet",
            "Add one from the Tasks view.",
            icon="list-checks",
            compact=True,
            plain=True,
        )
    return _html(
        f'<div class="dash-card dash-card--accent" style="--i:{index}">',
        '<div class="dash-card__head">',
        '<div class="dash-card__title">Tasks</div>',
        f'<div class="dash-card__figure">{d.done_tasks}/{d.total_tasks}</div>',
        "</div>",
        body,
        "</div>",
    )


def _feed_html(
    title: str, meta: str, events: Sequence[Event], icon: str, index: int
) -> str:
    items = []
    for event in events[:MAX_FEED_ITEMS]:
        when = event.when.strftime("%b %d") if event.when else "No date"
        items.append(
            _html(
                '<li class="dash-feed__item">',
                f'<span class="dash-feed__icon">{icon_html(icon, 16)}</span>',
                f'<span class="dash-feed__text">{_esc(event.title)}</span>',
                f'<span class="dash-chip dash-chip--quiet">{_esc(event.detail)}</span>',
                f'<span class="dash-feed__meta">{_esc(when)}</span>',
                "</li>",
            )
        )
    return _html(
        f'<div class="dash-card dash-card--wide" style="--i:{index}">',
        f'<div class="dash-card__head"><div class="dash-card__title">{_esc(title)}</div>',
        f'<div class="dash-card__meta">{_esc(meta)}</div></div>',
        f'<ul class="dash-list dash-list--feed">{"".join(items)}</ul>',
        "</div>",
    )


def _preferences_html(d: DashboardData, index: int) -> str:
    rows = []
    for key, value in list(d.settings.items())[:8]:
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, bool):
            tone = "success" if value else "neutral"
            shown = (
                f'<span class="status-pill status-pill--{tone}">'
                f'<span class="status-pill__dot"></span>{"On" if value else "Off"}</span>'
            )
        else:
            shown = f'<span class="dash-row__value">{_esc(value)}</span>'
        rows.append(f'<div class="dash-row"><span>{_esc(label)}</span>{shown}</div>')
    body = "".join(rows) or empty_state_html(
        "No preferences", "Defaults are in use.", icon="gear", compact=True, plain=True
    )
    return _html(
        f'<div class="dash-card dash-card--wide" style="--i:{index}">',
        '<div class="dash-card__head"><div class="dash-card__title">Workspace preferences</div></div>',
        f'<div class="dash-rows">{body}</div>',
        "</div>",
    )


# ---------------------------------------------------------------------------
# Styles (tokens only - see styles/theme.css)
# ---------------------------------------------------------------------------
DASHBOARD_CSS = """
@property --pct { syntax: '<number>'; inherits: false; initial-value: 0; }

@keyframes dash-rise { from { opacity: 0; transform: translateY(10px); } }
@keyframes dash-bar  { from { transform: scaleY(0); } }
@keyframes dash-grow { from { width: 0; } }
@keyframes dash-ring { from { --pct: 0; } }

.dash-section {
  margin: var(--space-6) 0 var(--space-1);
  color: var(--color-text-muted); font-size: var(--font-size-xs);
  font-weight: var(--font-weight-semibold); letter-spacing: var(--letter-spacing-wide); text-transform: uppercase;
}

/* ---- Headline st.metric tiles ------------------------------------------ */
[class*="st-key-dash-headline"] [data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--color-surface-raised) 0%, var(--color-surface) 70%);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-sm);
  padding: var(--space-5);
  transition: transform var(--duration-base) var(--ease-standard),
              box-shadow var(--duration-base) var(--ease-standard),
              border-color var(--duration-base) var(--ease-standard);
  animation: dash-rise var(--duration-slow) var(--ease-standard) both;
}
[class*="st-key-dash-headline"] [data-testid="stMetric"]:hover {
  transform: translateY(-3px); box-shadow: var(--shadow-lg); border-color: var(--color-accent-border);
}
[class*="st-key-dash-headline"] [data-testid="stMetricValue"] { font-size: var(--font-size-3xl); }

/* ---- Progress bars ----------------------------------------------------- */
.dash-bars { display: flex; flex-wrap: wrap; gap: var(--space-4) var(--space-6); max-width: 40rem; }
.dash-bar { flex: 1 1 14rem; min-width: 0; display: flex; flex-direction: column; gap: var(--space-2); }
.dash-bar__head {
  display: flex; justify-content: space-between; align-items: baseline;
  color: var(--color-text-secondary); font-size: var(--font-size-xs); font-weight: var(--font-weight-medium);
}
.dash-bar__head strong { color: var(--color-text); font-size: var(--font-size-sm); }
.dash-bar__track { height: 0.625rem; border-radius: var(--radius-full); background: var(--color-surface-raised); border: var(--border); }
.dash-bar__fill {
  height: 100%; width: calc(var(--pct) * 1%); border-radius: var(--radius-full);
  background: linear-gradient(90deg, var(--color-accent-active), var(--color-accent));
  animation: dash-grow 900ms var(--ease-standard) both;
}
.dash-figure__value {
  color: var(--color-text); font-size: var(--font-size-3xl); font-weight: var(--font-weight-bold);
  letter-spacing: var(--letter-spacing-tight); line-height: var(--line-height-tight); margin-top: var(--space-2);
}

/* ---- Card base --------------------------------------------------------- */
.dash-card {
  position: relative; display: flex; flex-direction: column; aspect-ratio: 1 / 1;
  padding: var(--space-5);
  background: linear-gradient(180deg, var(--color-surface-raised) 0%, var(--color-surface) 55%);
  border: var(--border); border-radius: var(--radius-xl); box-shadow: var(--shadow-sm);
  transition: transform var(--duration-base) var(--ease-standard),
              box-shadow var(--duration-base) var(--ease-standard),
              border-color var(--duration-base) var(--ease-standard);
  animation: dash-rise var(--duration-slow) var(--ease-standard) both;
  animation-delay: calc(var(--i, 0) * 60ms);
}
.dash-card:hover { transform: translateY(-3px); box-shadow: var(--shadow-lg); border-color: var(--color-border-strong); }
.dash-card--accent {
  background: linear-gradient(160deg, var(--color-accent-soft) 0%, var(--color-surface) 70%);
  border-color: var(--color-accent-border);
}
.dash-card--wide { aspect-ratio: auto; min-height: 18rem; }
.dash-card .empty-state { flex: 1 1 auto; margin-top: var(--space-3); border-color: transparent; }

.dash-card__head { display: flex; align-items: flex-start; justify-content: space-between; gap: var(--space-3); }
.dash-card__title { color: var(--color-text); font-size: var(--font-size-lg); font-weight: var(--font-weight-semibold); letter-spacing: var(--letter-spacing-tight); }
.dash-card__figure { color: var(--color-accent); font-size: var(--font-size-xl); font-weight: var(--font-weight-bold); }
.dash-card__meta { color: var(--color-text-muted); font-size: var(--font-size-sm); }
.dash-card__text { margin-top: var(--space-1); color: var(--color-text-secondary); font-size: var(--font-size-sm); line-height: var(--line-height-normal); }
.dash-card__text--center { text-align: center; margin-top: auto; }

/* ---- Clickable cards ----------------------------------------------------
   The container holds the card and one Streamlit button. Streamlit wraps the button in its
   own element container, so THAT wrapper is what must be stretched over the card (stretching
   only the inner button leaves it inside a zero-height box). The button is invisible but
   stays in the page, focusable and labelled for assistive technology. */
[class*="st-key-dash-link-"] {
  position: relative; gap: 0 !important; border-radius: var(--radius-xl);
  transition: transform var(--duration-base) var(--ease-standard);
}
[class*="st-key-dash-link-"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
  position: absolute; inset: 0; z-index: 3; width: 100%; height: 100%; margin: 0;
}
[class*="st-key-dash-link-"] [data-testid="stButton"] { width: 100%; height: 100%; margin: 0; }
[class*="st-key-dash-link-"] [data-testid="stButton"] button {
  width: 100%; height: 100%; min-height: 0; padding: 0; opacity: 0; cursor: pointer; border-radius: var(--radius-xl);
}
[class*="st-key-dash-link-"]:hover { transform: translateY(-4px); }
[class*="st-key-dash-link-"] .dash-card:hover { transform: none; }
[class*="st-key-dash-link-"]:hover .dash-card,
[class*="st-key-dash-link-"]:hover [data-testid="stMetric"] {
  box-shadow: var(--shadow-lg); border-color: var(--color-accent-border);
}
[class*="st-key-dash-link-"] [data-testid="stMetric"]:hover { transform: none; }
[class*="st-key-dash-link-"]:has(button:focus-visible) { outline: 2px solid var(--color-accent); outline-offset: 2px; }
[class*="st-key-dash-link-"]:has(button:disabled) { opacity: 0.55; }
[class*="st-key-dash-link-"]:has(button:disabled) [data-testid="stButton"] button { cursor: not-allowed; }

.dash-module__icon {
  display: grid; place-items: center; width: 2.75rem; height: 2.75rem; border-radius: var(--radius-md);
  background: var(--color-accent-soft); border: var(--border-width) solid var(--color-accent-border); color: var(--color-accent);
}
.dash-module__go {
  display: grid; place-items: center; width: 2rem; height: 2rem; border-radius: var(--radius-full);
  border: var(--border); color: var(--color-text-secondary);
  transition: background var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard), transform var(--duration-base) var(--ease-standard);
}
[class*="st-key-dash-link-"]:hover .dash-module__go { background: var(--color-accent); color: var(--color-text-on-accent); border-color: transparent; transform: translate(2px, -2px); }
.dash-module__body { margin-top: auto; display: flex; flex-direction: column; align-items: flex-start; gap: var(--space-1); }
.dash-chip {
  margin-top: var(--space-3); padding: 2px var(--space-3); border-radius: var(--radius-full);
  background: var(--color-accent-soft); color: var(--color-accent); font-size: var(--font-size-xs); font-weight: var(--font-weight-semibold);
}
.dash-chip--quiet { margin-top: 0; flex: none; background: var(--color-surface-raised); color: var(--color-text-secondary); }

/* ---- "Open ..." links under scrolling cards ------------------------------ */
[class*="st-key-dash-more-"] { margin-top: var(--space-2); }
[class*="st-key-dash-more-"] [data-testid="stButton"] button {
  width: 100%; min-height: 2.25rem; background: transparent; color: var(--color-text-secondary);
  border: var(--border); border-radius: var(--radius-md); font-size: var(--font-size-sm); font-weight: var(--font-weight-medium);
  transition: color var(--duration-fast) var(--ease-standard), border-color var(--duration-fast) var(--ease-standard),
              background var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-standard);
}
[class*="st-key-dash-more-"] [data-testid="stButton"] button:hover:not(:disabled) {
  color: var(--color-accent); border-color: var(--color-accent-border); background: var(--color-accent-soft);
}
[class*="st-key-dash-more-"] [data-testid="stButton"] button:active:not(:disabled) { transform: scale(0.98); }
[class*="st-key-dash-more-"] [data-testid="stButton"] button:focus-visible { outline: 2px solid var(--color-accent); outline-offset: 2px; }
[class*="st-key-dash-more-"] [data-testid="stButton"] button p { margin: 0; color: inherit; }

/* ---- Profile ----------------------------------------------------------- */
.dash-profile { display: flex; flex-direction: column; align-items: center; text-align: center; gap: var(--space-1); margin: var(--space-3) 0; }
.dash-avatar {
  display: grid; place-items: center; width: 4.5rem; height: 4.5rem; border-radius: var(--radius-full);
  background: linear-gradient(135deg, var(--color-accent), var(--color-accent-active));
  color: var(--color-text-on-accent); font-size: var(--font-size-xl); font-weight: var(--font-weight-bold); box-shadow: var(--shadow-glow);
}
.dash-profile__name { margin-top: var(--space-2); color: var(--color-text); font-size: var(--font-size-lg); font-weight: var(--font-weight-semibold); }
.dash-mini { display: grid; grid-template-columns: 1fr 1fr; gap: var(--space-2); margin-top: auto; }
.dash-mini__item { display: flex; flex-direction: column; align-items: center; padding: var(--space-2) var(--space-3); border-radius: var(--radius-md); background: var(--color-surface-sunken); border: var(--border); }
.dash-mini__item strong { color: var(--color-text); font-size: var(--font-size-md); }
.dash-mini__item span { color: var(--color-text-muted); font-size: var(--font-size-xs); }

/* ---- Weekly chart ------------------------------------------------------ */
.dash-chart { display: flex; align-items: flex-end; gap: var(--space-2); margin-top: auto; padding-top: var(--space-3); }
.dash-chart__col { flex: 1 1 0; display: flex; flex-direction: column; align-items: center; gap: var(--space-2); }
.dash-chart__track { height: 6rem; width: 100%; display: flex; align-items: flex-end; justify-content: center; }
.dash-chart__bar {
  width: 100%; max-width: 1.25rem; height: max(6px, calc(var(--h) * 0.06rem)); border-radius: var(--radius-full);
  background: var(--color-border-strong); transform-origin: bottom;
  transition: background var(--duration-base) var(--ease-standard);
  animation: dash-bar 700ms var(--ease-standard) both; animation-delay: calc(var(--i, 0) * 70ms);
}
.dash-chart__col:hover .dash-chart__bar { background: var(--color-accent-hover); }
.dash-chart__col.is-today .dash-chart__bar { background: var(--color-accent); }
.dash-chart__day { color: var(--color-text-muted); font-size: var(--font-size-xs); }
.dash-chart__col.is-today .dash-chart__day { color: var(--color-accent); font-weight: var(--font-weight-semibold); }

/* ---- Completion ring --------------------------------------------------- */
.dash-ring {
  position: relative; display: grid; place-items: center; width: min(100%, 9.5rem); aspect-ratio: 1 / 1;
  margin: var(--space-3) auto; border-radius: var(--radius-full);
  background: conic-gradient(var(--color-accent) calc(var(--pct) * 1%), var(--color-surface-raised) 0);
  animation: dash-ring 1s var(--ease-standard) both;
}
.dash-ring::before { content: ""; position: absolute; inset: 0.7rem; border-radius: var(--radius-full); background: var(--color-surface); }
.dash-ring__value { position: relative; color: var(--color-text); font-size: var(--font-size-2xl); font-weight: var(--font-weight-bold); letter-spacing: var(--letter-spacing-tight); }

/* ---- Lists ------------------------------------------------------------- */
.dash-list {
  list-style: none; margin: var(--space-3) 0 0; padding: 0 var(--space-1) 0 0;
  display: flex; flex-direction: column; gap: var(--space-2);
  flex: 1 1 auto; min-height: 0; overflow-y: auto; overscroll-behavior: auto;
}
.dash-list--feed { max-height: 15rem; }
.dash-task {
  display: flex; align-items: center; gap: var(--space-3); padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md); background: var(--color-surface-sunken); border: var(--border);
  transition: border-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-standard);
}
.dash-task:hover { border-color: var(--color-accent-border); transform: translateX(2px); }
.dash-task__mark {
  flex: none; display: grid; place-items: center; width: 1.25rem; height: 1.25rem; border-radius: var(--radius-full);
  border: var(--border-width-thick) solid var(--color-border-strong); color: var(--color-text-on-accent);
}
.dash-task.is-done .dash-task__mark { background: var(--color-accent); border-color: var(--color-accent); }
.dash-task__text { min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--color-text); font-size: var(--font-size-sm); font-weight: var(--font-weight-medium); }
.dash-task.is-done .dash-task__text { color: var(--color-text-muted); text-decoration: line-through; }

.dash-feed__item {
  display: flex; align-items: center; gap: var(--space-3); padding: var(--space-3); border-radius: var(--radius-md);
  background: var(--color-surface-sunken); border: var(--border);
  transition: border-color var(--duration-fast) var(--ease-standard);
}
.dash-feed__item:hover { border-color: var(--color-accent-border); }
.dash-feed__icon { flex: none; display: grid; place-items: center; color: var(--color-accent); }
.dash-feed__text { flex: 1 1 auto; min-width: 0; color: var(--color-text); font-size: var(--font-size-sm); }
.dash-feed__meta { flex: none; color: var(--color-text-muted); font-size: var(--font-size-xs); }

.dash-rows { margin-top: var(--space-2); display: flex; flex-direction: column; max-height: 15rem; overflow-y: auto; }
.dash-row {
  display: flex; align-items: center; justify-content: space-between; gap: var(--space-3);
  padding: var(--space-3) 0; border-bottom: var(--border); color: var(--color-text); font-size: var(--font-size-sm);
}
.dash-row:last-child { border-bottom: 0; }
.dash-row__value { color: var(--color-text-secondary); }

@media (max-width: 640px) {
  .dash-card { aspect-ratio: auto; min-height: 12rem; }
}

@media (prefers-reduced-motion: reduce) {
  .dash-card, .dash-bar__fill, .dash-chart__bar, .dash-ring,
  [class*="st-key-dash-headline"] [data-testid="stMetric"] { animation: none; }
  [class*="st-key-dash-link-"], [class*="st-key-dash-link-"]:hover { transition: none; transform: none; }
}
"""


def _link_icon_css() -> str:
    """Phosphor arrow on the 'Open ...' links (a CSS mask, so it follows the text colour)."""
    uri = icon_data_uri("arrow-up-right")
    if uri is None:
        return ""  # missing asset: the links simply render without an icon
    return (
        '[class*="st-key-dash-more-"] [data-testid="stButton"] button p::after {'
        "content:'';display:inline-block;width:0.95rem;height:0.95rem;margin-left:var(--space-2);"
        "vertical-align:-0.15em;background-color:currentColor;"
        f"-webkit-mask:url('{uri}') center/contain no-repeat;"
        f"mask:url('{uri}') center/contain no-repeat;}}"
    )


def _inject_css() -> None:
    st.markdown(
        f"<style>{DASHBOARD_CSS}{_link_icon_css()}</style>", unsafe_allow_html=True
    )


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------
def _render_headline_metrics(d: DashboardData) -> None:
    # No ``help=`` tooltips here: the whole tile is a link and its overlay would cover the
    # tooltip icon. The definitions are in the "How these metrics are calculated" expander.
    with st.container(key="dash-headline"):
        c1, c2, c3 = st.columns(3, gap="medium")
        with c1, _linked("conversations", "Open AI Assistant"):
            st.metric("Conversations", d.chats)
        with c2, _linked("tasks-completed", "Open Tasks"):
            if _SUPPORTS_DELTA_ARROW:
                st.metric(
                    "Tasks completed",
                    d.done_tasks,
                    delta=f"{d.open_tasks} open",
                    delta_color="off",
                    delta_arrow="off",
                )
            else:
                # No way to hide the arrow on this Streamlit version, so avoid a delta.
                st.metric("Tasks completed", d.done_tasks)
                st.caption(f"{d.open_tasks} open")
        with c3, _linked("assistant-interactions", "Open AI Assistant"):
            st.metric("Assistant interactions", d.assistant_replies)


def _render_modules(d: DashboardData) -> None:
    cards = ds.module_cards(d)
    for index, (column, card) in enumerate(
        zip(st.columns(len(cards), gap="medium"), cards)
    ):
        with column, _linked(card.card_id, f"Open {card.title}"):
            st.markdown(_module_html(card, index), unsafe_allow_html=True)


def _render_performance(
    d: DashboardData, all_events: Sequence[Event], today: date
) -> None:
    st.markdown(_bars_html(d), unsafe_allow_html=True)

    _section("Derived metrics")
    messages_week = [
        count for _, count, _ in ds.week_series((m.when for m in d.messages), today)
    ]
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, _linked("completion-rate", "Open Tasks"):
        render_metric_card(
            "Completion rate",
            d.completion_pct if d.total_tasks else None,
            icon="check",
            suffix="%" if d.total_tasks else "",
            progress=d.completion_pct if d.total_tasks else None,
            variant="accent",
            caption="Done tasks as a share of all tasks",
        )
    with c2, _linked("messages-per-chat", "Open AI Assistant"):
        render_metric_card(
            "Messages per chat",
            d.messages_per_chat,
            icon="chat-circle-dots",
            decimals=1,
            trend=messages_week if sum(messages_week) else None,
            caption="Stored messages divided by conversations",
        )
    with c3, _linked("replies-per-prompt", "Open AI Assistant"):
        render_metric_card(
            "Replies per prompt",
            d.replies_per_prompt,
            icon="chart-bar",
            decimals=1,
            caption="Assistant replies for each of your messages",
        )

    _section("Insights")
    week = ds.week_series((e.when for e in all_events), today)
    profile_col, weekly_col, completion_col, tasks_col = st.columns(4, gap="medium")
    with profile_col, _linked("profile", "Open Settings"):
        st.markdown(_profile_html(d, 4), unsafe_allow_html=True)
    with weekly_col:
        st.markdown(_activity_html(d, week, 5), unsafe_allow_html=True)
    with completion_col, _linked("completion", "Open Tasks"):
        st.markdown(_completion_html(d, 6), unsafe_allow_html=True)
    with tasks_col:
        st.markdown(_tasks_html(d, 7), unsafe_allow_html=True)
        _more_link("tasks", "Open Tasks")


def _render_activity(
    filtered: Sequence[Event], all_events: Sequence[Event], filters: Filters
) -> None:
    messages = [e for e in filtered if e.kind == "Messages"]
    tasks = [e for e in filtered if e.kind == "Tasks"]
    left, right = st.columns(2, gap="medium")
    with left:
        if messages:
            st.markdown(
                _feed_html(
                    "Recent conversations",
                    ds.count_label(len(messages), "message"),
                    messages,
                    "chat-circle-dots",
                    0,
                ),
                unsafe_allow_html=True,
            )
            _more_link("recent-conversations", "Open AI Assistant")
        else:
            _no_results("Messages", all_events, filters, "dash_activity_messages")
    with right:
        if tasks:
            st.markdown(
                _feed_html(
                    "Task activity",
                    ds.count_label(len(tasks), "task"),
                    tasks,
                    "list-checks",
                    1,
                ),
                unsafe_allow_html=True,
            )
            _more_link("task-activity", "Open Tasks")
        else:
            _no_results("Tasks", all_events, filters, "dash_activity_tasks")


def _csv_cell(value: Any) -> str:
    """Neutralize spreadsheet formula injection."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def _render_reports(
    d: DashboardData,
    filtered: Sequence[Event],
    all_events: Sequence[Event],
    filters: Filters,
) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        _section("Summary (all data)")
        _show_dataframe(
            [
                {"Metric": label, "Value": str(value)}
                for label, value in ds.summary_rows(d)
            ]
        )
    with right:
        _section("Configuration")
        st.markdown(_preferences_html(d, 0), unsafe_allow_html=True)

    _section("Detail (filtered)")
    if not filtered:
        _no_results(None, all_events, filters, "dash_reports")
        return

    rows = [
        {
            "Date": event.when.isoformat() if event.when else "No date",
            "Type": event.kind,
            "Item": event.title,
            "Detail": event.detail,
        }
        for event in filtered
    ]
    _show_dataframe(rows)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Date", "Type", "Item", "Detail"])
    for event in filtered:
        writer.writerow(
            [
                _csv_cell(event.when.isoformat() if event.when else ""),
                _csv_cell(event.kind),
                _csv_cell(event.title),
                _csv_cell(event.detail),
            ]
        )
    # UTF-8 with a byte-order mark so Excel reads non-English text correctly.
    csv_bytes = ("﻿" + buffer.getvalue()).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name="digital-twin-report.csv",
        mime="text/csv",
        key="dash_download_csv",
    )


def _render_expanders(d: DashboardData) -> None:
    with st.expander("How these metrics are calculated"):
        st.markdown(
            "- **Conversations** is the session's chat counter. It is not derived from stored messages.\n"
            "- **Tasks completed** counts tasks whose status is done.\n"
            "- **Assistant interactions** counts messages whose role is assistant.\n"
            "- **Completion rate** is completed tasks divided by all tasks.\n"
            "- **Messages per chat** is stored messages divided by conversations.\n"
            "- **Replies per prompt** is assistant replies divided by your messages.\n"
            "- **Weekly events** counts one event per dated message and per dated task over the last 7 days. "
            "A task is dated by its completion date when it has one, otherwise its creation date, so it is never counted twice.\n"
            "- Click any card to open the view it summarises."
        )

    with st.expander("Recently completed tasks"):
        done = sorted(
            (t for t in d.tasks if t.done),
            key=lambda t: t.when or date.min,
            reverse=True,
        )[:5]
        if done:
            items = "".join(
                _html(
                    '<li class="dash-task is-done">',
                    f'<span class="dash-task__mark">{icon_html("check", 12)}</span>',
                    f'<span class="dash-task__text">{_esc(t.label)}</span>',
                    "</li>",
                )
                for t in done
            )
            st.markdown(f'<ul class="dash-list">{items}</ul>', unsafe_allow_html=True)
        else:
            st.caption("No completed tasks yet.")

    with st.expander("Assistant usage breakdown"):
        body = "".join(
            f'<div class="dash-row"><span>{_esc(label)}</span><span class="dash-row__value">{value}</span></div>'
            for label, value in ds.usage_rows(d)
        )
        st.markdown(f'<div class="dash-rows">{body}</div>', unsafe_allow_html=True)
        ratio = d.replies_per_prompt
        st.caption(
            f"{ratio} assistant replies per message from you."
            if ratio is not None
            else "No messages from you yet."
        )

    with st.expander("Data notes"):
        st.markdown("\n".join(f"- {note}" for note in ds.data_notes(d)))


# ---------------------------------------------------------------------------
# View
# ---------------------------------------------------------------------------
def render() -> None:
    """Render the Overview dashboard."""
    _inject_css()
    _inject_component_styles()

    today = _today()
    data = _collect_data()
    filters = _render_filters(data, today)
    all_events = ds.build_events(data)
    filtered = ds.apply_filters(all_events, filters)

    render_header(
        f"Welcome back, {data.name}",
        "Your workspace at a glance.",
        eyebrow="Overview",
        status="Online",
        status_tone="success",
    )

    if data.tasks_unavailable:
        st.warning(
            "Tasks can't be loaded right now, so task figures show as zero until they can."
        )

    _render_headline_metrics(data)

    _section("Modules")
    _render_modules(data)

    if data.is_empty:
        first = ds.module_cards(data)[0]
        can_navigate = first.route_id in ROUTES_BY_ID
        st.write("")
        render_empty_state(
            "Your workspace is empty",
            "Start a conversation, add a task or run a calculation and your metrics will appear here.",
            icon="chart-bar",
            hints=[card.title for card in ds.module_cards(data)[:3]],
            action_label=f"Open {first.title}" if can_navigate else None,
            on_click=_navigate if can_navigate else None,
            args=(first.route_id,),
            key="dash_page",
        )
    else:
        st.write("")
        performance, activity, reports = st.tabs(["Performance", "Activity", "Reports"])
        with performance:
            _render_performance(data, all_events, today)
        with activity:
            _render_activity(filtered, all_events, filters)
        with reports:
            _render_reports(data, filtered, all_events, filters)

    st.write("")
    _render_expanders(data)
