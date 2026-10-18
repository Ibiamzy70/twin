from __future__ import annotations

import csv
import io
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import streamlit as st

from services import task_service
from services.task_service import TaskError

logger = logging.getLogger("twin_tasks")
logger.propagate = False
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MAX_TEXT = 200  # characters per task
MAX_ASSIGNEE = 40
TABLE_PAGE = 8  # rows shown in the Overview table before "Show more"
SCHEDULE_MAX = 5  # items shown in "Today's schedule"
WEEK = 7  # days used for "this week" figures

TABS: Tuple[str, ...] = ("Overview", "List", "Board", "Timeline", "Files")
VIEW_OPTIONS: Tuple[str, ...] = ("All tasks", "Active", "Done")
SORT_OPTIONS: Tuple[str, ...] = ("Newest", "Due date", "Name", "Status")
SCHEDULE_FILTERS: Tuple[str, ...] = ("All", "Open", "Done")


@dataclass(frozen=True)
class Status:
    id: str
    label: str


STATUSES: Dict[str, Status] = {
    "todo": Status("todo", "To do"),
    "in_progress": Status("in_progress", "In Progress"),
    "in_review": Status("in_review", "In Review"),
    "done": Status("done", "Done"),
}
STATUS_ORDER: Tuple[str, ...] = tuple(STATUSES)

# session_state keys (the tasks themselves live in services.task_service, not in session_state)
KEY_CONFIRM = "confirm_clear"
KEY_FLASH = "flash"
KEY_PREFS = "tasks_prefs"
KEY_LIMIT = "tasks_limit"
KEY_SHOW_ADD = "tasks_show_add"
KEY_SHOW_SEARCH = "tasks_show_search"
# widget keys that mirror a remembered preference
W_TAB, W_VIEW, W_SORT = "tasks_tab", "tasks_view", "tasks_sort"
W_STATUSES, W_QUERY, W_SCHED = "tasks_statuses", "tasks_query", "tasks_sched"
# per-task widget key prefixes (a task can appear in several places at once)
CHECK_PREFIXES: Tuple[str, ...] = ("chk", "sched", "lchk", "tl")


# ---------------------------------------------------------------------------
# Pure helpers (no Streamlit: importable and testable on their own)
# ---------------------------------------------------------------------------
def clean_text(raw: Any, limit: int = MAX_TEXT) -> str:
    """Collapse whitespace and cap the length. Non-strings become ''."""
    if not isinstance(raw, str):
        return ""
    return " ".join(raw.split())[:limit]


def _as_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def normalize_todo(todo: Dict[str, Any], now: Optional[datetime] = None) -> Dict[str, Any]:
    """Bring a task (possibly saved by the earlier version) up to the current shape, in place.

    ``done`` wins when it disagrees with ``status``: the old page only ever set ``done``.
    """
    done = bool(todo.get("done"))
    status = todo.get("status")
    if status not in STATUSES:
        status = "done" if done else "todo"
    if done and status != "done":
        status = "done"
    elif not done and status == "done":
        status = "todo"
    todo["status"], todo["done"] = status, status == "done"
    todo["text"] = clean_text(todo.get("text", ""))
    todo["assignee"] = clean_text(todo.get("assignee", ""), MAX_ASSIGNEE)
    todo["due"] = _as_date(todo.get("due"))
    if not isinstance(todo.get("created"), datetime):
        todo["created"] = now or datetime.now()
    if not isinstance(todo.get("completed_at"), datetime):
        todo["completed_at"] = None
    if todo.get("prev_status") not in STATUSES or todo.get("prev_status") == "done":
        todo["prev_status"] = "todo"
    return todo


def format_due(due: Optional[date]) -> str:
    """dd/mm/yyyy, as in the design; an em dash when there is no date."""
    return due.strftime("%d/%m/%Y") if due else "\u2014"


def due_state(todo: Dict[str, Any], today: date) -> str:
    """'none' | 'done' | 'overdue' | 'today' | 'upcoming'."""
    if todo.get("done"):
        return "done"
    due = todo.get("due")
    if due is None:
        return "none"
    if due < today:
        return "overdue"
    return "today" if due == today else "upcoming"


def tasks_label(count: int) -> str:
    return f"{count} Task" if count == 1 else f"{count} Tasks"


def compute_stats(todos: Sequence[Dict[str, Any]], now: datetime) -> Dict[str, int]:
    """Everything the summary cards and progress need, derived from the tasks."""
    today = now.date()
    week_ago, week_ahead = now - timedelta(days=WEEK), today + timedelta(days=WEEK)
    by_status = Counter(t["status"] for t in todos)
    total = len(todos)
    completed = by_status["done"]
    return {
        "total": total,
        "completed": completed,
        "remaining": total - completed,
        "in_progress": by_status["in_progress"],
        "in_review": by_status["in_review"],
        "todo": by_status["todo"],
        "pct": int(completed / total * 100) if total else 0,
        "added_week": sum(1 for t in todos if t["created"] >= week_ago),
        "completed_week": sum(1 for t in todos if t["completed_at"] and t["completed_at"] >= week_ago),
        "progress_due_soon": sum(
            1 for t in todos if t["status"] == "in_progress" and t["due"] and today <= t["due"] <= week_ahead
        ),
        "review_overdue": sum(
            1 for t in todos if t["status"] == "in_review" and t["due"] and t["due"] < today
        ),
        "overdue": sum(1 for t in todos if t["status"] != "done" and t["due"] and t["due"] < today),
    }


def filter_todos(
    todos: Iterable[Dict[str, Any]],
    view: str = "All tasks",
    query: str = "",
    statuses: Sequence[str] = (),
) -> List[Dict[str, Any]]:
    """Apply the segmented view (All tasks / Active / Done), the status filter and the search."""
    q = query.strip().lower()
    out = []
    for t in todos:
        if view == "Active" and t["done"]:
            continue
        if view == "Done" and not t["done"]:
            continue
        if statuses and t["status"] not in statuses:
            continue
        if q and q not in t["text"].lower() and q not in t["assignee"].lower():
            continue
        out.append(t)
    return out


def sort_todos(todos: Iterable[Dict[str, Any]], key: str = "Newest") -> List[Dict[str, Any]]:
    items = list(todos)
    if key == "Due date":  # soonest first, undated last
        return sorted(items, key=lambda t: (t["due"] is None, t["due"] or date.max, -t["id"]))
    if key == "Name":
        return sorted(items, key=lambda t: (t["text"].casefold(), t["id"]))
    if key == "Status":
        return sorted(items, key=lambda t: (STATUS_ORDER.index(t["status"]), -t["id"]))
    return sorted(items, key=lambda t: (t["created"], t["id"]), reverse=True)


def schedule_items(
    todos: Iterable[Dict[str, Any]], today: date, show: str = "All", limit: int = SCHEDULE_MAX
) -> Tuple[List[Dict[str, Any]], int]:
    """Overdue open tasks first, then everything due today. Returns (items, hidden_count)."""
    overdue = sorted(
        (t for t in todos if due_state(t, today) == "overdue"), key=lambda t: (t["due"], t["id"])
    )
    due_today = sorted((t for t in todos if t["due"] == today), key=lambda t: t["id"])
    items = overdue + due_today
    if show == "Open":
        items = [t for t in items if not t["done"]]
    elif show == "Done":
        items = [t for t in items if t["done"]]
    return items[:limit], max(0, len(items) - limit)


def group_by_due(todos: Iterable[Dict[str, Any]], today: date) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """Open tasks grouped for the timeline, in the order they need attention."""
    week_ahead = today + timedelta(days=WEEK)
    groups: Dict[str, List[Dict[str, Any]]] = {
        "Overdue": [], "Today": [], "Next 7 days": [], "Later": [], "No due date": [],
    }
    for t in sorted((t for t in todos if not t["done"]), key=lambda t: (t["due"] or date.max, t["id"])):
        if t["due"] is None:
            groups["No due date"].append(t)
        elif t["due"] < today:
            groups["Overdue"].append(t)
        elif t["due"] == today:
            groups["Today"].append(t)
        elif t["due"] <= week_ahead:
            groups["Next 7 days"].append(t)
        else:
            groups["Later"].append(t)
    return [(label, items) for label, items in groups.items() if items]


def _csv_cell(value: Any) -> str:
    """Neutralise spreadsheet formulas: a cell starting with = + - @ is treated as a formula by Excel."""
    text = "" if value is None else str(value)
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text


def todos_to_csv(todos: Iterable[Dict[str, Any]]) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "task", "status", "assignee", "due", "created", "completed"])
    for t in todos:
        writer.writerow([
            t["id"], _csv_cell(t["text"]), STATUSES[t["status"]].label, _csv_cell(t["assignee"]),
            t["due"].isoformat() if t["due"] else "", t["created"].isoformat(timespec="minutes"),
            t["completed_at"].isoformat(timespec="minutes") if t["completed_at"] else "",
        ])
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Clock
# ---------------------------------------------------------------------------
def _now() -> datetime:
    """Now in the visitor's timezone when Streamlit can tell us, else server time.

    Matters for "today" and for due dates: a server running in UTC would otherwise
    flip to tomorrow hours early or late for the person using the app.
    """
    try:
        tz_name = st.context.timezone
        if tz_name:
            from zoneinfo import ZoneInfo

            return datetime.now(ZoneInfo(tz_name)).replace(tzinfo=None)
    except Exception:  # old Streamlit, unknown zone, or no tzdata installed
        pass
    return datetime.now()


# ---------------------------------------------------------------------------
# State and actions (used as callbacks so state updates before the rerun renders)
# ---------------------------------------------------------------------------
def init_state() -> None:
    """Seed UI-only state. Tasks are not kept here: services.task_service is the single store."""
    defaults = {
        KEY_CONFIRM: False,  # two-step "clear all" confirmation
        KEY_PREFS: {},  # remembered tab / filter / sort / search
        KEY_LIMIT: TABLE_PAGE,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _service() -> task_service.TaskService:
    return task_service.get_service()


def _todos() -> List[Dict[str, Any]]:
    """Every task, newest first, in the dict shape the renderers use. Read fresh on each run so a
    task the assistant created shows up here without any sync step."""
    return [t.to_view_dict() for t in _service().list_tasks()]


def _attempt(action: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a service call from a callback; turn a task error into a flash message, not a crash."""
    try:
        return action(*args, **kwargs)
    except TaskError as exc:
        logger.warning("task action failed: %s", exc)
        _flash("error", str(exc))
        return None


def _forget_widgets(todo_id: int) -> None:
    for prefix in CHECK_PREFIXES:
        st.session_state.pop(f"{prefix}_{todo_id}", None)
    st.session_state.pop(f"lstatus_{todo_id}", None)


def _flash(kind: str, message: str) -> None:
    st.session_state[KEY_FLASH] = (kind, message)


def add_todo() -> None:
    ss = st.session_state
    if not clean_text(ss.get("new_todo", "")):
        _flash("warning", "Please enter a task first.")
        return
    status = ss.get("new_status", "todo")
    task = _attempt(
        _service().create_task,
        ss.get("new_todo", ""),
        assignee=ss.get("new_assignee", ""),
        due=_as_date(ss.get("new_due")),
        status=status if status in STATUSES else "todo",
    )
    if task is not None:
        _flash("success", "Task added.")
        logger.info("added task %s (%s)", task.id, task.status)


def set_status(todo_id: int, status: str) -> None:
    """Change a task's status; the service keeps ``done``, ``completed_at`` and ``prev_status`` consistent."""
    if status not in STATUSES:
        return
    _attempt(_service().set_status, todo_id, status)


def toggle_todo(todo_id: int, widget_key: Optional[str] = None) -> None:
    """Checkbox callback: tick -> done, untick -> back to the status it had before."""
    key = widget_key or f"chk_{todo_id}"
    _attempt(_service().toggle_done, todo_id, bool(st.session_state.get(key, False)))


def move_todo(todo_id: int, step: int) -> None:
    """Board buttons: move a task one column left (-1) or right (+1)."""
    task = _attempt(_service().get_task, todo_id)
    if task is None:
        return
    index = STATUS_ORDER.index(task.status) + step
    if 0 <= index < len(STATUS_ORDER):
        set_status(todo_id, STATUS_ORDER[index])


def set_status_from_widget(todo_id: int, widget_key: str) -> None:
    set_status(todo_id, st.session_state.get(widget_key, ""))


def delete_todo(todo_id: int) -> None:
    _attempt(_service().delete_task, todo_id)
    _forget_widgets(todo_id)


def clear_completed() -> None:
    done_ids = [t["id"] for t in _todos() if t["done"]]
    if _attempt(_service().clear_completed) is not None:
        for todo_id in done_ids:
            _forget_widgets(todo_id)


def request_clear_all() -> None:
    st.session_state[KEY_CONFIRM] = True


def cancel_clear_all() -> None:
    st.session_state[KEY_CONFIRM] = False


def clear_all() -> None:
    ids = [t["id"] for t in _todos()]
    if _attempt(_service().clear_all) is not None:
        for todo_id in ids:
            _forget_widgets(todo_id)
    st.session_state[KEY_CONFIRM] = False


def _toggle_flag(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)


def _show_more() -> None:
    st.session_state[KEY_LIMIT] = st.session_state.get(KEY_LIMIT, TABLE_PAGE) + TABLE_PAGE


# -- remembered preferences ---------------------------------------------------
def _prefs() -> Dict[str, Any]:
    return st.session_state.setdefault(KEY_PREFS, {})


def _remember(pref: str, widget_key: str) -> None:
    _prefs()[pref] = st.session_state[widget_key]
    st.session_state[KEY_LIMIT] = TABLE_PAGE  # a new filter/sort starts from the first page


def _restore(pref: str, widget_key: str, default: Any, options: Optional[Sequence[str]] = None) -> None:
    """Seed a widget from the remembered value (before the widget is created)."""
    value = _prefs().get(pref, default)
    if options is not None:
        if isinstance(default, list):
            value = [v for v in (value if isinstance(value, list) else []) if v in options]
        elif value not in options:
            value = default
    st.session_state[widget_key] = value


def _go_to_tab(tab: str) -> None:
    _prefs()["tab"] = tab


def _drill(statuses: List[str], view: str = "All tasks") -> None:
    """Summary-card menu: show the table filtered to that card's tasks."""
    _prefs()["statuses"] = statuses
    _prefs()["view"] = view
    st.session_state[KEY_LIMIT] = TABLE_PAGE


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
TASKS_CSS = """
.st-key-tasks-root {
  --tk-panel: #272727;
  --tk-card: #353535;
  --tk-card-line: #444444;
  --tk-line: #404040;
  --tk-text: #FFFFFF;
  --tk-muted: #A0A0A0;
  --tk-btn: #3E3E3E;
  --tk-btn-line: #595959;
  --tk-seg-track: #444444;
  --tk-seg-active: #777777;
  --tk-accent: var(--color-accent, #CAAA01);
  --tk-error: var(--color-error, #FF8A80);
  --tk-pill-todo: #555555;
  --tk-pill-progress: #6E3575;
  --tk-pill-review: #94603F;
  --tk-pill-done: #4A7857;
  --tk-s1: var(--space-1, 0.25rem); --tk-s2: var(--space-2, 0.5rem); --tk-s3: var(--space-3, 0.75rem);
  --tk-s4: var(--space-4, 1rem); --tk-s5: var(--space-5, 1.25rem); --tk-s6: var(--space-6, 1.5rem);
  --tk-r-md: var(--radius-md, 0.5rem); --tk-r-lg: var(--radius-lg, 0.75rem); --tk-r-xl: var(--radius-xl, 1.25rem);
  --tk-r-full: var(--radius-full, 999px);
  --tk-fast: var(--duration-fast, 120ms); --tk-ease: var(--ease-standard, ease);

  background: var(--tk-panel);
  border-radius: var(--tk-r-xl);
  padding: var(--tk-s6) var(--tk-s6) var(--tk-s5);
  gap: var(--tk-s4) !important;
  color: var(--tk-text);
}

/* ======================= typography & layout ======================= */
.st-key-tasks-root h1.tk-title {
  margin: 0; padding: 0; color: var(--tk-text);
  font-size: 1.9rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1.15;
}
.st-key-tasks-root [data-testid="stHorizontalBlock"] { gap: var(--tk-s3); }
.st-key-tasks-root [data-testid="stColumn"] { min-width: 0; }
.tk-section {
  margin: var(--tk-s2) 0 var(--tk-s1); color: var(--tk-text);
  font-size: 1.05rem; font-weight: 600;
}
.tk-section__count { color: var(--tk-muted); font-weight: 500; }
.tk-cell { color: var(--tk-text); font-size: 0.9rem; overflow-wrap: anywhere; }
.tk-meta { color: var(--tk-muted); font-size: 0.78rem; margin-top: 2px; }
.tk-th { color: var(--tk-muted); font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }

/* ======================= buttons ======================= */
.st-key-tasks-root [data-testid="stButton"] button,
.st-key-tasks-root [data-testid="stDownloadButton"] button,
.st-key-tasks-root [data-testid="stPopoverButton"],
.st-key-tasks-root [data-testid="stFormSubmitButton"] button {
  background: var(--tk-btn); color: var(--tk-text);
  border: 1px solid var(--tk-btn-line); border-radius: var(--tk-r-md);
  min-height: 2.4rem; padding: 0 var(--tk-s3); font-weight: 500; font-size: 0.9rem;
  white-space: nowrap; box-shadow: none;
  transition: border-color var(--tk-fast) var(--tk-ease), background-color var(--tk-fast) var(--tk-ease), transform var(--tk-fast) var(--tk-ease);
}
.st-key-tasks-root [data-testid="stButton"] button p,
.st-key-tasks-root [data-testid="stDownloadButton"] button p,
.st-key-tasks-root [data-testid="stPopoverButton"] p { margin: 0; color: inherit; font-size: inherit; line-height: 1.2; white-space: nowrap; }
.st-key-tasks-root [data-testid="stButton"] button:hover:not(:disabled),
.st-key-tasks-root [data-testid="stDownloadButton"] button:hover,
.st-key-tasks-root [data-testid="stPopoverButton"]:hover,
.st-key-tasks-root [data-testid="stFormSubmitButton"] button:hover { border-color: var(--tk-accent); color: var(--tk-accent); background: var(--tk-btn); }
.st-key-tasks-root [data-testid="stButton"] button:active:not(:disabled) { transform: scale(0.97); }
.st-key-tasks-root button:focus-visible { outline: 2px solid var(--tk-accent); outline-offset: 2px; }
.st-key-tasks-root button:disabled { opacity: 0.4; cursor: not-allowed; }
.st-key-tasks-root [data-testid="stBaseButton-primary"],
.st-key-tasks-root [data-testid="stBaseButton-primaryFormSubmit"] { background: var(--tk-accent) !important; color: #000 !important; border-color: transparent !important; }

/* header actions: Search / Add / Export share the row evenly */
.st-key-tasks-actions [data-testid="stButton"],
.st-key-tasks-actions [data-testid="stDownloadButton"] { width: 100%; }
.st-key-tasks-actions button { width: 100%; }

/* icon-only buttons: keep the label for screen readers, show just the icon */
.st-key-tasks-root [class*="st-key-tasks_drill_"] button [data-testid="stMarkdownContainer"],
.st-key-tasks-root [class*="st-key-del_"] button [data-testid="stMarkdownContainer"],
.st-key-tasks-root [class*="st-key-ldel_"] button [data-testid="stMarkdownContainer"],
.st-key-tasks-root [class*="st-key-mvl_"] button [data-testid="stMarkdownContainer"],
.st-key-tasks-root [class*="st-key-mvr_"] button [data-testid="stMarkdownContainer"] {
  position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap;
}
.st-key-tasks-root [class*="st-key-del_"] button,
.st-key-tasks-root [class*="st-key-ldel_"] button,
.st-key-tasks-root [class*="st-key-mvl_"] button,
.st-key-tasks-root [class*="st-key-mvr_"] button { min-width: 2.2rem; width: 2.2rem; padding: 0; justify-content: center; background: transparent; border-color: transparent; color: var(--tk-muted); }
.st-key-tasks-root [class*="st-key-del_"] button:hover:not(:disabled),
.st-key-tasks-root [class*="st-key-ldel_"] button:hover:not(:disabled) { color: var(--tk-error); border-color: var(--tk-error); }

/* ======================= inputs ======================= */
.st-key-tasks-root [data-baseweb="input"],
.st-key-tasks-root [data-baseweb="select"] > div,
.st-key-tasks-root [data-baseweb="base-input"] { background: var(--tk-btn) !important; border-color: var(--tk-btn-line) !important; border-radius: var(--tk-r-md) !important; }
.st-key-tasks-root input, .st-key-tasks-root textarea { color: var(--tk-text) !important; }
.st-key-tasks-root [data-baseweb="input"]:focus-within { border-color: var(--tk-accent) !important; }
.st-key-tasks-root label[data-testid="stWidgetLabel"] p { color: var(--tk-muted); font-size: 0.82rem; }
.st-key-tasks-add {
  background: var(--tk-card); border: 1px solid var(--tk-card-line); border-radius: var(--tk-r-lg); padding: var(--tk-s4);
}
.st-key-tasks-root [data-testid="stForm"] { border: 0; padding: 0; }

/* ======================= segmented radios (tabs, view, schedule) ======================= */
.st-key-tasks_tab [data-testid="stRadioGroup"],
.st-key-tasks_view [data-testid="stRadioGroup"],
.st-key-tasks_sched [data-testid="stRadioGroup"] {
  display: inline-flex; gap: 2px; padding: 4px; background: var(--tk-seg-track); border-radius: var(--tk-r-full); width: fit-content; max-width: 100%; flex-wrap: wrap;
}
.st-key-tasks_tab [data-testid="stRadioOption"],
.st-key-tasks_view [data-testid="stRadioOption"],
.st-key-tasks_sched [data-testid="stRadioOption"] {
  margin: 0; padding: 0.35rem 1rem; border-radius: var(--tk-r-full); cursor: pointer;
  transition: background-color var(--tk-fast) var(--tk-ease);
}
.st-key-tasks_tab [data-testid="stRadioOption"] > div > div:first-child,
.st-key-tasks_view [data-testid="stRadioOption"] > div > div:first-child,
.st-key-tasks_sched [data-testid="stRadioOption"] > div > div:first-child { display: none; }
.st-key-tasks_tab [data-testid="stRadioOption"] p,
.st-key-tasks_view [data-testid="stRadioOption"] p,
.st-key-tasks_sched [data-testid="stRadioOption"] p { margin: 0; color: var(--tk-muted); font-size: 0.88rem; font-weight: 500; white-space: nowrap; }
.st-key-tasks_tab [data-testid="stRadioOption"][data-selected="true"],
.st-key-tasks_view [data-testid="stRadioOption"][data-selected="true"],
.st-key-tasks_sched [data-testid="stRadioOption"][data-selected="true"] { background: var(--tk-seg-active); }
.st-key-tasks_tab [data-testid="stRadioOption"][data-selected="true"] p,
.st-key-tasks_view [data-testid="stRadioOption"][data-selected="true"] p,
.st-key-tasks_sched [data-testid="stRadioOption"][data-selected="true"] p { color: var(--tk-text); font-weight: 600; }
.st-key-tasks_tab [data-testid="stRadioOption"]:has(input:focus-visible),
.st-key-tasks_view [data-testid="stRadioOption"]:has(input:focus-visible),
.st-key-tasks_sched [data-testid="stRadioOption"]:has(input:focus-visible) { outline: 2px solid var(--tk-accent); }

/* ======================= cards ======================= */
[class*="st-key-tasks-stat-"], .st-key-tasks-table, .st-key-tasks-schedule, [class*="st-key-tasks-card-"], [class*="st-key-tasks-list-"] {
  background: var(--tk-card); border: 1px solid var(--tk-card-line); border-radius: var(--tk-r-lg);
}
[class*="st-key-tasks-stat-"] { position: relative; padding: var(--tk-s4); gap: var(--tk-s1) !important; transition: border-color var(--tk-fast) var(--tk-ease), transform var(--tk-fast) var(--tk-ease); }
[class*="st-key-tasks-stat-"]:hover { border-color: var(--tk-accent); transform: translateY(-2px); }
.tk-stat__label { color: var(--tk-muted); font-size: 0.82rem; font-weight: 600; padding-right: 2.4rem; }
.tk-stat__value { color: var(--tk-text); font-size: 1.55rem; font-weight: 700; letter-spacing: -0.02em; margin-top: 2px; }
.tk-stat__desc { color: var(--tk-muted); font-size: 0.8rem; line-height: 1.35; }
.tk-stat__foot { display: flex; align-items: center; gap: var(--tk-s2); margin-top: var(--tk-s2); color: var(--tk-accent); font-size: 0.8rem; font-weight: 500; }
.tk-stat__foot svg { flex: none; }
[class*="st-key-tasks-stat-"] [data-testid="stElementContainer"]:has([data-testid="stButton"]) { position: absolute; top: var(--tk-s2); right: var(--tk-s2); width: auto; height: auto; z-index: 1; }
[class*="st-key-tasks-stat-"] [data-testid="stButton"] button { min-width: 2.2rem; width: 2.2rem; min-height: 2.2rem; padding: 0; justify-content: center; background: transparent; border-color: transparent; color: var(--tk-muted); }

.st-key-tasks-table { padding: var(--tk-s4); gap: var(--tk-s2) !important; }
[class*="st-key-tasks-row-"], [class*="st-key-tasks-tl-"], [class*="st-key-tasks-sched-"] {
  border-top: 1px solid var(--tk-line); padding: var(--tk-s2) 0; gap: 0 !important;
}
[class*="st-key-tasks-row-"]:hover, [class*="st-key-tasks-tl-"]:hover { background: rgba(255,255,255,0.03); }
.st-key-tasks-schedule { padding: var(--tk-s4); gap: var(--tk-s2) !important; }
.tk-sched__title { color: var(--tk-text); font-size: 1.05rem; font-weight: 700; }
.tk-sched__date { color: var(--tk-muted); font-size: 0.82rem; margin-bottom: var(--tk-s1); }
.tk-sched__name { color: var(--tk-text); font-size: 0.9rem; font-weight: 500; overflow-wrap: anywhere; }
.tk-sched__meta { color: var(--tk-muted); font-size: 0.76rem; margin-top: 2px; }
[class*="st-key-tasks-list-"] { padding: var(--tk-s3) var(--tk-s4); }

/* ======================= checkboxes ======================= */
.st-key-tasks-root [data-testid="stCheckbox"] label { align-items: center; gap: var(--tk-s2); }
.st-key-tasks-root [data-testid="stCheckbox"] p { color: var(--tk-text); font-size: 0.92rem; margin: 0; overflow-wrap: anywhere; }
.st-key-tasks-root [data-testid="stCheckbox"] label:has(input:checked) p { color: var(--tk-muted); text-decoration: line-through; }

/* ======================= pills, due dates ======================= */
.tk-pill { display: inline-flex; align-items: center; gap: 0.4rem; padding: 0.15rem 0.65rem 0.15rem 0.5rem; border-radius: var(--tk-r-full); font-size: 0.78rem; font-weight: 600; color: #fff; white-space: nowrap; }
.tk-pill__dot { width: 0.45rem; height: 0.45rem; border-radius: 50%; background: currentColor; opacity: 0.85; }
.tk-pill--todo { background: var(--tk-pill-todo); }
.tk-pill--in_progress { background: var(--tk-pill-progress); }
.tk-pill--in_review { background: var(--tk-pill-review); }
.tk-pill--done { background: var(--tk-pill-done); }
.tk-due { font-size: 0.86rem; white-space: nowrap; font-variant-numeric: tabular-nums; }
.tk-due--none, .tk-due--upcoming { color: var(--tk-muted); }
.tk-due--done { color: var(--tk-muted); text-decoration: line-through; }
.tk-due--today { color: var(--tk-accent); font-weight: 600; }
.tk-due--overdue { color: var(--tk-error); font-weight: 600; }

/* ======================= empty states, progress ======================= */
.tk-empty { text-align: center; padding: var(--tk-s6) var(--tk-s4); border: 1px dashed var(--tk-line); border-radius: var(--tk-r-lg); margin-top: var(--tk-s2); }
.tk-empty__title { color: var(--tk-text); font-size: 1rem; font-weight: 600; }
.tk-empty__sub { color: var(--tk-muted); font-size: 0.85rem; margin-top: 2px; }
.tk-progress__meta { display: flex; justify-content: space-between; color: var(--tk-muted); font-size: 0.85rem; font-weight: 500; margin-bottom: var(--tk-s2); }
.tk-progress__track { height: 0.6rem; background: var(--tk-seg-track); border-radius: var(--tk-r-full); overflow: hidden; }
.tk-progress__fill { height: 100%; background: var(--tk-accent); border-radius: var(--tk-r-full); transition: width 400ms var(--tk-ease); }

/* ======================= board ======================= */
.tk-board__head { display: flex; align-items: center; justify-content: space-between; margin-bottom: var(--tk-s2); }
.tk-board__count { color: var(--tk-muted); font-size: 0.85rem; font-weight: 600; }
.tk-board__empty { color: var(--tk-muted); font-size: 0.82rem; text-align: center; padding: var(--tk-s4); border: 1px dashed var(--tk-line); border-radius: var(--tk-r-md); }
[class*="st-key-tasks-card-"] { padding: var(--tk-s3); gap: var(--tk-s2) !important; }
.tk-card__name { color: var(--tk-text); font-size: 0.92rem; font-weight: 600; overflow-wrap: anywhere; }
.tk-card__meta { display: flex; justify-content: space-between; gap: var(--tk-s2); color: var(--tk-muted); font-size: 0.8rem; margin-top: 4px; }

/* ======================= popovers (filter / sort) ======================= */
[data-testid="stPopoverBody"] { background: var(--tk-card, #353535); }

/* ======================= small screens ======================= */
@media (max-width: 900px) {
  .st-key-tasks-root { padding: var(--tk-s4); }
  .st-key-tasks_tab [data-testid="stRadioGroup"] { border-radius: var(--tk-r-lg); }
  .st-key-tasks-root [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
  .st-key-tasks-root [data-testid="stColumn"] { flex: 1 1 100% !important; width: 100% !important; }
  .st-key-tasks-root [class*="st-key-tasks-row-"] [data-testid="stHorizontalBlock"],
  .st-key-tasks-root [class*="st-key-tasks-tl-"] [data-testid="stHorizontalBlock"] { gap: var(--tk-s1); }
}
"""


def _inject_css() -> None:
    st.markdown(f"<style>{TASKS_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Small HTML helpers
# ---------------------------------------------------------------------------
def _esc(value: Any) -> str:
    """Escape for inline HTML; also neutralises markdown math and blank lines."""
    return escape(" ".join(str(value).split())).replace("$", "&#36;")


def _md_escape(text: Any) -> str:
    """Show user text literally where Streamlit renders markdown (checkbox labels)."""
    import re

    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>~$<:])", r"\\\1", " ".join(str(text).split()))


_SVG = (
    '<svg viewBox="0 0 24 24" width="{s}" height="{s}" fill="none" stroke="currentColor" stroke-width="1.6" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">{p}</svg>'
)
ICONS = {
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "check_square": '<polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
    "loader": (
        '<line x1="12" y1="2" x2="12" y2="6"/><line x1="12" y1="18" x2="12" y2="22"/>'
        '<line x1="4.93" y1="4.93" x2="7.76" y2="7.76"/><line x1="16.24" y1="16.24" x2="19.07" y2="19.07"/>'
        '<line x1="2" y1="12" x2="6" y2="12"/><line x1="18" y1="12" x2="22" y2="12"/>'
        '<line x1="4.93" y1="19.07" x2="7.76" y2="16.24"/><line x1="16.24" y1="7.76" x2="19.07" y2="4.93"/>'
    ),
    "alert_circle": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
}


def _icon(name: str, size: int = 18) -> str:
    return _SVG.format(s=size, p=ICONS[name])


def pill_html(status: str) -> str:
    return (
        f'<span class="tk-pill tk-pill--{status}"><span class="tk-pill__dot"></span>'
        f"{_esc(STATUSES[status].label)}</span>"
    )


def due_html(todo: Dict[str, Any], today: date) -> str:
    return f'<span class="tk-due tk-due--{due_state(todo, today)}">{format_due(todo["due"])}</span>'


def _empty_html(title: str, sub: str) -> str:
    return (
        f'<div class="tk-empty"><div class="tk-empty__title">{_esc(title)}</div>'
        f'<div class="tk-empty__sub">{_esc(sub)}</div></div>'
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StatCard:
    id: str
    label: str
    description: str
    icon: str
    statuses: Tuple[str, ...]  # what the card's menu filters the table to
    view: str = "All tasks"


STAT_CARDS: Tuple[StatCard, ...] = (
    StatCard("total", "Total Tasks", "All tasks across your projects", "zap", ()),
    StatCard("completed", "Completed", "Successfully finished and approved", "check_square", ("done",), "Done"),
    StatCard("in_progress", "In Progress", "Tasks currently being worked on", "loader", ("in_progress",)),
    StatCard("in_review", "Pending Review", "Waiting for feedback or approval", "alert_circle", ("in_review",)),
)


def stat_footer(card_id: str, s: Dict[str, int]) -> str:
    """The small line under each card, derived from the tasks (no invented trends)."""
    if card_id == "total":
        return f"+{s['added_week']} added this week" if s["added_week"] else "Nothing added this week"
    if card_id == "completed":
        return f"+{s['completed_week']} finished this week" if s["completed_week"] else "None finished this week"
    if card_id == "in_progress":
        n = s["progress_due_soon"]
        return f"{n} due in the next {WEEK} days" if n else f"None due in the next {WEEK} days"
    n = s["review_overdue"]
    return f"{n} overdue" if n else "None overdue"


def stat_value(card_id: str, s: Dict[str, int]) -> int:
    return {"total": s["total"], "completed": s["completed"], "in_progress": s["in_progress"],
            "in_review": s["in_review"]}[card_id]


def _render_flash() -> None:
    flash = st.session_state.pop(KEY_FLASH, None)
    if flash:
        kind, message = flash
        show, icon = {"success": (st.success, "\u2705"), "error": (st.error, "\u274c")}.get(kind, (st.warning, "\u26a0\ufe0f"))
        show(message, icon=icon)


def _render_header(todos: List[Dict[str, Any]]) -> None:
    title, actions = st.columns([5, 3], vertical_alignment="center")
    title.markdown('<h1 class="tk-title">Task Manager</h1>', unsafe_allow_html=True)
    with actions:
        with st.container(key="tasks-actions"):
            a, b, c = st.columns(3, gap="small")
            a.button("Search", key="tasks_btn_search", icon=":material/search:",
                     on_click=_toggle_flag, args=(KEY_SHOW_SEARCH,))
            b.button("Add", key="tasks_btn_add", icon=":material/add:",
                     on_click=_toggle_flag, args=(KEY_SHOW_ADD,))
            c.download_button("Export", data=("\ufeff" + todos_to_csv(todos)).encode("utf-8"), file_name="tasks.csv", mime="text/csv",
                              key="tasks_export", icon=":material/ios_share:")

    if st.session_state.get(KEY_SHOW_SEARCH):
        _restore("query", W_QUERY, "")
        st.text_input("Search tasks", key=W_QUERY, placeholder="Search by task or assignee",
                      label_visibility="collapsed", max_chars=MAX_TEXT, on_change=_remember, args=("query", W_QUERY))
    if st.session_state.get(KEY_SHOW_ADD):
        with st.container(key="tasks-add"):
            with st.form("tasks_add_form", clear_on_submit=True, border=False):
                st.text_input("Task", key="new_todo", placeholder="What needs to be done?", max_chars=MAX_TEXT)
                c1, c2, c3 = st.columns(3)
                c1.text_input("Assignee", key="new_assignee", placeholder="Optional", max_chars=MAX_ASSIGNEE)
                c2.selectbox("Status", STATUS_ORDER, format_func=lambda s: STATUSES[s].label, key="new_status")
                c3.date_input("Due date", value=None, key="new_due", format="DD/MM/YYYY")
                st.form_submit_button("Add task", on_click=add_todo)
    _render_flash()


def _render_tabs() -> str:
    _restore("tab", W_TAB, TABS[0], TABS)
    return st.radio("Section", TABS, key=W_TAB, horizontal=True, label_visibility="collapsed",
                    on_change=_remember, args=("tab", W_TAB))


def _task_checkbox(todo: Dict[str, Any], prefix: str, *, hide_label: bool = False) -> None:
    key = f"{prefix}_{todo['id']}"
    st.session_state[key] = bool(todo["done"])
    label = f"Mark done: {_md_escape(todo['text'])}" if hide_label else _md_escape(todo["text"])
    st.checkbox(label, key=key, on_change=toggle_todo, args=(todo["id"], key),
                label_visibility="collapsed" if hide_label else "visible")


def _render_stat_cards(stats: Dict[str, int]) -> None:
    for col, card in zip(st.columns(4, gap="small"), STAT_CARDS):
        with col:
            with st.container(key=f"tasks-stat-{card.id}"):
                st.markdown(
                    "".join([
                        f'<div class="tk-stat__label">{_esc(card.label)}</div>',
                        f'<div class="tk-stat__value">{tasks_label(stat_value(card.id, stats))}</div>',
                        f'<div class="tk-stat__desc">{_esc(card.description)}</div>',
                        f'<div class="tk-stat__foot">{_icon(card.icon)}<span>{_esc(stat_footer(card.id, stats))}</span></div>',
                    ]),
                    unsafe_allow_html=True,
                )
                st.button(f"Show {card.label} tasks", key=f"tasks_drill_{card.id}", icon=":material/more_vert:",
                          help=f"Show {card.label.lower()} in the table",
                          on_click=_drill, args=(list(card.statuses), card.view))


TABLE_RATIOS = [3.6, 1.6, 1.6, 1.3, 0.55]


def _render_table(todos: List[Dict[str, Any]], today: date) -> None:
    _restore("view", W_VIEW, VIEW_OPTIONS[0], VIEW_OPTIONS)
    _restore("sort", W_SORT, SORT_OPTIONS[0], SORT_OPTIONS)
    _restore("statuses", W_STATUSES, [], STATUS_ORDER)
    view, sort = st.session_state[W_VIEW], st.session_state[W_SORT]
    statuses, query = st.session_state[W_STATUSES], _prefs().get("query", "")
    visible = sort_todos(filter_todos(todos, view, query, statuses), sort)

    with st.container(key="tasks-table"):
        seg, _gap, flt, srt = st.columns([3, 2.4, 1.1, 1.1], vertical_alignment="center", gap="small")
        with seg:
            st.radio("View", VIEW_OPTIONS, key=W_VIEW, horizontal=True, label_visibility="collapsed",
                     on_change=_remember, args=("view", W_VIEW))
        with flt:
            with st.container(key="tasks-filter"):
                with st.popover("Filter", icon=":material/filter_list:"):
                    st.multiselect("Status", STATUS_ORDER, key=W_STATUSES, format_func=lambda s: STATUSES[s].label,
                                   on_change=_remember, args=("statuses", W_STATUSES))
        with srt:
            with st.container(key="tasks-sort"):
                with st.popover("Sort", icon=":material/swap_vert:"):
                    st.radio("Sort by", SORT_OPTIONS, key=W_SORT, on_change=_remember, args=("sort", W_SORT))

        head = st.columns(TABLE_RATIOS, vertical_alignment="center")
        for col, name in zip(head, ("Task name", "Status", "Assignee", "Due date", "")):
            col.markdown(f'<div class="tk-th">{name}</div>', unsafe_allow_html=True)

        if not visible:
            if not todos:
                text = ("No tasks yet", "Use Add to create your first task.")
            elif query or statuses:
                text = ("No matching tasks", "Try clearing the search or the status filter.")
            elif view == "Active":
                text = ("All caught up!", "Nothing left to do. Nice work.")
            else:
                text = ("Nothing completed yet", "Finish a task and it will show up here.")
            st.markdown(_empty_html(*text), unsafe_allow_html=True)
            return

        limit = st.session_state.get(KEY_LIMIT, TABLE_PAGE)
        for todo in visible[:limit]:
            with st.container(key=f"tasks-row-{todo['id']}"):
                c_name, c_status, c_who, c_due, c_act = st.columns(TABLE_RATIOS, vertical_alignment="center")
                with c_name:
                    _task_checkbox(todo, "chk")
                c_status.markdown(pill_html(todo["status"]), unsafe_allow_html=True)
                c_who.markdown(f'<span class="tk-cell">{_esc(todo["assignee"]) or "&mdash;"}</span>', unsafe_allow_html=True)
                c_due.markdown(due_html(todo, today), unsafe_allow_html=True)
                with c_act:
                    st.button("Delete task", key=f"del_{todo['id']}", icon=":material/delete:", help="Delete task",
                              on_click=delete_todo, args=(todo["id"],))
        if len(visible) > limit:
            st.button(f"Show more ({len(visible) - limit} hidden)", key="tasks_more", on_click=_show_more)


def _render_schedule(todos: List[Dict[str, Any]], now: datetime) -> None:
    today = now.date()
    _restore("sched", W_SCHED, SCHEDULE_FILTERS[0], SCHEDULE_FILTERS)
    with st.container(key="tasks-schedule"):
        st.markdown(
            f'<div class="tk-sched__title">Today\'s schedule</div>'
            f'<div class="tk-sched__date">{_esc(now.strftime("%A, %d %B"))}</div>',
            unsafe_allow_html=True,
        )
        st.radio("Show", SCHEDULE_FILTERS, key=W_SCHED, horizontal=True, label_visibility="collapsed",
                 on_change=_remember, args=("sched", W_SCHED))
        items, hidden = schedule_items(todos, today, st.session_state[W_SCHED])
        if not items:
            st.markdown(_empty_html("Nothing scheduled", "Tasks due today will appear here."), unsafe_allow_html=True)
        for todo in items:
            with st.container(key=f"tasks-sched-{todo['id']}"):
                text_col, check_col = st.columns([5, 1], vertical_alignment="center")
                meta = "Overdue \u00b7 " + format_due(todo["due"]) if due_state(todo, today) == "overdue" else (
                    "Today \u00b7 " + STATUSES[todo["status"]].label)
                text_col.markdown(
                    f'<div class="tk-sched__name">{_esc(todo["text"])}</div><div class="tk-sched__meta">{_esc(meta)}</div>',
                    unsafe_allow_html=True,
                )
                with check_col:
                    _task_checkbox(todo, "sched", hide_label=True)
        if hidden:
            st.caption(f"+{hidden} more")
        st.button("View all tasks", key="tasks_sched_all", icon=":material/chevron_right:",
                  on_click=_go_to_tab, args=("List",))


def _render_overview(todos: List[Dict[str, Any]], now: datetime) -> None:
    stats = compute_stats(todos, now)
    st.markdown(
        f'<div class="tk-section">Task Overview <span class="tk-section__count">({stats["total"]})</span></div>',
        unsafe_allow_html=True,
    )
    _render_stat_cards(stats)
    table_col, side_col = st.columns([3.1, 1.15], gap="small")
    with table_col:
        _render_table(todos, now.date())
    with side_col:
        _render_schedule(todos, now)


def _render_list(todos: List[Dict[str, Any]], now: datetime) -> None:
    today = now.date()
    stats = compute_stats(todos, now)
    st.markdown(
        f'<div class="tk-progress"><div class="tk-progress__meta"><span>Progress</span><span>{stats["pct"]}%</span></div>'
        f'<div class="tk-progress__track"><div class="tk-progress__fill" style="width:{stats["pct"]}%"></div></div></div>',
        unsafe_allow_html=True,
    )
    _restore("view", W_VIEW, VIEW_OPTIONS[0], VIEW_OPTIONS)
    st.radio("View", VIEW_OPTIONS, key=W_VIEW, horizontal=True, label_visibility="collapsed",
             on_change=_remember, args=("view", W_VIEW))
    visible = sort_todos(filter_todos(todos, st.session_state[W_VIEW], _prefs().get("query", "")), "Newest")
    st.markdown(f'<div class="tk-section">{_esc(st.session_state[W_VIEW])} <span class="tk-section__count">({len(visible)})</span></div>',
                unsafe_allow_html=True)
    if not visible:
        st.markdown(_empty_html("No tasks here", "Add a task, or change the view above."), unsafe_allow_html=True)
    for todo in visible:
        with st.container(key=f"tasks-list-{todo['id']}", border=True):
            c_check, c_meta, c_status, c_del = st.columns([4, 2.4, 1.6, 0.6], vertical_alignment="center")
            with c_check:
                _task_checkbox(todo, "lchk")
            c_meta.markdown(
                f'<div class="tk-cell">{_esc(todo["assignee"]) or "&mdash;"}</div>'
                f'<div class="tk-meta">Added {todo["created"].strftime("%b %d, %H:%M")} &middot; Due {format_due(todo["due"])}</div>',
                unsafe_allow_html=True)
            with c_status:
                skey = f"lstatus_{todo['id']}"
                st.session_state[skey] = todo["status"]
                st.selectbox("Status", STATUS_ORDER, format_func=lambda s: STATUSES[s].label, key=skey,
                             label_visibility="collapsed", on_change=set_status_from_widget, args=(todo["id"], skey))
            with c_del:
                st.button("Delete task", key=f"ldel_{todo['id']}", icon=":material/delete:", help="Delete task",
                          on_click=delete_todo, args=(todo["id"],))
    _render_bulk_actions(todos, stats)


def _render_bulk_actions(todos: List[Dict[str, Any]], stats: Dict[str, int]) -> None:
    if not todos:
        return
    st.write("")
    if st.session_state.get(KEY_CONFIRM):
        st.warning(f"Delete all {stats['total']} tasks? This can't be undone.", icon="\u26a0\ufe0f")
        b1, b2 = st.columns(2)
        b1.button("Yes, clear everything", key="tasks_clear_yes", on_click=clear_all, type="primary")
        b2.button("Cancel", key="tasks_clear_cancel", on_click=cancel_clear_all)
    else:
        b1, b2 = st.columns(2)
        b1.button("Clear completed", key="tasks_clear_done", on_click=clear_completed, disabled=stats["completed"] == 0)
        b2.button("Clear all", key="tasks_clear_all", on_click=request_clear_all)


def _render_board(todos: List[Dict[str, Any]], today: date) -> None:
    visible = filter_todos(todos, query=_prefs().get("query", ""))
    for col, status_id in zip(st.columns(len(STATUS_ORDER), gap="small"), STATUS_ORDER):
        items = sorted((t for t in visible if t["status"] == status_id), key=lambda t: (t["due"] or date.max, t["id"]))
        with col:
            st.markdown(f'<div class="tk-board__head">{pill_html(status_id)}<span class="tk-board__count">{len(items)}</span></div>',
                        unsafe_allow_html=True)
            if not items:
                st.markdown('<div class="tk-board__empty">No tasks</div>', unsafe_allow_html=True)
            for todo in items:
                with st.container(key=f"tasks-card-{todo['id']}"):
                    st.markdown(
                        f'<div class="tk-card__name">{_esc(todo["text"])}</div>'
                        f'<div class="tk-card__meta"><span>{_esc(todo["assignee"]) or "&mdash;"}</span>{due_html(todo, today)}</div>',
                        unsafe_allow_html=True)
                    left, right, _sp = st.columns([1, 1, 2], gap="small")
                    left.button("Move left", key=f"mvl_{todo['id']}", icon=":material/chevron_left:", help="Move left",
                                on_click=move_todo, args=(todo["id"], -1), disabled=status_id == STATUS_ORDER[0])
                    right.button("Move right", key=f"mvr_{todo['id']}", icon=":material/chevron_right:", help="Move right",
                                 on_click=move_todo, args=(todo["id"], 1), disabled=status_id == STATUS_ORDER[-1])


def _render_timeline(todos: List[Dict[str, Any]], today: date) -> None:
    groups = group_by_due(filter_todos(todos, query=_prefs().get("query", "")), today)
    if not groups:
        st.markdown(_empty_html("Nothing coming up", "Open tasks will be grouped here by due date."), unsafe_allow_html=True)
        return
    for label, items in groups:
        st.markdown(f'<div class="tk-section">{_esc(label)} <span class="tk-section__count">({len(items)})</span></div>',
                    unsafe_allow_html=True)
        for todo in items:
            with st.container(key=f"tasks-tl-{todo['id']}"):
                c_check, c_status, c_who, c_due = st.columns([3.6, 1.6, 1.6, 1.3], vertical_alignment="center")
                with c_check:
                    _task_checkbox(todo, "tl")
                c_status.markdown(pill_html(todo["status"]), unsafe_allow_html=True)
                c_who.markdown(f'<span class="tk-cell">{_esc(todo["assignee"]) or "&mdash;"}</span>', unsafe_allow_html=True)
                c_due.markdown(due_html(todo, today), unsafe_allow_html=True)


def render() -> None:
    """Render the tasks page (called by app.py for the 'tasks' route)."""
    init_state()
    _inject_css()
    try:
        todos = _todos()
    except TaskError as exc:  # e.g. the database file is locked or unreadable
        logger.error("could not load tasks: %s", exc)
        st.error("Your tasks can't be loaded right now. Please try again in a moment.", icon="\u274c")
        return
    now = _now()
    with st.container(key="tasks-root"):
        _render_header(todos)
        tab = _render_tabs()
        if tab == "Overview":
            _render_overview(todos, now)
        elif tab == "List":
            _render_list(todos, now)
        elif tab == "Board":
            _render_board(todos, now.date())
        elif tab == "Timeline":
            _render_timeline(todos, now.date())
        else:
            st.markdown(_empty_html("No files yet", "File attachments aren't supported yet."), unsafe_allow_html=True)