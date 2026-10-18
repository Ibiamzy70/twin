"""
Dashboard data layer (no Streamlit, no network).

The Overview view asks this module for everything it shows, so the numbers can be
tested without rendering a page and the view only has to draw them.

What it provides
----------------
  ``DashboardService.load(session)``   read the session (messages, chat counter, profile,
                                       settings, calculator history) and the shared task
                                       store into one immutable ``DashboardData``
  ``build_events`` / ``apply_filters`` one event per message and per task; the sidebar
                                       filters (status, type, date range)
  ``week_series``                      events per day for the last seven days
  ``date_bounds``                      default range and picker limits for the date filter
  ``module_cards``                     the four navigation cards with their live counts
  ``CARD_ROUTES``                      which view each dashboard card opens
  ``summary_rows`` / ``usage_rows`` / ``data_notes``   the Reports table and the expanders

Definitions
-----------
  Conversations           the chat counter in the session
  Tasks completed         tasks whose status is done
  Assistant interactions  messages whose role is assistant
  Events                  one per dated message and one per dated task (a task is dated by its
                          completion date if it has one, otherwise its creation date)

Tasks come from ``services.task_service``, the store the Tasks page and the assistant use,
so a task created by the twin is counted here without any sync step. If that store cannot
be read the dashboard still loads, with no tasks and ``tasks_unavailable=True``.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

logger = logging.getLogger("digital_twin.dashboard_service")

__all__ = [
    "WEEK_DAYS",
    "PREVIEW_CHARS",
    "DATE_RANGE_YEARS",
    "STATUS_OPTIONS",
    "KIND_OPTIONS",
    "USER_ROLES",
    "ASSISTANT_ROLES",
    "CARD_ROUTES",
    "SessionKeys",
    "MessageItem",
    "TaskItem",
    "Event",
    "Filters",
    "ModuleCard",
    "DashboardData",
    "DashboardService",
    "count_label",
    "module_cards",
    "build_events",
    "apply_filters",
    "is_narrowed",
    "week_series",
    "date_bounds",
    "summary_rows",
    "usage_rows",
    "data_notes",
    "configure",
    "get_service",
    "load",
]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
WEEK_DAYS = 7
PREVIEW_CHARS = 110
DATE_RANGE_YEARS = 10

STATUS_OPTIONS: Tuple[str, ...] = ("Open", "Done")
KIND_OPTIONS: Tuple[str, ...] = ("Messages", "Tasks")

# Message roles counted as "you" and "assistant". If a chat view stores roles under other
# names, add them here (``data_notes`` lists any role that is not recognised).
USER_ROLES = frozenset({"user", "human"})
ASSISTANT_ROLES = frozenset({"assistant", "bot", "ai"})

# Where each dashboard card leads. Values are route ids from components/sidebar.py::ROUTES;
# the view refuses to navigate to an id the sidebar does not know, and a test checks this
# table against the real routes.
CARD_ROUTES: Mapping[str, str] = MappingProxyType(
    {
        # navigation cards
        "module-assistant": "assistant",
        "module-calculator": "calculator",
        "module-tasks": "tasks",
        "module-settings": "settings",
        # headline metrics
        "conversations": "assistant",
        "tasks-completed": "tasks",
        "assistant-interactions": "assistant",
        # derived metrics
        "completion-rate": "tasks",
        "messages-per-chat": "assistant",
        "replies-per-prompt": "assistant",
        # insights and activity
        "profile": "settings",
        "completion": "tasks",
        "tasks": "tasks",
        "recent-conversations": "assistant",
        "task-activity": "tasks",
    }
)


@dataclass(frozen=True)
class SessionKeys:
    """Session-state keys read by ``load`` (defaults match ``state/session.py``)."""

    messages: str = "messages"
    chat_count: str = "chat_count"
    profile: str = "user_profile"
    settings: str = "settings"
    calc_history: str = "calc_history"


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MessageItem:
    role: str
    text: str
    when: Optional[date]


@dataclass(frozen=True)
class TaskItem:
    label: str
    done: bool
    when: Optional[date]


@dataclass(frozen=True)
class Event:
    kind: str  # "Messages" or "Tasks"
    title: str
    detail: str  # "You" / "Assistant" for messages, "Open" / "Done" for tasks
    when: Optional[date]


@dataclass(frozen=True)
class Filters:
    statuses: frozenset
    kinds: frozenset
    start: Optional[date]
    end: Optional[date]
    narrowed: bool


@dataclass(frozen=True)
class ModuleCard:
    """A navigation card: which view it opens and the live figure shown on it."""

    card_id: str
    route_id: str
    title: str
    description: str
    icon: str  # file name (no extension) in assets/icons/
    metric: str


@dataclass(frozen=True)
class DashboardData:
    name: str
    role: str
    initials: str
    chats: int
    messages: Tuple[MessageItem, ...]
    tasks: Tuple[TaskItem, ...]
    calculations: int
    settings: Dict[str, Any]
    tasks_unavailable: bool = False

    @property
    def total_tasks(self) -> int:
        return len(self.tasks)

    @property
    def done_tasks(self) -> int:
        return sum(1 for t in self.tasks if t.done)

    @property
    def open_tasks(self) -> int:
        return self.total_tasks - self.done_tasks

    @property
    def completion_pct(self) -> int:
        return (
            round(self.done_tasks / self.total_tasks * 100) if self.total_tasks else 0
        )

    @property
    def open_pct(self) -> int:
        return 100 - self.completion_pct if self.total_tasks else 0

    @property
    def user_prompts(self) -> int:
        return sum(1 for m in self.messages if m.role in USER_ROLES)

    @property
    def assistant_replies(self) -> int:
        return sum(1 for m in self.messages if m.role in ASSISTANT_ROLES)

    @property
    def other_messages(self) -> int:
        return len(self.messages) - self.user_prompts - self.assistant_replies

    @property
    def unknown_roles(self) -> List[str]:
        """Sanitised role names that are neither user nor assistant."""
        known = USER_ROLES | ASSISTANT_ROLES
        return sorted(
            {
                re.sub(r"[^\w\- ]", "", m.role)[:20] or "(missing)"
                for m in self.messages
                if m.role not in known
            }
        )

    @property
    def messages_per_chat(self) -> Optional[float]:
        return round(len(self.messages) / self.chats, 1) if self.chats else None

    @property
    def replies_per_prompt(self) -> Optional[float]:
        return (
            round(self.assistant_replies / self.user_prompts, 1)
            if self.user_prompts
            else None
        )

    @property
    def undated_items(self) -> int:
        return sum(1 for m in self.messages if m.when is None) + sum(
            1 for t in self.tasks if t.when is None
        )

    @property
    def dated_dates(self) -> List[date]:
        dates = [m.when for m in self.messages if m.when is not None]
        dates += [t.when for t in self.tasks if t.when is not None]
        return dates

    @property
    def is_empty(self) -> bool:
        return not (self.chats or self.messages or self.tasks or self.calculations)


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------
def count_label(n: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _to_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def _first_date(item: Any, keys: Tuple[str, ...]) -> Optional[date]:
    if not isinstance(item, dict):
        return None
    for key in keys:
        parsed = _to_date(item.get(key))
        if parsed:
            return parsed
    return None


def _task_done(task: Any) -> bool:
    if not isinstance(task, dict):
        return False
    for key in ("done", "completed"):
        if key in task:
            return bool(task[key])
    return str(task.get("status", "")).lower() in {"done", "complete", "completed"}


def _task_label(task: Any) -> str:
    if isinstance(task, dict):
        for key in ("title", "text", "name", "description"):
            value = task.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return "Untitled task"
    return str(task)


def _initials(name: str) -> str:
    letters = [part[0] for part in name.split() if part]
    return "".join(letters[:2]).upper() or "?"


def truncate(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _shared_task_store() -> Any:
    """The task service the Tasks page and the assistant use (``task_service``, or its
    former name ``todo_service``)."""
    try:
        from services import task_service as module
    except ImportError:
        from services import todo_service as module  # type: ignore[no-redef]
    return module.get_service()


def _task_as_dict(task: Any) -> Dict[str, Any]:
    if isinstance(task, dict):
        return task
    to_view = getattr(task, "to_view_dict", None)
    if callable(to_view):
        return to_view()
    return {}


class DashboardService:
    """Builds ``DashboardData`` from the session and the shared task store."""

    def __init__(
        self,
        tasks: Any = None,
        *,
        keys: SessionKeys = SessionKeys(),
    ) -> None:
        # ``tasks``: any object with ``list_tasks()``. None means the shared store, looked up
        # on every load so a reconfigured store (tests, a new database path) is picked up.
        self._tasks = tasks
        self.keys = keys

    def _task_store(self) -> Any:
        return self._tasks if self._tasks is not None else _shared_task_store()

    def load_tasks(self) -> Tuple[Tuple[TaskItem, ...], bool]:
        """(tasks, unavailable). A storage problem must not take the dashboard down."""
        try:
            raw = self._task_store().list_tasks()
        except Exception:  # noqa: BLE001 - any storage failure degrades to "no tasks"
            logger.exception("dashboard could not load tasks")
            return (), True
        items: List[TaskItem] = []
        for task in raw:
            data = _task_as_dict(task)
            done = _task_done(data)
            created = _first_date(data, ("created", "created_at"))
            completed = _first_date(data, ("completed_at",))
            items.append(
                TaskItem(
                    label=_task_label(data),
                    done=done,
                    when=(completed if done and completed else created) or completed,
                )
            )
        return tuple(items), False

    def load(self, session: Mapping[str, Any]) -> DashboardData:
        keys = self.keys
        raw_profile = session.get(keys.profile)
        profile = raw_profile if isinstance(raw_profile, dict) else {}
        raw_settings = session.get(keys.settings)
        settings = raw_settings if isinstance(raw_settings, dict) else {}
        name = str(profile.get("display_name") or "Guest")

        messages: List[MessageItem] = []
        for message in _as_list(session.get(keys.messages)):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            messages.append(
                MessageItem(
                    role=str(message.get("role", "")).lower(),
                    text=content if isinstance(content, str) else "",
                    when=_first_date(message, ("timestamp", "created", "created_at")),
                )
            )

        tasks, unavailable = self.load_tasks()
        try:
            chats = max(0, int(session.get(keys.chat_count) or 0))
        except (TypeError, ValueError):
            chats = 0

        return DashboardData(
            name=name,
            role=str(profile.get("role") or "Workspace owner"),
            initials=_initials(name),
            chats=chats,
            messages=tuple(messages),
            tasks=tasks,
            calculations=len(_as_list(session.get(keys.calc_history))),
            settings=settings,
            tasks_unavailable=unavailable,
        )


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------
def module_cards(data: DashboardData) -> Tuple[ModuleCard, ...]:
    """The four navigation cards with their live figures."""
    return (
        ModuleCard(
            "module-assistant",
            CARD_ROUTES["module-assistant"],
            "AI Assistant",
            "Chat with your twin and review past conversations.",
            "chat-circle-dots",
            count_label(data.chats, "conversation"),
        ),
        ModuleCard(
            "module-calculator",
            CARD_ROUTES["module-calculator"],
            "Calculator",
            "Quick calculations with a running history.",
            "calculator",
            count_label(data.calculations, "calculation"),
        ),
        ModuleCard(
            "module-tasks",
            CARD_ROUTES["module-tasks"],
            "Tasks",
            "Plan, track and complete your to-do list.",
            "list-checks",
            f"{data.open_tasks} open",
        ),
        ModuleCard(
            "module-settings",
            CARD_ROUTES["module-settings"],
            "Settings",
            "Profile, preferences and workspace controls.",
            "gear",
            "Preferences",
        ),
    )


# ---------------------------------------------------------------------------
# Events, filters, series
# ---------------------------------------------------------------------------
def build_events(data: DashboardData) -> List[Event]:
    """One event per message and per task, newest first (undated items last)."""
    events: List[Event] = []
    for message in reversed(data.messages):
        if message.role in USER_ROLES:
            who = "You"
        elif message.role in ASSISTANT_ROLES:
            who = "Assistant"
        else:
            who = message.role.title() or "Message"
        events.append(
            Event(
                "Messages",
                truncate(message.text, PREVIEW_CHARS) or "(empty message)",
                who,
                message.when,
            )
        )
    for task in reversed(data.tasks):
        events.append(
            Event("Tasks", task.label, "Done" if task.done else "Open", task.when)
        )
    events.sort(key=lambda e: e.when or date.min, reverse=True)  # stable
    return events


def apply_filters(events: Iterable[Event], filters: Filters) -> List[Event]:
    """Undated items cannot be placed in a date range, so they are always kept."""
    result: List[Event] = []
    for event in events:
        if event.kind not in filters.kinds:
            continue
        if event.kind == "Tasks" and event.detail not in filters.statuses:
            continue
        if (
            event.when is not None
            and filters.start is not None
            and filters.end is not None
            and not (filters.start <= event.when <= filters.end)
        ):
            continue
        result.append(event)
    return result


def is_narrowed(
    statuses: Sequence[str],
    kinds: Sequence[str],
    start: Optional[date],
    end: Optional[date],
    default_range: Optional[Tuple[date, date]],
) -> bool:
    """True when the user has restricted anything compared with the defaults."""
    return (
        len(statuses) < len(STATUS_OPTIONS)
        or len(kinds) < len(KIND_OPTIONS)
        or (default_range is not None and (start, end) != default_range)
    )


def week_series(
    dates: Iterable[Optional[date]], today: Optional[date] = None
) -> List[Tuple[str, int, bool]]:
    """(weekday initial, count, is_today) for the last ``WEEK_DAYS`` days, oldest first."""
    today = today or date.today()
    days = [today - timedelta(days=offset) for offset in range(WEEK_DAYS - 1, -1, -1)]
    counts = {day: 0 for day in days}
    for when in dates:
        if when in counts:
            counts[when] += 1
    return [(day.strftime("%a")[0], counts[day], day == today) for day in days]


def date_bounds(
    dated: Sequence[date], today: Optional[date] = None
) -> Optional[Tuple[Tuple[date, date], date, date]]:
    """(default_range, min_value, max_value) for the date picker, or None without dates.

    The limits always contain the data, so very old or future-dated items can never fall
    outside the picker.
    """
    if not dated:
        return None
    today = today or date.today()
    span = timedelta(days=365 * DATE_RANGE_YEARS)
    earliest, latest = min(dated), max(max(dated), today)
    return (
        (earliest, latest),
        min(today - span, earliest),
        max(today + span, latest),
    )


# ---------------------------------------------------------------------------
# Report and notes content
# ---------------------------------------------------------------------------
def summary_rows(data: DashboardData) -> List[Tuple[str, Any]]:
    return [
        ("Conversations", data.chats),
        ("Messages", len(data.messages)),
        ("Your messages", data.user_prompts),
        ("Assistant interactions", data.assistant_replies),
        ("Tasks", data.total_tasks),
        ("Tasks completed", data.done_tasks),
        ("Tasks open", data.open_tasks),
        ("Completion rate", f"{data.completion_pct}%" if data.total_tasks else "n/a"),
        ("Calculations", data.calculations),
    ]


def usage_rows(data: DashboardData) -> List[Tuple[str, int]]:
    return [
        ("Your messages", data.user_prompts),
        ("Assistant replies", data.assistant_replies),
        ("Other messages", data.other_messages),
        ("Total messages", len(data.messages)),
    ]


def data_notes(data: DashboardData) -> List[str]:
    notes = [
        "Messages, conversations and calculations live in this browser session. "
        "Refreshing or closing the tab clears them. Tasks are stored and are not cleared.",
        "Undated messages and tasks cannot be placed in a date range, so date filters never hide them.",
        "Filters apply to Activity and Reports. Headline metrics and Performance always use all data.",
    ]
    if data.tasks_unavailable:
        notes.append(
            "Tasks could not be loaded just now, so task figures show as zero. Try again in a moment."
        )
    if data.undated_items:
        notes.append(
            f"{count_label(data.undated_items, 'item')} currently have no timestamp."
        )
    if data.chats and not data.messages:
        notes.append(
            f"The chat counter is {data.chats} but no messages are stored, so history may have been cleared."
        )
    if data.unknown_roles:
        shown = ", ".join(f"`{role}`" for role in data.unknown_roles[:5])
        notes.append(
            f"Some messages use roles this dashboard does not recognise ({shown}), so "
            "Assistant interactions may be undercounted. Add them to USER_ROLES or "
            "ASSISTANT_ROLES in services/dashboard_service.py."
        )
    if data.messages and not data.chats:
        notes.append(
            "Messages exist but the chat counter is 0; conversations may not be counted by the assistant view."
        )
    return notes


# ---------------------------------------------------------------------------
# Default service + module-level functions
# ---------------------------------------------------------------------------
_default: Optional[DashboardService] = None
_default_lock = threading.Lock()


def get_service() -> DashboardService:
    """The shared service, created on first use."""
    global _default
    with _default_lock:
        if _default is None:
            _default = DashboardService()
        return _default


def configure(
    tasks: Any = None, *, keys: SessionKeys = SessionKeys()
) -> DashboardService:
    """Replace the shared service (a fake task store or other session keys, for example)."""
    global _default
    with _default_lock:
        _default = DashboardService(tasks, keys=keys)
        return _default


def load(session: Mapping[str, Any]) -> DashboardData:
    return get_service().load(session)
