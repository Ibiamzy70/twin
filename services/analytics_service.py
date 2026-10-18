"""Analytics service: turns the twin's activity history and the user's tasks into chart data.

The numbers are computed here, from stored data, by plain code. A language model can ask for a
chart by *source name* (``make_chart`` in ``tools.py``) but never supplies the values, labels or
title, so a chart can only ever show what is actually stored. The view draws the result.

A chart is a plain, JSON-safe dict (an "attachment")::

    {"type": "chart", "kind": "bar" | "line", "source": "activity_per_week",
     "title": "Activity per week", "subtitle": "Aug 10 to Aug 31, 2026",
     "x_label": "Week starting", "y_label": "Activities",
     "labels": ["Aug 10", "Aug 17"], "series": [{"name": "Activities", "values": [5, 6]}],
     "total": 11, "summary": "11 activities in 2 weeks. Busiest: Aug 17 (6)."}

``validate_chart`` is the single gatekeeper: the tool runs it before handing a chart out, the chat
service re-checks size, and the view runs it again before drawing. It limits kinds, point counts,
string lengths and number ranges and drops unknown keys.

No Streamlit import. Dependencies (twin data, task service, clock) are injectable for tests.

Sources
-------
activity_per_day       entries per calendar day (at most 31 days at a time)
activity_per_week      entries per week (Monday start)
activity_by_category   entries grouped by simple keyword categories
tasks_by_status        tasks per status (to do, in progress, in review, done)
tasks_by_due           open tasks by when they are due (overdue, this week, next week, ...)
"""

from __future__ import annotations

import json
import logging
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
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
    Union,
)

logger = logging.getLogger("digital_twin.analytics_service")

__all__ = [
    "AnalyticsError",
    "AnalyticsService",
    "CHART_KINDS",
    "SOURCES",
    "MAX_POINTS",
    "MAX_SERIES",
    "categorize",
    "validate_chart",
    "build_chart",
    "get_service",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "twin_data.json"

MAX_POINTS = 31
MAX_SERIES = 3
MAX_LABEL_CHARS = 40
MAX_TITLE_CHARS = 80
MAX_SUMMARY_CHARS = 240
MAX_ABS_VALUE = 1_000_000_000
CHART_KINDS = ("bar", "line")
_KEYS = (
    "type",
    "kind",
    "source",
    "title",
    "subtitle",
    "x_label",
    "y_label",
    "labels",
    "series",
    "total",
    "summary",
)


class AnalyticsError(ValueError):
    """A problem that is safe to show: ``code`` + ``message`` (no data, no paths)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    description: str
    time_based: bool  # the line kind makes sense, and date_from / date_to apply
    uses_dates: bool


SOURCES: Dict[str, Source] = {
    s.name: s
    for s in (
        Source(
            "activity_per_day",
            "Activity per day",
            "Saved activity entries per day.",
            True,
            True,
        ),
        Source(
            "activity_per_week",
            "Activity per week",
            "Saved activity entries per week.",
            True,
            True,
        ),
        Source(
            "activity_by_category",
            "Activity by category",
            "Saved activity entries by topic.",
            False,
            True,
        ),
        Source(
            "tasks_by_status",
            "Tasks by status",
            "The user's tasks per status.",
            False,
            False,
        ),
        Source(
            "tasks_by_due",
            "Open tasks by due date",
            "Open tasks by when they are due.",
            False,
            False,
        ),
    )
}

# ---------------------------------------------------------------------------
# Categories (keyword based; first match wins, so order matters)
# ---------------------------------------------------------------------------
DEFAULT_CATEGORIES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("Presentations", ("presentation", "slide", "deck", "demo", "talking")),
    ("Debugging", ("debug", "bug", "fix", "incident", "leak", "crash", "error")),
    ("Testing", ("test", "unit", "qa", "benchmark")),
    ("Code review", ("review", "pull", "pr", "merge")),
    (
        "Research and reading",
        ("research", "read", "paper", "study", "learn", "course", "article"),
    ),
    (
        "Planning and meetings",
        ("plan", "meeting", "sync", "standup", "roadmap", "retro", "interview"),
    ),
    (
        "Writing and docs",
        ("write", "wrote", "doc", "documentation", "document", "blog", "note"),
    ),
    (
        "Development",
        (
            "build",
            "built",
            "implement",
            "develop",
            "refactor",
            "deploy",
            "code",
            "api",
            "feature",
            "service",
            "backend",
            "frontend",
            "create",
            "design",
            "optimize",
        ),
    ),
)
OTHER = "Other"
_WORD = re.compile(r"[a-z0-9]+")


def _stem(word: str) -> str:
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    for suffix in (
        "ing",
        "ed",
    ):  # "debugged" -> "debugg" would miss "debug"; handle a doubled letter
        if len(word) > len(suffix) + 3 and word.endswith(suffix):
            word = word[: -len(suffix)]
            if len(word) > 3 and word[-1] == word[-2]:
                word = word[:-1]
            return word
    return word


def categorize(
    text: str, categories: Sequence[Tuple[str, Sequence[str]]] = DEFAULT_CATEGORIES
) -> str:
    """The first category whose keyword matches a word of ``text`` (after simple stemming)."""
    words = {_stem(w) for w in _WORD.findall(str(text).lower())}
    for name, keywords in categories:
        if any(_stem(k) in words for k in keywords):
            return name
    return OTHER


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
_ISO = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def _to_date(value: Any) -> Optional[date]:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        m = _ISO.match(value.strip())
        if m:
            try:
                return date(int(m[1]), int(m[2]), int(m[3]))
            except ValueError:
                return None
    return None


def _strict_date(value: Any, name: str) -> Optional[date]:
    if value is None or value == "":
        return None
    parsed = _to_date(value)
    if parsed is None:
        raise AnalyticsError(
            "invalid_arguments", f"{name} must be a date written as YYYY-MM-DD."
        )
    return parsed


def _day(d: date) -> str:
    return f"{d:%b} {d.day}"


def _span(start: date, end: date) -> str:
    if start == end:
        return f"{_day(start)}, {start.year}"
    if start.year == end.year:
        return f"{_day(start)} to {_day(end)}, {end.year}"
    return f"{_day(start)}, {start.year} to {_day(end)}, {end.year}"


def _monday(d: date) -> date:
    return d - timedelta(days=d.weekday())


def _n(count: int, one: str, many: Optional[str] = None) -> str:
    """ "1 task", "3 tasks", "2 activities" (pass ``many`` for irregular plurals)."""
    return f"{count} {one if count == 1 else (many or one + 's')}"


def _clean_text(value: Any, limit: int, name: str) -> str:
    if not isinstance(value, str):
        raise AnalyticsError("invalid_chart", f"{name} must be text.")
    text = "".join(
        c for c in value if unicodedata.category(c) not in ("Cc", "Cf")
    ).strip()
    if len(text) > limit:
        raise AnalyticsError("invalid_chart", f"{name} is too long.")
    return text


# ---------------------------------------------------------------------------
# Validation (the only way a chart leaves this module, and the view runs it again)
# ---------------------------------------------------------------------------
def validate_chart(spec: Any) -> Dict[str, Any]:
    """Return a cleaned copy of ``spec`` or raise ``AnalyticsError("invalid_chart", ...)``."""
    if not isinstance(spec, Mapping) or spec.get("type") != "chart":
        raise AnalyticsError("invalid_chart", "Not a chart.")
    kind = spec.get("kind")
    if kind not in CHART_KINDS:
        raise AnalyticsError("invalid_chart", "Unsupported chart kind.")

    labels_in, series_in = spec.get("labels"), spec.get("series")
    if (
        not isinstance(labels_in, (list, tuple))
        or not 1 <= len(labels_in) <= MAX_POINTS
    ):
        raise AnalyticsError(
            "invalid_chart", f"A chart needs 1 to {MAX_POINTS} points."
        )
    labels = [
        _clean_text(
            str(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else x,
            MAX_LABEL_CHARS,
            "label",
        )
        for x in labels_in
    ]
    if not all(labels) or len(set(labels)) != len(labels):
        raise AnalyticsError("invalid_chart", "Labels must be non-empty and unique.")

    if (
        not isinstance(series_in, (list, tuple))
        or not 1 <= len(series_in) <= MAX_SERIES
    ):
        raise AnalyticsError(
            "invalid_chart", f"A chart needs 1 to {MAX_SERIES} series."
        )
    series: List[Dict[str, Any]] = []
    for item in series_in:
        if not isinstance(item, Mapping):
            raise AnalyticsError("invalid_chart", "Bad series.")
        values = item.get("values")
        if not isinstance(values, (list, tuple)) or len(values) != len(labels):
            raise AnalyticsError(
                "invalid_chart", "Each series needs one value per label."
            )
        clean_values: List[Union[int, float]] = []
        for v in values:
            if (
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or abs(v) > MAX_ABS_VALUE
            ):
                raise AnalyticsError(
                    "invalid_chart", "Chart values must be ordinary finite numbers."
                )
            clean_values.append(v)
        series.append(
            {
                "name": _clean_text(
                    item.get("name", ""), MAX_LABEL_CHARS, "series name"
                ),
                "values": clean_values,
            }
        )
    if len({s["name"] for s in series}) != len(series):
        raise AnalyticsError("invalid_chart", "Series names must be unique.")

    total = spec.get("total")
    if total is not None and (
        isinstance(total, bool)
        or not isinstance(total, (int, float))
        or not math.isfinite(total)
    ):
        raise AnalyticsError("invalid_chart", "Bad total.")
    source = spec.get("source", "")
    cleaned: Dict[str, Any] = {
        "type": "chart",
        "kind": kind,
        "source": (
            source
            if isinstance(source, str) and re.fullmatch(r"[a-z_]{0,40}", source)
            else ""
        ),
        "title": _clean_text(spec.get("title", ""), MAX_TITLE_CHARS, "title"),
        "subtitle": _clean_text(
            spec.get("subtitle", ""), MAX_TITLE_CHARS * 2, "subtitle"
        ),
        "x_label": _clean_text(spec.get("x_label", ""), MAX_LABEL_CHARS, "x label"),
        "y_label": _clean_text(spec.get("y_label", ""), MAX_LABEL_CHARS, "y label"),
        "labels": labels,
        "series": series,
        "total": total,
        "summary": _clean_text(spec.get("summary", ""), MAX_SUMMARY_CHARS, "summary"),
    }
    if not cleaned["title"]:
        raise AnalyticsError("invalid_chart", "A chart needs a title.")
    return cleaned


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_twin_data() -> Mapping[str, Any]:
    """``config.load_twin_data()``, or the JSON file directly when the config cannot be imported
    (it raises when GROQ_API_KEY is unset, which charts do not need)."""
    try:
        if str(PROJECT_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "src"))
        from digital_twin.config import load_twin_data

        return load_twin_data()
    except (RuntimeError, ImportError):
        with open(DATA_FILE, "r", encoding="utf-8") as handle:
            return json.load(handle)


def _default_tasks() -> Any:
    import importlib

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    for name in ("services.task_service", "services.todo_service"):
        try:
            return importlib.import_module(name).get_service()
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
    raise AnalyticsError("unavailable", "The task store is not available.")


History = Union[
    Iterable[Mapping[str, Any]], Callable[[], Iterable[Mapping[str, Any]]], None
]

_STATUS_LABELS = {
    "todo": "To do",
    "in_progress": "In progress",
    "in_review": "In review",
    "done": "Done",
}


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class AnalyticsService:
    """Builds validated chart specs. ``history`` and ``tasks`` can be replaced in tests.

    history   activity entries (``{"time": "2026-08-10", "activity": "..."}``), or a function that
              returns them; default: the twin's ``activity_history``
    tasks     an object with ``list_tasks()`` (and optionally ``now()``); default: the task service
    clock     ``clock() -> datetime`` for "today" in the task charts; default: ``tasks.now`` or now
    """

    def __init__(
        self,
        history: History = None,
        tasks: Any = None,
        clock: Optional[Callable[[], datetime]] = None,
        categories: Sequence[Tuple[str, Sequence[str]]] = DEFAULT_CATEGORIES,
    ) -> None:
        self._history, self._tasks, self._clock, self._categories = (
            history,
            tasks,
            clock,
            categories,
        )

    # -- inputs ---------------------------------------------------------------------------
    def _entries(self) -> List[Tuple[Optional[date], str]]:
        raw = self._history
        if callable(raw):
            raw = raw()
        elif raw is None:
            raw = _load_twin_data().get("activity_history") or []
        entries: List[Tuple[Optional[date], str]] = []
        for item in raw or []:
            if not isinstance(item, Mapping):
                continue
            text = item.get("activity") or item.get("text")
            if isinstance(text, str) and text.strip():
                entries.append((_to_date(item.get("time")), text))
        return entries

    def _task_list(self) -> List[Any]:
        tasks = self._tasks if self._tasks is not None else _default_tasks()
        return list(tasks.list_tasks())

    def _today(self) -> date:
        if self._clock is not None:
            return self._clock().date()
        now = getattr(self._tasks, "now", None) if self._tasks is not None else None
        return (now() if callable(now) else datetime.now()).date()

    # -- public ---------------------------------------------------------------------------
    def describe_sources(self) -> List[Dict[str, Any]]:
        return [
            {
                "source": s.name,
                "description": s.description,
                "kinds": list(CHART_KINDS if s.time_based else ("bar",)),
            }
            for s in SOURCES.values()
        ]

    def build_chart(
        self,
        source: str,
        *,
        kind: Optional[str] = None,
        date_from: Any = None,
        date_to: Any = None,
    ) -> Dict[str, Any]:
        """A validated chart spec for ``source``. Raises ``AnalyticsError`` (never anything else
        for expected problems): ``invalid_arguments``, ``no_data``, ``too_many_points``.
        """
        meta = SOURCES.get(source) if isinstance(source, str) else None
        if meta is None:
            raise AnalyticsError(
                "invalid_arguments",
                f"Unknown chart source. Use one of: {', '.join(SOURCES)}.",
            )
        kind = kind or "bar"
        if kind not in CHART_KINDS:
            raise AnalyticsError(
                "invalid_arguments", f"kind must be one of: {', '.join(CHART_KINDS)}."
            )
        if kind == "line" and not meta.time_based:
            raise AnalyticsError(
                "invalid_arguments",
                "A line chart only fits activity_per_day and activity_per_week; use bar.",
            )
        start, end = _strict_date(date_from, "date_from"), _strict_date(
            date_to, "date_to"
        )
        if start and end and start > end:
            raise AnalyticsError("invalid_arguments", "date_from is after date_to.")
        if (start or end) and not meta.uses_dates:
            raise AnalyticsError(
                "invalid_arguments", "That chart source does not take a date range."
            )

        build = {
            "activity_per_day": self._per_day,
            "activity_per_week": self._per_week,
            "activity_by_category": self._by_category,
            "tasks_by_status": self._tasks_by_status,
            "tasks_by_due": self._tasks_by_due,
        }[source]
        spec = build(start, end) if meta.uses_dates else build()
        spec.update({"type": "chart", "kind": kind, "source": source})
        return validate_chart(spec)

    # -- activity -------------------------------------------------------------------------
    def _dated(
        self, start: Optional[date], end: Optional[date]
    ) -> List[Tuple[date, str]]:
        rows = [(d, t) for d, t in self._entries() if d is not None]
        rows = [
            (d, t)
            for d, t in rows
            if (not start or d >= start) and (not end or d <= end)
        ]
        if not rows:
            raise AnalyticsError(
                "no_data", "There are no dated activity entries in that range."
            )
        return rows

    def _per_day(self, start: Optional[date], end: Optional[date]) -> Dict[str, Any]:
        rows = self._dated(start, end)
        first, last = start or min(d for d, _ in rows), end or max(d for d, _ in rows)
        days = (last - first).days + 1
        if days > MAX_POINTS:
            raise AnalyticsError(
                "too_many_points",
                f"That range is {days} days. Use activity_per_week or a range of at most {MAX_POINTS} days.",
            )
        counts = {first + timedelta(days=i): 0 for i in range(days)}
        for d, _ in rows:
            counts[d] += 1
        values = list(counts.values())
        busiest = max(counts, key=lambda d: (counts[d], -d.toordinal()))
        return {
            "title": "Activity per day",
            "subtitle": _span(first, last),
            "x_label": "Day",
            "y_label": "Activities",
            "labels": [_day(d) for d in counts],
            "series": [{"name": "Activities", "values": values}],
            "total": sum(values),
            "summary": f"{_n(sum(values), 'activity', 'activities')} over {_n(days, 'day')}. Busiest day: {_day(busiest)} ({counts[busiest]}).",
        }

    def _per_week(self, start: Optional[date], end: Optional[date]) -> Dict[str, Any]:
        rows = self._dated(start, end)
        first = _monday(start or min(d for d, _ in rows))
        last = _monday(end or max(d for d, _ in rows))
        weeks = (last - first).days // 7 + 1
        if weeks > MAX_POINTS:
            raise AnalyticsError(
                "too_many_points",
                f"That range is {weeks} weeks. Narrow it to at most {MAX_POINTS} weeks.",
            )
        counts = {first + timedelta(weeks=i): 0 for i in range(weeks)}
        for d, _ in rows:
            counts[_monday(d)] += 1
        busiest = max(counts, key=lambda d: (counts[d], -d.toordinal()))
        total = sum(counts.values())
        return {
            "title": "Activity per week",
            "subtitle": _span(min(d for d, _ in rows), max(d for d, _ in rows)),
            "x_label": "Week starting",
            "y_label": "Activities",
            "labels": [_day(d) for d in counts],
            "series": [{"name": "Activities", "values": list(counts.values())}],
            "total": total,
            "summary": f"{_n(total, 'activity', 'activities')} over {_n(weeks, 'week')}. Busiest week starts {_day(busiest)} ({counts[busiest]}).",
        }

    def _by_category(
        self, start: Optional[date], end: Optional[date]
    ) -> Dict[str, Any]:
        has_range = bool(start or end)
        rows = (
            self._dated(start, end)
            if has_range
            else [(d, t) for d, t in self._entries()]
        )
        if not rows:
            raise AnalyticsError("no_data", "There are no activity entries to chart.")
        counts: Dict[str, int] = {}
        for _, text in rows:
            name = categorize(text, self._categories)
            counts[name] = counts.get(name, 0) + 1
        ordered = sorted(counts.items(), key=lambda kv: (kv[0] == OTHER, -kv[1], kv[0]))
        dates = [d for d, _ in rows if d]
        total = sum(counts.values())
        top, top_n = ordered[0]
        return {
            "title": "Activity by category",
            "subtitle": (
                _span(min(dates), max(dates))
                if dates
                else _n(total, "entry", "entries")
            ),
            "x_label": "Category",
            "y_label": "Activities",
            "labels": [name for name, _ in ordered],
            "series": [{"name": "Activities", "values": [n for _, n in ordered]}],
            "total": total,
            "summary": f"{_n(total, 'activity', 'activities')} in {_n(len(ordered), 'category', 'categories')}. Most common: {top} ({top_n}).",
        }

    # -- tasks ----------------------------------------------------------------------------
    def _tasks_by_status(self) -> Dict[str, Any]:
        tasks = self._task_list()
        if not tasks:
            raise AnalyticsError("no_data", "There are no tasks to chart.")
        counts: Dict[str, int] = {key: 0 for key in _STATUS_LABELS}
        for t in tasks:
            key = str(getattr(t, "status", "todo"))
            counts[key] = counts.get(key, 0) + 1
        labels = [
            _STATUS_LABELS.get(k, k.replace("_", " ").capitalize()) for k in counts
        ]
        total = len(tasks)
        done = counts.get("done", 0)
        return {
            "title": "Tasks by status",
            "subtitle": _n(total, "task"),
            "x_label": "Status",
            "y_label": "Tasks",
            "labels": labels,
            "series": [{"name": "Tasks", "values": list(counts.values())}],
            "total": total,
            "summary": f"{_n(total, 'task')}: {done} done, {total - done} still open.",
        }

    def _tasks_by_due(self) -> Dict[str, Any]:
        tasks = [
            t for t in self._task_list() if str(getattr(t, "status", "")) != "done"
        ]
        if not tasks:
            raise AnalyticsError("no_data", "There are no open tasks to chart.")
        today = self._today()
        this_monday = _monday(today)
        names = [
            "Overdue",
            "This week",
            "Next week",
            "In 2 weeks",
            "In 3 weeks",
            "Later",
            "No due date",
        ]
        counts = {n: 0 for n in names}
        for t in tasks:
            due = _to_date(getattr(t, "due", None))
            if due is None:
                counts["No due date"] += 1
            elif due < today:
                counts["Overdue"] += 1
            else:
                weeks = (_monday(due) - this_monday).days // 7
                counts[names[1 + weeks] if weeks <= 3 else "Later"] += 1
        total = len(tasks)
        return {
            "title": "Open tasks by due date",
            "subtitle": f"{_n(total, 'open task')} as of {_day(today)}, {today.year}",
            "x_label": "Due",
            "y_label": "Tasks",
            "labels": names,
            "series": [{"name": "Open tasks", "values": list(counts.values())}],
            "total": total,
            "summary": f"{_n(total, 'open task')}: {counts['Overdue']} overdue, {counts['This week']} due this week.",
        }


# ---------------------------------------------------------------------------
# Default service + module-level helpers
# ---------------------------------------------------------------------------
_default = AnalyticsService()


def get_service() -> AnalyticsService:
    return _default


def build_chart(source: str, **kwargs: Any) -> Dict[str, Any]:
    return _default.build_chart(source, **kwargs)
