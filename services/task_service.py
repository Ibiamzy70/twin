from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger("digital_twin.todo_service")

__all__ = [
    "MAX_TEXT",
    "MAX_ASSIGNEE",
    "MAX_TASKS",
    "STATUS_LABELS",
    "STATUS_ORDER",
    "SORT_KEYS",
    "Task",
    "TaskService",
    "TaskError",
    "ValidationError",
    "TaskNotFoundError",
    "TaskStorageError",
    "configure",
    "get_service",
    "create_task",
    "get_task",
    "list_tasks",
    "update_task",
    "delete_task",
    "set_status",
    "toggle_done",
    "clear_completed",
    "clear_all",
    "get_stats",
    "import_tasks",
]

# ---------------------------------------------------------------------------
# Constants (same limits and statuses as the Tasks page)
# ---------------------------------------------------------------------------
MAX_TEXT = 200
MAX_ASSIGNEE = 40
MAX_TASKS = 5000
WEEK = 7
MAX_LIST_LIMIT = 1000

STATUS_LABELS: Dict[str, str] = {
    "todo": "To do",
    "in_progress": "In Progress",
    "in_review": "In Review",
    "done": "Done",
}
STATUS_ORDER: Tuple[str, ...] = tuple(STATUS_LABELS)

# Sort names accepted by list_tasks (the page's labels work too, case-insensitively).
SORT_KEYS: Tuple[str, ...] = ("newest", "due", "name", "status")
_SORT_ALIASES = {
    "newest": "newest",
    "due": "due",
    "due date": "due",
    "name": "name",
    "status": "status",
}

_UNSET: Any = object()
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_STATUS_LOOKUP = {
    **{sid: sid for sid in STATUS_LABELS},
    **{label.lower(): sid for sid, label in STATUS_LABELS.items()},
    **{label.lower().replace(" ", "_"): sid for sid, label in STATUS_LABELS.items()},
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class TaskError(Exception):
    """Base class for task service errors."""


class ValidationError(TaskError, ValueError):
    """The input is not a valid task, field or id."""


class TaskNotFoundError(TaskError, LookupError):
    """No task has that id."""


class TaskStorageError(TaskError):
    """The database could not be read or written."""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Task:
    id: int
    text: str
    status: str
    assignee: str
    due: Optional[date]
    created: datetime
    completed_at: Optional[datetime]
    prev_status: str

    @property
    def done(self) -> bool:
        return self.status == "done"

    @property
    def status_label(self) -> str:
        return STATUS_LABELS[self.status]

    def to_view_dict(self) -> Dict[str, Any]:
        """The dict shape ``views/tasks.py`` already renders (date / datetime objects)."""
        return {
            "id": self.id,
            "text": self.text,
            "done": self.done,
            "status": self.status,
            "assignee": self.assignee,
            "due": self.due,
            "created": self.created,
            "completed_at": self.completed_at,
            "prev_status": self.prev_status,
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for the agent: dates as ISO strings, plus the readable status."""
        return {
            "id": self.id,
            "text": self.text,
            "status": self.status,
            "status_label": self.status_label,
            "done": self.done,
            "assignee": self.assignee,
            "due": self.due.isoformat() if self.due else None,
            "created": self.created.isoformat(timespec="seconds"),
            "completed_at": (
                self.completed_at.isoformat(timespec="seconds")
                if self.completed_at
                else None
            ),
        }


# ---------------------------------------------------------------------------
# Validation helpers (pure)
# ---------------------------------------------------------------------------
def _squash(raw: str, limit: Optional[int] = None) -> str:
    text = " ".join(_CONTROL_CHARS.sub("", raw).split())
    return text[:limit] if limit is not None else text


def validate_text(raw: Any) -> str:
    if not isinstance(raw, str):
        raise ValidationError("Task text must be a string.")
    text = _squash(raw)
    if not text:
        raise ValidationError("Task text cannot be empty.")
    if len(text) > MAX_TEXT:
        raise ValidationError(
            f"Task text is too long ({len(text)} characters; maximum {MAX_TEXT})."
        )
    return text


def validate_assignee(raw: Any) -> str:
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raise ValidationError("Assignee must be a string.")
    text = _squash(raw)
    if len(text) > MAX_ASSIGNEE:
        raise ValidationError(
            f"Assignee is too long ({len(text)} characters; maximum {MAX_ASSIGNEE})."
        )
    return text


def validate_status(raw: Any) -> str:
    if isinstance(raw, str):
        key = " ".join(raw.strip().lower().split())
        found = _STATUS_LOOKUP.get(key) or _STATUS_LOOKUP.get(key.replace(" ", "_"))
        if found:
            return found
    raise ValidationError(
        f"Unknown status {raw!r}. Use one of: {', '.join(STATUS_ORDER)}."
    )


def validate_due(raw: Any) -> Optional[date]:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    if isinstance(raw, str):
        try:
            return date.fromisoformat(raw.strip()[:10])
        except ValueError:
            pass
    raise ValidationError("Due date must be a date or an ISO string (YYYY-MM-DD).")


def validate_id(raw: Any) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise ValidationError("Task id must be a positive integer.")
    return raw


def _parse_dt(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value).replace(tzinfo=None)
        except ValueError:
            return None
    return None


def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.replace(microsecond=0).isoformat() if dt else None


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    text         TEXT NOT NULL CHECK (length(text) BETWEEN 1 AND {MAX_TEXT}),
    status       TEXT NOT NULL DEFAULT 'todo'
                 CHECK (status IN ({", ".join(repr(s) for s in STATUS_ORDER)})),
    assignee     TEXT NOT NULL DEFAULT '',
    due          TEXT,
    created      TEXT NOT NULL,
    completed_at TEXT,
    prev_status  TEXT NOT NULL DEFAULT 'todo'
);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks (status);
CREATE INDEX IF NOT EXISTS idx_tasks_due ON tasks (due);
"""

_ORDER_SQL = {
    "newest": "created DESC, id DESC",
    "due": "due IS NULL, due ASC, id DESC",
    "name": "id ASC",  # re-sorted in Python: SQLite NOCASE only folds ASCII
    "status": (
        "CASE status "
        + " ".join(f"WHEN '{s}' THEN {i}" for i, s in enumerate(STATUS_ORDER))
        + " END ASC, id DESC"
    ),
}

StatusArg = Union[str, Iterable[str], None]


def default_db_path() -> Path:
    env = os.environ.get("TWIN_TASKS_DB")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "tasks.db"


class TaskService:
    """CRUD and queries over the task table. Safe to use from several threads."""

    def __init__(
        self,
        db_path: Union[str, Path, None] = None,
        *,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else default_db_path()
        self._clock = clock or datetime.now
        self._lock = threading.RLock()
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TaskStorageError(
                f"Cannot create the data folder {self.db_path.parent}: {exc}"
            ) from exc
        with self._read():
            pass  # creates the file and the schema now, so problems surface at start-up

    # -- plumbing -----------------------------------------------------------
    def now(self) -> datetime:
        return self._clock().replace(tzinfo=None, microsecond=0)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.Error:  # e.g. network drives; the default journal still works
            pass
        conn.executescript(_SCHEMA)
        return conn

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        try:
            with closing(self._connect()) as conn:
                yield conn
        except sqlite3.Error as exc:
            logger.exception("task database read failed")
            raise TaskStorageError("The task database could not be read.") from exc

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One atomic transaction; rolled back on any error."""
        with self._lock:
            try:
                conn = self._connect()
            except sqlite3.Error as exc:
                logger.exception("task database unavailable")
                raise TaskStorageError(
                    "The task database could not be opened."
                ) from exc
            try:
                conn.execute("BEGIN IMMEDIATE")
                yield conn
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._rollback(conn)
                logger.exception("task database write failed")
                raise TaskStorageError(
                    "The task database could not be updated."
                ) from exc
            except BaseException:
                self._rollback(conn)
                raise
            finally:
                conn.close()

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @staticmethod
    def _to_task(row: sqlite3.Row) -> Task:
        created = _parse_dt(row["created"]) or datetime.now().replace(microsecond=0)
        return Task(
            id=row["id"],
            text=row["text"],
            status=row["status"],
            assignee=row["assignee"],
            due=validate_due(row["due"]) if row["due"] else None,
            created=created,
            completed_at=_parse_dt(row["completed_at"]),
            prev_status=row["prev_status"],
        )

    def _fetch(self, conn: sqlite3.Connection, task_id: int) -> Task:
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        if row is None:
            raise TaskNotFoundError(f"No task with id {task_id}.")
        return self._to_task(row)

    # -- create -------------------------------------------------------------
    def create_task(
        self,
        text: str,
        *,
        assignee: Optional[str] = "",
        due: Any = None,
        status: str = "todo",
    ) -> Task:
        text_v = validate_text(text)
        assignee_v = validate_assignee(assignee)
        due_v = validate_due(due)
        status_v = validate_status(status)
        now = self.now()
        with self._write() as conn:
            if conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] >= MAX_TASKS:
                raise ValidationError(
                    f"Task limit reached ({MAX_TASKS}). Delete some tasks first."
                )
            cur = conn.execute(
                "INSERT INTO tasks (text, status, assignee, due, created, completed_at, prev_status) "
                "VALUES (?, ?, ?, ?, ?, ?, 'todo')",
                (
                    text_v,
                    status_v,
                    assignee_v,
                    due_v.isoformat() if due_v else None,
                    _iso(now),
                    _iso(now) if status_v == "done" else None,
                ),
            )
            task = self._fetch(conn, int(cur.lastrowid))
        logger.info("created task %s (%s)", task.id, task.status)
        return task

    # -- read ---------------------------------------------------------------
    def get_task(self, task_id: int) -> Task:
        task_id = validate_id(task_id)
        with self._read() as conn:
            return self._fetch(conn, task_id)

    def list_tasks(
        self,
        *,
        status: StatusArg = None,
        done: Optional[bool] = None,
        query: str = "",
        assignee: Optional[str] = None,
        overdue: bool = False,
        due_from: Any = None,
        due_to: Any = None,
        sort: str = "newest",
        limit: Optional[int] = None,
    ) -> List[Task]:
        """Filtered, sorted tasks. Every filter is optional and filters combine with AND.

        status    one status or several (ids or labels)
        done      True = only done, False = only open
        query     case-insensitive match on the task text or the assignee
        assignee  exact (case-insensitive) assignee
        overdue   open tasks whose due date is before today
        due_from / due_to   inclusive due-date range
        sort      newest | due | name | status
        limit     maximum number of tasks returned
        """
        sort_key = _SORT_ALIASES.get(str(sort).strip().lower())
        if sort_key is None:
            raise ValidationError(
                f"Unknown sort {sort!r}. Use one of: {', '.join(SORT_KEYS)}."
            )
        if limit is not None:
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
                raise ValidationError("limit must be a positive integer.")
            limit = min(limit, MAX_LIST_LIMIT)
        if not isinstance(query, str):
            raise ValidationError("query must be a string.")

        where: List[str] = []
        params: List[Any] = []
        if status is not None:
            raw = [status] if isinstance(status, str) else list(status)
            statuses = sorted({validate_status(s) for s in raw})
            if statuses:
                where.append(f"status IN ({', '.join('?' * len(statuses))})")
                params += statuses
        if done is not None:
            where.append("status = 'done'" if done else "status != 'done'")
        if overdue:
            where.append("status != 'done' AND due IS NOT NULL AND due < ?")
            params.append(self.now().date().isoformat())
        start, end = validate_due(due_from), validate_due(due_to)
        if start:
            where.append("due IS NOT NULL AND due >= ?")
            params.append(start.isoformat())
        if end:
            where.append("due IS NOT NULL AND due <= ?")
            params.append(end.isoformat())

        sql = "SELECT * FROM tasks"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" ORDER BY {_ORDER_SQL[sort_key]}"
        with self._read() as conn:
            tasks = [self._to_task(r) for r in conn.execute(sql, params)]

        if sort_key == "name":
            tasks.sort(key=lambda t: (t.text.casefold(), t.id))

        # Text filters run in Python so matching is correct for non-ASCII text too.
        needle = _squash(query).casefold()
        if needle:
            tasks = [
                t
                for t in tasks
                if needle in t.text.casefold() or needle in t.assignee.casefold()
            ]
        if assignee is not None:
            who = validate_assignee(assignee).casefold()
            tasks = [t for t in tasks if t.assignee.casefold() == who]
        return tasks[:limit] if limit else tasks

    def count(self) -> int:
        with self._read() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])

    # -- update -------------------------------------------------------------
    def update_task(
        self,
        task_id: int,
        *,
        text: Any = _UNSET,
        assignee: Any = _UNSET,
        due: Any = _UNSET,
        status: Any = _UNSET,
    ) -> Task:
        """Change only the fields you pass. ``due=None`` clears the due date."""
        task_id = validate_id(task_id)
        changes: Dict[str, Any] = {}
        if text is not _UNSET:
            changes["text"] = validate_text(text)
        if assignee is not _UNSET:
            changes["assignee"] = validate_assignee(assignee)
        if due is not _UNSET:
            parsed = validate_due(due)
            changes["due"] = parsed.isoformat() if parsed else None
        new_status = validate_status(status) if status is not _UNSET else None
        if not changes and new_status is None:
            raise ValidationError(
                "Nothing to update: pass text, assignee, due or status."
            )

        with self._write() as conn:
            current = self._fetch(conn, task_id)
            if new_status is not None and new_status != current.status:
                changes.update(self._status_columns(current, new_status))
            if changes:
                cols = ", ".join(
                    f"{c} = ?" for c in changes
                )  # column names are our own constants
                conn.execute(
                    f"UPDATE tasks SET {cols} WHERE id = ?",
                    [*changes.values(), task_id],
                )
            task = self._fetch(conn, task_id)
        logger.info("updated task %s (%s)", task_id, ", ".join(changes) or "no change")
        return task

    def _status_columns(self, current: Task, new_status: str) -> Dict[str, Any]:
        """Columns to change when a status changes, keeping done / completed_at consistent."""
        if new_status == "done":
            return {
                "status": "done",
                "prev_status": current.status,
                "completed_at": _iso(self.now()),
            }
        return {"status": new_status, "completed_at": None}

    def set_status(self, task_id: int, status: str) -> Task:
        return self.update_task(task_id, status=status)

    def toggle_done(self, task_id: int, done: Optional[bool] = None) -> Task:
        """Tick -> done; untick -> back to the status the task had before it was done.

        ``done=None`` flips the current state; ``True`` / ``False`` set it explicitly.
        """
        task_id = validate_id(task_id)
        with self._write() as conn:
            current = self._fetch(conn, task_id)
            target = (not current.done) if done is None else bool(done)
            if target == current.done:
                return current
            new_status = (
                "done"
                if target
                else (current.prev_status if current.prev_status != "done" else "todo")
            )
            cols = self._status_columns(current, new_status)
            conn.execute(
                f"UPDATE tasks SET {', '.join(f'{c} = ?' for c in cols)} WHERE id = ?",
                [*cols.values(), task_id],
            )
            return self._fetch(conn, task_id)

    # -- delete -------------------------------------------------------------
    def delete_task(self, task_id: int) -> Task:
        """Delete exactly one task and return it. Unknown id -> ``TaskNotFoundError``."""
        task_id = validate_id(task_id)
        with self._write() as conn:
            task = self._fetch(conn, task_id)
            conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        logger.info("deleted task %s", task_id)
        return task

    def clear_completed(self) -> int:
        with self._write() as conn:
            removed = conn.execute("DELETE FROM tasks WHERE status = 'done'").rowcount
        logger.info("cleared %s completed tasks", removed)
        return removed

    def clear_all(self) -> int:
        with self._write() as conn:
            removed = conn.execute("DELETE FROM tasks").rowcount
        logger.info("cleared all tasks (%s)", removed)
        return removed

    # -- stats --------------------------------------------------------------
    def get_stats(self) -> Dict[str, int]:
        """Counts for the summary cards, derived from the stored tasks."""
        tasks = self.list_tasks()
        now = self.now()
        today = now.date()
        week_ago, week_ahead = now - timedelta(days=WEEK), today + timedelta(days=WEEK)
        by_status = {s: 0 for s in STATUS_ORDER}
        for t in tasks:
            by_status[t.status] += 1
        total = len(tasks)
        completed = by_status["done"]
        return {
            "total": total,
            "completed": completed,
            "remaining": total - completed,
            "todo": by_status["todo"],
            "in_progress": by_status["in_progress"],
            "in_review": by_status["in_review"],
            "pct": int(completed / total * 100) if total else 0,
            "overdue": sum(1 for t in tasks if not t.done and t.due and t.due < today),
            "due_today": sum(1 for t in tasks if not t.done and t.due == today),
            "added_week": sum(1 for t in tasks if t.created >= week_ago),
            "completed_week": sum(
                1 for t in tasks if t.completed_at and t.completed_at >= week_ago
            ),
            "progress_due_soon": sum(
                1
                for t in tasks
                if t.status == "in_progress" and t.due and today <= t.due <= week_ahead
            ),
            "review_overdue": sum(
                1 for t in tasks if t.status == "in_review" and t.due and t.due < today
            ),
        }

    # -- migration ----------------------------------------------------------
    def import_tasks(self, items: Iterable[Mapping[str, Any]]) -> Tuple[int, int]:
        """Bulk-insert task dicts (e.g. the old ``st.session_state`` list). Returns (imported, skipped).

        Lenient on purpose: over-long text is trimmed, ``done`` wins over ``status``
        (the old page only set ``done``), and unusable records are skipped. New ids are
        assigned; created / completed times are preserved when present.
        """
        rows: List[Tuple[Any, ...]] = []
        skipped = 0
        for item in items:
            if not isinstance(item, Mapping) or not isinstance(item.get("text"), str):
                skipped += 1
                continue
            text = _squash(item["text"], MAX_TEXT)
            if not text:
                skipped += 1
                continue
            done = bool(item.get("done")) or item.get("status") == "done"
            raw_status = item.get("status")
            status = (
                raw_status
                if raw_status in STATUS_LABELS
                else ("done" if done else "todo")
            )
            if done:
                status = "done"
            elif status == "done":
                status = "todo"
            assignee = item.get("assignee")
            assignee = (
                _squash(assignee, MAX_ASSIGNEE) if isinstance(assignee, str) else ""
            )
            try:
                due = validate_due(item.get("due"))
            except ValidationError:
                due = None
            created = _parse_dt(item.get("created")) or self.now()
            completed = _parse_dt(item.get("completed_at")) or (
                created if status == "done" else None
            )
            prev = item.get("prev_status")
            prev = prev if prev in STATUS_LABELS and prev != "done" else "todo"
            rows.append(
                (
                    text,
                    status,
                    assignee,
                    due.isoformat() if due else None,
                    _iso(created),
                    _iso(completed) if status == "done" else None,
                    prev,
                )
            )
        if not rows:
            return 0, skipped
        with self._write() as conn:
            room = MAX_TASKS - conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
            if len(rows) > room:
                skipped += len(rows) - max(room, 0)
                rows = rows[: max(room, 0)]
            conn.executemany(
                "INSERT INTO tasks (text, status, assignee, due, created, completed_at, prev_status) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.info("imported %s tasks (%s skipped)", len(rows), skipped)
        return len(rows), skipped


# ---------------------------------------------------------------------------
# Default service + module-level functions (what views and tools import)
# ---------------------------------------------------------------------------
_default: Optional[TaskService] = None
_default_lock = threading.Lock()


def get_service() -> TaskService:
    """The shared service, created on first use."""
    global _default
    with _default_lock:
        if _default is None:
            _default = TaskService()
        return _default


def configure(
    db_path: Union[str, Path, None] = None,
    *,
    clock: Optional[Callable[[], datetime]] = None,
) -> TaskService:
    """Replace the shared service (a different database path, or a fake clock in tests)."""
    global _default
    with _default_lock:
        _default = TaskService(db_path, clock=clock)
        return _default


def create_task(
    text: str, *, assignee: Optional[str] = "", due: Any = None, status: str = "todo"
) -> Task:
    return get_service().create_task(text, assignee=assignee, due=due, status=status)


def get_task(task_id: int) -> Task:
    return get_service().get_task(task_id)


def list_tasks(**filters: Any) -> List[Task]:
    return get_service().list_tasks(**filters)


def update_task(task_id: int, **fields: Any) -> Task:
    return get_service().update_task(task_id, **fields)


def delete_task(task_id: int) -> Task:
    return get_service().delete_task(task_id)


def set_status(task_id: int, status: str) -> Task:
    return get_service().set_status(task_id, status)


def toggle_done(task_id: int, done: Optional[bool] = None) -> Task:
    return get_service().toggle_done(task_id, done)


def clear_completed() -> int:
    return get_service().clear_completed()


def clear_all() -> int:
    return get_service().clear_all()


def get_stats() -> Dict[str, int]:
    return get_service().get_stats()


def import_tasks(items: Iterable[Mapping[str, Any]]) -> Tuple[int, int]:
    return get_service().import_tasks(items)
