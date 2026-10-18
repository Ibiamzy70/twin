"""Tests for services/task_service.py (the SQLite-backed task service).

Every test uses its own temporary database and a fixed clock: no network, no shared state.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone

import pytest

from services import task_service as ts
from services.task_service import (
    TaskNotFoundError,
    TaskService,
    TaskStorageError,
    ValidationError,
)

NOW = datetime(2026, 10, 16, 9, 0, 0)
TODAY = NOW.date()


class Clock:
    def __init__(self, start: datetime = NOW) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kw) -> None:
        self.now += timedelta(**kw)


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "data" / "tasks.db"


@pytest.fixture
def svc(db_path, clock) -> TaskService:
    return TaskService(db_path, clock=clock)


@pytest.fixture
def seeded(svc, clock) -> TaskService:
    """Six tasks (ids 1..6) covering every status and due-date state."""
    d = timedelta
    svc.create_task("Plan sprint")  # 1
    clock.advance(minutes=1)
    svc.create_task(
        "Design landing page",
        assignee="Ada",
        status="in_progress",
        due=TODAY + d(days=2),
    )  # 2
    clock.advance(minutes=1)
    svc.create_task("Write API docs", assignee="Grace", due=TODAY)  # 3
    clock.advance(minutes=1)
    svc.create_task(
        "Review pull request",
        assignee="Linus",
        status="in_review",
        due=TODAY - d(days=1),
    )  # 4 overdue
    clock.advance(minutes=1)
    svc.create_task(
        "Fix login bug", assignee="Grace", status="in_progress", due=TODAY + d(days=20)
    )  # 5
    clock.advance(minutes=1)
    svc.create_task(
        "Ship v1.0", assignee="Ada", status="done", due=TODAY - d(days=3)
    )  # 6 done
    return svc


def texts(tasks):
    return [t.text for t in tasks]


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
class TestValidators:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("  a   b \n", "a b"),
            ("x\x00y\x07z", "xyz"),
            ("tab\tsep", "tab sep"),
            ("é☕", "é☕"),
        ],
    )
    def test_text_is_tidied(self, raw, expected):
        assert ts.validate_text(raw) == expected

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t", "\x00\x01", None, 5, ["x"]])
    def test_text_rejects(self, bad):
        with pytest.raises(ValidationError):
            ts.validate_text(bad)

    def test_text_limit_is_exact(self):
        assert ts.validate_text("x" * ts.MAX_TEXT) == "x" * ts.MAX_TEXT
        with pytest.raises(ValidationError, match="too long"):
            ts.validate_text("x" * (ts.MAX_TEXT + 1))

    def test_assignee(self):
        assert ts.validate_assignee(None) == ""
        assert ts.validate_assignee("  Ada   L ") == "Ada L"
        with pytest.raises(ValidationError):
            ts.validate_assignee(5)
        with pytest.raises(ValidationError, match="too long"):
            ts.validate_assignee("a" * (ts.MAX_ASSIGNEE + 1))

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("todo", "todo"),
            ("in_progress", "in_progress"),
            ("In Progress", "in_progress"),
            ("in progress", "in_progress"),
            ("  DONE ", "done"),
            ("In   Review", "in_review"),
            ("to do", "todo"),
        ],
    )
    def test_status_accepts_ids_and_labels(self, raw, expected):
        assert ts.validate_status(raw) == expected

    @pytest.mark.parametrize("bad", ["blocked", "", None, 3, ["todo"]])
    def test_status_rejects(self, bad):
        with pytest.raises(ValidationError):
            ts.validate_status(bad)

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (None, None),
            ("", None),
            ("   ", None),
            ("2026-10-20", date(2026, 10, 20)),
            ("2026-10-20T08:30:00", date(2026, 10, 20)),
            (date(2026, 10, 20), date(2026, 10, 20)),
            (datetime(2026, 10, 20, 23, 59), date(2026, 10, 20)),
        ],
    )
    def test_due_accepts(self, raw, expected):
        assert ts.validate_due(raw) == expected

    @pytest.mark.parametrize("bad", ["tomorrow", "20/10/2026", "2026-13-40", 5, 1.5])
    def test_due_rejects(self, bad):
        with pytest.raises(ValidationError):
            ts.validate_due(bad)

    @pytest.mark.parametrize("bad", [0, -1, True, False, 1.0, "1", None])
    def test_id_rejects(self, bad):
        with pytest.raises(ValidationError):
            ts.validate_id(bad)

    def test_errors_are_catchable_as_builtins(self):
        assert issubclass(ValidationError, ValueError) and issubclass(
            TaskNotFoundError, LookupError
        )
        assert issubclass(TaskStorageError, ts.TaskError)


# ---------------------------------------------------------------------------
# construction and storage
# ---------------------------------------------------------------------------
class TestStorage:
    def test_creates_folder_file_and_schema(self, db_path, clock):
        assert not db_path.parent.exists()
        TaskService(db_path, clock=clock)
        assert db_path.is_file()
        with sqlite3.connect(db_path) as conn:
            names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
        assert {"tasks", "idx_tasks_status", "idx_tasks_due"} <= names

    def test_data_survives_a_new_service(self, db_path, clock):
        TaskService(db_path, clock=clock).create_task(
            "persist me", assignee="Ada", due="2026-10-30"
        )
        again = TaskService(db_path, clock=clock)
        (task,) = again.list_tasks()
        assert (task.text, task.assignee, task.due) == (
            "persist me",
            "Ada",
            date(2026, 10, 30),
        )

    def test_ids_are_never_reused(self, svc):
        a, b = svc.create_task("a"), svc.create_task("b")
        svc.delete_task(b.id)
        assert svc.create_task("c").id == b.id + 1
        svc.clear_all()
        assert svc.create_task("d").id > b.id + 1  # AUTOINCREMENT survives a full clear

    def test_database_constraints_back_up_validation(self, svc):
        with sqlite3.connect(svc.db_path) as conn:
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(
                    "INSERT INTO tasks (text, status, created) VALUES ('x', 'bogus', 'now')"
                )
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("INSERT INTO tasks (text, created) VALUES ('', 'now')")

    def test_unwritable_location_is_a_storage_error(self, tmp_path, clock):
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file, not a folder")
        with pytest.raises(TaskStorageError):
            TaskService(blocker / "tasks.db", clock=clock)

    def test_corrupt_database_is_a_storage_error_not_a_crash(self, tmp_path, clock):
        bad = tmp_path / "bad.db"
        bad.write_bytes(b"this is not an sqlite database" * 50)
        with pytest.raises(TaskStorageError):
            TaskService(bad, clock=clock)

    def test_clock_is_reduced_to_naive_whole_seconds(self, db_path):
        aware = datetime(2026, 10, 16, 9, 0, 0, 123456, tzinfo=timezone.utc)
        s = TaskService(db_path, clock=lambda: aware)
        assert s.now() == datetime(2026, 10, 16, 9, 0, 0)
        assert s.create_task("x").created == datetime(2026, 10, 16, 9, 0, 0)

    def test_concurrent_writers_do_not_lose_or_duplicate_rows(self, svc):
        errors: list = []

        def worker(n: int) -> None:
            try:
                for i in range(10):
                    svc.create_task(f"w{n}-{i}")
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        ids = [t.id for t in svc.list_tasks()]
        assert len(ids) == 80 and len(set(ids)) == 80


# ---------------------------------------------------------------------------
# create / get
# ---------------------------------------------------------------------------
class TestCreateGet:
    def test_defaults(self, svc):
        t = svc.create_task("  Write   the report ")
        assert (t.id, t.text, t.status, t.assignee, t.due) == (
            1,
            "Write the report",
            "todo",
            "",
            None,
        )
        assert (t.created, t.completed_at, t.prev_status) == (NOW, None, "todo")
        assert t.done is False and t.status_label == "To do"

    def test_all_fields_and_label_status(self, svc):
        t = svc.create_task(
            "x", assignee=" Ada  L ", due="2026-10-30", status="In Review"
        )
        assert (t.assignee, t.due, t.status) == (
            "Ada L",
            date(2026, 10, 30),
            "in_review",
        )

    def test_created_as_done_is_completed_now(self, svc):
        t = svc.create_task("x", status="done")
        assert t.done and t.completed_at == NOW

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"text": ""},
            {"text": None},
            {"text": "x", "status": "blocked"},
            {"text": "x", "due": "soon"},
            {"text": "x", "assignee": 5},
            {"text": "x" * 201},
        ],
    )
    def test_invalid_input_stores_nothing(self, svc, kwargs):
        with pytest.raises(ValidationError):
            svc.create_task(**kwargs)
        assert svc.count() == 0

    def test_task_limit(self, svc, monkeypatch):
        monkeypatch.setattr(ts, "MAX_TASKS", 3)
        for i in range(3):
            svc.create_task(f"t{i}")
        with pytest.raises(ValidationError, match="limit"):
            svc.create_task("one too many")
        assert svc.count() == 3
        svc.delete_task(1)
        svc.create_task("room again")

    def test_sql_metacharacters_are_data(self, svc):
        nasty = "'; DROP TABLE tasks; --"
        t = svc.create_task(nasty, assignee='" OR 1=1 --')
        assert svc.get_task(t.id).text == nasty
        assert svc.count() == 1

    def test_get_task(self, seeded):
        assert seeded.get_task(3).text == "Write API docs"
        with pytest.raises(TaskNotFoundError):
            seeded.get_task(999)
        with pytest.raises(ValidationError):
            seeded.get_task("3")

    def test_task_is_immutable(self, svc):
        t = svc.create_task("x")
        with pytest.raises(Exception):
            t.text = "changed"  # frozen dataclass


# ---------------------------------------------------------------------------
# list / filter / sort
# ---------------------------------------------------------------------------
class TestList:
    def test_default_is_newest_first(self, seeded):
        assert texts(seeded.list_tasks())[0] == "Ship v1.0"
        assert len(seeded.list_tasks()) == 6

    def test_status_single_multiple_and_labels(self, seeded):
        assert texts(seeded.list_tasks(status="in_progress", sort="name")) == [
            "Design landing page",
            "Fix login bug",
        ]
        assert len(seeded.list_tasks(status=["in_progress", "done"])) == 3
        assert len(seeded.list_tasks(status=["In Progress", "Done"])) == 3
        assert (
            len(seeded.list_tasks(status=[])) == 6
        )  # an empty filter means "no filter"

    def test_status_filter_rejects_unknown(self, seeded):
        with pytest.raises(ValidationError):
            seeded.list_tasks(status="nope")

    def test_done_flag(self, seeded):
        assert texts(seeded.list_tasks(done=True)) == ["Ship v1.0"]
        assert len(seeded.list_tasks(done=False)) == 5

    def test_query_matches_text_and_assignee_case_insensitively(self, seeded):
        assert texts(seeded.list_tasks(query="LOGIN")) == ["Fix login bug"]
        assert sorted(texts(seeded.list_tasks(query="grace"))) == [
            "Fix login bug",
            "Write API docs",
        ]
        assert seeded.list_tasks(query="zzz") == []

    def test_query_is_unicode_aware(self, svc):
        svc.create_task("Prépare l'Éclair")
        svc.create_task("Straße fix")
        assert texts(svc.list_tasks(query="éclair")) == ["Prépare l'Éclair"]
        assert texts(svc.list_tasks(query="STRASSE")) == [
            "Straße fix"
        ]  # casefold, not lower

    def test_assignee_filter_is_exact_and_case_insensitive(self, seeded):
        assert sorted(texts(seeded.list_tasks(assignee="GRACE"))) == [
            "Fix login bug",
            "Write API docs",
        ]
        assert seeded.list_tasks(assignee="Gra") == []
        assert len(seeded.list_tasks(assignee="")) == 1  # unassigned: "Plan sprint"

    def test_overdue_excludes_done_and_undated(self, seeded):
        assert texts(seeded.list_tasks(overdue=True)) == ["Review pull request"]

    def test_due_range_is_inclusive(self, seeded):
        got = seeded.list_tasks(
            due_from=TODAY, due_to=TODAY + timedelta(days=2), sort="due"
        )
        assert texts(got) == ["Write API docs", "Design landing page"]
        assert texts(seeded.list_tasks(due_from=TODAY + timedelta(days=20))) == [
            "Fix login bug"
        ]
        assert "Plan sprint" not in texts(
            seeded.list_tasks(due_to=TODAY + timedelta(days=100))
        )  # undated never match

    def test_filters_combine_with_and(self, seeded):
        assert texts(seeded.list_tasks(status="in_progress", assignee="grace")) == [
            "Fix login bug"
        ]
        assert seeded.list_tasks(done=True, status="todo") == []

    @pytest.mark.parametrize("alias", ["due", "Due", "due date", "DUE DATE"])
    def test_sort_due_accepts_page_labels(self, seeded, alias):
        got = texts(seeded.list_tasks(sort=alias))
        assert (
            got[0] == "Ship v1.0" and got[-1] == "Plan sprint"
        )  # soonest first, undated last

    def test_sort_name_is_casefolded(self, svc):
        for n in ("banana", "Apple", "cherry", "éclair", "Zebra"):
            svc.create_task(n)
        # case is ignored; accented letters sort by code point, so "é" lands after "z"
        assert texts(svc.list_tasks(sort="name")) == [
            "Apple",
            "banana",
            "cherry",
            "Zebra",
            "éclair",
        ]

    def test_sort_status_follows_board_order(self, seeded):
        order = [t.status for t in seeded.list_tasks(sort="status")]
        assert order == sorted(order, key=ts.STATUS_ORDER.index)

    def test_sort_newest_breaks_ties_by_id(self, svc):
        a, b = svc.create_task("a"), svc.create_task("b")  # same clock second
        assert [t.id for t in svc.list_tasks()] == [b.id, a.id]

    def test_unknown_sort(self, seeded):
        with pytest.raises(ValidationError):
            seeded.list_tasks(sort="random")

    def test_limit(self, seeded):
        assert len(seeded.list_tasks(limit=2)) == 2
        assert len(seeded.list_tasks(limit=10_000)) == 6  # capped, not an error

    @pytest.mark.parametrize("bad", [0, -1, True, 1.5, "2"])
    def test_bad_limit(self, seeded, bad):
        with pytest.raises(ValidationError):
            seeded.list_tasks(limit=bad)

    def test_limit_applies_after_text_filters(self, svc):
        for i in range(5):
            svc.create_task(f"match {i}")
        svc.create_task("other")
        assert len(svc.list_tasks(query="match", limit=3)) == 3

    def test_query_must_be_text(self, seeded):
        with pytest.raises(ValidationError):
            seeded.list_tasks(query=5)

    def test_count(self, seeded):
        assert seeded.count() == 6


# ---------------------------------------------------------------------------
# update / status
# ---------------------------------------------------------------------------
class TestUpdate:
    def test_update_only_what_is_passed(self, svc):
        t = svc.create_task("old", assignee="Ada", due="2026-10-20")
        u = svc.update_task(t.id, text="  new  text ")
        assert (u.text, u.assignee, u.due) == ("new text", "Ada", date(2026, 10, 20))

    def test_update_all_fields(self, svc):
        t = svc.create_task("old")
        u = svc.update_task(
            t.id, text="new", assignee="Grace", due="2026-11-01", status="in_review"
        )
        assert (u.text, u.assignee, u.due, u.status) == (
            "new",
            "Grace",
            date(2026, 11, 1),
            "in_review",
        )

    def test_due_none_clears_and_assignee_can_be_blanked(self, svc):
        t = svc.create_task("x", assignee="Ada", due="2026-10-20")
        u = svc.update_task(t.id, due=None, assignee="")
        assert u.due is None and u.assignee == ""

    def test_nothing_to_update(self, svc):
        t = svc.create_task("x")
        with pytest.raises(ValidationError, match="Nothing to update"):
            svc.update_task(t.id)

    def test_invalid_field_changes_nothing(self, svc):
        t = svc.create_task("keep", assignee="Ada")
        with pytest.raises(ValidationError):
            svc.update_task(t.id, text="changed", status="blocked")
        with pytest.raises(ValidationError):
            svc.update_task(t.id, text="changed", due="garbage")
        after = svc.get_task(t.id)
        assert (after.text, after.assignee, after.status) == ("keep", "Ada", "todo")

    def test_unknown_and_invalid_ids(self, svc):
        with pytest.raises(TaskNotFoundError):
            svc.update_task(5, text="x")
        with pytest.raises(ValidationError):
            svc.update_task("5", text="x")

    def test_done_records_time_and_previous_status(self, svc, clock):
        t = svc.create_task("x", status="in_review")
        clock.advance(hours=2)
        d = svc.set_status(t.id, "done")
        assert (d.status, d.prev_status, d.completed_at) == (
            "done",
            "in_review",
            NOW + timedelta(hours=2),
        )

    def test_reopening_clears_completed_at(self, svc):
        t = svc.create_task("x", status="done")
        r = svc.set_status(t.id, "in_progress")
        assert (r.status, r.completed_at, r.done) == ("in_progress", None, False)

    def test_same_status_is_a_no_op(self, svc, clock):
        t = svc.create_task("x", status="done")
        clock.advance(days=1)
        assert svc.set_status(t.id, "done").completed_at == NOW  # not re-stamped

    def test_text_edit_keeps_status_data(self, svc):
        t = svc.create_task("x", status="done")
        u = svc.update_task(t.id, text="renamed")
        assert u.done and u.completed_at == NOW

    def test_done_and_edit_in_one_call(self, svc):
        t = svc.create_task("x", status="in_progress")
        u = svc.update_task(t.id, text="y", status="done")
        assert (u.text, u.status, u.prev_status) == ("y", "done", "in_progress")


class TestToggle:
    def test_tick_then_untick_restores_previous_status(self, svc):
        t = svc.create_task("x", status="in_review")
        assert svc.toggle_done(t.id, True).status == "done"
        back = svc.toggle_done(t.id, False)
        assert (back.status, back.done, back.completed_at) == ("in_review", False, None)

    def test_flip_without_argument(self, svc):
        t = svc.create_task("x")
        assert svc.toggle_done(t.id).done is True
        assert svc.toggle_done(t.id).status == "todo"

    def test_explicit_target_already_met_is_a_no_op(self, svc, clock):
        t = svc.create_task("x", status="in_progress")
        assert svc.toggle_done(t.id, False).status == "in_progress"
        d = svc.create_task("y", status="done")
        clock.advance(days=1)
        assert svc.toggle_done(d.id, True).completed_at == NOW

    def test_untick_task_created_as_done_goes_to_todo(self, svc):
        t = svc.create_task("x", status="done")
        assert svc.toggle_done(t.id, False).status == "todo"

    def test_unknown_task(self, svc):
        with pytest.raises(TaskNotFoundError):
            svc.toggle_done(42)


# ---------------------------------------------------------------------------
# delete / clear
# ---------------------------------------------------------------------------
class TestDelete:
    def test_delete_one(self, seeded):
        gone = seeded.delete_task(3)
        assert gone.text == "Write API docs" and seeded.count() == 5
        with pytest.raises(TaskNotFoundError):
            seeded.get_task(3)

    def test_delete_unknown_or_repeated(self, seeded):
        seeded.delete_task(1)
        with pytest.raises(TaskNotFoundError):
            seeded.delete_task(1)
        with pytest.raises(ValidationError):
            seeded.delete_task(0)

    def test_clear_completed(self, seeded):
        assert seeded.clear_completed() == 1
        assert seeded.count() == 5 and seeded.list_tasks(done=True) == []
        assert seeded.clear_completed() == 0

    def test_clear_all(self, seeded):
        assert seeded.clear_all() == 6
        assert seeded.count() == 0 and seeded.clear_all() == 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
class TestStats:
    def test_empty(self, svc):
        s = svc.get_stats()
        assert (s["total"], s["pct"], s["overdue"], s["due_today"]) == (0, 0, 0, 0)

    def test_seeded(self, seeded):
        s = seeded.get_stats()
        assert (s["total"], s["completed"], s["remaining"]) == (6, 1, 5)
        assert (s["todo"], s["in_progress"], s["in_review"]) == (2, 2, 1)
        assert s["pct"] == 16
        assert s["overdue"] == 1  # the done task's past due date does not count
        assert s["due_today"] == 1
        assert s["added_week"] == 6 and s["completed_week"] == 1
        assert (
            s["progress_due_soon"] == 1
        )  # Design landing page (+2 days); the +20 day one is outside
        assert s["review_overdue"] == 1

    def test_week_window_follows_the_clock(self, seeded, clock):
        clock.advance(days=8)
        s = seeded.get_stats()
        assert s["added_week"] == 0 and s["completed_week"] == 0

    def test_keys_the_dashboard_cards_need(self, seeded):
        assert {
            "total",
            "completed",
            "remaining",
            "todo",
            "in_progress",
            "in_review",
            "pct",
            "overdue",
            "added_week",
            "completed_week",
            "progress_due_soon",
            "review_overdue",
        } <= set(seeded.get_stats())


# ---------------------------------------------------------------------------
# shapes
# ---------------------------------------------------------------------------
class TestShapes:
    def test_to_dict_is_json_safe(self, seeded):
        payload = [t.to_dict() for t in seeded.list_tasks()]
        json.dumps(payload)
        done = next(p for p in payload if p["text"] == "Ship v1.0")
        assert (
            done["status_label"] == "Done"
            and done["done"] is True
            and done["completed_at"]
        )
        assert next(p for p in payload if p["text"] == "Plan sprint")["due"] is None

    def test_to_view_dict_matches_the_tasks_page_shape(self, seeded):
        d = seeded.get_task(2).to_view_dict()
        assert set(d) == {
            "id",
            "text",
            "done",
            "status",
            "assignee",
            "due",
            "created",
            "completed_at",
            "prev_status",
        }
        assert isinstance(d["due"], date) and isinstance(d["created"], datetime)


# ---------------------------------------------------------------------------
# migration from the old session list
# ---------------------------------------------------------------------------
class TestImport:
    def test_imports_old_shapes_leniently(self, svc):
        created = datetime(2026, 10, 1, 8, 0, 0)
        items = [
            {"id": 7, "text": "legacy done", "done": True},
            {
                "text": "  spaced   out ",
                "status": "in_review",
                "assignee": "Ada",
                "due": "2026-10-20",
                "created": created,
            },
            {"text": "x" * 500},
            {"text": "bad due", "due": "garbage"},
            {"text": "bad status", "status": "nope"},
            {"text": "reopened", "status": "done", "done": False},
            {"text": "   "},
            {"text": 5},
            "not a mapping",
            None,
        ]
        imported, skipped = svc.import_tasks(items)
        assert (imported, skipped) == (6, 4)
        got = {t.text: t for t in svc.list_tasks()}
        legacy = got["legacy done"]
        assert legacy.done and legacy.completed_at is not None
        spaced = got["spaced out"]
        assert (spaced.status, spaced.assignee, spaced.due, spaced.created) == (
            "in_review",
            "Ada",
            date(2026, 10, 20),
            created,
        )
        assert len(got["x" * ts.MAX_TEXT].text) == ts.MAX_TEXT  # trimmed, not rejected
        assert got["bad due"].due is None
        assert got["bad status"].status == "todo"
        # NOTE: unlike the page's normaliser, a record with status "done" is treated as done here
        assert got["reopened"].status == "done"

    def test_new_ids_are_assigned_not_copied(self, svc):
        svc.import_tasks([{"id": 99, "text": "a"}, {"id": 99, "text": "b"}])
        assert sorted(t.id for t in svc.list_tasks()) == [1, 2]

    def test_nothing_usable(self, svc):
        assert svc.import_tasks([]) == (0, 0)
        assert svc.import_tasks([{"text": ""}, 3]) == (0, 2)
        assert svc.count() == 0

    def test_respects_the_task_limit(self, svc, monkeypatch):
        monkeypatch.setattr(ts, "MAX_TASKS", 3)
        svc.create_task("existing")
        imported, skipped = svc.import_tasks([{"text": f"t{i}"} for i in range(5)])
        assert (imported, skipped) == (2, 3) and svc.count() == 3

    def test_full_table_imports_nothing(self, svc, monkeypatch):
        monkeypatch.setattr(ts, "MAX_TASKS", 1)
        svc.create_task("only")
        assert svc.import_tasks([{"text": "a"}, {"text": "b"}]) == (0, 2)
        assert svc.count() == 1

    def test_accepts_a_generator(self, svc):
        assert svc.import_tasks({"text": f"g{i}"} for i in range(3)) == (3, 0)


# ---------------------------------------------------------------------------
# module-level functions
# ---------------------------------------------------------------------------
@pytest.fixture
def module_service(db_path, clock):
    service = ts.configure(db_path, clock=clock)
    yield service
    ts._default = None


class TestModuleFunctions:
    def test_configure_replaces_the_shared_service(self, module_service):
        assert ts.get_service() is module_service
        assert ts.get_service() is ts.get_service()

    def test_wrappers_round_trip(self, module_service):
        t = ts.create_task("via module", assignee="Ada", due="2026-10-30")
        assert ts.get_task(t.id).text == "via module"
        assert texts(ts.list_tasks(query="module")) == ["via module"]
        assert ts.update_task(t.id, text="renamed").text == "renamed"
        assert ts.set_status(t.id, "in_review").status == "in_review"
        assert ts.toggle_done(t.id).done is True
        assert ts.get_stats()["completed"] == 1
        assert ts.import_tasks([{"text": "imported"}]) == (1, 0)
        assert ts.clear_completed() == 1
        assert ts.delete_task(2).text == "imported"
        assert ts.clear_all() == 0

    def test_default_path_honours_the_environment(self, tmp_path, monkeypatch):
        target = tmp_path / "elsewhere" / "t.db"
        monkeypatch.setenv("TWIN_TASKS_DB", str(target))
        assert ts.default_db_path() == target
        monkeypatch.delenv("TWIN_TASKS_DB")
        assert (
            ts.default_db_path().name == "tasks.db"
            and ts.default_db_path().parent.name == "data"
        )

    def test_get_service_uses_the_environment_path(self, tmp_path, monkeypatch):
        target = tmp_path / "env" / "t.db"
        monkeypatch.setenv("TWIN_TASKS_DB", str(target))
        monkeypatch.setattr(ts, "_default", None)
        try:
            ts.create_task("env db")
            assert target.is_file()
        finally:
            ts._default = None
