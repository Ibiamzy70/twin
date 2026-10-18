"""services/dashboard_service.py (no Streamlit, no network). Run: python -m pytest tests -q"""

from __future__ import annotations

from datetime import date, datetime
from types import SimpleNamespace

import pytest

from services import dashboard_service as ds

TODAY = date(2026, 10, 18)  # a Sunday


class Task:
    """Stands in for services.task_service.Task."""

    def __init__(self, text, status="todo", created=None, completed=None):
        self.d = {
            "text": text,
            "done": status == "done",
            "status": status,
            "created": created,
            "completed_at": completed,
        }

    def to_view_dict(self):
        return dict(self.d)


class Store:
    def __init__(self, tasks=(), error=None):
        self.tasks, self.error = list(tasks), error

    def list_tasks(self):
        if self.error:
            raise self.error
        return list(self.tasks)


def session(**over):
    base = {
        "messages": [],
        "chat_count": 0,
        "user_profile": {"display_name": "Maze Ada", "role": "Builder"},
        "settings": {"compact_mode": False},
        "calc_history": [],
    }
    base.update(over)
    return base


def load(sess=None, tasks=()):
    return ds.DashboardService(Store(tasks)).load(
        sess if sess is not None else session()
    )


MESSAGES = [
    {"role": "user", "content": "hi", "timestamp": "2026-10-17T09:00:00"},
    {"role": "assistant", "content": "hello there", "created": "2026-10-17"},
    {"role": "user", "content": "chart please"},  # undated
    {
        "role": "Assistant",
        "content": "x" * 300,
        "created_at": datetime(2026, 10, 18, 8),
    },
    {"role": "tool", "content": "odd role"},
    "junk",
    {"role": "user", "content": 5},
]


# --- loading -------------------------------------------------------------------------------
def test_load_reads_session_and_tasks():
    d = load(
        session(messages=MESSAGES, chat_count=3, calc_history=[1, 2]),
        [
            Task("Write", "todo", created=datetime(2026, 10, 10, 9)),
            Task(
                "Ship",
                "done",
                created=datetime(2026, 10, 1),
                completed=datetime(2026, 10, 12),
            ),
        ],
    )
    assert (d.name, d.role, d.initials) == ("Maze Ada", "Builder", "MA")
    assert d.chats == 3 and d.calculations == 2
    assert len(d.messages) == 6  # the plain string is skipped
    assert (
        d.total_tasks,
        d.done_tasks,
        d.open_tasks,
        d.completion_pct,
        d.open_pct,
    ) == (2, 1, 1, 50, 50)
    # a done task is dated by its completion, an open one by its creation
    assert [t.when for t in d.tasks] == [date(2026, 10, 10), date(2026, 10, 12)]
    assert not d.tasks_unavailable and not d.is_empty


def test_message_counts_roles_and_ratios():
    d = load(session(messages=MESSAGES, chat_count=2))
    assert d.user_prompts == 3 and d.assistant_replies == 2 and d.other_messages == 1
    assert d.unknown_roles == ["tool"]
    assert d.messages_per_chat == 3.0 and d.replies_per_prompt == 0.7
    assert d.undated_items == 3


def test_ratios_are_none_without_a_denominator():
    d = load()
    assert d.messages_per_chat is None and d.replies_per_prompt is None
    assert (d.completion_pct, d.open_pct) == (0, 0) and d.is_empty


@pytest.mark.parametrize("bad", [None, "x", -4, 2.9, [], {}])
def test_bad_chat_counts_do_not_crash(bad):
    d = load(session(chat_count=bad))
    assert isinstance(d.chats, int) and d.chats >= 0


def test_malformed_session_values_fall_back():
    d = load(
        {"messages": "no", "user_profile": "no", "settings": 4, "calc_history": "no"}
    )
    assert (d.name, d.role, d.messages, d.calculations, d.settings) == (
        "Guest",
        "Workspace owner",
        (),
        0,
        {},
    )


def test_unreadable_task_store_degrades_to_no_tasks(caplog):
    svc = ds.DashboardService(Store(error=RuntimeError("database is locked")))
    d = svc.load(session())
    assert d.tasks == () and d.tasks_unavailable is True
    assert any("could not load tasks" in r.message for r in caplog.records)
    assert any("could not be loaded" in n for n in ds.data_notes(d))


def test_tasks_may_be_plain_dicts_and_status_strings():
    store = Store(
        [{"title": "A", "status": "Done", "created": "2026-10-01"}, {"name": "B"}, {}]
    )
    d = ds.DashboardService(store).load(session())
    assert [t.label for t in d.tasks] == ["A", "B", "Untitled task"]
    assert [t.done for t in d.tasks] == [True, False, False]


def test_default_store_is_the_shared_task_service(tmp_path):
    from services import task_service

    previous = task_service._default
    try:
        shared = task_service.configure(tmp_path / "t.db")
        shared.create_task("From the twin")
        d = ds.DashboardService().load(session())
        assert [t.label for t in d.tasks] == ["From the twin"]
        shared.create_task(
            "Another"
        )  # a later load sees it: the store is looked up each time
        assert ds.DashboardService().load(session()).total_tasks == 2
    finally:
        task_service._default = previous


def test_custom_session_keys():
    keys = ds.SessionKeys(
        messages="m", chat_count="c", profile="p", settings="s", calc_history="h"
    )
    d = ds.DashboardService(Store(), keys=keys).load(
        {"m": MESSAGES, "c": 2, "p": {"display_name": "Zed"}, "h": [1]}
    )
    assert (len(d.messages), d.chats, d.name, d.calculations) == (6, 2, "Zed", 1)


def test_module_level_helpers():
    assert ds.get_service() is ds.get_service()
    svc = ds.configure(Store([Task("x")]))
    assert ds.get_service() is svc and ds.load(session()).total_tasks == 1
    ds.configure()  # restore the default


# --- cards ---------------------------------------------------------------------------------------
def test_module_cards_have_live_figures_and_routes():
    d = load(
        session(chat_count=1, calc_history=[1, 2, 3]), [Task("a"), Task("b", "done")]
    )
    cards = {c.route_id: c for c in ds.module_cards(d)}
    assert list(cards) == ["assistant", "calculator", "tasks", "settings"]
    assert cards["assistant"].metric == "1 conversation"
    assert cards["calculator"].metric == "3 calculations"
    assert cards["tasks"].metric == "1 open"
    assert cards["settings"].metric == "Preferences"
    assert all(c.icon and c.title and c.description for c in cards.values())
    assert all(ds.CARD_ROUTES[c.card_id] == c.route_id for c in cards.values())


def test_every_card_has_a_route_and_the_table_is_read_only():
    assert set(ds.CARD_ROUTES.values()) <= {
        "assistant",
        "calculator",
        "tasks",
        "settings",
    }
    assert len(ds.CARD_ROUTES) == 15
    with pytest.raises(TypeError):
        ds.CARD_ROUTES["x"] = "tasks"  # type: ignore[index]


@pytest.mark.parametrize("n,expected", [(0, "0 chats"), (1, "1 chat"), (2, "2 chats")])
def test_count_label(n, expected):
    assert ds.count_label(n, "chat") == expected
    assert ds.count_label(2, "entry", "entries") == "2 entries"


# --- events and filters ------------------------------------------------------------------------------------
def events_data():
    return load(
        session(messages=MESSAGES, chat_count=1),
        [
            Task("Open one", "todo", created=datetime(2026, 10, 16)),
            Task(
                "Done one",
                "done",
                created=datetime(2026, 9, 1),
                completed=datetime(2026, 10, 18),
            ),
        ],
    )


def test_events_are_newest_first_with_undated_last_and_truncated_text():
    ev = ds.build_events(events_data())
    dated = [e.when for e in ev if e.when]
    assert dated == sorted(dated, reverse=True)
    assert ev[-1].when is None
    long = next(e for e in ev if e.title.endswith("…"))
    assert len(long.title) == ds.PREVIEW_CHARS and long.detail == "Assistant"
    assert {e.detail for e in ev if e.kind == "Messages"} == {
        "You",
        "Assistant",
        "Tool",
    }
    assert next(e for e in ev if e.title == "(empty message)").detail == "You"


def filters(**kw):
    base = dict(
        statuses=frozenset(ds.STATUS_OPTIONS),
        kinds=frozenset(ds.KIND_OPTIONS),
        start=None,
        end=None,
        narrowed=False,
    )
    base.update(kw)
    return ds.Filters(**base)


def test_filters_by_kind_status_and_range():
    ev = ds.build_events(events_data())
    assert len(ds.apply_filters(ev, filters())) == len(ev)
    assert {
        e.kind for e in ds.apply_filters(ev, filters(kinds=frozenset({"Tasks"})))
    } == {"Tasks"}
    only_done = ds.apply_filters(
        ev, filters(kinds=frozenset({"Tasks"}), statuses=frozenset({"Done"}))
    )
    assert [e.title for e in only_done] == ["Done one"]
    in_range = ds.apply_filters(
        ev, filters(start=date(2026, 10, 17), end=date(2026, 10, 18))
    )
    assert all(
        e.when is None or date(2026, 10, 17) <= e.when <= date(2026, 10, 18)
        for e in in_range
    )
    assert any(
        e.when is None for e in in_range
    )  # undated items are never hidden by a date range


def test_is_narrowed():
    full = (date(2026, 1, 1), TODAY)
    assert not ds.is_narrowed(
        ds.STATUS_OPTIONS, ds.KIND_OPTIONS, full[0], full[1], full
    )
    assert ds.is_narrowed(["Open"], ds.KIND_OPTIONS, full[0], full[1], full)
    assert ds.is_narrowed(ds.STATUS_OPTIONS, ["Tasks"], full[0], full[1], full)
    assert ds.is_narrowed(
        ds.STATUS_OPTIONS, ds.KIND_OPTIONS, date(2026, 5, 1), full[1], full
    )
    assert not ds.is_narrowed(ds.STATUS_OPTIONS, ds.KIND_OPTIONS, None, None, None)


def test_week_series_covers_seven_days_ending_today():
    week = ds.week_series(
        [TODAY, TODAY, date(2026, 10, 12), date(2026, 10, 11), None, date(2020, 1, 1)],
        TODAY,
    )
    assert len(week) == 7
    assert [w[0] for w in week] == list("MTWTFSS")
    assert [w[1] for w in week] == [
        1,
        0,
        0,
        0,
        0,
        0,
        2,
    ]  # Oct 12 is the oldest day shown, Oct 11 is out
    assert [w[2] for w in week] == [False] * 6 + [True]


def test_date_bounds():
    assert ds.date_bounds([], TODAY) is None
    (start, end), low, high = ds.date_bounds(
        [date(2026, 3, 1), date(2026, 10, 1)], TODAY
    )
    assert (start, end) == (date(2026, 3, 1), TODAY)
    assert low < start and high > end
    (s2, e2), low2, high2 = ds.date_bounds(
        [date(2012, 1, 1), date(2040, 1, 1)], TODAY
    )  # outside the 10 year window
    assert low2 <= s2 and high2 >= e2 == date(2040, 1, 1)


# --- report and notes --------------------------------------------------------------------------------------
def test_summary_and_usage_rows():
    d = events_data()
    summary = dict(ds.summary_rows(d))
    assert (
        summary["Tasks"] == 2
        and summary["Tasks completed"] == 1
        and summary["Completion rate"] == "50%"
    )
    assert dict(ds.summary_rows(load()))["Completion rate"] == "n/a"
    usage = dict(ds.usage_rows(d))
    assert (
        usage["Total messages"] == 6
        and usage["Your messages"]
        + usage["Assistant replies"]
        + usage["Other messages"]
        == 6
    )


def test_data_notes_cover_each_situation():
    quiet = ds.data_notes(
        load(
            session(
                messages=[{"role": "user", "content": "a", "created": "2026-10-01"}],
                chat_count=1,
            )
        )
    )
    assert len(quiet) == 3 and "Tasks are stored" in quiet[0]
    noisy = " ".join(ds.data_notes(events_data()))
    assert (
        "no timestamp" in noisy
        and "`tool`" in noisy
        and "chat counter is 0" not in noisy
    )
    assert "history may have been cleared" in " ".join(
        ds.data_notes(load(session(chat_count=4)))
    )
    assert "chat counter is 0" in " ".join(
        ds.data_notes(load(session(messages=MESSAGES)))
    )
