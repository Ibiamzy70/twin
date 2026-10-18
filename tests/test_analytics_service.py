"""Stage 3a: services/analytics_service.py (no Streamlit, no network)."""

from __future__ import annotations

import copy
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from services import analytics_service as A

HISTORY = [
    {"time": "2026-08-10", "activity": "reviewed pull requests for the backend team"},
    {
        "time": "2026-08-11",
        "activity": "practiced a technical presentation with the team",
    },
    {
        "time": "2026-08-12",
        "activity": "created a presentation comparing cloud providers",
    },
    {"time": "2026-08-13", "activity": "wrote unit tests for the payment module"},
    {"time": "2026-08-18", "activity": "debugged a memory leak in the worker service"},
    {
        "time": "2026-08-18",
        "activity": "prepared talking points for a stakeholder demo",
    },
    {"time": "2026-08-25", "activity": "went for a 45 minute run before breakfast"},
    {"time": "not a date", "activity": "undated thing"},
    "junk",
    {"time": "2026-08-26"},
]


def task(status="todo", due=None):
    return SimpleNamespace(status=status, due=due)


class FakeTasks:
    def __init__(self, tasks):
        self._tasks = tasks

    def list_tasks(self):
        return list(self._tasks)

    def now(self):
        return datetime(2026, 10, 14, 9, 0)  # a Wednesday


@pytest.fixture()
def svc():
    return A.AnalyticsService(history=HISTORY, tasks=FakeTasks([]))


def values(chart):
    return dict(zip(chart["labels"], chart["series"][0]["values"]))


# --- categories -------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,category",
    [
        ("reviewed pull requests for the backend team", "Code review"),
        ("practiced a technical presentation with the team", "Presentations"),
        ("created slides on edge computing", "Presentations"),
        ("prepared talking points for a stakeholder demo", "Presentations"),
        ("debugged a memory leak in the worker service", "Debugging"),
        ("wrote unit tests for the payment module", "Testing"),
        ("read a research paper on vector databases", "Research and reading"),
        ("went for a 45 minute run before breakfast", "Other"),
        ("", "Other"),
    ],
)
def test_categorize(text, category):
    assert A.categorize(text) == category


# --- activity charts -----------------------------------------------------------------------------
def test_per_day_fills_empty_days_and_counts(svc):
    chart = svc.build_chart(
        "activity_per_day", date_from="2026-08-10", date_to="2026-08-19"
    )
    assert (
        chart["labels"][0] == "Aug 10"
        and chart["labels"][-1] == "Aug 19"
        and len(chart["labels"]) == 10
    )
    v = values(chart)
    assert v["Aug 10"] == 1 and v["Aug 14"] == 0 and v["Aug 18"] == 2
    assert (
        chart["total"] == 6
        and chart["kind"] == "bar"
        and chart["source"] == "activity_per_day"
    )
    assert (
        chart["subtitle"] == "Aug 10 to Aug 19, 2026"
        and "Busiest day: Aug 18 (2)" in chart["summary"]
    )


def test_undated_and_invalid_entries_are_skipped(svc):
    assert svc.build_chart("activity_per_week")["total"] == 7  # 7 dated entries


def test_per_day_refuses_long_ranges(svc):
    with pytest.raises(A.AnalyticsError) as e:
        svc.build_chart(
            "activity_per_day", date_from="2026-06-01", date_to="2026-08-30"
        )
    assert e.value.code == "too_many_points" and "activity_per_week" in e.value.message


def test_per_week_uses_monday_buckets(svc):
    chart = svc.build_chart("activity_per_week", kind="line")
    assert chart["kind"] == "line" and chart["labels"] == ["Aug 10", "Aug 17", "Aug 24"]
    assert (
        chart["series"][0]["values"] == [4, 2, 1]
        and chart["x_label"] == "Week starting"
    )
    assert "Busiest week starts Aug 10 (4)" in chart["summary"]


def test_per_week_date_filter(svc):
    chart = svc.build_chart("activity_per_week", date_from="2026-08-17")
    assert chart["labels"] == ["Aug 17", "Aug 24"] and chart["total"] == 3


def test_by_category_sorted_with_other_last(svc):
    chart = svc.build_chart("activity_by_category")
    v = chart["series"][0]["values"]
    assert (
        chart["labels"][0] == "Presentations" and v[0] == 3
    )  # 2 presentations + the demo
    assert chart["labels"][-1] == "Other" and v == sorted(v[:-1], reverse=True) + [
        v[-1]
    ]
    assert (
        chart["total"] == 8
    )  # includes the undated entry: categories do not need a date unless a range is given


def test_by_category_with_a_range_uses_only_dated_entries(svc):
    chart = svc.build_chart(
        "activity_by_category", date_from="2026-08-10", date_to="2026-08-13"
    )
    assert chart["total"] == 4 and "Other" not in chart["labels"]


def test_no_data_errors():
    empty = A.AnalyticsService(history=[], tasks=FakeTasks([]))
    for source in (
        "activity_per_day",
        "activity_per_week",
        "activity_by_category",
        "tasks_by_status",
        "tasks_by_due",
    ):
        with pytest.raises(A.AnalyticsError) as e:
            empty.build_chart(source)
        assert e.value.code == "no_data"
    with pytest.raises(A.AnalyticsError) as e:
        A.AnalyticsService(history=HISTORY).build_chart(
            "activity_per_week", date_from="2030-01-01"
        )
    assert e.value.code == "no_data"


def test_history_can_be_a_function():
    assert (
        A.AnalyticsService(history=lambda: HISTORY).build_chart("activity_per_week")[
            "total"
        ]
        == 7
    )


# --- task charts -----------------------------------------------------------------------------------
def test_tasks_by_status():
    tasks = [
        task("todo"),
        task("todo"),
        task("in_progress"),
        task("done"),
        task("weird_status"),
    ]
    chart = A.AnalyticsService(tasks=FakeTasks(tasks)).build_chart("tasks_by_status")
    v = values(chart)
    assert (
        v["To do"] == 2
        and v["In progress"] == 1
        and v["In review"] == 0
        and v["Done"] == 1
        and v["Weird status"] == 1
    )
    assert chart["total"] == 5 and "1 done, 4 still open" in chart["summary"]


def test_tasks_by_due_buckets():
    tasks = [
        task(due=date(2026, 10, 13)),  # yesterday: overdue
        task(due=date(2026, 10, 14)),  # today: this week
        task(due=date(2026, 10, 18)),  # Sunday: this week
        task(due=date(2026, 10, 19)),  # next Monday
        task(due=date(2026, 11, 2)),  # 3 weeks on
        task(due=date(2026, 12, 25)),  # later
        task(),  # no date
        task("done", due=date(2026, 10, 1)),  # finished: not counted
    ]
    chart = A.AnalyticsService(tasks=FakeTasks(tasks)).build_chart("tasks_by_due")
    assert values(chart) == {
        "Overdue": 1,
        "This week": 2,
        "Next week": 1,
        "In 2 weeks": 0,
        "In 3 weeks": 1,
        "Later": 1,
        "No due date": 1,
    }
    assert chart["total"] == 7 and "1 overdue" in chart["summary"]


# --- arguments ----------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "args",
    [
        {"source": "salaries"},
        {"source": None},
        {"source": "activity_per_week", "kind": "pie"},
        {"source": "activity_by_category", "kind": "line"},
        {"source": "tasks_by_status", "kind": "line"},
        {"source": "activity_per_week", "date_from": "yesterday"},
        {
            "source": "activity_per_week",
            "date_from": "2026-09-01",
            "date_to": "2026-08-01",
        },
        {"source": "tasks_by_status", "date_from": "2026-08-01"},
    ],
)
def test_bad_arguments_raise_invalid_arguments(svc, args):
    with pytest.raises(A.AnalyticsError) as e:
        svc.build_chart(**args)
    assert e.value.code == "invalid_arguments"


def test_every_built_chart_is_json_safe_and_valid(svc):
    for source in A.SOURCES:
        chart = (
            svc.build_chart(source)
            if source.startswith("activity")
            else A.AnalyticsService(
                tasks=FakeTasks([task(), task("done")])
            ).build_chart(source)
        )
        assert A.validate_chart(json.loads(json.dumps(chart))) == chart


# --- validate_chart --------------------------------------------------------------------------------------
GOOD = {
    "type": "chart",
    "kind": "bar",
    "title": "T",
    "labels": ["a", "b"],
    "series": [{"name": "n", "values": [1, 2.5]}],
}


def test_validate_chart_accepts_and_cleans():
    cleaned = A.validate_chart({**GOOD, "evil": "<script>", "title": " T​\x00 "})
    assert (
        "evil" not in cleaned
        and cleaned["title"] == "T"
        and set(cleaned) == set(A._KEYS)
    )
    assert GOOD["title"] == "T"  # input untouched


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.update(type="table"),
        lambda c: c.update(kind="pie"),
        lambda c: c.update(title=""),
        lambda c: c.update(title="x" * 81),
        lambda c: c.update(labels=[]),
        lambda c: c.update(
            labels=[str(i) for i in range(32)],
            series=[{"name": "n", "values": [1] * 32}],
        ),
        lambda c: c.update(labels=["a", "a"]),
        lambda c: c.update(labels=["a", ""]),
        lambda c: c.update(labels=["x" * 41, "b"]),
        lambda c: c.update(labels="ab"),
        lambda c: c.update(series=[]),
        lambda c: c.update(
            series=[{"name": str(i), "values": [1, 2]} for i in range(4)]
        ),
        lambda c: c.update(series=[{"name": "n", "values": [1]}]),
        lambda c: c.update(series=[{"name": "n", "values": [1, math.nan]}]),
        lambda c: c.update(series=[{"name": "n", "values": [1, math.inf]}]),
        lambda c: c.update(series=[{"name": "n", "values": [True, 2]}]),
        lambda c: c.update(series=[{"name": "n", "values": ["1", 2]}]),
        lambda c: c.update(series=[{"name": "n", "values": [1, 10**12]}]),
        lambda c: c.update(
            series=[{"name": "n", "values": [1, 2]}, {"name": "n", "values": [3, 4]}]
        ),
        lambda c: c.update(total=math.nan),
        lambda c: c.update(summary="x" * 241),
    ],
)
def test_validate_chart_rejects(mutate):
    chart = copy.deepcopy(GOOD)
    mutate(chart)
    with pytest.raises(A.AnalyticsError) as e:
        A.validate_chart(chart)
    assert e.value.code == "invalid_chart"


@pytest.mark.parametrize("value", [None, "chart", 5, [], {"kind": "bar"}])
def test_validate_chart_rejects_non_charts(value):
    with pytest.raises(A.AnalyticsError):
        A.validate_chart(value)


# --- default data loading ---------------------------------------------------------------------------------
def test_default_history_comes_from_the_twin_data(monkeypatch):
    monkeypatch.setattr(A, "_load_twin_data", lambda: {"activity_history": HISTORY})
    assert (
        A.AnalyticsService(tasks=FakeTasks([])).build_chart("activity_per_week")[
            "total"
        ]
        == 7
    )


def test_data_file_fallback_when_config_cannot_load(monkeypatch, tmp_path):
    data = tmp_path / "twin_data.json"
    data.write_text(json.dumps({"activity_history": HISTORY}), encoding="utf-8")
    monkeypatch.setattr(A, "DATA_FILE", data)
    monkeypatch.setitem(
        sys.modules, "digital_twin.config", None
    )  # makes the import fail
    assert A._load_twin_data()["activity_history"] == HISTORY


def test_module_level_helpers_exist():
    assert A.get_service() is A.get_service() and callable(A.build_chart)
    assert {d["source"] for d in A.AnalyticsService().describe_sources()} == set(
        A.SOURCES
    )
