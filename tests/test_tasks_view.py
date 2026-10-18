"""views/tasks.py and the twin's task tools share one store (no network, no API key).

A task the twin creates through its tools must show on the Tasks page, and a change made on
the page must be visible to the twin. Run: python -m pytest tests -q
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from digital_twin import tools as T  # noqa: E402
from services import calculator_service, task_service  # noqa: E402

NOW = datetime(2026, 10, 18, 9, 0)


def script() -> None:
    from views import tasks

    tasks.render()


@pytest.fixture()
def store(tmp_path):
    """The shared service, pointed at a throwaway database for this test."""
    previous = task_service._default
    svc = task_service.configure(tmp_path / "tasks.db", clock=lambda: NOW)
    yield svc
    task_service._default = previous


@pytest.fixture()
def twin_tools(store):
    # No ``todos=`` argument: the tools must find the same shared service the page uses.
    return T.build_default_registry(calculator=calculator_service)


def page() -> AppTest:
    return AppTest.from_function(script, default_timeout=30)


def page_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.markdown) + " ".join(
        c.label for c in at.checkbox
    )


def test_tools_and_page_use_the_same_service(store, twin_tools):
    assert twin_tools.execute("create_task", {"text": "Ping from the twin"}).ok
    assert [t.text for t in task_service.get_service().list_tasks()] == [
        "Ping from the twin"
    ]


def test_task_created_by_the_twin_shows_on_the_page(store, twin_tools):
    at = page().run()
    assert not at.exception
    assert "Ping from the twin" not in page_text(at)

    result = twin_tools.execute(
        "create_task",
        {
            "text": "Ping from the twin",
            "assignee": "Maze",
            "due": "2026-10-20",
            "status": "in_progress",
        },
    )
    assert result.ok

    at = page().run()  # what opening or refreshing the Tasks page does
    assert not at.exception
    text = page_text(at)
    assert "Ping from the twin" in text and "Maze" in text and "20/10/2026" in text
    assert "In Progress" in text


def test_a_running_page_picks_the_task_up_on_its_next_rerun(store, twin_tools):
    at = page().run()
    twin_tools.execute("create_task", {"text": "Added while the page was open"})
    at.run()
    assert "Added while the page was open" in page_text(at)


def test_ticking_a_task_on_the_page_is_visible_to_the_twin(store, twin_tools):
    task_id = twin_tools.execute("create_task", {"text": "Tick me"}).data["id"]
    at = page().run()
    at.checkbox(key=f"chk_{task_id}").check().run()
    assert not at.exception
    assert task_service.get_service().get_task(task_id).status == "done"
    listed = twin_tools.execute("get_task", {"id": task_id})
    assert listed.ok and listed.data["status"] == "done"


def test_adding_on_the_page_is_visible_to_the_twin(store, twin_tools):
    at = page().run()
    at.button(key="tasks_btn_add").click().run()
    # Inside a form the typed value only reaches the app together with the submit click.
    at.text_input(key="new_todo").input("Typed on the page")
    next(b for b in at.button if b.label == "Add task").click().run()
    assert not at.exception
    out = twin_tools.execute("list_tasks", {})
    assert out.ok and "Typed on the page" in str(out.data)


def test_empty_store_shows_the_empty_state(store):
    at = page().run()
    assert not at.exception
    assert "No tasks yet" in page_text(at)


def test_storage_failure_shows_a_friendly_error(store, monkeypatch):
    def boom(*a, **k):
        raise task_service.TaskStorageError("locked")

    monkeypatch.setattr(task_service.TaskService, "list_tasks", boom)
    at = page().run()
    assert not at.exception
    assert any("can't be loaded" in e.value for e in at.error)
