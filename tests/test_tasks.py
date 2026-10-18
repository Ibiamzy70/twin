"""The Tasks page and the assistant's tools must share ONE store (services.task_service)."""

from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from services import task_service

SCRIPT = "import views.tasks as t\nt.render()\n"


@pytest.fixture
def svc(tmp_path):
    return task_service.configure(tmp_path / "tasks.db")


def run() -> AppTest:
    at = AppTest.from_string(SCRIPT, default_timeout=30)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_task_created_elsewhere_appears_on_the_page(svc):
    svc.create_task("Created by the assistant", assignee="Ada")
    at = AppTest.from_string(SCRIPT, default_timeout=30)
    at.session_state["tasks_prefs"] = {"tab": "List"}
    at.run()
    assert not at.exception
    assert any("Created by the assistant" in c.label for c in at.checkbox)
    assert "Ada" in " ".join(m.value for m in at.markdown)


def test_adding_on_the_page_writes_to_the_service(svc):
    at = AppTest.from_string(
        "import streamlit as st, views.tasks as t\n"
        "st.session_state.setdefault('new_todo', 'From the page')\n"
        "st.session_state.setdefault('new_status', 'in_progress')\n"
        "st.button('go', on_click=t.add_todo)\n"
        "t.render()\n",
        default_timeout=30,
    ).run()
    at.button[0].click().run()
    assert not at.exception
    assert [(x.text, x.status) for x in svc.list_tasks()] == [
        ("From the page", "in_progress")
    ]


def test_toggle_move_delete_hit_the_service(svc):
    import views.tasks as t

    a = svc.create_task("A")
    b = svc.create_task("B")
    c = svc.create_task("C", status="in_review")

    at = AppTest.from_string(
        "import streamlit as st, views.tasks as t\n"
        "st.session_state['chk_%d'] = True\n"
        "t.toggle_todo(%d)\n"
        "t.move_todo(%d, 1)\n"
        "t.delete_todo(%d)\n" % (a.id, a.id, b.id, c.id),
        default_timeout=30,
    ).run()
    assert not at.exception
    assert svc.get_task(a.id).status == "done"
    assert svc.get_task(b.id).status == "in_progress"
    assert [x.id for x in svc.list_tasks()] == [b.id, a.id]


def test_untick_restores_previous_status(svc):
    a = svc.create_task("A", status="in_review")
    at = AppTest.from_string(
        "import streamlit as st, views.tasks as t\n"
        "st.session_state['chk_%d'] = True\nt.toggle_todo(%d)\n"
        "st.session_state['chk_%d'] = False\nt.toggle_todo(%d)\n" % ((a.id,) * 4),
        default_timeout=30,
    ).run()
    assert not at.exception
    assert svc.get_task(a.id).status == "in_review"


def test_clear_completed_and_clear_all(svc):
    svc.create_task("open")
    d = svc.create_task("done", status="done")
    AppTest.from_string("import views.tasks as t\nt.clear_completed()\n").run()
    assert [x.text for x in svc.list_tasks()] == ["open"]
    AppTest.from_string("import views.tasks as t\nt.clear_all()\n").run()
    assert svc.list_tasks() == []


def test_stale_id_is_a_message_not_a_crash(svc):
    at = AppTest.from_string(
        "import views.tasks as t\nt.delete_todo(999)\nt.set_status(999,'done')\nt.move_todo(999,1)\n"
    ).run()
    assert not at.exception


def test_all_tabs_render_with_data(svc):
    svc.create_task("One", due="2026-10-18")
    svc.create_task("Two", status="done", assignee="Bo")
    for tab in ("Overview", "List", "Board", "Timeline"):
        at = AppTest.from_string(SCRIPT, default_timeout=30)
        at.session_state["tasks_prefs"] = {"tab": tab}
        at.run()
        assert not at.exception, (tab, [e.value for e in at.exception])


def test_storage_failure_shows_friendly_error(svc, monkeypatch):
    def boom(*a, **k):
        raise task_service.TaskStorageError("db locked")

    monkeypatch.setattr(task_service.TaskService, "list_tasks", boom)
    at = AppTest.from_string(SCRIPT, default_timeout=30).run()
    assert not at.exception
    assert any("can't be loaded" in e.value for e in at.error)
