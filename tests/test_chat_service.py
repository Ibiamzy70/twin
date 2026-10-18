"""Tests for services/chat_service.py (no Streamlit, no network). Run: python -m pytest tests -q"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest

from services import chat_service as cs

REPLY = {
    "user": "Ada",
    "current_context": {"summary": "s", "observed_signals": ["a"]},
    "predicted_next_action": [],
    "missing_information": [],
    "response_quality": {"grounded_in_data": True, "contains_guessing": False},
}


@dataclass
class FakeReply:  # same attribute names as twin_adapter.TwinReply
    ok: bool = True
    status: str = "ok"
    text: str = "readable"
    reply: Dict[str, Any] = field(default_factory=lambda: dict(REPLY))
    message: str = ""
    notices: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    retry_after: float = 0.0


class Recorder:
    def __init__(self, reply=None, exc=None):
        self.reply, self.exc, self.calls = reply or FakeReply(), exc, []

    def __call__(self, prompt, history, *, identifier):
        self.calls.append((prompt, list(history), identifier))
        if self.exc:
            raise self.exc
        return self.reply


def test_success_content_is_the_markdown_and_records():
    rec = Recorder(FakeReply(notices=["redacted"]))
    r = cs.ChatService(rec).send_message("  hello  ", [], session_id="s1")
    assert r.ok and r.level == "ok" and r.prompt == "hello"
    assert (
        r.content == "readable"
        and r.markdown == "readable"
        and r.reply["user"] == "Ada"
    )
    u, a = r.records()
    assert u["include_in_context"] and a["include_in_context"] and not a["is_error"]
    assert a["notices"] == ["redacted"]
    assert rec.calls == [("hello", [], "s1")]


def test_current_prompt_is_never_in_its_own_history():
    transcript = []
    rec = Recorder()
    svc = cs.ChatService(rec)
    for text in ("one", "two"):
        cs.append_exchange(transcript, svc.send_message(text, transcript))
    assert rec.calls[1][1] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": rec.calls[1][1][1]["content"]},
    ]
    assert all(h["content"] != "two" for h in rec.calls[1][1])
    assert len(transcript) == 4  # each exchange stored exactly once


@pytest.mark.parametrize(
    "status,level",
    [
        ("blocked_injection", "warning"),
        ("rate_limited", "warning"),
        ("too_long", "warning"),
        ("blocked_output", "error"),
        ("upstream_busy", "error"),
        ("error", "error"),
    ],
)
def test_failed_exchanges_are_shown_but_not_reused(status, level):
    transcript = []
    bad = FakeReply(ok=False, status=status, message="nope", retry_after=3.0)
    r = cs.ChatService(Recorder(bad)).send_message("bad", transcript)
    assert (
        not r.ok and r.level == level and r.content == "nope" and r.retry_after == 3.0
    )
    cs.append_exchange(transcript, r)
    assert [m["include_in_context"] for m in transcript] == [False, False]
    assert transcript[1]["is_error"] is True
    assert cs.build_context(transcript) == []


def test_empty_prompt_does_not_call_agent_or_store():
    rec = Recorder()
    transcript = []
    for empty in ("", "   ", None, 5):
        r = cs.ChatService(rec).send_message(empty, transcript)
        assert r.status == "invalid_input" and not r.should_store
        cs.append_exchange(transcript, r)
    assert rec.calls == [] and transcript == []


def test_exceptions_become_friendly_error():
    r = cs.ChatService(
        Recorder(exc=RuntimeError("GROQ_API_KEY is not set."))
    ).send_message("hi", [])
    assert (r.ok, r.status, r.level) == (False, "error", "error")
    assert "GROQ" not in r.content and r.content == cs.MESSAGES["error"]


def test_malformed_adapter_reply_is_an_error():
    r = cs.ChatService(Recorder(reply=FakeReply(reply={}, text="  "))).send_message(
        "hi", []
    )
    assert r.status == "error"


def test_build_context_filters_and_trims():
    t = [
        {"role": "assistant", "content": "orphan"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "failed", "include_in_context": False},
        {"role": "assistant", "content": "err", "is_error": True},
        {"role": "system", "content": "no"},
        {"role": "user", "content": 5},
        "junk",
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]
    before = [dict(x) if isinstance(x, dict) else x for x in t]
    assert [h["content"] for h in cs.build_context(t)] == ["q1", "a1", "q2", "a2"]
    assert t == before
    long = [
        {"role": r, "content": f"{r}{i}"}
        for i in range(20)
        for r in ("user", "assistant")
    ]
    ctx = cs.build_context(long, turns=2)
    assert len(ctx) == 4 and ctx[0]["role"] == "user"
    assert cs.build_context(long, turns=0) == []


def test_append_exchange_caps_transcript():
    transcript = []
    ok = cs.ChatService(Recorder())
    for i in range(10):
        cs.append_exchange(
            transcript, ok.send_message(f"m{i}", transcript), max_messages=7
        )
    assert len(transcript) <= 7 and transcript[0]["role"] == "user"


def test_session_identifier_is_stable_and_unique():
    a, b = {}, {}
    assert cs.session_identifier(a) == cs.session_identifier(a)
    assert cs.session_identifier(a) != cs.session_identifier(b)


def test_lazy_adapter_loading(monkeypatch):
    fake = types.ModuleType("digital_twin.twin_adapter")
    fake.send_message = lambda prompt, history, identifier="default": FakeReply()
    pkg = types.ModuleType("digital_twin")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "digital_twin", pkg)
    monkeypatch.setitem(sys.modules, "digital_twin.twin_adapter", fake)
    monkeypatch.setattr(cs, "_adapter_sender", None)
    assert cs.ChatService().send_message("hi", []).ok


def test_missing_adapter_is_a_friendly_error(monkeypatch):
    monkeypatch.setattr(cs, "_adapter_sender", None)
    monkeypatch.setattr(cs, "_ADAPTER_MODULES", ("no_such_pkg_xyz.twin_adapter",))
    r = cs.ChatService().send_message("hi", [])
    assert r.status == "error"
