"""GroqClient tests. Everything is offline except the one `integration` test."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

import groq
import httpx
import pytest

from digital_twin import config, groq_client
from digital_twin.groq_client import GroqClient


# --------------------------------------------------------------------- helpers
def _request() -> httpx.Request:
    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


def _status_error(status: int, headers: dict | None = None) -> groq.APIStatusError:
    response = httpx.Response(status, request=_request(), headers=headers or {})
    cls = {429: groq.RateLimitError, 400: groq.BadRequestError}.get(
        status, groq.InternalServerError if status >= 500 else groq.APIStatusError
    )
    return cls("boom", response=response, body=None)


def _reply(content="hi", tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class FakeCompletions:
    """Replays a script of replies/exceptions and records every request."""

    def __init__(self, script):
        self.script, self.calls = list(script), []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def make_client(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(groq_client.time, "sleep", sleeps.append)

    def build(script, **kwargs):
        client = GroqClient(api_key="test-key", **kwargs)
        client.client.chat = SimpleNamespace(completions=FakeCompletions(script))
        client.sleeps = sleeps
        client.calls = client.client.chat.completions.calls
        return client

    return build


USER = [{"role": "user", "content": "ping"}]


# ------------------------------------------------------------ construction/key
def test_initialization_defaults():
    client = GroqClient(api_key="k")
    assert client.model
    assert isinstance(client.max_tokens, int) and client.max_tokens > 0
    assert (
        client.client.max_retries == 0
    ), "SDK retries must be off; GroqClient owns retrying"


@pytest.mark.parametrize("key", ["", "   "])
def test_explicit_blank_key_is_rejected(key):
    with pytest.raises(ValueError, match="API key"):
        GroqClient(api_key=key)


def test_missing_key_everywhere_raises(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    with pytest.raises(ValueError, match="API key"):
        GroqClient()


def test_key_is_read_at_construction_not_import(monkeypatch):
    monkeypatch.setattr(config, "GROQ_API_KEY", "")
    monkeypatch.setenv("GROQ_API_KEY", "late-key")
    assert GroqClient().client.api_key == "late-key"


def test_config_key_is_used_when_env_is_empty(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(config, "GROQ_API_KEY", "cfg-key")
    assert GroqClient().client.api_key == "cfg-key"


def test_negative_max_retries_rejected():
    with pytest.raises(ValueError):
        GroqClient(api_key="k", max_retries=-1)


# ------------------------------------------------------------------ validation
@pytest.mark.parametrize(
    "bad, exc",
    [
        ([], ValueError),
        (None, TypeError),
        ("hello", TypeError),
        (["hello"], TypeError),
        ([{"content": "x"}], ValueError),
        ([{"role": "robot", "content": "x"}], ValueError),
        ([{"role": "user"}], ValueError),
        ([{"role": "user", "content": None}], ValueError),
        ([{"role": "user", "content": 5}], TypeError),
    ],
)
def test_invalid_messages_fail_locally_without_a_request(make_client, bad, exc):
    client = make_client([_reply()])
    with pytest.raises(exc):
        client.chat(bad)
    assert client.calls == [], "validation must happen before any network call"


def test_assistant_tool_call_message_without_content_is_valid(make_client):
    client = make_client([_reply()])
    history = USER + [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "1", "type": "function"}],
        },
        {"role": "tool", "tool_call_id": "1", "content": "{}"},
    ]
    assert client.chat(history) == "hi"


# ------------------------------------------------------------------- happy path
def test_chat_returns_text_and_passes_settings(make_client):
    client = make_client([_reply("hello")])
    assert client.chat(USER, temperature=0.1) == "hello"
    call = client.calls[0]
    assert call["model"] == client.model
    assert call["max_tokens"] == client.max_tokens
    assert call["temperature"] == 0.1


def test_chat_without_choices_or_content_raises(make_client):
    with pytest.raises(RuntimeError, match="no completion choices"):
        make_client([SimpleNamespace(choices=[])]).chat(USER)
    with pytest.raises(RuntimeError, match="no assistant message content"):
        make_client([_reply(None)]).chat(USER)


def test_chat_with_tools_sends_tools_and_returns_message(make_client):
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    client = make_client([_reply(None, tool_calls=["call"])])
    message = client.chat_with_tools(USER, tools=tools)
    assert message.tool_calls == ["call"]
    assert client.calls[0]["tools"] == tools
    assert client.calls[0]["tool_choice"] == "auto"


def test_chat_with_tools_omits_tool_args_when_none_or_empty(make_client):
    for tools in (None, []):
        client = make_client([_reply("text")])
        assert client.chat_with_tools(USER, tools=tools).content == "text"
        assert "tools" not in client.calls[0] and "tool_choice" not in client.calls[0]


# ---------------------------------------------------------------------- retries
@pytest.mark.parametrize("status", [429, 500, 503])
def test_retries_transient_status_then_succeeds(make_client, status):
    client = make_client([_status_error(status), _status_error(status), _reply("ok")])
    assert client.chat(USER) == "ok"
    assert len(client.calls) == 3 and len(client.sleeps) == 2


def test_retries_connection_and_timeout_errors(make_client):
    client = make_client(
        [
            groq.APIConnectionError(request=_request()),
            groq.APITimeoutError(request=_request()),
            _reply("ok"),
        ]
    )
    assert client.chat(USER) == "ok"
    assert len(client.calls) == 3


def test_does_not_retry_client_errors(make_client):
    client = make_client([_status_error(400), _reply()])
    with pytest.raises(groq.BadRequestError):
        client.chat(USER)
    assert len(client.calls) == 1 and client.sleeps == []


def test_gives_up_after_max_retries_and_reraises(make_client):
    client = make_client([_status_error(503)] * 3, max_retries=2)
    with pytest.raises(groq.InternalServerError):
        client.chat(USER)
    assert len(client.calls) == 3  # first try + 2 retries


def test_zero_retries_means_single_attempt(make_client):
    client = make_client([_status_error(503)], max_retries=0)
    with pytest.raises(groq.InternalServerError):
        client.chat(USER)
    assert len(client.calls) == 1


def test_backoff_grows_has_jitter_and_is_capped(make_client):
    client = make_client([_status_error(503)] * 5 + [_reply()], max_retries=5)
    client.chat(USER)
    for attempt, delay in enumerate(client.sleeps):
        base = min(2**attempt, groq_client.MAX_BACKOFF)
        assert 0.5 * base <= delay <= base


def test_retry_after_header_is_honoured_and_capped(make_client):
    client = make_client(
        [
            _status_error(429, {"retry-after": "7"}),
            _status_error(429, {"retry-after": "9999"}),
            _reply(),
        ]
    )
    client.chat(USER)
    assert client.sleeps == [7.0, groq_client.MAX_BACKOFF]


# ------------------------------------------------------- end to end via tools.py
def _tool_call(call_id, name, args):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=json.dumps(args)),
    )


def test_tool_loop_end_to_end_creates_a_real_task(make_client, tmp_path):
    """GroqClient -> make_generate -> run_tool_loop -> registry -> SQLite, with a scripted model."""
    from digital_twin import tools
    from services import task_service

    svc = task_service.TaskService(tmp_path / "tasks.db")
    registry = tools.build_default_registry(todos=svc)
    client = make_client(
        [
            _reply(
                None,
                [
                    _tool_call(
                        "c1", "create_task", {"text": "Email Ada", "assignee": "Ada"}
                    )
                ],
            ),
            _reply("Done - I added 'Email Ada'."),
        ]
    )
    runs = []
    generate = tools.make_generate(
        registry=registry,
        client=client,
        system_prompt="You are a twin.",
        policy=tools.ToolPolicy.from_env({"TWIN_TOOLS_MODE": "full"}),
        on_run=runs.append,
    )

    answer = generate("add a task to email Ada", [])

    assert answer == "Done - I added 'Email Ada'."
    tasks = svc.list_tasks()
    assert [(t.text, t.assignee) for t in tasks] == [("Email Ada", "Ada")]
    assert runs and runs[0].calls[0].name == "create_task"
    # the 2nd request carried the tool result back to the model
    second = client.calls[1]["messages"]
    assert second[-1]["role"] == "tool" and second[-1]["tool_call_id"] == "c1"
    assert "tools" in client.calls[0]


# --------------------------------------------------------------------- live API
@pytest.mark.integration
def test_live_chat_returns_text():
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set; skipping the live Groq call.")
    reply = GroqClient().chat(
        [{"role": "user", "content": "Respond with the single word: OK"}]
    )
    assert isinstance(reply, str) and reply.strip()


@pytest.mark.integration
def test_live_tool_call_round_trip(tmp_path):
    """One real model call that must choose the create_task tool and write to SQLite."""
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set; skipping the live Groq call.")
    from digital_twin import tools
    from services import task_service

    svc = task_service.TaskService(tmp_path / "tasks.db")
    generate = tools.make_generate(
        registry=tools.build_default_registry(todos=svc),
        client=GroqClient(),
        system_prompt="You are a helpful assistant.",
        policy=tools.ToolPolicy.from_env({"TWIN_TOOLS_MODE": "full"}),
    )
    answer = generate("Please add a task: buy printer paper", [])
    assert answer.strip()
    assert any("printer paper" in t.text.lower() for t in svc.list_tasks())
