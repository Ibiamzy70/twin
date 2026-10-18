"""
Digital Twin Assistant — Streamlit view.

Responsibilities:
- Render the assistant page and conversation history.
- Use Streamlit's native bottom-positioned chat input.
- Send messages through services.chat_service (safety pipeline + agent).
- Display structured JSON responses and ordinary text responses.
- Handle loading states and agent errors without crashing the application.

Run from the project root:
    python -m streamlit run app.py
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

import state
from services import chat_service

# Same key the dashboard and Settings ("Clear chat") read, so counts stay in sync.
MESSAGES_KEY = state.MESSAGES
# Single source of truth: matches SafetyConfig.max_input_chars.
MAX_MESSAGE_LENGTH = chat_service.MAX_MESSAGE_CHARS

# Problems the user can fix (shown as warnings); anything else is a system error.
_WARNING_STATUSES = frozenset(
    {
        "invalid_input",
        "too_long",
        "rate_limited",
        "blocked_injection",
        "blocked_content",
        "blocked_crisis",
    }
)


def _initialize_messages() -> None:
    """Initialize this view's conversation state without resetting it."""
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []


def _render_content(content: str) -> None:
    """Render JSON as structured data and other responses as Markdown."""
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        st.markdown(content)
        return

    if isinstance(parsed, (dict, list)):
        st.json(parsed)
    else:
        st.markdown(content)


def _render_notices(notices: Any) -> None:
    for note in notices or []:
        st.caption(str(note))


def _render_failure(content: str, status: str | None) -> None:
    """User-fixable problems (too long, blocked, rate limited) are warnings; the rest errors."""
    if status in _WARNING_STATUSES:
        st.warning(content)
    else:
        st.error(content)


def _render_history() -> None:
    """Render persisted messages in chronological order."""
    for message in st.session_state[MESSAGES_KEY]:
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue  # ignore malformed records instead of crashing the page

        with st.chat_message(role):
            if message.get("is_error"):
                _render_failure(content, message.get("status"))
            elif role == "user":
                st.markdown(content)
            else:
                _render_content(content)
            if role == "assistant":
                _render_notices(message.get("notices"))


def _process_prompt(prompt: str) -> None:
    """Send one user message through the safety pipeline via the chat service."""
    messages = st.session_state[MESSAGES_KEY]

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Your digital twin is thinking..."):
            result = chat_service.send_message(
                prompt,
                messages,
                session_id=chat_service.session_identifier(st.session_state),
            )

        if result.is_error:
            _render_failure(result.content, result.status)
        else:
            _render_content(result.content)
        _render_notices(result.notices)

    chat_service.append_exchange(messages, result)


def render() -> None:
    """
    Public view entry point called by app.py:

        "assistant": assistant.render
    """
    _initialize_messages()

    st.title("Digital Twin Assistant")
    st.caption(
        "An AI assistant powered by your digital twin's profile, "
        "stored context, and existing agent."
    )

    # Native Streamlit chat input stays at the bottom of the page.
    hint = st.empty()
    if not st.session_state[MESSAGES_KEY]:
        hint.info(
            "Start a conversation with your digital twin. "
            "Ask about your current context, priorities, or next steps."
        )

    _render_history()

    prompt = st.chat_input(
        "Message your digital twin...",
        max_chars=MAX_MESSAGE_LENGTH,
        key="assistant_chat_input",
    )

    if prompt is not None:
        prompt = prompt.strip()

        if not prompt:
            st.warning("Enter a message before sending.")
        else:
            hint.empty()  # the welcome note should not sit above the first exchange
            _process_prompt(prompt)
