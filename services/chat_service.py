"""Chat service: the application-facing interface of the assistant.

Responsibilities
----------------
* ``views/assistant.py``  draws the chat and handles clicks. It calls this module only.
* ``chat_service.py``     (this file) validates the prompt, builds the model context from the
                          stored transcript, calls the twin through the adapter, turns the
                          outcome into one ``ChatResult``, and records the exchange.
* ``twin_adapter.py``     stable interface to the agent (``send_message`` -> ``safe_chat``).
* ``agent.py`` / ``safety.py``  reasoning, generation and the safety pipeline. The service does
                          not repeat those checks; it only decides what is stored and reused.

Design rules
------------
* No Streamlit import: importable and unit-testable. The view passes ``st.session_state``
  values in and receives plain data back.
* The current prompt is NEVER part of the history sent with it. History is built from the
  transcript as it was *before* this exchange, and the exchange is appended once, afterwards,
  by ``append_exchange``. That is what prevents duplicated turns.
* Only successful exchanges become context. Blocked, rate-limited, failed or invalid turns are
  stored for display (``include_in_context=False``) but are never sent to the model again.
* ``send_message`` never raises for expected problems (bad input, safety blocks, rate limits,
  model errors, a missing API key). Every outcome is a ``ChatResult`` with a ``status``.
* The agent is imported lazily, because ``digital_twin.config`` raises at import time when
  ``GROQ_API_KEY`` is missing. That must show as a friendly error, not crash the app.

Transcript record (same keys the existing assistant view already uses)::

    {"role": "user" | "assistant", "content": str,
     "include_in_context": bool, "is_error": bool,          # is_error only on assistant records
     "status": str, "notices": [str], "ts": ISO-8601}
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, MutableMapping, Optional, Sequence

logger = logging.getLogger("digital_twin.chat_service")

__all__ = [
    "MAX_MESSAGE_CHARS",
    "CONTEXT_TURNS",
    "MAX_STORED_MESSAGES",
    "MAX_ATTACHMENTS",
    "ChatResult",
    "ChatService",
    "build_context",
    "append_exchange",
    "session_identifier",
    "send_message",
    "get_service",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

# Matches SafetyConfig.max_input_chars. Use it for st.chat_input(max_chars=...) so the box
# never accepts text the safety pipeline will reject.
MAX_MESSAGE_CHARS = 2000
# User+assistant pairs kept as context. Matches agent.MAX_HISTORY_TURNS (safe_chat trims again).
CONTEXT_TURNS = 6
# Upper bound for the stored transcript so a long session cannot grow without limit.
MAX_STORED_MESSAGES = 200

# Charts kept per answer, and the size (as JSON) of one stored attachment.
MAX_ATTACHMENTS = 3
MAX_ATTACHMENT_JSON_CHARS = 20_000

SESSION_ID_KEY = "chat_session_id"

# Statuses caused by the user's input or usage (show as a warning) vs. system problems.
_USER_STATUSES = frozenset(
    {
        "invalid_input",
        "too_long",
        "rate_limited",
        "blocked_injection",
        "blocked_content",
        "blocked_crisis",
    }
)

MESSAGES = {
    "invalid_input": "Please type a message.",
    "error": "The twin is temporarily unavailable. Please try again shortly.",
}


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChatResult:
    """Everything the view needs from one exchange.

    ok          True only for a validated model answer
    status      ok | invalid_input | too_long | rate_limited | blocked_injection |
                blocked_content | blocked_crisis | blocked_output | upstream_busy | error
    prompt      the user's message (stripped)
    content     text to store and show, as markdown (draw it with ``st.markdown``). On success: the
                twin's answer in plain language (or, in "analysis" mode, the structured answer
                rendered as markdown). On failure: a user-facing sentence.
    reply       the structured answer: {} for a plain-language answer, a fallback structure when not ok
    markdown    the same as ``content`` (kept for callers that used it before)
    attachments charts the tools produced for this answer: plain dicts such as
                ``{"type": "chart", "kind": "bar", ...}``, at most ``MAX_ATTACHMENTS``; [] when not ok
    mode        "conversation" or "analysis" ("" when no answer was requested)
    notices     short notes for the user, e.g. "Personal details were redacted ..."
    violations  machine-readable safety codes (for logs and tests, not for the UI)
    retry_after seconds to wait when rate limited
    """

    ok: bool
    status: str
    prompt: str
    content: str
    reply: Dict[str, Any] = field(default_factory=dict)
    markdown: str = ""
    notices: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    retry_after: float = 0.0
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    mode: str = ""
    created: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    @property
    def is_error(self) -> bool:
        return not self.ok

    @property
    def level(self) -> str:
        """'ok' | 'warning' (the user can fix it) | 'error' (the system could not answer)."""
        if self.ok:
            return "ok"
        return "warning" if self.status in _USER_STATUSES else "error"

    @property
    def should_store(self) -> bool:
        """An empty submission is not an exchange and is never added to the transcript."""
        return self.status != "invalid_input"

    def user_record(self) -> Dict[str, Any]:
        return {
            "role": "user",
            "content": self.prompt,
            "include_in_context": self.ok,
            "status": self.status,
            "ts": self.created,
        }

    def assistant_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "role": "assistant",
            "content": self.content,
            "include_in_context": self.ok,
            "is_error": self.is_error,
            "status": self.status,
            "notices": list(self.notices),
            "ts": self.created,
        }
        if self.attachments:
            record["attachments"] = [dict(a) for a in self.attachments]
        return record

    def records(self) -> List[Dict[str, Any]]:
        return [self.user_record(), self.assistant_record()]


def _clean_attachments(items: Any) -> List[Dict[str, Any]]:
    """Keep up to ``MAX_ATTACHMENTS`` JSON-safe dicts of a sane size; drop anything else.

    The tools already validated these; this is the last check before they are stored in the
    session transcript and handed to the view.
    """
    clean: List[Dict[str, Any]] = []
    if not isinstance(items, (list, tuple)):
        return clean
    for item in items:
        if len(clean) >= MAX_ATTACHMENTS:
            break
        if not isinstance(item, dict):
            continue
        try:
            text = json.dumps(item, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError):
            continue
        if len(text) <= MAX_ATTACHMENT_JSON_CHARS:
            clean.append(json.loads(text))  # a detached, JSON-only copy
    return clean


def _failure(status: str, prompt: str, message: str, **extra: Any) -> ChatResult:
    return ChatResult(
        ok=False,
        status=status,
        prompt=prompt,
        content=message,
        markdown=message,
        **extra,
    )


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def build_context(
    transcript: Optional[Sequence[Any]],
    *,
    turns: int = CONTEXT_TURNS,
) -> List[Dict[str, str]]:
    """Turn the stored transcript into the history the agent expects.

    Includes only records that are dicts with role 'user'/'assistant', string content,
    ``include_in_context`` true (default) and not an error. Keeps the newest ``turns``
    exchanges and never starts on an assistant turn. The input is not modified.
    """
    history: List[Dict[str, str]] = []
    for item in transcript or []:
        if not isinstance(item, dict):
            continue
        if item.get("include_in_context", True) is not True or item.get("is_error"):
            continue
        role, content = item.get("role"), item.get("content")
        if (
            role in ("user", "assistant")
            and isinstance(content, str)
            and content.strip()
        ):
            history.append({"role": role, "content": content})

    history = history[-(2 * turns) :] if turns > 0 else []
    while history and history[0]["role"] != "user":
        history.pop(0)
    return history


def append_exchange(
    transcript: List[Dict[str, Any]],
    result: ChatResult,
    *,
    max_messages: int = MAX_STORED_MESSAGES,
) -> None:
    """Add the user and assistant records for ``result`` to ``transcript`` (in place), once.

    Call exactly once per ``send_message``. Empty submissions are skipped. The oldest
    records are dropped past ``max_messages``.
    """
    if not result.should_store:
        return
    transcript.extend(result.records())
    overflow = len(transcript) - max_messages
    if overflow > 0:
        del transcript[:overflow]
        while transcript and transcript[0].get("role") != "user":
            transcript.pop(0)


def session_identifier(state: MutableMapping[str, Any]) -> str:
    """A stable per-browser-session id for the rate limiter.

    Pass ``st.session_state``. Without this every visitor would share one rate-limit bucket.
    """
    value = state.get(SESSION_ID_KEY)
    if not isinstance(value, str) or not value:
        value = uuid.uuid4().hex
        state[SESSION_ID_KEY] = value
    return value


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------
_ADAPTER_MODULES = ("digital_twin.twin_adapter", "services.twin_adapter")
Sender = Callable[..., Any]
_adapter_sender: Optional[Sender] = None


def _load_sender() -> Sender:
    """Import the adapter's ``send_message`` on first use (and cache it).

    Importing the agent reads the API key and the twin's data, so failures here are
    configuration problems; the caller turns them into a friendly ``error`` result.
    """
    global _adapter_sender
    if _adapter_sender is not None:
        return _adapter_sender
    if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    last_error: Optional[BaseException] = None
    for name in _ADAPTER_MODULES:
        try:
            module = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name and name.startswith(
                exc.name
            ):  # the adapter itself is not there
                last_error = exc
                continue
            raise  # a dependency inside it is missing (e.g. groq): surface it
        sender = getattr(module, "send_message", None)
        if callable(sender):
            _adapter_sender = sender
            return sender
        last_error = ImportError(f"{name} does not define send_message().")
    raise ImportError("Could not load the twin adapter.") from last_error


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
class ChatService:
    """Coordinates one chat exchange. ``sender`` can be replaced in tests."""

    def __init__(self, sender: Optional[Sender] = None) -> None:
        self._sender = sender

    def send_message(
        self,
        prompt: Any,
        transcript: Optional[Sequence[Any]] = None,
        *,
        session_id: str = "default",
        mode: Optional[str] = None,
    ) -> ChatResult:
        """Send ``prompt`` to the twin and return a :class:`ChatResult`. Does not raise.

        prompt      the user's newest message
        transcript  the stored messages BEFORE this exchange (do not append the prompt first)
        session_id  rate-limit bucket; use ``session_identifier(st.session_state)``
        mode        "conversation" (plain language) or "analysis" (structured JSON, rendered as
                    markdown). None leaves the choice to the adapter (``TWIN_REPLY_MODE``, default
                    conversation). Only passed on when given, so simple senders still work.
        """
        text = prompt.strip() if isinstance(prompt, str) else ""
        if not text:
            return _failure("invalid_input", "", MESSAGES["invalid_input"])

        history = build_context(transcript)
        try:
            sender = self._sender or _load_sender()
            extra = {"mode": mode} if mode is not None else {}
            reply = sender(
                text, history, identifier=str(session_id) or "default", **extra
            )
        except Exception:
            # Detail goes to the server log only (never the prompt text, never to the UI).
            logger.exception("chat request failed before a reply was produced")
            return _failure("error", text, MESSAGES["error"])

        try:
            return self._to_result(text, reply)
        except Exception:
            logger.exception("could not read the adapter's reply")
            return _failure("error", text, MESSAGES["error"])

    @staticmethod
    def _to_result(prompt: str, reply: Any) -> ChatResult:
        ok = bool(getattr(reply, "ok", False))
        status = str(getattr(reply, "status", "") or ("ok" if ok else "error"))
        structured = getattr(reply, "reply", None)
        structured = dict(structured) if isinstance(structured, dict) else {}
        notices = [str(n) for n in (getattr(reply, "notices", None) or []) if n]
        violations = [str(v) for v in (getattr(reply, "violations", None) or [])]
        retry_after = float(getattr(reply, "retry_after", 0.0) or 0.0)
        display = str(getattr(reply, "text", "") or "")
        mode = str(getattr(reply, "mode", "") or "")

        if ok:
            # Markdown for the view. The structured answer (if any) travels in ``reply``.
            content = display or (
                json.dumps(structured, ensure_ascii=False) if structured else ""
            )
            if not content.strip():
                return _failure("error", prompt, MESSAGES["error"])
            logger.info(
                "chat ok (prompt %d chars, %d notices)", len(prompt), len(notices)
            )
            return ChatResult(
                ok=True,
                status="ok",
                prompt=prompt,
                content=content,
                reply=structured,
                markdown=content,
                notices=notices,
                violations=violations,
                attachments=_clean_attachments(getattr(reply, "attachments", None)),
                mode=mode,
            )

        message = str(getattr(reply, "message", "") or display or MESSAGES["error"])
        logger.info(
            "chat not ok: status=%s violations=%s", status, ",".join(violations) or "-"
        )
        return ChatResult(
            ok=False,
            status=status,
            prompt=prompt,
            content=message,
            reply=structured,
            markdown=message,
            notices=notices,
            violations=violations,
            retry_after=retry_after,
            mode=mode,
        )


# ---------------------------------------------------------------------------
# Default service + module-level functions
# ---------------------------------------------------------------------------
_default = ChatService()


def get_service() -> ChatService:
    return _default


def send_message(
    prompt: Any,
    transcript: Optional[Sequence[Any]] = None,
    *,
    session_id: str = "default",
    mode: Optional[str] = None,
) -> ChatResult:
    return _default.send_message(prompt, transcript, session_id=session_id, mode=mode)
