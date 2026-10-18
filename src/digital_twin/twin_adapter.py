"""Stable interface between the Streamlit application and the digital twin agent.

The UI and the services layer import this module and nothing else from
``digital_twin``. They never see prompts, the Groq client, the safety pipeline or the
shape of ``SafeChatResult``, so any of those can change without touching a view.

Which agent function is exposed
-------------------------------
``agent.chat()`` is the raw model call and runs no safety checks (its own docstring says
anything a user can reach must go through ``safe_chat()``). The adapter therefore:

* ``send_message()``      -> ``agent.safe_chat()``  (input checks, PII redaction, rate
                             limiting, output validation). Use this from the UI.
                             ``mode="conversation"`` (default) answers in plain language;
                             ``mode="analysis"`` keeps the structured JSON contract.
* ``send_raw_message()``  -> ``agent.chat()``       (unguarded; tests and internal tools only).

No Streamlit import here: the module is pure Python and unit-testable.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from .agent import (
    VALID_HISTORY_ROLES,
    chat as _agent_chat,
    clear_caches as _agent_clear_caches,
    safe_chat as _agent_safe_chat,
)

logger = logging.getLogger("digital_twin.twin_adapter")

__all__ = [
    "TwinReply",
    "MODE_ANALYSIS",
    "MODE_CONVERSATION",
    "MAX_ATTACHMENTS",
    "send_message",
    "send_raw_message",
    "history_from_messages",
    "format_reply",
    "reset",
]

Message = Dict[str, str]

GENERIC_ERROR = "The twin is temporarily unavailable. Please try again shortly."

# Same strings as agent.MODE_*; repeated here so this module needs nothing extra from the agent.
MODE_ANALYSIS = "analysis"
MODE_CONVERSATION = "conversation"
_MODES = (MODE_ANALYSIS, MODE_CONVERSATION)
MODE_ENV = "TWIN_REPLY_MODE"  # app-wide default: "conversation" (default) or "analysis"
MAX_ATTACHMENTS = 3  # charts the UI will be given for one answer

# Role names used by the UI layers -> roles the agent accepts.
_ROLE_ALIASES = {
    "user": "user",
    "human": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "ai": "assistant",
}
_CONTENT_KEYS = ("content", "text", "message")

_CONFIDENCE_ORDER = {"high": 0, "medium": 1, "low": 2}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TwinReply:
    """What the UI gets back from one exchange.

    ok            True only when ``reply`` is a validated model answer
    status        ok | invalid_input | too_long | rate_limited | blocked_injection |
                  blocked_content | blocked_crisis | blocked_output | upstream_busy | error
    text          ready-to-display text (formatted answer, or the explanation when not ok)
    reply         the structured answer (a fallback structure when ok is False)
    message       user-facing explanation when not ok, else ""
    notices       short notes to show the user (e.g. "personal details were redacted")
    violations    machine-readable codes the safety pipeline flagged
    retry_after   seconds to wait when rate limited
    sent_message  the redacted text actually sent to the model ("" if never sent)
    mode          "conversation" or "analysis": how the answer was produced
    attachments   charts and similar items the tools produced for this answer (plain, validated
                  dicts such as ``{"type": "chart", ...}``); empty unless the answer succeeded
    """

    ok: bool
    status: str
    text: str
    reply: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    notices: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)
    retry_after: float = 0.0
    sent_message: str = ""
    mode: str = MODE_CONVERSATION
    attachments: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "status": self.status,
            "text": self.text,
            "reply": self.reply,
            "message": self.message,
            "notices": list(self.notices),
            "violations": list(self.violations),
            "retry_after": self.retry_after,
            "sent_message": self.sent_message,
            "mode": self.mode,
            "attachments": [dict(a) for a in self.attachments],
        }


def _tools_enabled(flag: Optional[bool]) -> bool:
    """Tools are on by default. ``TWIN_ENABLE_TOOLS=0`` turns them off for the whole app."""
    if flag is not None:
        return bool(flag)
    return os.environ.get("TWIN_ENABLE_TOOLS", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _resolve_mode(mode: Optional[str]) -> str:
    """An explicit ``mode`` wins; otherwise ``TWIN_REPLY_MODE``; otherwise conversation."""
    if mode is not None:
        if mode not in _MODES:
            raise ValueError(f"mode must be one of: {', '.join(_MODES)}.")
        return mode
    value = os.environ.get(MODE_ENV, MODE_CONVERSATION).strip().lower()
    if value not in _MODES:
        logger.warning("Unknown %s %r; using %r.", MODE_ENV, value, MODE_CONVERSATION)
        return MODE_CONVERSATION
    return value


def _tool_generate(
    identifier: str,
    used: List[str],
    attachments: Optional[List[Dict[str, Any]]] = None,
    mode: str = MODE_CONVERSATION,
) -> Optional[Callable[..., Any]]:
    """The tool-calling ``generate`` for ``safe_chat``; ``used`` collects the tools that ran
    and ``attachments`` the charts they produced.

    Returns None (plain chat, no tools) if the tools module cannot be loaded, so a tools problem
    never takes the whole assistant down.
    """
    try:
        from .tools import make_generate

        def remember(run: Any) -> None:
            for name in run.tools_used:
                if name not in used:
                    used.append(name)
            if attachments is not None:
                for item in getattr(run, "attachments", None) or ():
                    if isinstance(item, Mapping):
                        attachments.append(dict(item))

        return make_generate(identifier=identifier, on_run=remember, mode=mode)
    except Exception:
        logger.exception("tools unavailable; continuing without them")
        return None


def _failure(status: str, message: str) -> TwinReply:
    return TwinReply(ok=False, status=status, text=message, message=message)


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
def history_from_messages(messages: Optional[Iterable[Any]]) -> List[Message]:
    """Convert UI messages into the history format the agent requires.

    Accepts dicts with ``role`` (user/human/assistant/bot/ai) and ``content`` (or
    ``text``/``message``). Anything else (system turns, empty text, wrong types) is
    dropped rather than raised on, because a stray UI record must not break the chat.
    The input is never mutated.
    """
    history: List[Message] = []
    for item in messages or []:
        if not isinstance(item, Mapping):
            continue
        role = _ROLE_ALIASES.get(str(item.get("role", "")).strip().lower())
        if role is None or role not in VALID_HISTORY_ROLES:
            continue
        content = next(
            (item[k] for k in _CONTENT_KEYS if isinstance(item.get(k), str)), None
        )
        if content is None or not content.strip():
            continue
        history.append({"role": role, "content": content})
    return history


# ---------------------------------------------------------------------------
# Reply formatting
# ---------------------------------------------------------------------------
def _clean(value: Any) -> str:
    return " ".join(str(value).split()) if value is not None else ""


def _str_list(value: Any) -> List[str]:
    if isinstance(value, (list, tuple)):
        return [s for s in (_clean(v) for v in value) if s]
    text = _clean(value)
    return [text] if text else []


def format_reply(reply: Mapping[str, Any]) -> str:
    """Render the twin's structured JSON answer as readable markdown.

    Defensive by design: unknown or missing fields are skipped, never raised on.
    """
    if not isinstance(reply, Mapping):
        return _clean(reply)

    parts: List[str] = []

    context = reply.get("current_context")
    if isinstance(context, Mapping):
        summary = _clean(context.get("summary"))
        if summary:
            parts.append(f"**Current context**\n{summary}")
        signals = _str_list(context.get("observed_signals"))
        if signals:
            parts.append(
                "**Observed signals**\n" + "\n".join(f"- {s}" for s in signals)
            )

    actions = reply.get("predicted_next_action")
    if isinstance(actions, Mapping):
        actions = [actions]
    if isinstance(actions, (list, tuple)):
        rows = [
            a for a in actions if isinstance(a, Mapping) and _clean(a.get("action"))
        ]
        rows.sort(
            key=lambda a: _CONFIDENCE_ORDER.get(_clean(a.get("confidence")).lower(), 3)
        )
        lines: List[str] = []
        for number, row in enumerate(rows, start=1):
            confidence = _clean(row.get("confidence")).lower()
            tag = f" ({confidence} confidence)" if confidence else ""
            lines.append(f"{number}. **{_clean(row['action'])}**{tag}")
            why = _clean(row.get("why"))
            if why:
                lines.append(f"   {why}")
            for item in _str_list(row.get("supporting_data")):
                lines.append(f"   - {item}")
        if lines:
            parts.append("**Likely next actions**\n" + "\n".join(lines))

    missing = _str_list(reply.get("missing_information"))
    if missing:
        parts.append("**Missing information**\n" + "\n".join(f"- {m}" for m in missing))

    quality = reply.get("response_quality")
    if isinstance(quality, Mapping) and quality.get("contains_guessing") is True:
        parts.append("_Note: part of this answer is inferred rather than observed._")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def send_message(
    user_message: str,
    history: Optional[Iterable[Any]] = None,
    *,
    identifier: str = "default",
    use_tools: Optional[bool] = None,
    mode: Optional[str] = None,
) -> TwinReply:
    """Send one message through the guarded agent and return a :class:`TwinReply`.

    user_message  the user's newest message
    history       earlier turns; dicts with role/content (see ``history_from_messages``)
    identifier    rate-limit bucket: pass a per-session or per-user id in production
    use_tools     let the model call the registered tools (tasks, calculator, date). None = follow
                  the ``TWIN_ENABLE_TOOLS`` environment variable (default on). Tool calls happen
                  inside ``safe_chat``, so input checks and output validation still apply.
    mode          "conversation" (plain-language answer, the default) or "analysis" (the structured
                  JSON contract). None = follow ``TWIN_REPLY_MODE``. An unknown value raises ValueError.

    Never raises for user-caused or model-caused problems: check ``reply.ok``.
    """
    if not isinstance(user_message, str):
        return _failure("invalid_input", "Please type a message.")

    chosen = _resolve_mode(mode)
    used: List[str] = []
    charts: List[Dict[str, Any]] = []
    ident = str(identifier) or "default"
    options: Dict[str, Any] = {"mode": chosen}
    if _tools_enabled(use_tools):
        generate = _tool_generate(ident, used, charts, chosen)
        if generate is not None:
            options["generate"] = generate

    try:
        result = _agent_safe_chat(
            user_message,
            history_from_messages(history),
            identifier=ident,
            **options,
        )
    except Exception:  # last line of defence: the UI must never see a traceback
        logger.exception("twin agent failed unexpectedly")
        return _failure("error", GENERIC_ERROR)

    if result.ok:
        # Conversation mode: the answer is already text. If the model used the JSON contract
        # anyway (or in analysis mode) the structured reply is rendered as markdown.
        text = format_reply(result.reply) or result.reply_text
    else:
        text = result.message or GENERIC_ERROR

    return TwinReply(
        ok=bool(result.ok),
        status=str(result.status),
        text=text,
        reply=dict(result.reply or {}),
        message=result.message,
        notices=list(result.notices)
        + ([f"Tools used: {', '.join(used)}."] if used else []),
        violations=list(result.violations),
        retry_after=float(result.retry_after or 0.0),
        sent_message=result.sent_message,
        mode=str(getattr(result, "mode", chosen) or chosen),
        attachments=charts[:MAX_ATTACHMENTS] if result.ok else [],
    )


def send_raw_message(
    user_message: str,
    history: Optional[Iterable[Any]] = None,
    *,
    mode: str = MODE_ANALYSIS,
) -> str:
    """Call ``agent.chat()`` directly and return the raw model text.

    No input moderation, PII redaction, rate limiting or output validation. Keep this
    out of anything a user can reach. Raises ``ValueError`` / ``TypeError`` on a bad
    message and lets model errors propagate.
    """
    if mode == MODE_ANALYSIS:
        return _agent_chat(user_message, history_from_messages(history))
    return _agent_chat(
        user_message, history_from_messages(history), mode=_resolve_mode(mode)
    )


def reset() -> None:
    """Drop the agent's cached prompt, client and safety pipeline.

    Call after the twin's data or API key changes, and between tests.
    """
    _agent_clear_caches()
