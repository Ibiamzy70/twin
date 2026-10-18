"""Thin, hardened wrapper around the Groq chat-completions API.

Design notes
------------
* The API key is resolved when a client is *constructed*, never at import time,
  so tests and Streamlit reruns see the current environment.
* We own the retry policy: the SDK's own retries are switched off so attempts
  do not multiply (SDK retries x ours).
* Messages are validated locally so a malformed request fails fast, for free,
  with a clear error instead of a 400 from the network.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any

import groq
from groq import Groq

from . import config
from .config import GROQ_MODEL, MAX_TOKENS

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0  # seconds, per attempt
MAX_BACKOFF = 30.0  # seconds
_VALID_ROLES = frozenset({"system", "user", "assistant", "tool", "developer"})
_RETRYABLE_STATUS = frozenset({408, 409, 429})


def _default_api_key() -> str:
    """Current key: the environment wins, then the config module."""
    return (
        os.getenv("GROQ_API_KEY") or getattr(config, "GROQ_API_KEY", "") or ""
    ).strip()


def _validate_messages(messages: Any) -> None:
    """Reject obviously malformed payloads before spending a network call."""
    if not isinstance(messages, list):
        raise TypeError("messages must be a list of dicts.")
    if not messages:
        raise ValueError("messages must not be empty.")
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise TypeError(f"messages[{i}] must be a dict, got {type(msg).__name__}.")
        role = msg.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"messages[{i}] has an invalid role: {role!r}.")
        content = msg.get("content")
        # An assistant message that only carries tool_calls may have no content.
        if content is None and not (role == "assistant" and msg.get("tool_calls")):
            raise ValueError(f"messages[{i}] is missing content.")
        if content is not None and not isinstance(content, (str, list)):
            raise TypeError(f"messages[{i}].content must be a string or list.")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and (status in _RETRYABLE_STATUS or status >= 500)


def _retry_after(exc: Exception) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


class GroqClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = GROQ_MODEL,
        max_tokens: int = MAX_TOKENS,
        max_retries: int = 3,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        key = _default_api_key() if api_key is None else api_key.strip()
        if not key:
            raise ValueError("Groq API key is required.")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0.")

        # max_retries=0: this class is the single retry layer.
        self.client = Groq(api_key=key, timeout=timeout, max_retries=0)
        self.model = model
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout

    def _create(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        """Send one chat completion, retrying transient failures with
        exponential backoff plus jitter. Shared by chat() and chat_with_tools()."""
        _validate_messages(messages)

        for attempt in range(self.max_retries + 1):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    **kwargs,
                )
            except Exception as exc:
                if not _is_retryable(exc) or attempt >= self.max_retries:
                    raise
                delay = _retry_after(exc)
                if delay is None:
                    delay = min(2**attempt, MAX_BACKOFF) * random.uniform(0.5, 1.0)
                delay = min(delay, MAX_BACKOFF)
                logger.warning(
                    "Groq request failed (%s); retry %d/%d in %.1fs",
                    type(exc).__name__,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError("Groq request failed after all retries.")  # pragma: no cover

    def chat(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        """Return the assistant's reply text."""
        response = self._create(messages, **kwargs)
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Groq returned no assistant message content.")
        return content

    def chat_with_tools(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        tool_choice: str = "auto",
        **kwargs: Any,
    ) -> Any:
        """Return the assistant MESSAGE object (it carries `tool_calls`).

        `content` may be None on such a message. With no `tools` the request is
        sent without them, so the model must answer in text.
        """
        if tools:
            kwargs["tools"] = tools
            kwargs.setdefault("tool_choice", tool_choice)
        response = self._create(messages, **kwargs)
        if not response.choices:
            raise RuntimeError("Groq returned no completion choices.")
        return response.choices[0].message
