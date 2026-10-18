from __future__ import annotations

import base64
import logging
import re
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ICON_DIR = (Path(__file__).resolve().parent.parent / "assets" / "icons").resolve()

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_COLOR_PATTERN = re.compile(r"^(currentColor|var\(--[a-z0-9-]+\))$")
_FORBIDDEN_MARKUP = (
    "<script",
    "javascript:",
    "<foreignobject",
    "onload=",
    "onerror=",
    "<iframe",
)

MIN_SIZE = 8
MAX_SIZE = 128


@lru_cache(maxsize=None)
def _read_svg(name: str) -> bytes:
    """Read and validate a bundled SVG. Raises ValueError / OSError on problems."""
    if not _NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid icon name: {name!r}")

    path = (ICON_DIR / f"{name}.svg").resolve()
    if ICON_DIR not in path.parents:
        raise ValueError(f"Icon path escapes the icon directory: {name!r}")

    data = path.read_bytes()
    text = data.decode("utf-8").lower()
    if "<svg" not in text:
        raise ValueError(f"{path.name} is not an SVG file")
    if any(token in text for token in _FORBIDDEN_MARKUP):
        raise ValueError(f"{path.name} contains disallowed markup")
    return data


@lru_cache(maxsize=None)
def icon_data_uri(name: str) -> Optional[str]:
    """Return a base64 ``data:`` URI for a bundled icon, or None if unavailable."""
    try:
        encoded = base64.b64encode(_read_svg(name)).decode("ascii")
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        logger.warning("Icon %r could not be loaded: %s", name, exc)
        return None
    return f"data:image/svg+xml;base64,{encoded}"


def icon_html(
    name: str,
    size: int = 20,
    color: str = "currentColor",
    label: Optional[str] = None,
) -> str:
    """
    Return an HTML snippet that renders a bundled Phosphor icon.

    Args:
        name:  File name without extension, e.g. ``"gear"`` for ``assets/icons/gear.svg``.
        size:  Pixel width/height (clamped to 8-128).
        color: ``"currentColor"`` or a CSS variable such as ``"var(--color-accent)"``.
        label: Accessible name. When omitted the icon is treated as decorative.

    Returns an empty string if the icon is missing, so a missing asset never
    breaks the page.
    """
    uri = icon_data_uri(name)
    if uri is None:
        return ""

    if not _COLOR_PATTERN.fullmatch(color):
        color = "currentColor"
    size = max(MIN_SIZE, min(MAX_SIZE, int(size)))

    style = (
        "display:inline-block;flex:none;vertical-align:middle;"
        f"width:{size}px;height:{size}px;background-color:{color};"
        f"-webkit-mask:url('{uri}') center/contain no-repeat;"
        f"mask:url('{uri}') center/contain no-repeat;"
    )
    a11y = (
        f'role="img" aria-label="{escape(label, quote=True)}"'
        if label
        else 'aria-hidden="true"'
    )
    return f'<span class="ph-icon" {a11y} style="{style}"></span>'
