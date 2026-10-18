"""
Metric card.

A premium stat tile in the same visual language as the dashboard: soft gradient
surface, large rounded corners, a big number, a small delta pill and an optional
sparkline or progress bar. Three looks:

* ``default``  light surface
* ``accent``   filled with the accent colour (use for the one hero metric)
* ``dark``     inverted surface for contrast

Two entry points:

* ``metric_card_html(...)``    pure function, returns markup (no Streamlit calls).
* ``render_metric_card(...)``  renders the card; pass ``on_click`` and ``key`` to
                                make the whole card clickable, like the dashboard cards.

All text is escaped. Numbers are sanitised (NaN/inf are treated as missing).
Styling uses only theme tokens from ``styles/theme.css``.
"""

from __future__ import annotations

import math
from html import escape
from typing import Any, Callable, Optional, Sequence, Union

import streamlit as st

from components.icons import icon_html

Number = Union[int, float]

VARIANTS = ("default", "accent", "dark")
TONES = ("positive", "negative", "neutral")

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
_METRIC_CSS = """
.metric-card {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
  min-height: 11rem;
  padding: var(--space-5);
  overflow: hidden;
  background: linear-gradient(180deg, var(--color-surface) 0%, var(--color-sidebar) 100%);
  border: var(--border);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-md);
  color: var(--color-text);
  animation: metric-rise var(--duration-slow) var(--ease-standard) both;
  transition: transform var(--duration-base) var(--ease-standard),
              box-shadow var(--duration-base) var(--ease-standard),
              border-color var(--duration-base) var(--ease-standard);
}
.metric-card--accent {
  background: linear-gradient(160deg, var(--color-accent) 0%, var(--color-accent-border) 160%);
  border-color: transparent;
  color: var(--color-text-on-accent);
}
.metric-card--dark {
  background: linear-gradient(180deg, var(--color-text) 0%, var(--color-text-secondary) 220%);
  border-color: transparent;
  color: var(--color-surface);
}

.metric-card:hover,
[class*="st-key-metric_card_"]:hover .metric-card {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}
.metric-card--default:hover,
[class*="st-key-metric_card_"]:hover .metric-card--default { border-color: var(--color-accent-border); }

/* head: icon chip, label, optional action arrow */
.metric-card__head { display: flex; align-items: center; justify-content: space-between; gap: var(--space-3); }
.metric-card__lead { display: flex; align-items: center; gap: var(--space-3); min-width: 0; }
.metric-card__chip {
  display: grid; place-items: center; flex: none;
  width: 2.25rem; height: 2.25rem;
  border-radius: var(--radius-md);
  background: var(--color-accent-soft);
  color: var(--color-accent);
  transition: background var(--duration-base) var(--ease-standard), color var(--duration-base) var(--ease-standard);
}
.metric-card--accent .metric-card__chip { background: var(--color-text-on-accent); color: var(--color-accent); }
.metric-card--dark .metric-card__chip { background: var(--color-accent); color: var(--color-text-on-accent); }
.metric-card--default:hover .metric-card__chip,
[class*="st-key-metric_card_"]:hover .metric-card--default .metric-card__chip {
  background: var(--color-accent); color: var(--color-text-on-accent);
}

.metric-card__label {
  margin: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  color: inherit; opacity: 0.8;
  font-size: var(--font-size-sm); font-weight: var(--font-weight-medium); line-height: var(--line-height-tight);
}

.metric-card__go {
  position: relative; flex: none; width: 2rem; height: 2rem;
  border: var(--border-width) solid currentColor; border-radius: var(--radius-full); opacity: 0.55;
  transition: transform var(--duration-base) var(--ease-standard), opacity var(--duration-base) var(--ease-standard),
              background var(--duration-base) var(--ease-standard);
}
.metric-card__go::after {
  content: ""; position: absolute; top: 50%; left: 50%; width: 0.45rem; height: 0.45rem;
  border-top: 2px solid currentColor; border-right: 2px solid currentColor;
  transform: translate(-60%, -40%);
}
[class*="st-key-metric_card_"]:hover .metric-card__go { opacity: 1; transform: rotate(45deg); }

/* value row */
.metric-card__body { display: flex; align-items: flex-end; justify-content: space-between; gap: var(--space-3); margin-top: auto; }
.metric-card__value {
  display: flex; align-items: baseline; gap: var(--space-1); margin: 0; min-width: 0;
  font-size: 2.75rem; font-weight: var(--font-weight-semibold);
  letter-spacing: var(--letter-spacing-tight); line-height: 1;
}
.metric-card__affix { font-size: 1.25rem; font-weight: var(--font-weight-medium); opacity: 0.7; }
.metric-card__value--empty { opacity: 0.45; }

/* delta pill */
.metric-card__delta {
  display: inline-flex; align-items: center; gap: var(--space-1); flex: none;
  padding: var(--space-1) var(--space-3);
  border-radius: var(--radius-full);
  font-size: var(--font-size-xs); font-weight: var(--font-weight-semibold); white-space: nowrap;
  background: var(--color-surface-raised); color: var(--color-text-secondary);
}
.metric-card__delta svg { width: 0.7rem; height: 0.7rem; }
.metric-card__delta--positive { background: var(--color-success-soft); color: var(--color-success); }
.metric-card__delta--negative { background: var(--color-danger-soft, var(--color-surface-raised)); color: var(--color-danger, var(--color-text-secondary)); }
.metric-card--accent .metric-card__delta,
.metric-card--dark .metric-card__delta { background: var(--color-surface); }

.metric-card__caption { margin: 0; color: inherit; opacity: 0.7; font-size: var(--font-size-xs); line-height: var(--line-height-normal); }

/* sparkline */
.metric-card__spark { width: 100%; height: 2.25rem; margin-top: var(--space-1); color: var(--color-accent); }
.metric-card--accent .metric-card__spark { color: var(--color-text-on-accent); }
.metric-card__spark svg { width: 100%; height: 100%; overflow: visible; }
.metric-card__spark polyline { fill: none; stroke: currentColor; stroke-width: 2; stroke-linecap: round; stroke-linejoin: round; vector-effect: non-scaling-stroke; }
.metric-card__spark polygon { fill: currentColor; opacity: 0.12; }

/* progress bar */
.metric-card__bar {
  height: 0.6rem; border-radius: var(--radius-full); overflow: hidden;
  background: repeating-linear-gradient(135deg, var(--color-border) 0 2px, transparent 2px 8px);
}
.metric-card__bar-fill {
  height: 100%; border-radius: var(--radius-full); background: var(--color-accent);
  transform-origin: left center; animation: metric-grow 0.9s var(--ease-standard) both 200ms;
}
.metric-card--accent .metric-card__bar-fill { background: var(--color-text-on-accent); }

/* whole-card click target (same pattern as the dashboard cards) */
[class*="st-key-metric_card_"] { position: relative; }
[class*="st-key-metric_open_"] { position: absolute; inset: 0; z-index: 3; margin: 0; width: 100% !important; height: 100% !important; }
[class*="st-key-metric_open_"] * { width: 100%; height: 100%; min-height: 0; }
[class*="st-key-metric_open_"] button { opacity: 0; cursor: pointer; padding: 0; border: 0; }
[class*="st-key-metric_card_"]:has(button:focus-visible) .metric-card { outline: 2px solid var(--color-accent); outline-offset: 3px; }

@keyframes metric-rise { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes metric-grow { from { transform: scaleX(0); opacity: 0; } to { transform: scaleX(1); opacity: 1; } }

@media (prefers-reduced-motion: reduce) {
  .metric-card, .metric-card__bar-fill { animation: none; transition: none; }
}
@media (max-width: 640px) { .metric-card__value { font-size: 2.25rem; } }
"""

_ARROW_UP = '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 1.5 9 7H1z" fill="currentColor"/></svg>'
_ARROW_DOWN = '<svg viewBox="0 0 10 10" aria-hidden="true"><path d="M5 8.5 1 3h8z" fill="currentColor"/></svg>'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _clean(text: Any) -> str:
    return " ".join(str(text or "").split())


def _safe_icon(name: Optional[str], size: int) -> str:
    if not name:
        return ""
    try:
        return icon_html(name, size=size) or ""
    except (KeyError, ValueError, TypeError):
        return ""


def _format_value(value: Any, decimals: Optional[int]) -> str:
    """Numbers get thousands separators; text is shown as given; None/NaN shows a dash."""
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    if _is_number(value):
        if decimals is None:
            decimals = 0 if float(value).is_integer() else 1
        return f"{value:,.{decimals}f}"
    return _clean(value) or "—"


def _delta_pill(
    delta: Any, tone: Optional[str], suffix: str, label: str, invert: bool
) -> str:
    if delta is None or (isinstance(delta, float) and not math.isfinite(delta)):
        return ""

    if _is_number(delta):
        if tone not in TONES:
            sign = (delta > 0) - (delta < 0)
            if invert:
                sign = -sign
            tone = "positive" if sign > 0 else "negative" if sign < 0 else "neutral"
        arrow = _ARROW_UP if delta > 0 else _ARROW_DOWN if delta < 0 else ""
        magnitude = abs(delta)
        text = (
            f"{magnitude:,.0f}{suffix}"
            if float(magnitude).is_integer()
            else f"{magnitude:,.1f}{suffix}"
        )
    else:
        text = _clean(delta)
        if not text:
            return ""
        arrow = ""
        tone = tone if tone in TONES else "neutral"

    tail = f" {escape(_clean(label))}" if _clean(label) else ""
    return f'<span class="metric-card__delta metric-card__delta--{tone}">{arrow}{escape(text)}{tail}</span>'


def sparkline_svg(values: Optional[Sequence[Any]]) -> str:
    """Tiny trend line. Needs at least two finite numbers, otherwise returns ''."""
    points = [float(v) for v in (values or []) if _is_number(v)][-40:]
    if len(points) < 2:
        return ""
    low, high = min(points), max(points)
    span = (high - low) or 1.0
    step = 100 / (len(points) - 1)
    coords = [
        (
            f"{i * step:.2f},{30 - ((p - low) / span) * 28:.2f}"
            if high != low
            else f"{i * step:.2f},16"
        )
        for i, p in enumerate(points)
    ]
    line = " ".join(coords)
    area = f"0,32 {line} 100,32"
    return (
        '<div class="metric-card__spark" aria-hidden="true">'
        '<svg viewBox="0 0 100 32" preserveAspectRatio="none">'
        f'<polygon points="{area}"/><polyline points="{line}"/></svg></div>'
    )


def _progress_bar(progress: Any) -> str:
    if not _is_number(progress):
        return ""
    pct = max(0.0, min(100.0, float(progress)))
    return (
        f'<div class="metric-card__bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" '
        f'aria-valuenow="{pct:.0f}"><div class="metric-card__bar-fill" style="width:{pct:.1f}%"></div></div>'
    )


def _styles_once() -> None:
    """Inject the stylesheet once per script run (a cheap no-op on later calls).

    Streamlit creates a fresh ``cursors`` mapping for every script run, so holding a
    reference to it in session state identifies "this run". If Streamlit's internals
    change, the stylesheet is simply injected every time, which is safe, only wasteful.
    """
    marker = "_metric_card_css_run"
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        ctx = get_script_run_ctx()
        run_token = getattr(ctx, "cursors", None) if ctx else None
        if run_token is not None:
            if st.session_state.get(marker) is run_token:
                return
            st.session_state[marker] = run_token
    except Exception:  # pragma: no cover - never let styling break the page
        pass
    st.markdown(f"<style>{_METRIC_CSS}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def metric_card_html(
    label: str,
    value: Any,
    *,
    icon: Optional[str] = None,
    prefix: str = "",
    suffix: str = "",
    decimals: Optional[int] = None,
    delta: Any = None,
    delta_tone: Optional[str] = None,
    delta_suffix: str = "%",
    delta_label: str = "",
    invert_delta: bool = False,
    caption: str = "",
    trend: Optional[Sequence[Any]] = None,
    progress: Optional[Number] = None,
    variant: str = "default",
    clickable: bool = False,
) -> str:
    """Return the card markup (no styles, no Streamlit calls).

    ``delta`` may be a number (sign picks the colour, ``invert_delta`` flips it for
    metrics where lower is better) or a short string such as ``"On track"``.
    ``trend`` is a list of numbers for the sparkline; ``progress`` is 0-100.
    """
    if variant not in VARIANTS:
        variant = "default"
    label_text = _clean(label) or "Metric"

    chip = _safe_icon(icon, 18)
    lead = (
        f'<span class="metric-card__chip" aria-hidden="true">{chip}</span>'
        if chip
        else ""
    ) + (f'<p class="metric-card__label">{escape(label_text)}</p>')
    go = '<span class="metric-card__go" aria-hidden="true"></span>' if clickable else ""

    shown = _format_value(value, decimals)
    empty_cls = " metric-card__value--empty" if shown == "—" else ""
    pre = (
        f'<span class="metric-card__affix">{escape(_clean(prefix))}</span>'
        if _clean(prefix)
        else ""
    )
    suf = (
        f'<span class="metric-card__affix">{escape(_clean(suffix))}</span>'
        if _clean(suffix)
        else ""
    )
    pill = _delta_pill(delta, delta_tone, delta_suffix, delta_label, invert_delta)

    parts = [
        f'<div class="metric-card metric-card--{variant}" role="group" aria-label="{escape(label_text, quote=True)}">',
        f'<div class="metric-card__head"><div class="metric-card__lead">{lead}</div>{go}</div>',
        f'<div class="metric-card__body"><p class="metric-card__value{empty_cls}">{pre}{escape(shown)}{suf}</p>{pill}</div>',
    ]
    parts.append(_progress_bar(progress))
    parts.append(sparkline_svg(trend))
    if _clean(caption):
        parts.append(f'<p class="metric-card__caption">{escape(_clean(caption))}</p>')
    parts.append("</div>")
    return "".join(parts)


def render_metric_card(
    label: str,
    value: Any,
    *,
    key: Optional[str] = None,
    on_click: Optional[Callable[..., None]] = None,
    args: Sequence[Any] = (),
    action_label: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Render a metric card. Accepts every option of ``metric_card_html``.

    Give ``key`` (unique per page) and ``on_click`` to make the whole card a
    button, exactly like the dashboard's module cards.
    """
    _styles_once()
    clickable = on_click is not None and bool(key)
    markup = metric_card_html(label, value, clickable=clickable, **kwargs)

    if not clickable:
        st.markdown(markup, unsafe_allow_html=True)
        return

    suffix = "".join(c if c.isalnum() or c == "_" else "_" for c in str(key))
    with st.container(key=f"metric_card_{suffix}"):
        st.markdown(markup, unsafe_allow_html=True)
        st.button(
            action_label or f"Open {_clean(label) or 'metric'}",
            key=f"metric_open_{suffix}",
            on_click=on_click,
            args=tuple(args),
        )
