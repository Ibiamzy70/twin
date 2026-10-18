from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from html import escape
from typing import Any, Callable, Dict, List, Optional, Tuple

import streamlit as st

try:
    from components.icons import icon_data_uri
except (
    ImportError
):  # pragma: no cover - the calculator must still work without icon support

    def icon_data_uri(name: str) -> Optional[str]:  # type: ignore[misc]
        return None


# ---------------------------------------------------------------------
# Logging (production apps shouldn't rely on print(); this goes to the
# server's stdout/log file wherever the app is hosted)
# ---------------------------------------------------------------------
logger = logging.getLogger("twin_calculator")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
    )
    logger.addHandler(_h)

MAX_HISTORY = 50  # the dashboard counts len(calc_history), so this is also its ceiling
DISPLAY_SCI_THRESHOLD = 1e15  # above this magnitude, switch to scientific notation
DEFAULT_PRECISION = 4
MAX_DIGITS = 15  # digits accepted per number while typing


# ---------------------------------------------------------------------
# Calculation logic - kept separate from the UI so it's independently
# testable (e.g. with pytest) without spinning up Streamlit.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Operation:
    symbol: str
    label: str
    glyph: str  # what the user sees in expressions and history
    fn: Callable[[float, float], float]


def _raise_zero_div():
    raise ZeroDivisionError("Cannot divide by zero.")


OPERATIONS = {
    "+": Operation("+", "Add", "+", lambda a, b: a + b),
    "-": Operation("-", "Subtract", "−", lambda a, b: a - b),
    "*": Operation("*", "Multiply", "×", lambda a, b: a * b),
    "/": Operation(
        "/", "Divide", "÷", lambda a, b: a / b if b != 0 else _raise_zero_div()
    ),
    "^": Operation("^", "Power", "^", lambda a, b: a**b),
    "%": Operation(
        "%", "Modulo", "%", lambda a, b: a % b if b != 0 else _raise_zero_div()
    ),
}


class CalculationError(Exception):
    """Raised for a result that computed without a Python exception but
    still isn't something the UI should display as a normal number
    (currently unused directly, but keeps the error hierarchy extensible -
    e.g. a future domain-restricted operation can raise this)."""


def compute(num1: float, num2: float, op_symbol: str):
    """Runs the operation and returns the raw result. Lets the caller
    decide how to format/display it - this function only computes."""
    if op_symbol not in OPERATIONS:
        raise ValueError(f"Unknown operation '{op_symbol}'.")
    return OPERATIONS[op_symbol].fn(num1, num2)


def _trim(text: str) -> str:
    """Drop trailing zeros from a plain decimal string ('261.4000' -> '261.4')."""
    if "." in text and "e" not in text.lower():
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-0") else text


def format_result(result, precision: int = DEFAULT_PRECISION, trim: bool = True) -> str:
    """Turns a raw numeric result into a safe, readable string. Handles the
    edge cases a plain f'{result:,.4f}' would crash or mislead on:
      - complex numbers (e.g. a negative base with a fractional exponent -
        (-8) ** 0.5 does NOT raise in Python, it silently returns a complex
        number; formatting that with :,.4f raises TypeError)
      - inf / -inf / nan
      - very large magnitudes, shown in scientific notation instead of a
        hundreds-of-digits-long string
      - a huge Python int (e.g. 10 ** 400 computed from two ints): the 'f'
        format spec forces a float() conversion even for ints, which raises
        OverflowError for a value that large. st.number_input always yields
        floats so this path isn't hit from the UI, but compute()/
        format_result() are public functions this guards them regardless.

    ``precision`` is the number of decimal places; ``trim`` removes trailing
    zeros so 261.4 is shown as '261.4' rather than '261.4000'.
    """
    precision = max(0, int(precision))

    def fixed(value: float) -> str:
        text = f"{value:,.{precision}f}"
        return _trim(text) if trim else text

    if isinstance(result, complex):
        real = fixed(result.real) if result.real else ""
        imag_sign = "+" if result.imag >= 0 else "-"
        lead = imag_sign if (real or imag_sign == "-") else ""
        return f"{real}{lead}{fixed(abs(result.imag))}i  (complex result)"
    if isinstance(result, float):
        if math.isnan(result):
            return "undefined (NaN)"
        if math.isinf(result):
            return "∞" if result > 0 else "-∞"
        if abs(result) >= DISPLAY_SCI_THRESHOLD:
            return f"{result:.6e}"
        if precision > 0 and result != 0 and abs(result) < 10**-precision:
            return f"{result:.6e}"  # would otherwise round to a misleading 0
        return fixed(result)
    if isinstance(
        result, int
    ):  # bool is an int subclass but never reaches here from OPERATIONS
        if abs(result) >= DISPLAY_SCI_THRESHOLD:
            try:
                return f"{float(result):.6e}"
            except OverflowError:
                # beyond float's ~1.8e308 range entirely (e.g. 10 ** 400):
                # format from the decimal string directly, never via float().
                return _format_big_int_scientific(result)
        return f"{result:,}"
    return str(result)


def _format_big_int_scientific(n: int, significant_digits: int = 6) -> str:
    """Scientific notation for an int too large to ever pass through
    float() -- built from its decimal digits directly, not a float cast."""
    sign = "-" if n < 0 else ""
    digits = str(abs(n))
    exponent = len(digits) - 1
    mantissa = digits[:significant_digits]
    if len(mantissa) > 1:
        mantissa = f"{mantissa[0]}.{mantissa[1:]}"
    return f"{sign}{mantissa}e+{exponent}"


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------
KEY_ENTRY = "calc_entry"  # the number being typed (raw string)
KEY_ACC = "calc_acc"  # left operand once an operator is pressed
KEY_OP = "calc_op"  # pending operator symbol ('+', '-', '*', '/', '^', '%')
KEY_FRESH = "calc_fresh"  # True: the next digit starts a new number
KEY_RESULT = "calc_result"  # raw result shown on the display (or None)
KEY_EXPR = "calc_expr"  # the small expression line above the result
KEY_ERROR = "calc_error"  # error message shown instead of a result
KEY_HISTORY = "calc_history"  # shared with the dashboard (its length is counted)
KEY_PRECISION = "calc_precision"  # sidebar slider

_KEYPAD_DEFAULTS: Dict[str, Callable[[], Any]] = {
    KEY_ENTRY: lambda: "0",
    KEY_ACC: lambda: None,
    KEY_OP: lambda: None,
    KEY_FRESH: lambda: True,
    KEY_RESULT: lambda: None,
    KEY_EXPR: lambda: "",
    KEY_ERROR: lambda: None,
}


def init_calculator_state() -> None:
    """Create any missing keys. Safe on every rerun: nothing is overwritten."""
    for key, factory in _KEYPAD_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = factory()
    if not isinstance(st.session_state.get(KEY_HISTORY), list):
        st.session_state[KEY_HISTORY] = []


def _reset_keypad() -> None:
    for key, factory in _KEYPAD_DEFAULTS.items():
        st.session_state[key] = factory()


def _clock(moment: datetime) -> str:
    """'8:15 AM' - built by hand because '%-I' is not portable to Windows."""
    return f"{moment.hour % 12 or 12}:{moment:%M} {moment:%p}"


def add_to_history(num1: float, op_symbol: str, num2: float, display: str) -> None:
    now = datetime.now()
    history = st.session_state.get(KEY_HISTORY)
    if not isinstance(history, list):
        history = []
    history.insert(
        0,
        {
            "time": _clock(now),
            "expr": f"{_num(num1)} {OPERATIONS[op_symbol].glyph} {_num(num2)}",
            "result": display,
            "created": now,
        },
    )
    st.session_state[KEY_HISTORY] = history[:MAX_HISTORY]


def _clear_history() -> None:
    st.session_state[KEY_HISTORY] = []


# ---------------------------------------------------------------------
# Number helpers
# ---------------------------------------------------------------------
def _num(value: float) -> str:
    """Compact number for expressions and history (256.1, not 256.100000)."""
    return "0" if format(value, ".10g") == "-0" else format(value, ".10g")


def _plain(value: float) -> str:
    """Raw string used to keep editing a result as the next operand."""
    text = format(value, ".15g")
    return "0" if text == "-0" else text


def _to_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _group(entry: str) -> str:
    """Thousands separators for the number being typed ('1234.5' -> '1,234.5')."""
    sign = "-" if entry.startswith("-") else ""
    body = entry.lstrip("-")
    if "e" in body.lower():
        return entry
    head, dot, tail = body.partition(".")
    try:
        return f"{sign}{int(head or '0'):,}{dot}{tail}"
    except ValueError:
        return entry


def _precision() -> int:
    try:
        value = int(st.session_state.get(KEY_PRECISION, DEFAULT_PRECISION))
    except (TypeError, ValueError):
        return DEFAULT_PRECISION
    return max(0, min(8, value))


# ---------------------------------------------------------------------
# Keypad behaviour (button callbacks run before the rerun renders)
# ---------------------------------------------------------------------
def _leave_error() -> None:
    if st.session_state.get(KEY_ERROR):
        _reset_keypad()


def _fail(message: str, expr: str = "") -> None:
    _reset_keypad()
    st.session_state[KEY_ERROR] = message
    st.session_state[KEY_EXPR] = expr


def _press_digit(digit: str) -> None:
    s = st.session_state
    _leave_error()
    if s[KEY_FRESH]:
        s[KEY_ENTRY] = digit
        s[KEY_FRESH] = False
        s[KEY_RESULT] = None
        if s[KEY_OP] is None:
            s[KEY_EXPR] = ""  # a brand-new calculation
        return
    entry = s[KEY_ENTRY]
    if len(entry.replace("-", "").replace(".", "")) >= MAX_DIGITS:
        return
    s[KEY_ENTRY] = digit if entry == "0" else entry + digit


def _press_dot() -> None:
    s = st.session_state
    _leave_error()
    if s[KEY_FRESH]:
        s[KEY_ENTRY] = "0."
        s[KEY_FRESH] = False
        s[KEY_RESULT] = None
        if s[KEY_OP] is None:
            s[KEY_EXPR] = ""
    elif "." not in s[KEY_ENTRY]:
        s[KEY_ENTRY] = s[KEY_ENTRY] + "."


def _press_backspace() -> None:
    s = st.session_state
    if s.get(KEY_ERROR):
        _reset_keypad()
        return
    if s[KEY_FRESH]:
        return  # nothing typed yet (a result is on screen)
    entry = s[KEY_ENTRY][:-1]
    s[KEY_ENTRY] = entry if entry not in ("", "-") else "0"


def _press_clear() -> None:
    _reset_keypad()


def _evaluate() -> Optional[float]:
    """Run the pending operation. Returns the numeric result when it is a finite
    real number (so a chain can continue), otherwise None."""
    s = st.session_state
    op, left = s[KEY_OP], s[KEY_ACC]
    if op is None or left is None:
        return None
    right = _to_float(s[KEY_ENTRY])
    expr = f"{_num(left)} {OPERATIONS[op].glyph} {_num(right)}"

    try:
        result = compute(left, right, op)
        display = format_result(result, _precision())
    except ZeroDivisionError as exc:
        _fail(str(exc) or "Cannot divide by zero.", expr)
        logger.warning("division by zero attempted: %s %s %s", left, op, right)
        return None
    except OverflowError:
        _fail("That result is too large to represent.", expr)
        logger.warning("overflow: %s %s %s", left, op, right)
        return None
    except Exception:  # last-resort guard; still logged for debugging
        _fail("Something went wrong.", expr)
        logger.exception("unexpected error computing %s %s %s", left, op, right)
        return None

    add_to_history(left, op, right, display)
    logger.debug("calculated %s %s %s = %s", left, op, right, display)

    is_real = (
        isinstance(result, (int, float))
        and not isinstance(result, bool)
        and math.isfinite(result)
    )
    s[KEY_RESULT] = result
    s[KEY_EXPR] = expr
    s[KEY_ACC] = None
    s[KEY_OP] = None
    s[KEY_FRESH] = True
    s[KEY_ENTRY] = _plain(float(result)) if is_real else "0"
    return float(result) if is_real else None


def _press_equals() -> None:
    _leave_error()
    if st.session_state[KEY_OP] is not None:
        _evaluate()


def _press_op(op: str) -> None:
    s = st.session_state
    _leave_error()
    glyph = OPERATIONS[op].glyph
    pending = s[KEY_OP]

    if pending is not None and s[KEY_FRESH]:
        # operator pressed twice in a row: just swap it
        s[KEY_OP] = op
        s[KEY_EXPR] = f"{_num(s[KEY_ACC])} {glyph}"
        return

    if pending is not None:
        value = _evaluate()  # chain: 2 + 3 + ... evaluates 2 + 3 first
        if value is None:
            return
    else:
        value = _to_float(s[KEY_ENTRY])

    s[KEY_ACC] = value
    s[KEY_OP] = op
    s[KEY_ENTRY] = _plain(value)
    s[KEY_FRESH] = True
    s[KEY_EXPR] = f"{_num(value)} {glyph}"
    if pending is None:
        s[KEY_RESULT] = None  # in a chain the running total stays on screen


# ---------------------------------------------------------------------
# Keypad layout (DOM order matters: the CSS grid auto-places everything
# except "=" and "x^y", which are pinned explicitly)
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Key:
    id: str
    label: str  # visible text, and the fallback when the icon is missing
    kind: str  # digit | op | fn | eq   (drives the colour)
    handler: Callable[..., None]
    args: Tuple[Any, ...] = ()
    icon: Optional[str] = None  # Phosphor file name in assets/icons/


KEYS: Tuple[Key, ...] = (
    Key("clear", "C", "fn", _press_clear),
    Key("div", "÷", "op", _press_op, ("/",), icon="divide"),
    Key("mul", "×", "op", _press_op, ("*",), icon="x"),
    Key("back", "⌫", "fn", _press_backspace, icon="backspace"),
    Key("7", "7", "digit", _press_digit, ("7",)),
    Key("8", "8", "digit", _press_digit, ("8",)),
    Key("9", "9", "digit", _press_digit, ("9",)),
    Key("sub", "−", "op", _press_op, ("-",), icon="minus"),
    Key("4", "4", "digit", _press_digit, ("4",)),
    Key("5", "5", "digit", _press_digit, ("5",)),
    Key("6", "6", "digit", _press_digit, ("6",)),
    Key("add", "+", "op", _press_op, ("+",), icon="plus"),
    Key("1", "1", "digit", _press_digit, ("1",)),
    Key("2", "2", "digit", _press_digit, ("2",)),
    Key("3", "3", "digit", _press_digit, ("3",)),
    Key("mod", "%", "digit", _press_op, ("%",)),
    Key("0", "0", "digit", _press_digit, ("0",)),
    Key("dot", ".", "digit", _press_dot),
    Key("equals", "=", "eq", _press_equals, icon="equals"),
    Key("pow", "xʸ", "op", _press_op, ("^",)),
)


def _key_name(key: Key) -> str:
    return f"calc_{key.kind}_{key.id}"


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
CALCULATOR_CSS = """
@keyframes calc-rise { from { opacity: 0; transform: translateY(10px); } }
@keyframes calc-pop  { from { opacity: 0.55; transform: translateY(3px); } }

.st-key-calc-root {
  /* local palette from the reference; re-maps the theme accent inside this view */
  --calc-accent: #E6B800;
  --calc-bg: #000000;
  --calc-panel: #0B0B0C;
  --calc-key-digit: #2B2B2E;
  --calc-key-digit-hover: #38383C;
  --calc-key-op: #1C1C1E;
  --calc-key-op-hover: #2A2A2D;
  --calc-item: #141416;
  --calc-line: #000000;
  --calc-muted: #8E8E93;
  --color-accent: var(--calc-accent);
  --color-accent-hover: #F5CB1F;
  --color-accent-active: #C9A100;
  --color-accent-soft: rgba(230, 184, 0, 0.14);
  --color-accent-border: rgba(230, 184, 0, 0.4);
  --color-text-on-accent: #000000;
  --calc-key-size: 1.35rem;

  position: relative;
  max-width: 62rem;
  margin: 0 auto;
  gap: 0 !important;
  overflow: hidden;
  background: var(--calc-bg);
  border: var(--border-width-thick) solid var(--color-border-strong);
  border-radius: var(--radius-xl);
  box-shadow: var(--shadow-lg);
  animation: calc-rise var(--duration-slow) var(--ease-standard) both;
}
.st-key-calc-root [data-testid="stHorizontalBlock"] { gap: 0 !important; align-items: stretch; }
.st-key-calc-root [data-testid="stColumn"],
.st-key-calc-root [data-testid="column"] { min-width: 0; }
.st-key-calc-root [data-testid="stColumn"]:has(.st-key-calc-history),
.st-key-calc-root [data-testid="column"]:has(.st-key-calc-history) {
  background: var(--calc-panel);
  border-left: var(--border);
}

/* ---- Display ----------------------------------------------------------- */
.st-key-calc-main { gap: 0 !important; background: var(--calc-bg); }
.calc-display {
  display: flex; flex-direction: column; min-height: 15rem;
  padding: var(--space-5) var(--space-6) var(--space-6);
  background: var(--calc-bg);
}
.calc-display__title {
  align-self: flex-start;
  color: var(--calc-accent); font-size: var(--font-size-sm); font-weight: var(--font-weight-bold);
  text-decoration: underline; text-underline-offset: 4px;
}
.calc-display__expr {
  margin-top: auto; min-height: 1.75rem; text-align: right;
  color: var(--calc-muted); font-size: var(--font-size-xl); font-variant-numeric: tabular-nums;
  overflow-wrap: anywhere;
}
.calc-display__value {
  text-align: right; color: var(--color-text);
  font-size: clamp(2.5rem, 6vw, 3.75rem); font-weight: var(--font-weight-semibold);
  line-height: var(--line-height-tight); letter-spacing: var(--letter-spacing-tight);
  font-variant-numeric: tabular-nums; overflow-wrap: anywhere;
  animation: calc-pop 140ms var(--ease-standard) both;
}
.calc-display__value.is-long { font-size: clamp(1.75rem, 4vw, 2.5rem); }
.calc-display__value.is-error { color: var(--color-error); font-size: var(--font-size-xl); font-weight: var(--font-weight-medium); }

/* ---- Keypad (CSS grid of st.button widgets) ---------------------------- */
.st-key-calc-keypad {
  display: grid !important;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  grid-template-rows: repeat(5, minmax(4.25rem, 1fr)) 3rem;
  gap: 1px !important;
  background: var(--calc-line);
}
.st-key-calc-keypad > div { display: flex; width: 100% !important; min-width: 0; margin: 0; }
.st-key-calc-keypad [data-testid="stButton"] { display: flex; flex: 1; width: 100% !important; }

.st-key-calc-keypad [data-testid="stButton"] button {
  position: relative; display: flex; flex: 1; align-items: center; justify-content: center;
  width: 100%; min-height: 4.25rem; padding: 0;
  background: var(--calc-key-digit); color: var(--color-text);
  border: 0; border-radius: 0; box-shadow: none;
  font-size: var(--calc-key-size); font-weight: var(--font-weight-medium);
  transition: background-color var(--duration-fast) var(--ease-standard),
              transform var(--duration-fast) var(--ease-standard);
}
.st-key-calc-keypad [data-testid="stButton"] button p {
  margin: 0; color: inherit; font-size: var(--calc-key-size); font-weight: inherit; line-height: 1;
}
.st-key-calc-keypad [data-testid="stButton"] button:hover { background: var(--calc-key-digit-hover); color: var(--color-text); }
.st-key-calc-keypad [data-testid="stButton"] button:active { transform: scale(0.97); }
.st-key-calc-keypad [data-testid="stButton"] button:focus-visible {
  outline: none; box-shadow: inset 0 0 0 2px var(--calc-accent); z-index: 1;
}

.st-key-calc-keypad [class*="st-key-calc_op_"] button,
.st-key-calc-keypad [class*="st-key-calc_fn_"] button { background: var(--calc-key-op); color: var(--calc-accent); }
.st-key-calc-keypad [class*="st-key-calc_op_"] button:hover,
.st-key-calc-keypad [class*="st-key-calc_fn_"] button:hover { background: var(--calc-key-op-hover); color: var(--color-accent-hover); }

.st-key-calc-keypad .st-key-calc_eq_equals {
  grid-column: 4; grid-row: 4 / span 2; align-self: stretch; height: 100%;
}
.st-key-calc-keypad .st-key-calc_eq_equals [data-testid="stButton"],
.st-key-calc-keypad .st-key-calc_eq_equals [data-testid="stButton"] button { height: 100%; }
/* black glyph on yellow: the light theme text colour is only ~1.8:1 on this yellow */
.st-key-calc-keypad .st-key-calc_eq_equals button { background: var(--calc-accent); color: var(--color-text-on-accent); }
.st-key-calc-keypad .st-key-calc_eq_equals button:hover { background: var(--color-accent-hover); color: var(--color-text-on-accent); }
.st-key-calc-keypad .st-key-calc_eq_equals button:active { background: var(--color-accent-active); }

.st-key-calc-keypad .st-key-calc_op_pow { grid-column: 1 / -1; grid-row: 6; }
.st-key-calc-keypad .st-key-calc_op_pow button { min-height: 3rem; }
.st-key-calc-keypad .st-key-calc_op_pow button p { font-size: 1.05rem; }

/* ---- History panel ------------------------------------------------------ */
.st-key-calc-history { padding: var(--space-5); gap: var(--space-3) !important; }
.calc-hist__title { color: var(--color-text); font-size: var(--font-size-md); font-weight: var(--font-weight-semibold); }
.st-key-calc-history [data-testid="stColumn"]:last-child [data-testid="stButton"],
.st-key-calc-history [data-testid="column"]:last-child [data-testid="stButton"] { display: flex; justify-content: flex-end; }

.st-key-calc_clear_history button {
  display: inline-flex; align-items: center; gap: var(--space-1);
  min-height: 1.75rem; padding: 0 var(--space-3);
  background: var(--calc-accent); color: var(--color-text-on-accent);
  border: 0; border-radius: var(--radius-full); box-shadow: none;
  font-size: var(--font-size-xs); font-weight: var(--font-weight-semibold);
  transition: background-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-standard);
}
.st-key-calc_clear_history button p { margin: 0; color: inherit; font-size: inherit; font-weight: inherit; line-height: 1; }
.st-key-calc_clear_history button:hover { background: var(--color-accent-hover); color: var(--color-text-on-accent); }
.st-key-calc_clear_history button:active { transform: scale(0.96); }
.st-key-calc_clear_history button:disabled { background: var(--calc-item); color: var(--calc-muted); opacity: 1; }

.calc-hist {
  list-style: none; margin: 0; padding: 0 var(--space-1) 0 0;
  display: flex; flex-direction: column; gap: var(--space-2);
  max-height: 32rem; overflow-y: auto; overscroll-behavior: auto;
}
.calc-hist__item {
  display: flex; align-items: center; justify-content: space-between; gap: var(--space-3);
  padding: var(--space-3); border-radius: var(--radius-md);
  background: var(--calc-item); border: var(--border-width) solid transparent;
  transition: border-color var(--duration-fast) var(--ease-standard), transform var(--duration-fast) var(--ease-standard);
}
.calc-hist__item:hover { border-color: var(--color-accent-border); transform: translateX(2px); }
.calc-hist__time { flex: none; color: var(--calc-accent); font-size: var(--font-size-xs); white-space: nowrap; }
.calc-hist__calc { display: flex; flex-direction: column; align-items: flex-end; min-width: 0; text-align: right; }
.calc-hist__expr { color: var(--calc-muted); font-size: var(--font-size-xs); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.calc-hist__result { color: var(--color-text); font-size: var(--font-size-sm); font-weight: var(--font-weight-bold); font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.calc-hist__empty { padding: var(--space-6) var(--space-2); color: var(--calc-muted); font-size: var(--font-size-sm); text-align: center; }

@media (max-width: 640px) {
  .st-key-calc-root [data-testid="stHorizontalBlock"] { flex-direction: column; flex-wrap: wrap; }
  .st-key-calc-root [data-testid="stColumn"],
  .st-key-calc-root [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
  /* ...but the History title and Clear button stay side by side */
  .st-key-calc-history [data-testid="stHorizontalBlock"] { flex-direction: row; flex-wrap: nowrap; }
  .st-key-calc-history [data-testid="stColumn"],
  .st-key-calc-history [data-testid="column"] { width: auto !important; flex: 1 1 0 !important; }
  .st-key-calc-keypad { grid-template-rows: repeat(5, minmax(3.75rem, 1fr)) 3rem; }
  .st-key-calc-keypad [data-testid="stButton"] button { min-height: 3.75rem; }
  .st-key-calc-root [data-testid="stColumn"]:has(.st-key-calc-history),
  .st-key-calc-root [data-testid="column"]:has(.st-key-calc-history) { border-left: 0; border-top: var(--border); }
  .calc-display { min-height: 12rem; }
}
"""


@lru_cache(maxsize=1)
def _icon_css() -> str:
    """Swap a key's text glyph for its Phosphor icon when the file is available."""
    rules: List[str] = []
    masked = (
        "content:'';display:block;flex:none;width:1.6rem;height:1.6rem;"
        "background-color:currentColor;"
        "-webkit-mask:url('{uri}') center/contain no-repeat;"
        "mask:url('{uri}') center/contain no-repeat;"
    )
    for key in KEYS:
        if not key.icon:
            continue
        uri = icon_data_uri(key.icon)
        if uri is None:
            continue  # missing asset: the text glyph stays visible
        name = _key_name(key)
        rules.append(
            f".st-key-calc-keypad .st-key-{name} button>div"
            "{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap;}"
        )
        rules.append(
            f".st-key-calc-keypad .st-key-{name} button::before{{{masked.format(uri=uri)}}}"
        )

    clear_uri = icon_data_uri("x")
    if clear_uri:
        rules.append(
            ".st-key-calc_clear_history button::before{"
            "content:'';display:block;flex:none;width:0.7rem;height:0.7rem;background-color:currentColor;"
            f"-webkit-mask:url('{clear_uri}') center/contain no-repeat;"
            f"mask:url('{clear_uri}') center/contain no-repeat;}}"
        )
    return "".join(rules)


def _inject_css() -> None:
    st.markdown(f"<style>{CALCULATOR_CSS}{_icon_css()}</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
def _esc(value: Any) -> str:
    """Escape for inline HTML; also neutralizes markdown math and blank lines."""
    return escape(" ".join(str(value).split())).replace("$", "&#36;")


def _display_html(precision: int) -> str:
    s = st.session_state
    error = s.get(KEY_ERROR)
    result = s.get(KEY_RESULT)

    if error:
        value, css = str(error), "is-error"
    elif result is not None:
        value = format_result(result, precision)
        css = "is-long" if len(value) > 12 else ""
    else:
        value = _group(str(s.get(KEY_ENTRY, "0")))
        css = "is-long" if len(value) > 12 else ""

    return "".join(
        [
            '<div class="calc-display">',
            '<div class="calc-display__title">Calculator</div>',
            f'<div class="calc-display__expr">{_esc(s.get(KEY_EXPR, ""))}</div>',
            f'<div class="calc-display__value {css}" role="status" aria-live="polite">{_esc(value)}</div>',
            "</div>",
        ]
    )


def _history_html(history: List[Dict[str, Any]]) -> str:
    items: List[str] = []
    for row in history:
        if not isinstance(row, dict):
            continue
        items.append(
            "".join(
                [
                    '<li class="calc-hist__item">',
                    f'<span class="calc-hist__time">{_esc(row.get("time", ""))}</span>',
                    '<span class="calc-hist__calc">',
                    f'<span class="calc-hist__expr">{_esc(row.get("expr", ""))}</span>',
                    f'<span class="calc-hist__result">{_esc(row.get("result", ""))}</span>',
                    "</span>",
                    "</li>",
                ]
            )
        )
    if not items:
        return '<div class="calc-hist__empty">No calculations yet.</div>'
    return f'<ul class="calc-hist">{"".join(items)}</ul>'


def _render_settings() -> int:
    with st.sidebar:
        st.markdown(
            '<div class="sidebar-section-label">Calculator</div>',
            unsafe_allow_html=True,
        )
        return int(
            st.slider(
                "Decimal places",
                min_value=0,
                max_value=8,
                value=DEFAULT_PRECISION,
                key=KEY_PRECISION,
            )
        )


def _render_keypad() -> None:
    with st.container(key="calc-keypad"):
        for key in KEYS:
            st.button(
                key.label,
                key=_key_name(key),
                on_click=key.handler,
                args=key.args,
            )


def _render_history() -> None:
    history = st.session_state.get(KEY_HISTORY)
    history = history if isinstance(history, list) else []
    with st.container(key="calc-history"):
        title, action = st.columns([1, 1], vertical_alignment="center")
        title.markdown(
            '<div class="calc-hist__title">History</div>', unsafe_allow_html=True
        )
        action.button(
            "Clear",
            key="calc_clear_history",
            on_click=_clear_history,
            disabled=not history,
        )
        st.markdown(_history_html(history), unsafe_allow_html=True)


def render() -> None:
    """Render the calculator page (called by app.py for the 'calculator' route)."""
    init_calculator_state()
    precision = _render_settings()
    _inject_css()

    with st.container(key="calc-root"):
        main, side = st.columns([5, 2], gap="small")
        with main:
            with st.container(key="calc-main"):
                st.markdown(_display_html(precision), unsafe_allow_html=True)
                _render_keypad()
        with side:
            _render_history()
