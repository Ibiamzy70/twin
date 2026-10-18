"""
Digital Twin · Week 4.1 — Neon Calculator

A production-ready Streamlit calculator: unique widget keys (the bug that
caused StreamlitDuplicateElementId), real input/edge-case handling, a
calculation history, and a small logging setup for debugging in deployment.

Run with:
    streamlit run app.py
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional

import streamlit as st

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

MAX_HISTORY = 10
DISPLAY_SCI_THRESHOLD = 1e15  # above this magnitude, switch to scientific notation


# ---------------------------------------------------------------------
# Calculation logic — kept separate from the UI so it's independently
# testable (e.g. with pytest) without spinning up Streamlit.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class Operation:
    symbol: str
    label: str
    emoji: str
    fn: Callable[[float, float], float]


OPERATIONS = {
    "+": Operation("+", "Add", "➕", lambda a, b: a + b),
    "-": Operation("-", "Subtract", "➖", lambda a, b: a - b),
    "*": Operation("*", "Multiply", "✖️", lambda a, b: a * b),
    "/": Operation(
        "/", "Divide", "➗", lambda a, b: a / b if b != 0 else _raise_zero_div()
    ),
    "^": Operation("^", "Power", "⤴️", lambda a, b: a**b),
    "%": Operation(
        "%", "Modulo", "🔢", lambda a, b: a % b if b != 0 else _raise_zero_div()
    ),
}


def _raise_zero_div():
    raise ZeroDivisionError("Cannot divide by zero.")


class CalculationError(Exception):
    """Raised for a result that computed without a Python exception but
    still isn't something the UI should display as a normal number
    (currently unused directly, but keeps the error hierarchy extensible —
    e.g. a future domain-restricted operation can raise this)."""


def compute(num1: float, num2: float, op_symbol: str):
    """Runs the operation and returns the raw result. Lets the caller
    decide how to format/display it — this function only computes."""
    if op_symbol not in OPERATIONS:
        raise ValueError(f"Unknown operation '{op_symbol}'.")
    return OPERATIONS[op_symbol].fn(num1, num2)


def format_result(result) -> str:
    """Turns a raw numeric result into a safe, readable string. Handles the
    edge cases a plain f'{result:,.4f}' would crash or mislead on:
      - complex numbers (e.g. a negative base with a fractional exponent —
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
    """
    if isinstance(result, complex):
        real = f"{result.real:,.4f}" if result.real else ""
        imag_sign = "+" if result.imag >= 0 else "-"
        return f"{real}{imag_sign}{abs(result.imag):,.4f}i  (complex result)"
    if isinstance(result, float):
        if math.isnan(result):
            return "undefined (NaN)"
        if math.isinf(result):
            return "∞" if result > 0 else "-∞"
        if abs(result) >= DISPLAY_SCI_THRESHOLD:
            return f"{result:.6e}"
        return f"{result:,.4f}"
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
def init_state():
    if "history" not in st.session_state:
        st.session_state.history = []
    if "precision" not in st.session_state:
        st.session_state.precision = 4


def add_to_history(num1: float, op_symbol: str, num2: float, display: str):
    st.session_state.history.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "expr": f"{num1:g} {op_symbol} {num2:g}",
            "result": display,
        },
    )
    st.session_state.history = st.session_state.history[:MAX_HISTORY]


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
<style>
    .stApp { background: radial-gradient(circle at 20% 20%, #1e293b, #0b0f1a); }

    .calc-shell {
        max-width: 480px; margin: 0 auto;
        background: linear-gradient(145deg, #1f2b3b, #151f2c);
        border-radius: 2.5rem; padding: 2rem 1.8rem;
        box-shadow: 0 30px 40px -15px rgba(0,0,0,0.8),
                    inset 0 -4px 8px rgba(0,0,0,0.6),
                    inset 0 4px 6px rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.06);
    }
    .calc-title {
        text-align: center; color: #38bdf8; font-size: 1.8rem; font-weight: 700;
        letter-spacing: 2px; margin-bottom: 0.2rem; text-transform: uppercase;
    }
    .calc-subtitle {
        text-align: center; color: #64748b; font-size: 0.75rem; letter-spacing: 1.5px;
        text-transform: uppercase; margin-bottom: 1.8rem;
    }
    .display-screen {
        background: linear-gradient(180deg, #1b2733 0%, #0e151e 100%);
        border-radius: 1rem; padding: 1.2rem 1rem; margin-bottom: 1.8rem;
        border: 1px solid #2e4052; box-shadow: inset 0 0 10px #00000066; min-height: 110px;
    }
    .display-label {
        font-size: 0.7rem; letter-spacing: 2px; color: #6a7f94;
        text-transform: uppercase; margin-bottom: 0.3rem;
    }
    .display-value {
        font-family: 'Courier New', monospace; font-size: 2.6rem; font-weight: 700;
        color: #8be9fd; text-shadow: 0 0 6px #38bdf8, 0 0 20px #0ea5e9;
        line-height: 1.2; word-break: break-word;
    }
    div[data-testid="stNumberInput"] input {
        background-color: #0e1a24 !important; color: #f1f5f9 !important;
        font-family: 'Courier New', monospace !important; font-size: 1.3rem !important;
        font-weight: 600 !important; border: 1px solid #263846 !important;
        border-radius: 0.8rem !important; padding: 0.6rem 0.8rem !important;
    }
    div[data-testid="stNumberInput"] input:focus {
        border-color: #38bdf8 !important; box-shadow: 0 0 0 2px #38bdf855 !important;
    }
    div[data-testid="stNumberInput"] label {
        color: #7d93a8 !important; font-size: 0.7rem !important; letter-spacing: 1px !important;
        text-transform: uppercase !important; font-weight: 600 !important;
    }
    div[data-testid="stSelectbox"] > div > div {
        background-color: #0e1a24 !important; border: 1px solid #263846 !important;
        border-radius: 0.8rem !important; color: #f1f5f9 !important;
    }
    div[data-testid="stSelectbox"] label {
        color: #7d93a8 !important; font-size: 0.7rem !important; letter-spacing: 1px !important;
        text-transform: uppercase !important; font-weight: 600 !important;
    }
    .stButton > button {
        background: linear-gradient(145deg, #2d4258, #1e2e3e); color: #d5e9ff;
        font-weight: 800; font-size: 1.2rem; letter-spacing: 3px; text-transform: uppercase;
        border-radius: 1.2rem; height: 3.5rem; width: 100%; border: 1px solid #3f5569;
        box-shadow: 0 6px 0 #0e151f, 0 8px 16px #00000090, inset 0 2px 4px #ffffff30;
        transition: all 0.08s ease;
    }
    .stButton > button:hover { background: linear-gradient(145deg, #38536b, #243646); color: #fff; transform: translateY(-1px); }
    .stButton > button:active { transform: translateY(4px); box-shadow: 0 2px 0 #0e151f, 0 4px 8px #00000090, inset 0 2px 8px #00000080; }
    .result-box {
        background: linear-gradient(180deg, #1b2733 0%, #0e151e 100%); border: 1px solid #38bdf8;
        border-radius: 1rem; padding: 1.2rem; text-align: center; margin-top: 1.5rem;
        box-shadow: inset 0 0 10px #00000066;
    }
    .result-number {
        font-family: 'Courier New', monospace; color: #4ade80; font-size: 2.2rem;
        font-weight: 700; text-shadow: 0 0 8px #4ade8088;
    }
    .calc-footer {
        text-align: center; font-size: 0.65rem; color: #5a6d80; letter-spacing: 1.5px;
        text-transform: uppercase; margin-top: 1.5rem;
    }
    .history-row { font-family: 'Courier New', monospace; font-size: 0.85rem; color: #94a3b8; }
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
def render_sidebar():
    with st.sidebar:
        st.markdown("### Settings")
        st.session_state.precision = st.slider(
            "Decimal places",
            min_value=0,
            max_value=8,
            value=st.session_state.precision,
            key="precision_slider",
        )
        st.markdown("### History")
        if st.session_state.history:
            for row in st.session_state.history:
                st.markdown(
                    f'<div class="history-row">{row["time"]} &nbsp; {row["expr"]} = {row["result"]}</div>',
                    unsafe_allow_html=True,
                )
            if st.button("Clear history", key="clear_history_btn"):
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No calculations yet.")


def render_calculator():
    display_placeholder = st.empty()

    col1, col2 = st.columns(2)
    with col1:
        num1 = st.number_input(
            "Number A", value=0.0, step=1.0, format="%.4f", key="num_a"
        )
    with col2:
        num2 = st.number_input(
            "Number B", value=0.0, step=1.0, format="%.4f", key="num_b"
        )

    operation = st.selectbox(
        "Operation",
        options=list(OPERATIONS.keys()),
        format_func=lambda op: f"{OPERATIONS[op].emoji} {OPERATIONS[op].label}",
        key="operation_select",
    )

    calculate_clicked = st.button("Calculate", key="calculate_btn")

    if calculate_clicked:
        try:
            result = compute(num1, num2, operation)
            display = format_result(result)
            display_placeholder.markdown(
                f'<div class="display-screen"><div class="display-label">Result</div>'
                f'<div class="display-value">{display}</div></div>',
                unsafe_allow_html=True,
            )
            add_to_history(num1, operation, num2, display)
            logger.info("calculated %s %s %s = %s", num1, operation, num2, display)

        except ZeroDivisionError as e:
            display_placeholder.empty()
            st.error(f"⚠️ {e}")
            logger.warning(
                "division by zero attempted: %s %s %s", num1, operation, num2
            )
        except OverflowError:
            display_placeholder.empty()
            st.error("⚠️ That result is too large to represent.")
            logger.warning("overflow: %s %s %s", num1, operation, num2)
        except Exception as e:  # last-resort guard; still logged for debugging
            display_placeholder.empty()
            st.error(f"⚠️ Something went wrong: {e}")
            logger.exception(
                "unexpected error computing %s %s %s", num1, operation, num2
            )
    else:
        display_placeholder.markdown(
            '<div class="display-screen"><div class="display-label">Result</div>'
            '<div class="display-value">0.0000</div></div>',
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Twin Calculator", page_icon="🧮", layout="centered")
    init_state()
    inject_css()
    render_sidebar()

    st.markdown('<div class="calc-shell">', unsafe_allow_html=True)
    st.markdown('<div class="calc-title">🧮 Calculator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="calc-subtitle">Twin Calc · X1</div>', unsafe_allow_html=True
    )

    render_calculator()

    st.markdown(
        '<div class="calc-footer">Digital Twin · Week 4.1</div>', unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
