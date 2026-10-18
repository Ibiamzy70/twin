"""Calculator service: the single implementation of the calculator's arithmetic.

``views/calculator.py`` (the keypad UI) and the agent's ``tools.py`` both use this module,
so a calculation gives the same answer, with the same formatting and the same errors,
wherever it is asked for.

* No Streamlit import and no session state: pure functions, easy to test.
* ``compute`` / ``format_result`` / ``OPERATIONS`` / ``Operation`` / ``CalculationError`` are the
  calculator's existing logic, moved here unchanged in behaviour, so the view can import
  them instead of keeping its own copies.
* ``calculate`` and ``evaluate`` are the strict entry points for the agent: they validate
  every input, return a structured ``CalculationResult`` and raise a small set of typed errors.

Supported operators (same as the keypad): ``+  -  *  /  ^ (power)  % (modulo)``.
``%`` is modulo, not "percent".

Numbers are handled as floats so a request can never trigger an enormous integer
computation. Integers beyond 2**53 therefore lose precision.
"""

from __future__ import annotations

import ast
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple, Union

logger = logging.getLogger("digital_twin.calculator_service")

__all__ = [
    "DEFAULT_PRECISION",
    "MAX_PRECISION",
    "DISPLAY_SCI_THRESHOLD",
    "MAX_EXPRESSION_CHARS",
    "Operation",
    "OPERATIONS",
    "CalculationError",
    "InvalidOperatorError",
    "InvalidOperandError",
    "InvalidExpressionError",
    "DivisionByZeroError",
    "ResultTooLargeError",
    "CalculationResult",
    "compute",
    "format_result",
    "format_number",
    "plain_number",
    "format_expression",
    "normalize_operator",
    "parse_operand",
    "calculate",
    "evaluate",
    "list_operations",
]

DISPLAY_SCI_THRESHOLD = 1e15  # above this magnitude, switch to scientific notation
DEFAULT_PRECISION = 4
MAX_PRECISION = 8  # the UI slider's maximum
MAX_EXPRESSION_CHARS = 200
MAX_EXPRESSION_NODES = 60
MAX_OPERAND_TEXT = 40


# ---------------------------------------------------------------------------
# Errors. The subclasses also inherit the builtin the keypad code already catches
# (ZeroDivisionError, OverflowError, ValueError), so existing ``except`` clauses keep working.
# ---------------------------------------------------------------------------
class CalculationError(Exception):
    """Base class for every error raised by this service."""


class InvalidOperatorError(CalculationError, ValueError):
    """The operator is not one of the supported operations."""


class InvalidOperandError(CalculationError, ValueError):
    """An operand (or the precision) is not a usable number."""


class InvalidExpressionError(CalculationError, ValueError):
    """The expression is malformed, too long, or uses something that is not allowed."""


class DivisionByZeroError(CalculationError, ZeroDivisionError):
    """Division, modulo or a negative power of zero."""


class ResultTooLargeError(CalculationError, OverflowError):
    """The result is beyond what a float can represent."""


MSG_ZERO_DIV = "Cannot divide by zero."
MSG_TOO_LARGE = "That result is too large to represent."


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Operation:
    symbol: str
    label: str
    glyph: str  # what the user sees in expressions and history
    fn: Callable[[float, float], float]


def _raise_zero_div():
    raise DivisionByZeroError(MSG_ZERO_DIV)


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

_OPERATOR_ALIASES: Dict[str, str] = {
    "+": "+",
    "plus": "+",
    "add": "+",
    "-": "-",
    "−": "-",
    "–": "-",
    "minus": "-",
    "subtract": "-",
    "*": "*",
    "×": "*",
    "times": "*",
    "multiply": "*",
    "/": "/",
    "÷": "/",
    "divide": "/",
    "divided by": "/",
    "^": "^",
    "**": "^",
    "power": "^",
    "pow": "^",
    "%": "%",
    "mod": "%",
    "modulo": "%",
}


def list_operations() -> Tuple[Dict[str, str], ...]:
    """The supported operations, for building a tool description."""
    return tuple(
        {"symbol": op.symbol, "name": op.label.lower(), "glyph": op.glyph}
        for op in OPERATIONS.values()
    )


def normalize_operator(raw: Any) -> str:
    """Map a symbol, glyph or name ('+', '×', 'times', 'modulo') to the canonical symbol."""
    if isinstance(raw, str):
        found = _OPERATOR_ALIASES.get(" ".join(raw.strip().lower().split()))
        if found:
            return found
    supported = ", ".join(OPERATIONS)
    raise InvalidOperatorError(f"Unknown operation {raw!r}. Supported: {supported}.")


def compute(num1: float, num2: float, op_symbol: str):
    """Runs the operation and returns the raw result. Lets the caller
    decide how to format/display it - this function only computes."""
    if op_symbol not in OPERATIONS:
        raise InvalidOperatorError(f"Unknown operation '{op_symbol}'.")
    return OPERATIONS[op_symbol].fn(num1, num2)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
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


def format_number(value: float) -> str:
    """Compact number for expressions and history (256.1, not 256.100000)."""
    return "0" if format(value, ".10g") == "-0" else format(value, ".10g")


def plain_number(value: float) -> str:
    """Raw string used to keep editing a result as the next operand."""
    text = format(value, ".15g")
    return "0" if text == "-0" else text


def format_expression(left: float, op_symbol: str, right: float) -> str:
    """'256.1 + 5.3' with the display glyph for the operator (× ÷ −)."""
    return f"{format_number(left)} {OPERATIONS[normalize_operator(op_symbol)].glyph} {format_number(right)}"


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_NUMBER_TEXT = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def parse_operand(value: Any, name: str = "value") -> float:
    """Return a finite float from an int, float or plain numeric string.

    Rejects booleans, None, NaN, infinities, and text such as '1_000', '0x10' or '1,5'.
    """
    if isinstance(value, bool) or value is None:
        raise InvalidOperandError(f"{name} must be a number.")
    try:
        if isinstance(value, (int, float)):
            number = float(value)
        elif isinstance(value, str):
            text = value.strip()
            if not text or len(text) > MAX_OPERAND_TEXT or not _NUMBER_TEXT.match(text):
                raise InvalidOperandError(f"{name} must be a number.")
            number = float(text)
        else:
            raise InvalidOperandError(f"{name} must be a number.")
    except OverflowError:
        raise InvalidOperandError(f"{name} is too large.") from None
    if not math.isfinite(number):
        raise InvalidOperandError(f"{name} must be a finite number.")
    return number


def _check_precision(precision: Any) -> int:
    if (
        isinstance(precision, bool)
        or not isinstance(precision, int)
        or not 0 <= precision <= MAX_PRECISION
    ):
        raise InvalidOperandError(
            f"precision must be a whole number from 0 to {MAX_PRECISION}."
        )
    return precision


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CalculationResult:
    """A validated calculation.

    expression  readable form, e.g. '256.1 + 5.3'
    value       the raw result (float, or complex for e.g. (-8) ^ 0.5)
    display     ``format_result(value, precision)``: what the UI shows
    kind        real | complex | infinite | undefined
    operator    canonical symbol for a single-operation ``calculate`` (else None)
    left/right  the parsed operands for ``calculate`` (else None)
    """

    expression: str
    value: Union[float, complex]
    display: str
    kind: str
    operator: Optional[str] = None
    left: Optional[float] = None
    right: Optional[float] = None

    @property
    def is_real(self) -> bool:
        """True for a finite real number (a chain can continue from it)."""
        return self.kind == "real"

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for the agent (never contains NaN or Infinity)."""
        data: Dict[str, Any] = {
            "expression": self.expression,
            "result": self.display,
            "kind": self.kind,
            "value": None,
        }
        if isinstance(self.value, complex):
            data["real"] = self.value.real if math.isfinite(self.value.real) else None
            data["imag"] = self.value.imag if math.isfinite(self.value.imag) else None
        elif self.kind == "real":
            data["value"] = self.value
        return data


def _kind(value: Any) -> str:
    if isinstance(value, complex):
        return "complex"
    if isinstance(value, float):
        if math.isnan(value):
            return "undefined"
        if math.isinf(value):
            return "infinite"
    return "real"


def _make_result(
    expression: str, value: Any, precision: int, **extra: Any
) -> CalculationResult:
    return CalculationResult(
        expression=expression,
        value=value,
        display=format_result(value, precision),
        kind=_kind(value),
        **extra,
    )


def _guarded(fn: Callable[[], Any]) -> Any:
    """Run a computation and translate Python's arithmetic errors into ours."""
    try:
        return fn()
    except CalculationError:
        raise
    except ZeroDivisionError:
        raise DivisionByZeroError(MSG_ZERO_DIV) from None
    except OverflowError:
        raise ResultTooLargeError(MSG_TOO_LARGE) from None
    except TypeError:
        raise InvalidExpressionError(
            "That operation is not defined for complex numbers."
        ) from None


# ---------------------------------------------------------------------------
# calculate(): one operation
# ---------------------------------------------------------------------------
def calculate(
    left: Any,
    operator: Any,
    right: Any,
    *,
    precision: int = DEFAULT_PRECISION,
) -> CalculationResult:
    """Validate and run ``left <operator> right``.

    left / right   numbers, or plain numeric strings ('256.1'); finite only
    operator       + - * / ^ %  (also × ÷ − ** and names such as 'times', 'modulo')
    precision      decimal places for ``display`` (0-8)

    Raises ``InvalidOperandError``, ``InvalidOperatorError``, ``DivisionByZeroError`` or
    ``ResultTooLargeError`` (all ``CalculationError``).
    """
    a = parse_operand(left, "left")
    b = parse_operand(right, "right")
    symbol = normalize_operator(operator)
    places = _check_precision(precision)
    value = _guarded(lambda: compute(a, b, symbol))
    logger.debug("calculate %s %s %s", a, symbol, b)
    return _make_result(
        format_expression(a, symbol, b), value, places, operator=symbol, left=a, right=b
    )


# ---------------------------------------------------------------------------
# evaluate(): a whole expression, parsed safely (no eval)
# ---------------------------------------------------------------------------
_ALLOWED_CHARS = re.compile(r"^[0-9eE+\-*/%^().\s×÷−–]+$")
_BIN_SYMBOL = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
    ast.Pow: "^",
}


def _prepare(text: str) -> str:
    for old, new in (("×", "*"), ("÷", "/"), ("−", "-"), ("–", "-")):
        text = text.replace(old, new)
    return text.replace("^", "**")


def _eval_node(node: ast.AST) -> Any:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        try:
            number = float(node.value)
        except OverflowError:
            number = math.inf
        if not math.isfinite(number):  # e.g. 1e999 parses to inf without raising
            raise InvalidOperandError("A number in the expression is too large.")
        return number
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        operand = _eval_node(node.operand)
        return -operand if isinstance(node.op, ast.USub) else operand
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_SYMBOL:
        left, right = _eval_node(node.left), _eval_node(node.right)
        return _guarded(lambda: compute(left, right, _BIN_SYMBOL[type(node.op)]))
    raise InvalidExpressionError(
        "Only numbers, + - * / % ^ and parentheses are allowed."
    )


def evaluate(
    expression: Any, *, precision: int = DEFAULT_PRECISION
) -> CalculationResult:
    """Evaluate an arithmetic expression such as ``(256.1 + 5.3) * 2``.

    Allowed: numbers (including 1e3), ``+ - * / % ^`` (``**`` too), unary +/-, parentheses.
    Everything else is rejected before parsing; the text is never passed to ``eval``.
    Raises ``InvalidExpressionError`` (and the same errors as ``calculate``).
    """
    if not isinstance(expression, str):
        raise InvalidExpressionError("The expression must be text.")
    text = " ".join(expression.split())
    if not text:
        raise InvalidExpressionError("The expression is empty.")
    if len(text) > MAX_EXPRESSION_CHARS:
        raise InvalidExpressionError(
            f"The expression is too long (maximum {MAX_EXPRESSION_CHARS} characters)."
        )
    if not _ALLOWED_CHARS.match(text):
        raise InvalidExpressionError(
            "Only numbers, + - * / % ^ and parentheses are allowed."
        )
    places = _check_precision(precision)

    try:
        tree = ast.parse(_prepare(text), mode="eval")
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        raise InvalidExpressionError(
            "That is not a valid arithmetic expression."
        ) from None
    if sum(1 for _ in ast.walk(tree)) > MAX_EXPRESSION_NODES:
        raise InvalidExpressionError("The expression is too complex.")

    try:
        value = _eval_node(tree)
    except RecursionError:
        raise InvalidExpressionError("The expression is too complex.") from None
    logger.debug("evaluate %d chars", len(text))
    return _make_result(text, value, places)
