"""Tests for services/calculator_service.py (no Streamlit). Run: python -m pytest tests -q"""

from __future__ import annotations

import json
import math
import time

import pytest

from services import calculator_service as cs


# -- the existing calculator behaviour must be unchanged ----------------------------
@pytest.mark.parametrize(
    "value,expected",
    [
        (261.4, "261.4"),
        (1234567.891, "1,234,567.891"),
        (0.0, "0"),
        (-0.0, "0"),
        (2.0, "2"),
        (1e15, "1.000000e+15"),
        (1e-6, "1.000000e-06"),
        (float("inf"), "∞"),
        (float("-inf"), "-∞"),
        (float("nan"), "undefined (NaN)"),
        (1024, "1,024"),
        (10**400, "1.00000e+400"),
        (complex(0, 2), "2i  (complex result)"),
        (complex(1.5, -2), "1.5-2i  (complex result)"),
    ],
)
def test_format_result(value, expected):
    assert cs.format_result(value) == expected


def test_format_result_precision_and_trim():
    assert cs.format_result(2 / 3, 2) == "0.67"
    assert cs.format_result(261.4, 4, trim=False) == "261.4000"
    assert cs.format_result(2.5, 0) == "2"


def test_compute_and_operations():
    assert cs.compute(256.1, 5.3, "+") == pytest.approx(261.4)
    assert cs.compute(2, 10, "^") == 1024
    assert cs.compute(7, 3, "%") == 1
    assert cs.OPERATIONS["*"].glyph == "×" and cs.OPERATIONS["/"].glyph == "÷"
    with pytest.raises(ZeroDivisionError, match="Cannot divide by zero"):
        cs.compute(1, 0, "/")
    with pytest.raises(ZeroDivisionError):
        cs.compute(1, 0, "%")
    with pytest.raises(ValueError):
        cs.compute(1, 2, "?")


def test_number_helpers():
    assert cs.format_number(256.1) == "256.1" and cs.format_number(-0.0) == "0"
    assert cs.plain_number(0.1 + 0.2) == "0.3"
    assert cs.format_expression(8, "/", 2) == "8 ÷ 2"


# -- calculate ----------------------------------------------------------------------
def test_calculate_basic():
    r = cs.calculate("256.1", "+", 5.3)
    assert r.display == "261.4" and r.expression == "256.1 + 5.3" and r.is_real
    assert r.operator == "+" and r.left == 256.1 and r.right == 5.3
    assert cs.calculate(2, "^", 10).display == "1,024"
    assert cs.calculate(10, "/", 4, precision=0).display == "2"


@pytest.mark.parametrize(
    "alias,symbol",
    [
        ("plus", "+"),
        ("−", "-"),
        ("×", "*"),
        ("Times", "*"),
        ("÷", "/"),
        ("divided by", "/"),
        ("**", "^"),
        ("Modulo", "%"),
        (" mod ", "%"),
    ],
)
def test_operator_aliases(alias, symbol):
    assert cs.normalize_operator(alias) == symbol


@pytest.mark.parametrize("bad", ["", "x", "&&", None, 5, "drop table"])
def test_bad_operator(bad):
    with pytest.raises(cs.InvalidOperatorError):
        cs.calculate(1, bad, 2)


@pytest.mark.parametrize(
    "bad",
    [
        None,
        True,
        "",
        "abc",
        "1_000",
        "0x10",
        "1,5",
        "nan",
        "inf",
        "-Infinity",
        float("nan"),
        float("inf"),
        [1],
        {"a": 1},
        10**400,
        "9" * 41,
    ],
)
def test_bad_operands(bad):
    with pytest.raises(cs.InvalidOperandError):
        cs.calculate(bad, "+", 1)
    with pytest.raises(cs.InvalidOperandError):
        cs.calculate(1, "+", bad)


@pytest.mark.parametrize(
    "good,val", [("  3.5 ", 3.5), (".5", 0.5), ("-2e3", -2000.0), ("+7", 7.0), (4, 4.0)]
)
def test_good_operands(good, val):
    assert cs.parse_operand(good) == val


def test_errors_are_typed_and_compatible():
    with pytest.raises(cs.DivisionByZeroError) as e1:
        cs.calculate(1, "/", 0)
    assert isinstance(e1.value, ZeroDivisionError) and "divide by zero" in str(e1.value)
    with pytest.raises(cs.DivisionByZeroError):
        cs.calculate(0, "^", -1)
    with pytest.raises(cs.ResultTooLargeError) as e2:
        cs.calculate(10, "^", 1000)
    assert isinstance(e2.value, OverflowError)
    for exc in (
        cs.InvalidOperatorError,
        cs.InvalidOperandError,
        cs.InvalidExpressionError,
    ):
        assert issubclass(exc, cs.CalculationError) and issubclass(exc, ValueError)


def test_precision_validation():
    for bad in (-1, 9, 2.5, True, "2", None):
        with pytest.raises(cs.InvalidOperandError):
            cs.calculate(1, "+", 1, precision=bad)


def test_complex_and_infinite_results():
    r = cs.calculate(-8, "^", 0.5)
    assert r.kind == "complex" and "complex result" in r.display and not r.is_real
    d = r.to_dict()
    assert d["value"] is None and set(d) >= {"real", "imag"}
    big = cs.calculate(1e308, "*", 10)
    assert (
        big.kind == "infinite" and big.display == "∞" and big.to_dict()["value"] is None
    )


def test_to_dict_is_strict_json():
    for result in (
        cs.calculate(1, "+", 2),
        cs.calculate(-8, "^", 0.5),
        cs.calculate(1e308, "*", 10),
        cs.evaluate("(1+2)*3"),
    ):
        json.dumps(result.to_dict(), allow_nan=False)
    assert cs.calculate(1, "+", 2).to_dict() == {
        "expression": "1 + 2",
        "result": "3",
        "kind": "real",
        "value": 3.0,
    }


# -- evaluate -----------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("1 + 2 * 3", 7.0),
        ("(256.1 + 5.3) * 2", 522.8),
        ("2 ^ 3 ^ 2", 512.0),
        ("2 ** 10", 1024.0),
        ("-3 + 5", 2.0),
        ("--3", 3.0),
        ("10 % 4", 2.0),
        ("8 ÷ 2 × 3 − 1", 11.0),
        ("1e3 + 1", 1001.0),
        ("  7  ", 7.0),
    ],
)
def test_evaluate(text, expected):
    assert cs.evaluate(text).value == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        None,
        5,
        "abc",
        "__import__('os')",
        "1; 2",
        "2 +",
        "(1",
        "1 2",
        "1,000",
        "sqrt(4)",
        "x + 1",
        "True",
        "1 if 1 else 2",
        "[1]",
        "'a'",
        "1_0",
        "1 // 2",
        "~1",
        "1 << 2",
        "5!",
        "1.2.3",
        "e",
        "1 + e",
        "0x10",
        "1 & 2",
        "a.b",
        "lambda: 1",
        "10%",
        "💥",
    ],
)
def test_evaluate_rejects_everything_else(text):
    with pytest.raises(cs.InvalidExpressionError):
        cs.evaluate(text)


def test_evaluate_limits_and_errors():
    with pytest.raises(cs.InvalidExpressionError):
        cs.evaluate("1+" * 101 + "1")  # too long
    with pytest.raises(cs.InvalidExpressionError):
        cs.evaluate("+".join(["1"] * 61))  # too many nodes
    with pytest.raises(cs.InvalidExpressionError):
        cs.evaluate("(" * 150 + "1" + ")" * 150)
    with pytest.raises(cs.DivisionByZeroError):
        cs.evaluate("1 / (2 - 2)")
    with pytest.raises(cs.ResultTooLargeError):
        cs.evaluate("9 ^ 9 ^ 9")
    with pytest.raises(cs.InvalidOperandError):
        cs.evaluate("1e999 + 1")
    with pytest.raises(cs.InvalidExpressionError):
        cs.evaluate("((-8) ^ 0.5) % 2")  # complex has no modulo
    assert cs.evaluate("(-8) ^ 0.5").kind == "complex"
    with pytest.raises(cs.InvalidOperandError):
        cs.evaluate("1 + 1", precision=99)


def test_no_dos_on_huge_power():
    start = time.perf_counter()
    for text in ("9 ^ 9 ^ 9 ^ 9", "2 ^ 100000000", "10 ** 10 ** 10"):
        with pytest.raises(cs.CalculationError):
            cs.evaluate(text)
    with pytest.raises(cs.CalculationError):
        cs.calculate(10**300, "^", 10**300)
    assert time.perf_counter() - start < 1.0


def test_list_operations():
    ops = cs.list_operations()
    assert [o["symbol"] for o in ops] == ["+", "-", "*", "/", "^", "%"]
    assert all(set(o) == {"symbol", "name", "glyph"} for o in ops)
    assert math.isclose(
        cs.calculate(0.1, "+", 0.2, precision=8).value, 0.30000000000000004
    )
