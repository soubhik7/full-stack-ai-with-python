"""
02_calculator_server.py — Multi-Tool MCP Server
================================================
A calculator MCP server demonstrating:
  - Multiple tools with typed parameters
  - Optional parameters with defaults
  - Error handling (division by zero, invalid operations)
  - Returning structured data as JSON strings

Tools exposed:
  add(a, b)           → float
  subtract(a, b)      → float
  multiply(a, b)      → float
  divide(a, b)        → float  (raises on division by zero)
  power(base, exp)    → float
  percentage(value, pct) → float
  stats(numbers)      → dict  (mean, min, max, sum)
"""

import json
import math
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("calculator-server")


# ── Basic Arithmetic ─────────────────────────────────────────────────────────

@mcp.tool()
def add(a: float, b: float) -> str:
    """
    Add two numbers together.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The sum as a string.
    """
    result = a + b
    return f"{a} + {b} = {result}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """
    Subtract b from a.

    Args:
        a: The number to subtract from.
        b: The number to subtract.

    Returns:
        The difference as a string.
    """
    result = a - b
    return f"{a} - {b} = {result}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """
    Multiply two numbers.

    Args:
        a: First number.
        b: Second number.

    Returns:
        The product as a string.
    """
    result = a * b
    return f"{a} × {b} = {result}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """
    Divide a by b.

    Args:
        a: The dividend (number to be divided).
        b: The divisor (number to divide by). Cannot be zero.

    Returns:
        The quotient as a string.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    result = a / b
    return f"{a} ÷ {b} = {result:.6g}"


@mcp.tool()
def power(base: float, exponent: float) -> str:
    """
    Raise base to the power of exponent.

    Args:
        base: The base number.
        exponent: The exponent (can be fractional for roots).

    Returns:
        The result of base^exponent.
    """
    result = math.pow(base, exponent)
    return f"{base}^{exponent} = {result:.6g}"


# ── Advanced Calculations ────────────────────────────────────────────────────

@mcp.tool()
def percentage(value: float, percent: float) -> str:
    """
    Calculate what percent of value equals.

    Example: percentage(200, 15) → "15% of 200 = 30.0"

    Args:
        value: The base number.
        percent: The percentage to calculate (e.g. 15 for 15%).

    Returns:
        The percentage result.
    """
    result = (value * percent) / 100
    return f"{percent}% of {value} = {result}"


@mcp.tool()
def stats(numbers: list[float]) -> str:
    """
    Compute basic statistics for a list of numbers.

    Args:
        numbers: A list of numbers to analyse.

    Returns:
        JSON string with count, sum, mean, min, max, and range.
    """
    if not numbers:
        raise ValueError("Cannot compute stats on an empty list.")

    result = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": sum(numbers) / len(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers),
    }
    return json.dumps(result, indent=2)


@mcp.tool()
def square_root(n: float) -> str:
    """
    Compute the square root of a non-negative number.

    Args:
        n: A non-negative number.

    Returns:
        The square root.
    """
    if n < 0:
        raise ValueError(f"Cannot take square root of negative number: {n}")
    return f"√{n} = {math.sqrt(n):.6g}"


# ── Unit Conversion ───────────────────────────────────────────────────────────

@mcp.tool()
def celsius_to_fahrenheit(celsius: float) -> str:
    """
    Convert temperature from Celsius to Fahrenheit.

    Args:
        celsius: Temperature in Celsius.

    Returns:
        Temperature in Fahrenheit.
    """
    fahrenheit = (celsius * 9 / 5) + 32
    return f"{celsius}°C = {fahrenheit:.1f}°F"


@mcp.tool()
def km_to_miles(km: float) -> str:
    """
    Convert kilometres to miles.

    Args:
        km: Distance in kilometres.

    Returns:
        Distance in miles.
    """
    miles = km * 0.621371
    return f"{km} km = {miles:.4f} miles"


if __name__ == "__main__":
    print("Calculator MCP server starting...")
    print("Tools: add, subtract, multiply, divide, power, percentage,")
    print("       stats, square_root, celsius_to_fahrenheit, km_to_miles")
    mcp.run()
