"""Shared small numeric helpers for the reliability analysis.

Everything here is deterministic: no global random state, caller-supplied seeds.
"""

import math
from .errors import Failure


def log1mexp(log_value):
    """log(1 - exp(log_value)) for log_value <= 0, accurate near both ends."""
    if log_value >= 0:
        raise Failure("numeric", "log1mexp requires a non-positive argument")
    if log_value > -math.log(2.0):
        return math.log(-math.expm1(log_value))
    return math.log1p(-math.exp(log_value))


def finite_float(value, code="numeric", label="value"):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Failure(code, f"{label} must be numeric", **{"label": label})
    value = float(value)
    if not math.isfinite(value):
        raise Failure(code, f"{label} must be finite", **{"label": label})
    return value
