"""Input contract for a life experiment: temperatures, columns and analysis config.

Temperature inputs are always Celsius.  Kelvin is derived with the offset fixed
by the SI definition of the kelvin, ``T_K = T_C + 273.15``.  No rounded 273 is
used anywhere in the analysis.
"""

import math
from ..validation import (
    decimal,
    integer,
    object_value,
    list_value,
    name,
    reject,
)

KELVIN_OFFSET = 273.15
ABSOLUTE_ZERO_C = -273.15

DEFAULT_COLUMNS = {
    "specimen": "specimen",
    "batch": "batch",
    "temperature_c": "temperature_c",
    "entry_day": "entry_day",
    "inspection_day": "inspection_day",
    "result": "result",
    "exact": "exact",
    "removed": "removed",
}
REQUIRED_COLUMNS = ("specimen", "batch", "temperature_c", "entry_day", "inspection_day", "result")
SUMMARY_REQUIRED = ("specimen", "batch", "temperature_c", "entry_day")
OPTIONAL_COLUMNS = ("exact", "removed", "interval_left", "interval_right", "interval_kind")
RESULT_VALUES = {"normal", "failed"}


def celsius_to_kelvin(value):
    """Convert Celsius to kelvin exactly; the offset is 273.15 K."""
    return value + KELVIN_OFFSET


def validate_temperature(value, maximum_c, label="temperature"):
    """Return a problem code for a Celsius temperature or None if usable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "not_numeric"
    value = float(value)
    if not math.isfinite(value):
        return "not_finite"
    if value <= ABSOLUTE_ZERO_C:
        return "at_or_below_absolute_zero"
    if value > maximum_c:
        return "above_configured_maximum"
    return None


def parse_config(data):
    if not isinstance(data, dict):
        reject("input", "life analysis config must be an object")
    if data.get("_parsed_config") is True:
        return data
    allowed = {
        "use_temperature_c",
        "reference_temperature_c",
        "maximum_temperature_c",
        "validation_fraction",
        "validation_batches",
        "bootstrap_replicates",
        "seed",
        "confidence",
        "survival_points",
        "manual_exclusions",
        "columns",
        "input_format",
        "kind_choices",
    }
    object_value(data, allowed, {"use_temperature_c", "seed"})
    use = _number(data["use_temperature_c"], "use_temperature_c")
    maximum = _number(data.get("maximum_temperature_c", 1000.0), "maximum_temperature_c")
    if maximum <= ABSOLUTE_ZERO_C:
        reject("config", "maximum_temperature_c must be above absolute zero")
    reference = _number(data.get("reference_temperature_c", 25.0), "reference_temperature_c")
    for label, value in (
        ("use_temperature_c", use),
        ("reference_temperature_c", reference),
    ):
        problem = validate_temperature(value, maximum, label)
        if problem:
            reject(
                "config",
                "temperature is not physically usable",
                field=label,
                reason=problem,
            )
    fraction = _number(data.get("validation_fraction", 0.3), "validation_fraction")
    if not 0.0 < fraction < 1.0:
        reject("config", "validation_fraction must be strictly between zero and one")
    batches = data.get("validation_batches")
    if batches is not None:
        batches = [
            _text(item, "validation batch")
            for item in list_value(batches, 1, 10000)
        ]
        if len(set(batches)) != len(batches):
            reject("config", "validation batches must be unique")
    replicates = integer(data.get("bootstrap_replicates", 200), 1, 100000)
    seed = integer(data["seed"], 0, 2**63 - 1)
    confidence = _number(data.get("confidence", 0.95), "confidence")
    if not 0.5 < confidence < 0.9999:
        reject("config", "confidence must be strictly between 0.5 and 0.9999")
    points = integer(data.get("survival_points", 101), 11, 2001)
    exclusions = []
    raw_exclusions = data.get("manual_exclusions", [])
    for item in list_value(raw_exclusions, 0, 10000):
        object_value(item, {"specimen", "reason"}, {"specimen", "reason"})
        exclusions.append(
            {
                "specimen": _text(item["specimen"], "excluded specimen"),
                "reason": _reason(item["reason"]),
            }
        )
    if len({item["specimen"] for item in exclusions}) != len(exclusions):
        reject("config", "a specimen may only be manually excluded once")
    columns = dict(DEFAULT_COLUMNS)
    columns.update(
        {"interval_left": None, "interval_right": None, "interval_kind": None}
    )
    supplied = data.get("columns")
    if supplied is not None:
        object_value(supplied, set(DEFAULT_COLUMNS) | {"interval_left", "interval_right", "interval_kind"})
        for role, field in supplied.items():
            if field is None:
                continue
            columns[role] = name(field)
    input_format = data.get("input_format", "inspection_long")
    if input_format not in {"inspection_long", "interval_summary"}:
        reject("config", "input_format must be inspection_long or interval_summary")
    if input_format == "interval_summary":
        columns.update(
            {"inspection_day": None, "result": None, "interval_left": "left_day",
             "interval_right": "right_day", "interval_kind": "kind"}
        )
    elif input_format == "inspection_long":
        columns.update({"interval_left": None, "interval_right": None, "interval_kind": None})
    kind_choices = data.get("kind_choices", {})
    if not isinstance(kind_choices, dict):
        reject("config", "kind_choices must be an object")
    allowed_kind = {"exact", "interval", "right", "right_observed"}
    kind_map = {"exact": "exact", "interval": "interval", "right": "right_observed",
                "ongoing": "right_observed", "right_censored": "right_observed"}
    for key, value in kind_choices.items():
        if not isinstance(key, str) or not key or not isinstance(value, str):
            reject("config", "kind_choices maps bounded text to bounded text")
        if value not in allowed_kind:
            reject("config", "kind choice target must be exact, interval or right")
        kind_map[key] = value
    return {
        "_parsed_config": True,
        "input_format": input_format,
        "kind_choices": kind_map,
        "use_temperature_c": use,
        "reference_temperature_c": reference,
        "maximum_temperature_c": maximum,
        "validation_fraction": fraction,
        "validation_batches": batches,
        "bootstrap_replicates": replicates,
        "seed": seed,
        "confidence": confidence,
        "survival_points": points,
        "manual_exclusions": exclusions,
        "columns": columns,
    }


def check_columns(columns, schema, input_format="inspection_long"):
    """Validate that the configured columns exist and have compatible types."""
    if input_format == "interval_summary":
        required = SUMMARY_REQUIRED + ("interval_left", "interval_kind")
        if columns.get("interval_right") is not None:
            required = required + ("interval_right",)
    else:
        required = REQUIRED_COLUMNS
    for role in required:
        field = schema.by_name.get(columns[role])
        if field is None:
            reject(
                "config",
                "configured input column is absent from dataset contract",
                role=role,
                column=columns[role],
            )
        types = {
            "specimen": {"text"},
            "batch": {"text"},
            "temperature_c": {"decimal", "integer"},
            "entry_day": {"decimal", "integer"},
            "interval_left": {"decimal", "integer"},
            "interval_right": {"decimal", "integer"},
            "interval_kind": {"text", "enum"},
            "inspection_day": {"decimal", "integer"},
            "result": {"text", "enum"},
        }[role]
        if field.type not in types:
            reject(
                "config",
                "configured input column has incompatible type",
                role=role,
                column=field.name,
                type=field.type,
            )
    for role in OPTIONAL_COLUMNS:
        if role not in columns or columns[role] is None:
            continue
        field = schema.by_name.get(columns[role])
        if field is None:
            if role in columns and columns[role] == DEFAULT_COLUMNS[role]:
                columns.pop(role, None)
                continue
            reject(
                "config",
                "configured optional column is absent from dataset contract",
                role=role,
                column=columns[role],
            )
        if role in {"interval_left", "interval_right"}:
            if field.type not in {"decimal", "integer"}:
                reject(
                    "config",
                    "interval bound column must be numeric",
                    role=role,
                    column=field.name,
                )
        elif role == "interval_kind":
            if field.type not in {"text", "enum"}:
                reject(
                    "config",
                    "interval kind column must be text",
                    role=role,
                    column=field.name,
                )
        elif field.type != "boolean":
            reject(
                "config",
                "optional input column must be boolean",
                role=role,
                column=field.name,
            )


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        reject("config", f"{label} must be numeric", field=label)
    value = float(value)
    if not math.isfinite(value):
        reject("config", f"{label} must be finite", field=label)
    return value


def _text(value, label):
    if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
        reject("config", f"{label} must be bounded text")
    return value


def _reason(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        reject("config", "exclusion reason must be bounded text")
    return value.strip()
