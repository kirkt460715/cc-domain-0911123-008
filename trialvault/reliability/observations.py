"""Convert long-format inspection records into censored survival observations.

The input is one row per inspection: specimen, batch, Celsius temperature,
entry day and inspection day with a ``normal`` or ``failed`` result.  A failure
is only known to lie between the last normal inspection and the first failing
inspection, unless the input explicitly marks the failing inspection as exact.

Every row receives an inclusion decision before any model is fitted.  Problems
never silently change a boundary or a day; affected specimens are excluded with
a diagnostic code, and the decision is preserved for export.
"""

import math

from .errors import Failure
from .spec import celsius_to_kelvin, validate_temperature

RESULT_VALUES = {"normal", "failed"}

CHECK_ORDER = (
    ("inspection_out_of_order", "inspections were not listed in chronological order"),
    ("duplicate_inspection", "the same specimen/day/result was inspected twice"),
    ("conflicting_duplicate_inspection", "same specimen/day has conflicting results"),
    ("recovered_after_failure", "a normal inspection follows a failing inspection"),
    ("inspection_after_failure", "an inspection follows the first failing inspection"),
    ("invalid_temperature", "Celsius temperature is not physically usable"),
    ("inspection_before_entry", "inspection day precedes the specimen entry day"),
    ("left_bound_after_right_bound", "last normal inspection is later than first failure"),
    ("nonpositive_exact_time", "exact failure time is not after the study origin"),
    ("invalid_entry", "entry day is not a finite non-negative number"),
    ("invalid_result", "inspection result is neither normal nor failed"),
    ("metadata_conflict", "specimen metadata differs between inspection rows"),
    ("no_follow_up", "no inspection strictly after entry, so nothing is observed"),
    ("row_marked_removed", "input row is marked removed and was not used"),
    ("manual_exclusion", "specimen was excluded by analysis configuration"),
    ("failure_without_prior_inspection", "failure observed with no earlier normal inspection"),
    ("zero_width_interval", "interval endpoints coincide and were treated as exact"),
)
CHECK_CODES = {code for code, _ in CHECK_ORDER}


def _parse_number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parse_flag(row, column):
    if column not in row:
        return None
    value = row[column]
    return value if isinstance(value, bool) else None


def _decision(row_index, specimen, batch, day, result, code, message, included):
    return {
        "row": row_index,
        "specimen": specimen,
        "batch": batch,
        "inspection_day": _text(day),
        "result": result,
        "decision": "included" if included else "excluded",
        "code": code,
        "message": message,
    }


def _text(value):
    if value is None:
        return None
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return repr(value)
    return str(value)


def build_summary(rows, columns, maximum_temperature_c, manual_exclusions, kind_choices):
    """Build observations from one interval-summary row per specimen.

    Summary fields: left_day is the last known-normal inspection, right_day the
    first known-failed inspection (absent for a still-running specimen), and
    kind names ``exact``, ``interval`` or a right-censoring label.  Unknown kind
    labels are rejected; they are never silently coerced to a censoring type.
    """
    default_choices = {
        "exact": "exact",
        "interval": "interval",
        "right": "right_observed",
        "ongoing": "right_observed",
        "right_censored": "right_observed",
    }
    choices = dict(default_choices)
    choices.update(kind_choices)
    manual = {item["specimen"]: item["reason"] for item in manual_exclusions or []}
    decisions = []
    checks = {code: [] for code in CHECK_CODES}
    observations = []
    excluded = {}

    def note(code, row, specimen, message, batch=None, day=None, result=None):
        checks[code].append(
            {
                "row": row,
                "specimen": specimen,
                "batch": batch,
                "inspection_day": _text(day),
                "result": result,
                "message": message,
            }
        )

    def decision(row, specimen, batch, left, right, label, code, message, included):
        decisions.append(
            {
                "row": row,
                "specimen": specimen,
                "batch": batch,
                "inspection_day": _text(left) if left is not None else None,
                "result": label,
                "right_day": _text(right),
                "decision": "included" if included else "excluded",
                "code": code,
                "message": message,
            }
        )

    seen = {}
    for index, row in enumerate(rows, 1):
        specimen = row.get(columns["specimen"])
        batch = row.get(columns["batch"])
        removed = _parse_flag(row, columns.get("removed"))
        left = _parse_number(row.get(columns["interval_left"]))
        right = _parse_number(row.get(columns["interval_right"]))
        label = row.get(columns.get("interval_kind"))
        if specimen in seen:
            note(
                "duplicate_inspection",
                index,
                specimen,
                "interval summary repeats a specimen that already has a row",
                batch,
                left,
                label,
            )
            excluded.setdefault(
                specimen,
                ("duplicate_inspection", "specimen has more than one summary row"),
            )
        else:
            seen[specimen] = index
        if removed is True:
            note(
                "row_marked_removed",
                index,
                specimen,
                "row carries the removed marker",
                batch,
                left,
                label,
            )
            decision(
                index,
                specimen,
                batch,
                left,
                right,
                label,
                "row_marked_removed",
                "row is marked removed in the input",
                False,
            )
            excluded.setdefault(specimen, ("row_marked_removed", "row is marked removed"))
            continue
        if not isinstance(label, str) or label not in choices:
            problem = "invalid_result"
            note(
                problem,
                index,
                specimen,
                f"unknown interval kind label: {label!r}",
                batch,
                left,
                label,
            )
            excluded.setdefault(specimen, (problem, "unknown interval kind label"))
            continue
        target = choices[label]
        temperature = _parse_number(row.get(columns["temperature_c"]))
        entry = _parse_number(row.get(columns["entry_day"]))
        if entry is None or entry < 0:
            note(
                "invalid_entry",
                index,
                specimen,
                "entry day is not finite and non-negative",
                batch,
                left,
                label,
            )
            excluded.setdefault(specimen, ("invalid_entry", "entry day is invalid"))
            continue
        problem = (
            "invalid_temperature"
            if temperature is None
            else validate_temperature(temperature, maximum_temperature_c)
        )
        if problem is not None:
            note(
                "invalid_temperature",
                index,
                specimen,
                "Celsius temperature is not usable",
                batch,
                left,
                label,
            )
            excluded.setdefault(specimen, ("invalid_temperature", "temperature is unusable"))
            continue
        if left is None or left < entry:
            note(
                "inspection_before_entry",
                index,
                specimen,
                "left bound is missing or precedes entry",
                batch,
                left,
                label,
            )
            excluded.setdefault(
                specimen, ("inspection_before_entry", "left bound is missing or before entry")
            )
            continue
        if target == "exact":
            if right is None or right != left:
                note(
                    "left_bound_after_right_bound",
                    index,
                    specimen,
                    "exact row endpoints must coincide",
                    batch,
                    left,
                    label,
                )
                excluded.setdefault(
                    specimen, ("left_bound_after_right_bound", "exact endpoints differ")
                )
                continue
            if right - entry <= 0:
                note(
                    "nonpositive_exact_time",
                    index,
                    specimen,
                    "exact failure time is not after entry",
                    batch,
                    right,
                    label,
                )
                excluded.setdefault(
                    specimen, ("nonpositive_exact_time", "exact time is not after entry")
                )
                continue
            kind = "exact"
            right_value = right
            ongoing = False
        elif target == "interval":
            if right is None:
                note(
                    "left_bound_after_right_bound",
                    index,
                    specimen,
                    "interval row has no right bound",
                    batch,
                    left,
                    label,
                )
                excluded.setdefault(
                    specimen, ("left_bound_after_right_bound", "interval has no right bound")
                )
                continue
            if right < left:
                note(
                    "left_bound_after_right_bound",
                    index,
                    specimen,
                    "left bound is later than right bound",
                    batch,
                    right,
                    label,
                )
                excluded.setdefault(
                    specimen, ("left_bound_after_right_bound", "left bound later than right")
                )
                continue
            if right - entry <= 0:
                note(
                    "nonpositive_exact_time",
                    index,
                    specimen,
                    "interval right bound is not after entry",
                    batch,
                    right,
                    label,
                )
                excluded.setdefault(
                    specimen, ("nonpositive_exact_time", "right bound is not after entry")
                )
                continue
            kind = "interval"
            right_value = right
            ongoing = False
        else:
            if right is not None:
                note(
                    "metadata_conflict",
                    index,
                    specimen,
                    "a right-censored row also carries a right bound; right bound ignored",
                    batch,
                    left,
                    label,
                )
                excluded.setdefault(
                    specimen,
                    ("metadata_conflict", "right bound present on right-censored row"),
                )
                continue
            if left - entry <= 0:
                note(
                    "no_follow_up",
                    index,
                    specimen,
                    "right-censored observation ends at entry",
                    batch,
                    left,
                    label,
                )
                excluded.setdefault(
                    specimen, ("no_follow_up", "right-censored observation ends at entry")
                )
                continue
            kind = "right"
            right_value = None
            ongoing = True
        if specimen in excluded:
            continue
        if specimen in manual:
            note(
                "manual_exclusion",
                index,
                specimen,
                "manual exclusion: " + manual[specimen],
                batch,
                left,
                label,
            )
            excluded[specimen] = ("manual_exclusion", "manual exclusion")
            continue
        observations.append(
            {
                "specimen": specimen,
                "batch": batch,
                "temperature_c": temperature,
                "temperature_k": celsius_to_kelvin(temperature),
                "entry_day": entry,
                "delayed_entry": entry > 0,
                "kind": kind,
                "left_day": left,
                "right_day": right_value,
                "tau": None if right_value is None else right_value - entry,
                "lam": left - entry,
                "ongoing": ongoing,
                "exact_declared": kind == "exact",
                "row_indices": [index],
                "rows_inspected": 1,
                "summary_label": label,
            }
        )
        decision(
            index,
            specimen,
            batch,
            left,
            right_value,
            label,
            None,
            "frozen interval-summary observation",
            True,
        )

    for specimen, (code, message) in sorted(excluded.items()):
        for index, row in enumerate(rows, 1):
            if row.get(columns["specimen"]) != specimen:
                continue
            already = any(item["row"] == index for item in decisions)
            if already:
                continue
            decision(
                index,
                specimen,
                row.get(columns["batch"]),
                _parse_number(row.get(columns["interval_left"])),
                _parse_number(row.get(columns["interval_right"])),
                row.get(columns.get("interval_kind")),
                code,
                message,
                False,
            )
    observations = [
        item for item in observations if item["specimen"] not in excluded
    ]
    decisions.sort(key=lambda item: item["row"])
    counts = {
        "exact": sum(item["kind"] == "exact" for item in observations),
        "interval": sum(item["kind"] == "interval" for item in observations),
        "right": sum(item["kind"] == "right" for item in observations),
        "delayed_entry": sum(item["delayed_entry"] for item in observations),
        "ongoing": sum(item["ongoing"] for item in observations),
    }
    check_report = []
    for code, description in CHECK_ORDER:
        items = checks[code]
        check_report.append(
            {
                "code": code,
                "description": description,
                "status": "present" if items else "absent",
                "count": len(items),
                "specimens": sorted({item["specimen"] for item in items})[:20],
                "rows": sorted({item["row"] for item in items})[:20],
                "truncated": len(items) > 20,
            }
        )
    diagnostics = {
        "rows": len(rows),
        "specimens": len(seen),
        "included_specimens": len(observations),
        "excluded_specimens": len(seen) - len(observations),
        "rows_included": sum(item["decision"] == "included" for item in decisions),
        "rows_excluded": sum(item["decision"] == "excluded" for item in decisions),
        "observation_kinds": counts,
        "checks": check_report,
    }
    return observations, diagnostics, decisions


def build(
    rows,
    columns,
    maximum_temperature_c,
    manual_exclusions=None,
    kind_choices=None,
    input_format="inspection_long",
):
    """Return (observations, diagnostics, decisions).

    ``rows`` are canonical dataset rows.  ``manual_exclusions`` is a list of
    ``{"specimen": ..., "reason": ...}`` from the frozen analysis configuration.
    Long-format inspections are normalised here; pre-formed interval summaries
    are routed through ``build_summary`` so both share one observation schema.
    """
    if input_format == "interval_summary":
        return build_summary(
            rows,
            columns,
            maximum_temperature_c,
            manual_exclusions,
            kind_choices or {},
        )
    manual = {item["specimen"]: item["reason"] for item in manual_exclusions or []}
    decisions = []
    checks = {code: [] for code in CHECK_CODES}
    groups = {}
    excluded_specimens = {}

    def note(code, row, specimen, message, batch=None, day=None, result=None):
        checks[code].append(
            {
                "row": row,
                "specimen": specimen,
                "batch": batch,
                "inspection_day": _text(day),
                "result": result,
                "message": message,
            }
        )

    def exclude(specimen, code, message):
        excluded_specimens.setdefault(specimen, (code, message))

    # Index every row.  Rows explicitly marked removed by the data feed are
    # dropped before any observation is formed, but the decision is retained.
    for index, row in enumerate(rows, 1):
        specimen = row.get(columns["specimen"])
        batch = row.get(columns["batch"])
        removed = _parse_flag(row, columns.get("removed"))
        day = _parse_number(row.get(columns["inspection_day"]))
        result = row.get(columns["result"])
        if removed is True:
            note(
                "row_marked_removed",
                index,
                specimen,
                "row carries the removed marker",
                batch,
                day,
                result,
            )
            decisions.append(
                _decision(
                    index,
                    specimen,
                    batch,
                    day,
                    result,
                    "row_marked_removed",
                    "row is marked removed in the input",
                    False,
                )
            )
            continue
        groups.setdefault(specimen, []).append(
            {
                "row": index,
                "specimen": specimen,
                "batch": batch,
                "temperature": row.get(columns["temperature_c"]),
                "entry": row.get(columns["entry_day"]),
                "day": day,
                "day_raw": row.get(columns["inspection_day"]),
                "result": result,
                "exact": _parse_flag(row, columns.get("exact")),
            }
        )

    for specimen, reason in sorted(manual.items()):
        if specimen not in groups:
            raise Failure(
                "config",
                "manual exclusion names a specimen absent from the input version",
                specimen=specimen,
            )
        exclude(specimen, "manual_exclusion", "manual exclusion: " + reason)
        for record in groups[specimen]:
            note(
                "manual_exclusion",
                record["row"],
                specimen,
                "manual exclusion: " + reason,
                record["batch"],
                record["day"],
                record["result"],
            )

    observations = []
    for specimen, records in sorted(groups.items()):
        batch = records[0]["batch"]
        if specimen in excluded_specimens:
            code, message = excluded_specimens[specimen]
            for record in records:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        code,
                        message,
                        False,
                    )
                )
            continue

        # Metadata must be constant for the specimen; otherwise its censoring
        # boundaries cannot be attributed to one temperature or one entry time.
        temperatures = {_text(_parse_number(item["temperature"])) for item in records}
        entries = {_text(_parse_number(item["entry"])) for item in records}
        batches = {item["batch"] for item in records}
        if len(temperatures) != 1 or len(entries) != 1 or len(batches) != 1:
            exclude(
                specimen,
                "metadata_conflict",
                "batch, temperature or entry day differs between rows",
            )
            for record in records:
                note(
                    "metadata_conflict",
                    record["row"],
                    specimen,
                    "specimen metadata differs between inspection rows",
                    record["batch"],
                    record["day"],
                    record["result"],
                )
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "metadata_conflict",
                        "specimen metadata differs between inspection rows",
                        False,
                    )
                )
            continue

        temperature = _parse_number(records[0]["temperature"])
        entry = _parse_number(records[0]["entry"])
        if entry is None or entry < 0:
            problem = "invalid_entry"
            exclude(specimen, problem, "entry day is not finite and non-negative")
            for record in records:
                note(
                    problem,
                    record["row"],
                    specimen,
                    "entry day is not finite and non-negative",
                    record["batch"],
                    record["day"],
                    record["result"],
                )
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        problem,
                        "entry day is not finite and non-negative",
                        False,
                    )
                )
            continue
        problem = (
            None
            if temperature is None
            else validate_temperature(temperature, maximum_temperature_c)
        )
        if temperature is None:
            problem = "invalid_temperature"
        if problem is not None:
            exclude(specimen, "invalid_temperature", f"temperature: {problem}")
            for record in records:
                note(
                    "invalid_temperature",
                    record["row"],
                    specimen,
                    f"Celsius temperature is not usable: {problem}",
                    record["batch"],
                    record["day"],
                    record["result"],
                )
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "invalid_temperature",
                        f"Celsius temperature is not usable: {problem}",
                        False,
                    )
                )
            continue

        # An inspection day must be a finite number at or after entry.
        timed = []
        for record in records:
            if record["result"] not in RESULT_VALUES:
                note(
                    "invalid_result",
                    record["row"],
                    specimen,
                    "result is neither normal nor failed",
                    record["batch"],
                    record["day"],
                    record["result"],
                )
                exclude(
                    specimen,
                    "invalid_result",
                    "inspection result is neither normal nor failed",
                )
            if record["day"] is None:
                note(
                    "inspection_before_entry",
                    record["row"],
                    specimen,
                    "inspection day is not a finite number",
                    record["batch"],
                    record["day_raw"],
                    record["result"],
                )
                exclude(
                    specimen,
                    "inspection_before_entry",
                    "inspection day is not a finite number",
                )
                continue
            if record["day"] < entry:
                note(
                    "inspection_before_entry",
                    record["row"],
                    specimen,
                    "inspection day precedes entry day",
                    record["batch"],
                    record["day"],
                    record["result"],
                )
                exclude(
                    specimen,
                    "inspection_before_entry",
                    "inspection day precedes entry day",
                )
            timed.append(record)
        if specimen in excluded_specimens:
            code, message = excluded_specimens[specimen]
            for record in records:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        code,
                        message,
                        False,
                    )
                )
            continue

        input_order = [record["day"] for record in records]
        if any(
            input_order[index] < input_order[index - 1]
            for index in range(1, len(input_order))
        ):
            for record in records:
                note(
                    "inspection_out_of_order",
                    record["row"],
                    specimen,
                    "rows for this specimen were listed out of chronological order",
                    record["batch"],
                    record["day"],
                    record["result"],
                )
        timed.sort(key=lambda item: (item["day"], item["row"]))
        if not timed:
            exclude(specimen, "no_follow_up", "no usable inspection remains")
            for record in records:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "no_follow_up",
                        "no usable inspection remains",
                        False,
                    )
                )
            continue

        # Repeated inspections at one day: identical repeats are dropped with a
        # diagnostic; conflicting repeats make the specimen unusable.
        by_day = {}
        for record in timed:
            by_day.setdefault(record["day"], []).append(record)
        duplicates = set()
        conflict = False
        exact_conflict = False
        for day, repeated in sorted(by_day.items()):
            if len(repeated) == 1:
                continue
            signatures = {
                (item["result"], item["exact"] is True) for item in repeated
            }
            if len(signatures) != 1:
                conflict = True
                for item in repeated:
                    note(
                        "conflicting_duplicate_inspection",
                        item["row"],
                        specimen,
                        "same inspection day has conflicting result or exactness",
                        item["batch"],
                        day,
                        item["result"],
                    )
            else:
                for item in repeated[1:]:
                    duplicates.add(item["row"])
                    note(
                        "duplicate_inspection",
                        item["row"],
                        specimen,
                        "identical inspection repeats an earlier row for this day",
                        item["batch"],
                        day,
                        item["result"],
                    )
        if conflict:
            exclude(
                specimen,
                "conflicting_duplicate_inspection",
                "same day has conflicting inspection results",
            )
            for record in records:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "conflicting_duplicate_inspection",
                        "same day has conflicting inspection results",
                        False,
                    )
                )
            continue
        unique = [item for item in timed if item["row"] not in duplicates]
        if any(
            item["exact"] is True and item["result"] != "failed" for item in unique
        ):
            exact_conflict = True
            for item in unique:
                if item["exact"] is True and item["result"] != "failed":
                    note(
                        "invalid_result",
                        item["row"],
                        specimen,
                        "exact marker is only valid on a failing inspection",
                        item["batch"],
                        item["day"],
                        item["result"],
                    )
            exclude(
                specimen,
                "invalid_result",
                "exact marker appears on a normal inspection",
            )
            for record in records:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "invalid_result",
                        "exact marker appears on a normal inspection",
                        False,
                    )
                )
            continue

        exact_flag = any(item["exact"] is True for item in unique)
        failures = [index for index, item in enumerate(unique) if item["result"] == "failed"]
        if failures:
            first = failures[0]
            after = unique[first + 1 :]
            recovered = [item for item in after if item["result"] == "normal"]
            if recovered:
                exclude(
                    specimen,
                    "recovered_after_failure",
                    "normal inspection follows a failing inspection",
                )
                for item in recovered:
                    note(
                        "recovered_after_failure",
                        item["row"],
                        specimen,
                        "normal inspection follows a failing inspection",
                        item["batch"],
                        item["day"],
                        item["result"],
                    )
            elif after:
                # Repeated failures add no information beyond the first one.  The
                # specimen stays usable, but the redundant rows are flagged.
                for item in after:
                    note(
                        "inspection_after_failure",
                        item["row"],
                        specimen,
                        "inspection follows the first failing inspection and adds no information",
                        item["batch"],
                        item["day"],
                        item["result"],
                    )
            failure = unique[first]
            right = failure["day"]
            prior = [item for item in unique[:first] if item["result"] == "normal"]
            if not prior:
                note(
                    "failure_without_prior_inspection",
                    failure["row"],
                    specimen,
                    "first inspection already shows failure; interval starts at entry",
                    failure["batch"],
                    right,
                    failure["result"],
                )
            left = max(entry, max((item["day"] for item in prior), default=entry))
            if left > right:
                exclude(
                    specimen,
                    "left_bound_after_right_bound",
                    "last normal inspection is later than first failure",
                )
                note(
                    "left_bound_after_right_bound",
                    failure["row"],
                    specimen,
                    "last normal inspection is later than first failure",
                    failure["batch"],
                    failure["day"],
                    failure["result"],
                )
                for record in records:
                    decisions.append(
                        _decision(
                            record["row"],
                            specimen,
                            record["batch"],
                            record["day"],
                            record["result"],
                            "left_bound_after_right_bound",
                            "last normal inspection is later than first failure",
                            False,
                        )
                    )
                continue
            if exact_flag or left == right:
                kind = "exact"
                if not exact_flag:
                    note(
                        "zero_width_interval",
                        failure["row"],
                        specimen,
                        "interval endpoints coincide; treated as an exact failure",
                        failure["batch"],
                        right,
                        failure["result"],
                    )
                if right <= entry:
                    exclude(
                        specimen,
                        "nonpositive_exact_time",
                        "exact failure time is not after the specimen entry",
                    )
                    note(
                        "nonpositive_exact_time",
                        failure["row"],
                        specimen,
                        "exact failure time is not after the specimen entry",
                        failure["batch"],
                        right,
                        failure["result"],
                    )
                    for record in records:
                        decisions.append(
                            _decision(
                                record["row"],
                                specimen,
                                record["batch"],
                                record["day"],
                                record["result"],
                                "nonpositive_exact_time",
                                "exact failure time is not after the specimen entry",
                                False,
                            )
                        )
                    continue
                left = right
                used = unique[: first + 1]
            else:
                kind = "interval"
                used = unique[: first + 1]
            ongoing = False
        else:
            normal = [item for item in unique if item["result"] == "normal"]
            if not normal:
                exclude(specimen, "no_follow_up", "no normal inspection remains")
                for record in records:
                    decisions.append(
                        _decision(
                            record["row"],
                            specimen,
                            record["batch"],
                            record["day"],
                            record["result"],
                            "no_follow_up",
                            "no normal inspection remains",
                            False,
                        )
                    )
                continue
            left = max(item["day"] for item in normal)
            right = None
            if left <= entry:
                exclude(
                    specimen,
                    "no_follow_up",
                    "latest normal inspection is at the entry day",
                )
                for record in records:
                    decisions.append(
                        _decision(
                            record["row"],
                            specimen,
                            record["batch"],
                            record["day"],
                            record["result"],
                            "no_follow_up",
                            "latest normal inspection is at the entry day",
                            False,
                        )
                    )
                continue
            kind = "right"
            used = unique
            ongoing = True

        if specimen in excluded_specimens:
            code, message = excluded_specimens[specimen]
            for record in records:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        code,
                        message,
                        False,
                    )
                )
            continue

        used_rows = {item["row"] for item in used}
        for record in records:
            if record["row"] in used_rows:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        None,
                        "supports the frozen censoring observation",
                        True,
                    )
                )
            elif record["row"] in duplicates:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "duplicate_inspection",
                        "identical repeat at the same inspection day",
                        False,
                    )
                )
            else:
                decisions.append(
                    _decision(
                        record["row"],
                        specimen,
                        record["batch"],
                        record["day"],
                        record["result"],
                        "unused_inspection",
                        "inspection is outside the failure window",
                        False,
                    )
                )
        observations.append(
            {
                "specimen": specimen,
                "batch": batch,
                "temperature_c": temperature,
                "temperature_k": celsius_to_kelvin(temperature),
                "entry_day": entry,
                "delayed_entry": entry > 0,
                "kind": kind,
                "left_day": left,
                "right_day": right,
                "tau": None if right is None else right - entry,
                "lam": left - entry,
                "ongoing": ongoing,
                "exact_declared": bool(exact_flag),
                "row_indices": sorted(used_rows),
                "rows_inspected": len(unique),
            }
        )

    decided = {decision["row"] for decision in decisions}
    for record_group in groups.values():
        for record in record_group:
            if record["row"] not in decided:
                decisions.append(
                    _decision(
                        record["row"],
                        record["specimen"],
                        record.get("batch"),
                        record.get("day"),
                        record.get("result"),
                        "unused_inspection",
                        "row was not used by the frozen observation",
                        False,
                    )
                )
    decisions.sort(key=lambda item: item["row"])

    counts = {
        "exact": sum(item["kind"] == "exact" for item in observations),
        "interval": sum(item["kind"] == "interval" for item in observations),
        "right": sum(item["kind"] == "right" for item in observations),
        "delayed_entry": sum(item["delayed_entry"] for item in observations),
        "ongoing": sum(item["ongoing"] for item in observations),
    }
    check_report = []
    for code, description in CHECK_ORDER:
        items = checks[code]
        check_report.append(
            {
                "code": code,
                "description": description,
                "status": "present" if items else "absent",
                "count": len(items),
                "specimens": sorted({item["specimen"] for item in items})[:20],
                "rows": sorted({item["row"] for item in items})[:20],
                "truncated": len(items) > 20,
            }
        )
    diagnostics = {
        "rows": len(rows),
        "specimens": len(groups),
        "included_specimens": len(observations),
        "excluded_specimens": len(groups) - len(observations),
        "rows_included": sum(item["decision"] == "included" for item in decisions),
        "rows_excluded": sum(item["decision"] == "excluded" for item in decisions),
        "observation_kinds": counts,
        "checks": check_report,
    }
    return observations, diagnostics, decisions
