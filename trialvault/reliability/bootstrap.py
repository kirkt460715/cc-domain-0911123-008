"""Whole-batch bootstrap for parameter and life-quantile intervals.

Resampling happens at the material-batch level with replacement: every
inspection row and every censored observation of a resampled batch travels
together, so shared lots are never split.  Replicate fits are treated as
reportable outcomes; failed or non-converged replicates are counted and their
codes summarised instead of being dropped silently.
"""

import random

from .errors import Failure
from .fitting import maximise, starts_for, identification
from .likelihood import quantile_time


def _percentile(sorted_values, probability):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = probability * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = position - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def interval(values, confidence):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    alpha = (1.0 - confidence) / 2.0
    return {
        "lower": _percentile(values, alpha),
        "upper": _percentile(values, 1.0 - alpha),
        "replicates": len(values),
        "method": "percentile interval over whole-batch bootstrap replicates",
    }


def bootstrap(
    observations,
    model,
    reference_k,
    use_temperature_c,
    seed,
    replicates,
    confidence,
    survival_times,
):
    batches = sorted({item["batch"] for item in observations})
    grouped = {}
    for observation in observations:
        grouped.setdefault(observation["batch"], []).append(observation)
    generator = random.Random(seed)
    successful = []
    failures = []
    failure_codes = {}
    quantiles = {0.1: [], 0.5: []}
    survival_values = {time: [] for time in survival_times}
    if not batches:
        return {
            "attempted": 0,
            "succeeded": 0,
            "failed": 0,
            "failure_rate": None,
            "failure_codes": {},
            "parameters": {},
            "quantiles": {},
            "survival": {},
            "status": "unavailable",
            "reason": "no batch available to resample",
        }
    for replicate in range(replicates):
        drawn = [generator.choice(batches) for _ in range(len(batches))]
        sample = []
        for batch in drawn:
            for observation in grouped[batch]:
                sample.append(dict(observation))
        try:
            starts = starts_for(sample, model, reference_k)
            fit = maximise(sample, model, reference_k, starts, iterations=800)
            problems = identification(sample, model, reference_k, fit["parameters"])
            if not fit["converged"]:
                raise Failure(
                    "bootstrap",
                    "replicate fit did not converge",
                    reason=fit["reason"],
                    gradient_norm=fit["gradient_norm"],
                )
            if any(item["severity"] == "blocking" for item in problems):
                raise Failure(
                    "bootstrap",
                    "replicate fit is not identifiable",
                    problems=[item["code"] for item in problems],
                )
        except (Failure, OverflowError, ValueError, ZeroDivisionError) as exc:
            code = exc.code if isinstance(exc, Failure) else "numeric"
            failure_codes[code] = failure_codes.get(code, 0) + 1
            failures.append(
                {
                    "replicate": replicate + 1,
                    "code": code,
                    "message": str(exc),
                    "details": exc.details if isinstance(exc, Failure) else {},
                }
            )
            continue
        parameters = fit["parameters"]
        successful.append(
            {
                "replicate": replicate + 1,
                "parameters": parameters,
                "log_likelihood": fit["log_likelihood"],
            }
        )
        for probability in quantiles:
            quantiles[probability].append(
                quantile_time(
                    model,
                    parameters,
                    reference_k,
                    use_temperature_c,
                    probability,
                )
            )
        for time in survival_times:
            survival_values[time].append(
                _survival_at(model, parameters, reference_k, use_temperature_c, time)
            )
    attempted = replicates
    failed = len(failures)
    return {
        "attempted": attempted,
        "succeeded": len(successful),
        "failed": failed,
        "failure_rate": failed / attempted if attempted else None,
        "failure_codes": dict(sorted(failure_codes.items())),
        "failures": failures[:50],
        "failures_truncated": len(failures) > 50,
        "parameters": {
            key: interval(
                [item["parameters"][key] for item in successful], confidence
            )
            for key in ("shape", "log_scale", "slope")
            if all(key in item["parameters"] for item in successful)
        },
        "quantiles": {
            ("B10" if probability == 0.1 else "B50"): interval(values, confidence)
            for probability, values in quantiles.items()
        },
        "survival": {
            "t=" + repr(round(time, 6)): interval(values, confidence)
            for time, values in survival_values.items()
        },
        "resampling": "whole material batch with replacement",
        "seed": seed,
        "status": "ok" if successful else "all_replicates_failed",
    }


def _survival_at(model, parameters, reference_k, temperature_c, time):
    from .likelihood import survival

    return survival([], model, parameters, reference_k, temperature_c, time)
