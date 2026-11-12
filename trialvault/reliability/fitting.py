"""Null-space identification checks and bounded Nelder-Mead maximisation.

The optimiser is deliberately small and dependency free.  It reports iterations,
evaluations, the achieved gradient measure and why it stopped, so a caller can
tell convergence from exhaustion.  Starting points are chosen on a transformed
scale and every evaluation is guarded against non-finite values.
"""

import math

from .errors import Failure
from .numerics import finite_float


def finite_or_none(value):
    """JSON cannot carry inf/nan; report absence instead of a fake number."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


BOLTZMANN_EV_PER_K = 8.617333262e-5


def parameter_keys(model):
    """The Arrhenius model adds the activation slope; common scale does not."""
    return ("shape", "log_scale") if model == "common_scale_weibull" else (
        "shape",
        "log_scale",
        "slope",
    )


def gradient(observations, model, parameters, reference_k, step=1e-5):
    """Central-difference gradient of the log likelihood in raw parameters."""
    from .likelihood import log_likelihood

    base = log_likelihood(observations, model, parameters, reference_k)
    result = {}
    for key in parameter_keys(model):
        value = parameters[key]
        delta = step * max(1.0, abs(value))
        up = dict(parameters)
        down = dict(parameters)
        up[key] = value + delta
        down[key] = value - delta
        high = log_likelihood(observations, model, up, reference_k)
        low = log_likelihood(observations, model, down, reference_k)
        result[key] = (high - low) / (2.0 * delta)
    return result, base


def identification(observations, model, reference_k, parameters=None):
    """Detect parameters the data cannot identify.

    * no temperature variation means the Arrhenius slope has no support;
    * all observations ending in a censoring state with no failures means the
    * shape cannot be learned from the data;
    * zero-width intervals only contribute exact terms, so they are fine;
    * a single distinct failure time cannot pin both shape and scale on its own,
      but that is a weak-data warning rather than a null direction.
    """
    problems = []
    temperatures = {round(item["temperature_k"], 9) for item in observations}
    failures = [item for item in observations if item["kind"] in {"exact", "interval"}]
    if model == "arrhenius_weibull" and len(temperatures) < 2:
        problems.append(
            {
                "code": "no_temperature_variation",
                "severity": "blocking",
                "message": "Arrhenius slope is not identifiable from a single temperature",
            }
        )
    if not failures:
        problems.append(
            {
                "code": "no_failures",
                "severity": "blocking",
                "message": "only right-censored observations; Weibull shape is not identifiable",
            }
        )
    elif len(failures) < 2:
        problems.append(
            {
                "code": "single_failure",
                "severity": "blocking",
                "message": "one failure cannot identify shape and scale together",
            }
        )
    if parameters is not None:
        # A null direction shows up as a vanishing curvature in one parameter.
        hessian = _curvature(observations, model, parameters, reference_k)
        for key in parameter_keys(model):
            if hessian[key] is not None and abs(hessian[key]) < 1e-8 * max(
                1.0, abs(parameters[key])
            ):
                problems.append(
                    {
                        "code": "flat_" + key,
                        "severity": "blocking",
                        "message": f"likelihood curvature for {key} vanishes",
                        "curvature": hessian[key],
                    }
                )
    return problems


def _curvature(observations, model, parameters, reference_k):
    from .likelihood import log_likelihood

    result = {}
    for key in parameter_keys(model):
        value = parameters[key]
        delta = 1e-4 * max(1.0, abs(value))
        up = dict(parameters)
        down = dict(parameters)
        up[key] = value + delta
        down[key] = value - delta
        try:
            center = log_likelihood(observations, model, parameters, reference_k)
            high = log_likelihood(observations, model, up, reference_k)
            low = log_likelihood(observations, model, down, reference_k)
        except (Failure, OverflowError, ValueError):
            result[key] = None
            continue
        result[key] = (high - 2.0 * center + low) / (delta * delta)
    return result


def maximise(observations, model, reference_k, starts, iterations=2000, tolerance=1e-8):
    """Deterministic bounded Nelder-Mead over the model's free parameters."""
    from .likelihood import log_likelihood

    bounds = {
        "shape": (0.02, 50.0),
        "log_scale": (-30.0, 40.0),
        "slope": (-100000.0, 100000.0),
    }
    keys = parameter_keys(model)
    best = None
    attempts = []
    for start in starts:
        point = {
            key: min(max(float(start[key]), bounds[key][0]), bounds[key][1])
            for key in keys
        }
        outcome = _nelder_mead(
            observations, model, reference_k, point, bounds, iterations, tolerance
        )
        outcome = _pattern_refine(
            observations, model, reference_k, outcome, bounds, tolerance
        )
        attempts.append(
            {
                "start": point,
                "log_likelihood": finite_or_none(outcome["log_likelihood"]),
                "converged": outcome["converged"],
                "reason": outcome["reason"],
            }
        )
        if best is None or outcome["log_likelihood"] > best["log_likelihood"]:
            best = outcome
    if best is None:
        raise Failure("fit", "no optimisation attempt was started")
    gradient_vector, _ = gradient(
        observations, model, best["parameters"], reference_k
    )
    finite_gradient = [value for value in gradient_vector.values() if finite_or_none(value) is not None]
    gradient_norm = max(abs(value) for value in finite_gradient) if finite_gradient else None
    converged = (
        best["converged"]
        and gradient_norm is not None
        and gradient_norm < 1e-3
    )
    return {
        "parameters": best["parameters"],
        "log_likelihood": finite_or_none(best["log_likelihood"]),
        "iterations": best["iterations"],
        "evaluations": best["evaluations"],
        "converged": converged,
        "reason": (
            "gradient small"
            if converged and best["reason"] == "tolerance"
            else best["reason"]
        ),
        "gradient_norm": finite_or_none(gradient_norm),
        "gradient": {key: finite_or_none(value) for key, value in gradient_vector.items()},
        "attempts": attempts,
        "bounds": {key: list(value) for key, value in bounds.items()},
        "free_parameters": list(keys),
    }


def _nelder_mead(observations, model, reference_k, start, bounds, iterations, tolerance):
    from .likelihood import log_likelihood

    keys = parameter_keys(model)
    regulations = {
        "shape": (0.02, 50.0),
        "log_scale": (-30.0, 40.0),
        "slope": (-100000.0, 100000.0),
    }

    def evaluate(point):
        values = []
        for key in keys:
            value = finite_float(point[key], "fit", key)
            low, high = regulations[key]
            if value < low or value > high:
                return -math.inf
            values.append(value)
        params = dict(zip(keys, values))
        try:
            result = log_likelihood(observations, model, params, reference_k)
        except (OverflowError, ValueError, ZeroDivisionError):
            return -math.inf
        if not math.isfinite(result):
            return -math.inf
        return result

    steps = {
        "shape": max(0.15, 0.25 * abs(start["shape"])),
        "log_scale": 0.5,
        "slope": max(2.0, 0.25 * abs(start.get("slope", 0.0))),
    }
    simplex = [dict(start)]
    for key in keys:
        variant = dict(start)
        variant[key] = start[key] + steps[key]
        simplex.append(variant)
    values = [evaluate(point) for point in simplex]
    evaluations = len(simplex)
    converged = False
    reason = "iteration_limit"
    iteration = 0
    for iteration in range(1, iterations + 1):
        order = sorted(range(len(simplex)), key=lambda index: values[index], reverse=True)
        simplex = [simplex[index] for index in order]
        values = [values[index] for index in order]
        if values[0] == -math.inf:
            reason = "all_points_nonfinite"
            break
        spread = max(
            max(abs(simplex[i][key] - simplex[-1][key]) for i in range(len(simplex)))
            for key in keys
        )
        if spread < tolerance:
            converged = True
            reason = "tolerance"
            break
        centroid = {
            key: sum(point[key] for point in simplex[:-1]) / (len(simplex) - 1)
            for key in keys
        }
        reflected = {
            key: centroid[key] + 1.0 * (centroid[key] - simplex[-1][key]) for key in keys
        }
        reflected_value = evaluate(reflected)
        evaluations += 1
        if reflected_value > values[0]:
            expanded = {
                key: centroid[key] + 2.0 * (reflected[key] - centroid[key])
                for key in keys
            }
            expanded_value = evaluate(expanded)
            evaluations += 1
            if expanded_value > reflected_value:
                simplex[-1], values[-1] = expanded, expanded_value
            else:
                simplex[-1], values[-1] = reflected, reflected_value
        elif reflected_value > values[-2]:
            simplex[-1], values[-1] = reflected, reflected_value
        else:
            contracted = {
                key: centroid[key] + 0.5 * (simplex[-1][key] - centroid[key])
                for key in keys
            }
            contracted_value = evaluate(contracted)
            evaluations += 1
            if contracted_value > values[-1]:
                simplex[-1], values[-1] = contracted, contracted_value
            else:
                for index in range(1, len(simplex)):
                    simplex[index] = {
                        key: simplex[0][key]
                        + 0.5 * (simplex[index][key] - simplex[0][key])
                        for key in keys
                    }
                    values[index] = evaluate(simplex[index])
                    evaluations += 1
        if not all(math.isfinite(value) and value > -math.inf for value in values):
            reason = "nonfinite_evaluation"
            break
    best_index = max(range(len(simplex)), key=lambda index: values[index])
    best_point = simplex[best_index]
    return {
        "parameters": {key: best_point[key] for key in keys},
        "log_likelihood": values[best_index],
        "iterations": iteration,
        "evaluations": evaluations,
        "converged": converged,
        "reason": reason,
    }


def _pattern_refine(observations, model, reference_k, outcome, bounds, tolerance):
    """Coordinate pattern search from the simplex result.

    Nelder-Mead can stall on the strongly curved Arrhenius likelihood; a cheap
    axis search with shrinking steps removes that stall without changing the
    objective.  It stops when the step falls below ``tolerance`` or the log
    likelihood stops improving.
    """
    from .likelihood import log_likelihood

    keys = parameter_keys(model)
    parameters = dict(outcome["parameters"])
    value = outcome["log_likelihood"]
    evaluations = outcome["evaluations"]
    step = {
        "shape": max(0.01, 0.05 * abs(parameters["shape"])),
        "log_scale": 0.1,
        "slope": max(0.5, 0.05 * abs(parameters.get("slope", 0.0))),
    }
    for _ in range(400):
        improved = False
        for key in keys:
            for direction in (1.0, -1.0):
                candidate = dict(parameters)
                candidate[key] = parameters[key] + direction * step[key]
                low, high = bounds[key]
                if candidate[key] < low or candidate[key] > high:
                    continue
                try:
                    candidate_value = log_likelihood(
                        observations, model, candidate, reference_k
                    )
                except (OverflowError, ValueError, ZeroDivisionError):
                    candidate_value = -math.inf
                evaluations += 1
                if candidate_value > value:
                    parameters = candidate
                    value = candidate_value
                    improved = True
        if not improved:
            if max(step[key] / max(1.0, abs(parameters[key])) for key in keys) < tolerance:
                break
            for key in keys:
                step[key] *= 0.5
    if value > outcome["log_likelihood"]:
        return {
            **outcome,
            "parameters": parameters,
            "log_likelihood": value,
            "evaluations": evaluations,
            "refined": True,
        }
    return {**outcome, "evaluations": evaluations}


def starts_for(observations, model, reference_k):
    """Several deterministic starting values including small and large shapes."""
    logs = []
    for item in observations:
        if item["kind"] in {"exact", "interval"}:
            if item["tau"] and item["tau"] > 0:
                logs.append(math.log(item["tau"]))
        elif item["lam"] and item["lam"] > 0:
            logs.append(math.log(item["lam"]))
    if not logs:
        raise Failure("fit", "no positive exposure exists in the data")
    center = sum(logs) / len(logs)
    estimates = []
    for shape in (0.5, 1.0, 2.0, 5.0):
        estimates.append(
            {
                "shape": shape,
                "log_scale": center + 0.5772156649 / shape,
                "slope": 0.0,
            }
        )
    return estimates
