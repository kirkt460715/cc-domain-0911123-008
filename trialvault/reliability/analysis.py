"""Orchestrate one saved life-experiment analysis on a frozen release version.

The analysis is deterministic for fixed input version, configuration and seed.
It records every row-level decision, the validation split, per-model fit
outcomes, the selected model, the survival curve and life quantiles at the
configured use temperature, and whole-batch bootstrap intervals including the
share of failed replicates.  Restrictions are explicit; nothing is silently
recomputed from a newer dataset version.
"""

import math

from .bootstrap import bootstrap
from .errors import Failure
from .fitting import (
    BOLTZMANN_EV_PER_K,
    finite_or_none,
    identification,
    maximise,
    starts_for,
)
from .likelihood import quantile_time, scale_at, survival
from .observations import build
from .spec import celsius_to_kelvin, check_columns, parse_config
from .validation import (
    assess_temperature,
    batch_assignments,
    compare_models,
    fit_on,
)

ANALYSIS_KIND = "weibull-life-analysis"
ANALYSIS_VERSION = 1
MODELS = ("common_scale_weibull", "arrhenius_weibull")
MODEL_LABELS = {
    "common_scale_weibull": "Weibull with a common scale parameter at every temperature",
    "arrhenius_weibull": "Weibull accelerated life model with an Arrhenius temperature term",
}
BLOCKING_CHECKS = {
    "recovered_after_failure",
    "left_bound_after_right_bound",
    "invalid_temperature",
    "conflicting_duplicate_inspection",
    "invalid_result",
    "metadata_conflict",
    "inspection_before_entry",
    "invalid_entry",
    "nonpositive_exact_time",
}


def run_analysis(rows, schema, config):
    """Pure analysis over canonical rows; no persistence side effects."""
    config = parse_config(config)
    check_columns(config["columns"], schema, config["input_format"])
    observations, diagnostics, decisions = build(
        rows,
        config["columns"],
        config["maximum_temperature_c"],
        config["manual_exclusions"],
        config["kind_choices"],
        config["input_format"],
    )
    reference_c = config["reference_temperature_c"]
    reference_k = celsius_to_kelvin(reference_c)
    use_c = config["use_temperature_c"]

    blocking = [
        item
        for item in diagnostics["checks"]
        if item["code"] in BLOCKING_CHECKS and item["status"] == "present"
    ]
    warnings = [
        item
        for item in diagnostics["checks"]
        if item["code"] not in BLOCKING_CHECKS
        and item["status"] == "present"
        and item["code"] not in {"unused_inspection", "row_marked_removed", "manual_exclusion"}
    ]
    restrictions = []
    if diagnostics["excluded_specimens"]:
        restrictions.append(
            {
                "code": "specimens_excluded_before_fitting",
                "message": "input diagnostics excluded specimens; inspect row-level decisions",
                "excluded_specimens": diagnostics["excluded_specimens"],
                "checks": [item["code"] for item in blocking],
            }
        )
    if warnings:
        restrictions.append(
            {
                "code": "nonblocking_input_diagnostics",
                "message": "input diagnostics were recorded and the affected rows handled explicitly",
                "checks": [item["code"] for item in warnings],
            }
        )

    result = {
        "kind": ANALYSIS_KIND,
        "analysis_version": ANALYSIS_VERSION,
        "status": "deliverable",
        "restrictions": restrictions,
        "config": {key: value for key, value in config.items() if not key.startswith("_")},
        "temperature": {
            "input_unit": "celsius",
            "kelvin_offset": 273.15,
            "kelvin_definition": "T_K = T_C + 273.15 exactly; the offset is never rounded",
            "reference_temperature_c": reference_c,
            "reference_temperature_k": reference_k,
            "use_temperature_c": use_c,
            "use_temperature_k": celsius_to_kelvin(use_c),
            "parameterisation": {
                "common_scale_weibull": {
                    "eta": "eta = exp(log_scale), the same at every temperature",
                    "arrhenius": "not used",
                },
                "arrhenius_weibull": {
                    "z": "z = 1/T_K - 1/T_ref_K, both temperatures in kelvin",
                    "eta": "eta(T) = exp(log_scale + slope * z)",
                    "slope": "slope = Ea/kB in kelvin; Ea = slope * 8.617333262e-5 eV",
                },
            },
        },
        "model_definitions": {
            "common_scale_weibull": MODEL_LABELS["common_scale_weibull"],
            "arrhenius_weibull": MODEL_LABELS["arrhenius_weibull"],
        },
        "input": {
            "rows": diagnostics["rows"],
            "specimens": diagnostics["specimens"],
            "included_specimens": diagnostics["included_specimens"],
            "excluded_specimens": diagnostics["excluded_specimens"],
            "rows_included": diagnostics["rows_included"],
            "rows_excluded": diagnostics["rows_excluded"],
            "observation_kinds": diagnostics["observation_kinds"],
            "checks": diagnostics["checks"],
            "decisions": decisions,
            "observations": observations,
        },
        "diagnostics_status": "blocked" if blocking else ("warnings" if warnings else "ok"),
    }

    if not observations:
        result["status"] = "failed"
        result["restrictions"].append(
            {
                "code": "no_usable_observations",
                "message": "no specimen produced a usable survival observation",
            }
        )
        result["fit"] = {"status": "not_attempted"}
        return result

    batches = sorted({item["batch"] for item in observations})
    split = batch_assignments(
        batches,
        config["validation_fraction"],
        config["seed"],
        config["validation_batches"],
    )
    result["validation_split"] = {
        **split,
        "training_batches": split["train"],
        "validation_batches": split["validation"],
        "batch_isolation": "batches are never split across training and validation",
    }
    temperature = assess_temperature(observations, split, use_c)
    result["temperature_range"] = temperature
    if temperature["extrapolation"]:
        restrictions.append(
            {
                "code": "use_temperature_outside_training_range",
                "message": "use temperature lies outside the training temperature range; survival is an extrapolation",
                "training_range_c": temperature["training_range_c"],
                "use_temperature_c": use_c,
            }
        )

    comparison = compare_models(observations, MODELS, reference_k, split)
    result["model_comparison"] = comparison
    selected = comparison["selected"]
    if selected is None:
        result["status"] = "restricted"
        restrictions.append(
            {
                "code": "model_selection_unavailable",
                "message": "no candidate model could be compared on an isolated validation batch set",
                "reason": comparison.get("reason") or comparison.get("status"),
                "models": comparison["models"],
            }
        )
        result["fit"] = {"status": "not_attempted"}
        problems = [
            item
            for entry in comparison["models"]
            for item in entry.get("identifiability", [])
        ]
        result["identifiability"] = problems
        if any(item["severity"] == "blocking" for item in problems):
            result["status"] = "failed"
            restrictions.append(
                {
                    "code": "parameters_not_identifiable",
                    "message": "no candidate model has identifiable parameters on the training data",
                    "problems": sorted({item["code"] for item in problems}),
                }
            )
        return result
    if selected == "common_scale_weibull":
        restrictions.append(
            {
                "code": "selected_model_has_no_temperature_term",
                "message": "the selected common-scale model carries no temperature effect; the use-temperature result is not temperature-specific",
            }
        )

    training_observations = [
        item for item in observations if item["batch"] in set(split["train"])
    ]
    try:
        training_fit = fit_on(training_observations, selected, reference_k)
    except (Failure, OverflowError, ValueError) as exc:
        result["status"] = "failed"
        restrictions.append(
            {
                "code": "training_fit_failed",
                "message": "the selected model could not be fitted on the training batches",
                "error": _error(exc),
            }
        )
        result["fit"] = {"status": "failed"}
        return result
    result["training_fit"] = _fit_report(
        training_fit, selected, reference_k, use_c, scope="training_batches"
    )
    if not training_fit["converged"]:
        result["status"] = "restricted"
        restrictions.append(
            {
                "code": "training_fit_not_converged",
                "message": "the training fit did not meet the convergence criteria",
                "reason": training_fit["reason"],
                "gradient_norm": training_fit["gradient_norm"],
            }
        )

    # Final reporting fit uses all included observations after model selection
    # has been frozen on the training batches.  No decision above used
    # validation rows; the validation likelihood never feeds an optimiser.
    try:
        final_fit = maximise(
            observations,
            selected,
            reference_k,
            starts_for(observations, selected, reference_k),
        )
    except (Failure, OverflowError, ValueError) as exc:
        result["status"] = "failed"
        restrictions.append(
            {
                "code": "final_fit_failed",
                "message": "the selected model could not be fitted on all included observations",
                "error": _error(exc),
            }
        )
        result["fit"] = {"status": "failed"}
        return result

    problems = identification(
        observations, selected, reference_k, final_fit["parameters"]
    )
    result["identifiability"] = problems
    if any(item["severity"] == "blocking" for item in problems):
        result["status"] = "failed"
        restrictions.append(
            {
                "code": "parameters_not_identifiable",
                "message": "some model parameters are not identifiable from the available data",
                "problems": [item["code"] for item in problems],
            }
        )
        result["fit"] = _fit_report(
            final_fit, selected, reference_k, use_c, scope="all_included"
        )
        return result
    if not final_fit["converged"]:
        result["status"] = "restricted"
        restrictions.append(
            {
                "code": "optimisation_did_not_converge",
                "message": "the optimiser did not meet the convergence criteria",
                "reason": final_fit["reason"],
                "gradient_norm": final_fit["gradient_norm"],
            }
        )

    result["final_refit"] = True
    result["fit"] = _fit_report(
        final_fit, selected, reference_k, use_c, scope="all_included"
    )
    result["estimates"] = _estimates(selected, final_fit["parameters"], reference_k, use_c)
    result["estimates"]["at_tested_temperatures"] = _temperature_quantiles(
        selected,
        final_fit["parameters"],
        reference_k,
        sorted({item["temperature_c"] for item in observations}),
    )
    grid = _time_grid(result["estimates"]["B50_days"], config["survival_points"])
    result["survival_curve"] = {
        "use_temperature_c": use_c,
        "use_temperature_k": celsius_to_kelvin(use_c),
        "model": selected,
        "times_days": grid,
        "survival": [
            survival([], selected, final_fit["parameters"], reference_k, use_c, time)
            for time in grid
        ],
        "definition": "S(t) = exp(-(t / eta(T_use))^shape)",
    }

    boot = bootstrap(
        observations,
        selected,
        reference_k,
        use_c,
        config["seed"],
        config["bootstrap_replicates"],
        config["confidence"],
        grid,
    )
    result["bootstrap"] = boot
    if boot.get("status") != "ok":
        result["status"] = "restricted"
        restrictions.append(
            {
                "code": "bootstrap_all_failed",
                "message": "every whole-batch bootstrap replicate failed; no interval can be delivered",
                "failure_codes": boot.get("failure_codes", {}),
            }
        )
    elif (boot.get("failure_rate") or 0.0) > 0.1:
        result["status"] = "restricted"
        restrictions.append(
            {
                "code": "bootstrap_failure_rate_high",
                "message": "more than ten percent of whole-batch bootstrap replicates failed",
                "failure_rate": boot["failure_rate"],
                "failure_codes": boot["failure_codes"],
            }
        )

    if result["diagnostics_status"] == "blocked" and result["status"] == "deliverable":
        result["status"] = "restricted"
        restrictions.append(
            {
                "code": "blocking_input_diagnostics",
                "message": "blocking input diagnostics were present; affected specimens are excluded",
            }
        )
    if any(
        item["code"]
        in {
            "use_temperature_outside_training_range",
            "selected_model_has_no_temperature_term",
            "training_fit_not_converged",
            "optimisation_did_not_converge",
            "bootstrap_all_failed",
            "bootstrap_failure_rate_high",
        }
        for item in restrictions
    ) and result["status"] == "deliverable":
        result["status"] = "restricted"
    result["restrictions"] = restrictions
    return result


def _error(exc):
    return exc.as_dict() if isinstance(exc, Failure) else {"code": "numeric", "message": str(exc)}


def _fit_report(fit, model, reference_k, use_c, scope):
    parameters = fit["parameters"]
    report = {
        "model": model,
        "scope": scope,
        "parameters": parameters,
        "shape": parameters["shape"],
        "log_scale": parameters["log_scale"],
        "eta_days_at_reference": math.exp(parameters["log_scale"]),
        "reference_temperature_c": float(reference_k) - 273.15,
        "log_likelihood": finite_or_none(fit["log_likelihood"]),
        "converged": fit["converged"],
        "reason": fit["reason"],
        "iterations": fit["iterations"],
        "evaluations": fit["evaluations"],
        "gradient_norm": finite_or_none(fit["gradient_norm"]),
        "attempts": fit["attempts"],
        "optimiser": "bounded Nelder-Mead on (shape, log_scale[, slope])",
    }
    if model == "arrhenius_weibull":
        report["slope_kelvin"] = parameters["slope"]
        report["activation_energy_ev"] = parameters["slope"] * BOLTZMANN_EV_PER_K
    report["eta_days_at_use_temperature"] = scale_at(model, parameters, reference_k, use_c)
    return report


def _estimates(model, parameters, reference_k, use_c):
    return {
        "use_temperature_c": use_c,
        "use_temperature_k": celsius_to_kelvin(use_c),
        "eta_days": scale_at(model, parameters, reference_k, use_c),
        "B10_days": quantile_time(model, parameters, reference_k, use_c, 0.10),
        "B50_days": quantile_time(model, parameters, reference_k, use_c, 0.50),
        "B90_days": quantile_time(model, parameters, reference_k, use_c, 0.90),
        "quantile_definition": "Bp is t with F(t) = p, using the same likelihood model",
    }


def _temperature_quantiles(model, parameters, reference_k, temperatures_c):
    return {
        repr(float(temperature)): {
            "temperature_c": temperature,
            "temperature_k": celsius_to_kelvin(temperature),
            "eta_days": scale_at(model, parameters, reference_k, temperature),
            "B10_days": quantile_time(model, parameters, reference_k, temperature, 0.10),
            "B50_days": quantile_time(model, parameters, reference_k, temperature, 0.50),
        }
        for temperature in temperatures_c
    }


def verify_likelihood_terms():
    """Closed-form likelihood values for independent hand checking.

    Uses shape 2, eta 10 days at the reference temperature and slope 1500 K so
    that every term can be recomputed with a hand calculator.  Observed values
    are asserted in the test suite and exported for the demo.
    """
    reference_k = celsius_to_kelvin(25.0)
    parameters = {"shape": 2.0, "log_scale": math.log(10.0), "slope": 1500.0}
    observations = [
        _hand_observation("exact", 25.0, right=10.0),
        _hand_observation("right", 25.0, left=12.0),
        _hand_observation("interval", 25.0, left=9.0, right=15.0),
        _hand_observation("delayed_exact", 100.0, entry=4.0, right=7.0),
    ]
    terms = []
    for observation in observations:
        terms.append(
            {
                "specimen": observation["specimen"],
                "kind": observation["kind"],
                "temperature_k": observation["temperature_k"],
                "log_likelihood": _hand_term(parameters, reference_k, observation),
                "log_likelihood_delayed_entry": _hand_term(
                    parameters, reference_k, observation
                ),
            }
        )
    return {
        "reference_temperature_k": reference_k,
        "parameters": parameters,
        "hand_check": {
            "exact": "log f(t) = log beta + (beta-1)(log t - log eta) - log eta - (t/eta)^beta",
            "right": "log S(t) = -(t/eta)^beta",
            "interval": "log(S(left)-S(right)); no midpoint substitution",
            "delayed_entry": "t is measured from the specimen entry day",
            "arrhenius": "eta(T) = eta_ref exp(slope (1/T_K - 1/T_ref_K)), T_K = T_C + 273.15",
        },
        "terms": terms,
        "total": sum(item["log_likelihood"] for item in terms),
    }


def _hand_observation(specimen, temperature_c, entry=0.0, left=None, right=None):
    return {
        "specimen": specimen,
        "batch": "hand",
        "kind": "right" if right is None else ("exact" if left is None else "interval"),
        "temperature_c": temperature_c,
        "temperature_k": celsius_to_kelvin(temperature_c),
        "entry_day": entry,
        "left_day": entry if left is None else left,
        "right_day": right,
        "tau": None if right is None else right - entry,
        "lam": (entry if left is None else left) - entry,
    }


def _hand_term(parameters, reference_k, observation):
    beta = parameters["shape"]
    eta = math.exp(
        parameters["log_scale"]
        + parameters["slope"] * (1.0 / observation["temperature_k"] - 1.0 / reference_k)
    )
    if observation["kind"] == "exact":
        return (
            math.log(beta)
            + (beta - 1.0) * (math.log(observation["tau"]) - math.log(eta))
            - math.log(eta)
            - (observation["tau"] / eta) ** beta
        )
    if observation["kind"] == "right":
        return -((observation["lam"] / eta) ** beta)
    return math.log(
        math.exp(-((observation["lam"] / eta) ** beta))
        - math.exp(-((observation["tau"] / eta) ** beta))
    )


def _time_grid(b50, points):
    if b50 is None or b50 <= 0 or not math.isfinite(b50):
        return []
    start = max(0.05, b50 / 500.0)
    stop = max(start * 10.0, b50 * 3.0)
    ratio = (stop / start) ** (1.0 / (points - 1))
    return [start * (ratio**index) for index in range(points)]
