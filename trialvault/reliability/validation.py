"""Batch-isolated validation likelihood and model selection.

Batches are grouped, never split within a batch: a specimen never appears in
both the training and validation side, which keeps repeat inspections and
shared material lots isolated.  Validation batches are selected deterministically
from the seed and may also be pinned in the configuration.

Only the training portion is used for fitting, for holding out and for the
temperature-range decision.  The validation log likelihood is evaluated with the
training parameters and is never optimised on validation rows.
"""

import hashlib

from .errors import Failure
from .fitting import finite_or_none, identification, maximise, starts_for
from .likelihood import log_likelihood


def batch_assignments(batches, fraction, seed, pinned=None):
    batches = sorted(set(batches))
    if len(batches) < 2:
        return {
            "train": batches,
            "validation": [],
            "selection": "unavailable_single_batch",
        }
    if pinned:
        unknown = sorted(set(pinned) - set(batches))
        if unknown:
            raise Failure(
                "config",
                "pinned validation batches are absent",
                batches=unknown,
            )
        if len(pinned) >= len(batches):
            raise Failure(
                "config",
                "pinned validation batches leave no training batch",
            )
        validation = sorted(set(pinned))
        selection = "pinned"
    else:
        count = int(round(fraction * len(batches)))
        count = max(1, min(len(batches) - 1, count))
        ranked = sorted(
            batches,
            key=lambda batch: (
                hashlib.sha256((str(seed) + "|" + batch).encode()).hexdigest(),
                batch,
            ),
        )
        validation = sorted(ranked[:count])
        selection = "deterministic hash of seed and batch identifier"
    return {
        "train": [batch for batch in batches if batch not in validation],
        "validation": validation,
        "selection": selection,
    }


def fit_on(observations, model, reference_k):
    starts = starts_for(observations, model, reference_k)
    return maximise(observations, model, reference_k, starts)


def assess_temperature(dataset_observations, split, use_c):
    """Extrapolation uses the training-only range, never the validation rows."""
    training = [
        item
        for item in dataset_observations
        if item["batch"] in set(split["train"])
    ]
    if not training:
        return {
            "status": "unknown",
            "training_range_c": None,
            "use_temperature_c": use_c,
            "extrapolation": True,
            "message": "no training observation exists",
        }
    low = min(item["temperature_c"] for item in training)
    high = max(item["temperature_c"] for item in training)
    extrapolation = use_c < low or use_c > high
    return {
        "status": "extrapolation" if extrapolation else "interpolation",
        "training_range_c": {"minimum": low, "maximum": high},
        "validation_temperature_range_c": None,
        "use_temperature_c": use_c,
        "extrapolation": extrapolation,
        "message": (
            "use temperature lies outside the training temperature range"
            if extrapolation
            else "use temperature lies inside the training temperature range"
        ),
    }


def compare_models(observations, models, reference_k, split):
    training = [
        item for item in observations if item["batch"] in set(split["train"])
    ]
    validation = [
        item for item in observations if item["batch"] in set(split["validation"])
    ]
    if not validation:
        return {
            "status": "unavailable",
            "reason": "validation split has no held-out batches",
            "models": [],
            "selected": None,
        }
    results = []
    for model in models:
        entry = {
            "model": model,
            "training_observations": len(training),
            "training_failures": sum(
                item["kind"] in {"exact", "interval"} for item in training
            ),
            "validation_observations": len(validation),
            "validation_failures": sum(
                item["kind"] in {"exact", "interval"} for item in validation
            ),
            "fit": None,
            "validation_log_likelihood": None,
            "validation_mean_log_likelihood": None,
            "selectable": False,
        }
        try:
            fit = fit_on(training, model, reference_k)
        except Failure as exc:
            entry["status"] = "training_fit_failed"
            entry["failure"] = exc.as_dict()
            results.append(entry)
            continue
        entry["fit"] = _fit_summary(fit)
        if not fit["converged"]:
            entry["status"] = "training_fit_not_converged"
            results.append(entry)
            continue
        problems = identification(training, model, reference_k, fit["parameters"])
        if any(item["severity"] == "blocking" for item in problems):
            entry["status"] = "training_parameters_not_identifiable"
            entry["identifiability"] = problems
            results.append(entry)
            continue
        try:
            value = log_likelihood(validation, model, fit["parameters"], reference_k)
        except (Failure, OverflowError, ValueError):
            value = -float("inf")
        if finite_or_none(value) is None:
            entry["status"] = "validation_likelihood_failed"
            results.append(entry)
            continue
        entry["validation_log_likelihood"] = finite_or_none(value)
        entry["validation_mean_log_likelihood"] = finite_or_none(value / len(validation))
        entry["selectable"] = True
        entry["status"] = "ok"
        results.append(entry)
    selectable = [entry for entry in results if entry["selectable"]]
    if not selectable:
        return {
            "status": "no_comparable_model",
            "reason": "every candidate failed on the training or validation side",
            "models": results,
            "selected": None,
        }
    selected = max(
        selectable,
        key=lambda entry: (entry["validation_log_likelihood"], entry["model"]),
    )
    return {
        "status": "ok",
        "selected": selected["model"],
        "selected_by": "highest validation log likelihood on held-out batches",
        "models": results,
    }


def _fit_summary(fit):
    return {
        "parameters": fit["parameters"],
        "log_likelihood": finite_or_none(fit["log_likelihood"]),
        "iterations": fit["iterations"],
        "evaluations": fit["evaluations"],
        "converged": fit["converged"],
        "reason": fit["reason"],
        "gradient_norm": finite_or_none(fit["gradient_norm"]),
        "attempts": fit["attempts"],
    }
