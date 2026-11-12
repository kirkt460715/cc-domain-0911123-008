"""Weibull likelihoods with exact, interval, right-censored and delayed-entry data.

Time is measured on the experiment's day scale from the shared origin.  For a
specimen entering at ``entry_day`` and failing at absolute day ``T`` the
accumulated exposure is ``T - entry_day`` and the survival probability is
``S(T - entry_day)``.  Let ``tau = right - entry`` and ``lam = left - entry``:

* exact at ``right``:      ``f(tau)``
* interval ``(left, right]``: ``S(lam) - S(tau)``
* right-censored at ``left``: ``S(lam)``

No interval midpoint is ever substituted; the interval term is the exact
increment of the survival function.  The interval probability is evaluated as
``S(lam) * (1 - exp(log S(tau) - log S(lam)))`` so that tight inspection
intervals keep full relative precision.

Both models use the same log-linear Weibull parameterisation:

    log T ~ Gumbel(location = mu + slope * z, scale = 1 / shape)

where ``z = 1/T_K - 1/T_ref_K`` (Arrhenius) or ``z = 0`` (common scale).
``mu = log(eta_ref)`` is the log scale parameter at the reference temperature,
``shape = beta``, and ``slope = Ea / kB`` in kelvin.  Every reported hazard
multiplies by ``exp(slope * z)``.
"""

import math

from .errors import Failure
from .numerics import log1mexp
from .spec import celsius_to_kelvin


MODELS = ("common_scale_weibull", "arrhenius_weibull")
PARAMETER_NAMES = {
    "common_scale_weibull": ("shape", "log_scale", "slope"),
    "arrhenius_weibull": ("shape", "log_scale", "slope"),
}


def _prepare(observation, model):
    """Fill tau/lam once; tolerate observations carrying only absolute days."""
    entry = observation.get("entry_day", 0.0)
    tau = observation.get("tau")
    lam = observation.get("lam")
    if tau is None and observation.get("right_day") is not None:
        tau = observation["right_day"] - entry
    if lam is None and observation.get("left_day") is not None:
        lam = observation["left_day"] - entry
    if observation["kind"] == "exact" and tau is None:
        tau = lam
    if lam is None:
        raise Failure(
            "observation",
            "observation has no censoring time",
            specimen=observation.get("specimen"),
        )
    if tau is not None:
        if tau < lam:
            raise Failure(
                "observation",
                "interval right bound precedes left bound after entry adjustment",
                specimen=observation.get("specimen"),
            )
        if tau <= 0:
            raise Failure(
                "observation",
                "failure time is not after entry",
                specimen=observation.get("specimen"),
            )
    if lam < 0:
        raise Failure(
            "observation",
            "censoring time precedes entry",
            specimen=observation.get("specimen"),
        )
    return tau, lam


def _z(temperature_k, reference_k, model):
    if model == "arrhenius_weibull":
        return 1.0 / temperature_k - 1.0 / reference_k
    return 0.0


def _terms(observation, model, reference_k, shape, log_scale, slope):
    """Return the log-likelihood contribution and its log survival values."""
    beta = shape
    z = _z(observation["temperature_k"], reference_k, model)
    eta_log = log_scale + slope * z
    kind = observation["kind"]
    if kind == "exact":
        variation = observation["tau"]
    else:
        variation = observation["lam"]
    if variation is None or variation < 0:
        return -math.inf, None, None
    if variation == 0:
        # An observation entering at the origin has full survival so far.
        log_t = None
        log_s = 0.0
    else:
        log_t = math.log(variation)
        log_s = -math.exp(beta * (log_t - eta_log))
    if kind == "exact":
        if variation <= 0:
            return -math.inf, None, None
        log_f = (
            math.log(beta)
            + (beta - 1.0) * (log_t - eta_log)
            - eta_log
            + log_s
        )
        return log_f, log_s, None
    if kind == "right":
        return log_s, log_s, None
    tau = observation["tau"]
    if tau is None or tau <= 0:
        return -math.inf, None, None
    log_tau = math.log(tau)
    log_s_tau = -math.exp(beta * (log_tau - eta_log))
    log_increment = log_s + log1mexp(log_s_tau - log_s)
    return log_increment, log_s_tau, log_s


def log_likelihood(observations, model, parameters, reference_k):
    shape = parameters["shape"]
    log_scale = parameters["log_scale"]
    slope = parameters.get("slope", 0.0)
    total = 0.0
    for observation in observations:
        term, _, _ = _terms(observation, model, reference_k, shape, log_scale, slope)
        if term == -math.inf:
            return -math.inf
        total += term
    return total


def survival(observations, model, parameters, reference_k, temperature_c, time):
    """Survival at a Celsius use temperature and a day from entry."""
    shape = parameters["shape"]
    log_scale = parameters["log_scale"]
    slope = parameters.get("slope", 0.0)
    temperature_k = celsius_to_kelvin(temperature_c)
    z = _z(temperature_k, reference_k, model)
    eta_log = log_scale + slope * z
    if time <= 0:
        return 1.0
    return math.exp(-math.exp(shape * (math.log(time) - eta_log)))


def quantile_time(model, parameters, reference_k, temperature_c, probability):
    """Time t with F(t) = probability at a Celsius temperature."""
    shape = parameters["shape"]
    log_scale = parameters["log_scale"]
    slope = parameters.get("slope", 0.0)
    temperature_k = celsius_to_kelvin(temperature_c)
    z = _z(temperature_k, reference_k, model)
    eta_log = log_scale + slope * z
    return math.exp(
        eta_log + math.log(-math.log1p(-probability)) / shape
    )


def scale_at(model, parameters, reference_k, temperature_c):
    z = _z(celsius_to_kelvin(temperature_c), reference_k, model)
    return math.exp(parameters["log_scale"] + parameters.get("slope", 0.0) * z)
