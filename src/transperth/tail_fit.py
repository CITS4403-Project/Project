"""Finite-support discrete power-law MLE and refitted bootstrap goodness of fit.

This is a bounded analogue of Clauset et al. (2009), not their unbounded
power-law estimator. The physical avalanche bound is known from the graph.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp

from transperth.stats import validate_count


def _sample(values, upper: int) -> np.ndarray:
    upper = validate_count(upper, "upper", minimum=1)
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or not len(sample) or not np.isfinite(sample).all():
        raise ValueError("need a nonempty finite one-dimensional sample")
    if (
        np.any(sample < 0)
        or np.any(sample > upper)
        or np.any(sample != np.floor(sample))
    ):
        raise ValueError("sizes must be integers in [0, upper]")
    return sample.astype(int)


def discrete_pmf(
    exponent: float, xmin: int, upper: int
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise x**(-exponent) over every integer xmin..upper, including gaps."""
    xmin = validate_count(xmin, "xmin", minimum=1)
    upper = validate_count(upper, "upper", minimum=xmin)
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError("exponent must be finite and positive")
    support = np.arange(xmin, upper + 1)
    logits = -exponent * np.log(support)
    return support, np.exp(logits - logsumexp(logits))


def fit_discrete_tail(values, *, upper: int, min_tail: int = 50) -> dict:
    """Choose the eligible observed cutoff minimising discrete CDF KS distance."""
    sample = _sample(values, upper)
    min_tail = validate_count(min_tail, "min_tail", minimum=3)
    positive = sample[sample > 0]
    base = {
        "n": len(sample),
        "n_positive": len(positive),
        "p_zero": float(np.mean(sample == 0)),
        "upper": upper,
        "model": "finite_support_discrete_power_law",
        "min_tail": min_tail,
    }
    candidates = []
    for cutoff in np.unique(positive):
        tail = positive[positive >= cutoff]
        if len(tail) < min_tail or len(np.unique(tail)) < 3:
            continue
        support = np.arange(cutoff, upper + 1)
        log_support = np.log(support)
        log_total = float(np.log(tail).sum())
        objective = (
            lambda exponent, log_total=log_total, tail=tail, log_support=log_support: (
                exponent * log_total + len(tail) * logsumexp(-exponent * log_support)
            )
        )
        optimum = minimize_scalar(
            objective, bounds=(0.01, 20), method="bounded", options={"xatol": 1e-8}
        )
        if not optimum.success:
            raise RuntimeError("discrete likelihood optimisation failed")
        exponent = float(optimum.x)
        _, probability = discrete_pmf(exponent, int(cutoff), upper)
        empirical = np.cumsum(np.bincount(tail, minlength=upper + 1)[cutoff:]) / len(
            tail
        )
        ks = float(np.max(np.abs(empirical - np.cumsum(probability))))
        candidates.append(
            {
                "xmin": int(cutoff),
                "exponent": exponent,
                "ks": ks,
                "n_tail": len(tail),
                "log_likelihood": float(-optimum.fun),
                "exponent_at_bound": bool(exponent < 0.0101 or exponent > 19.9999),
            }
        )
    if not candidates:
        return {
            **base,
            "status": "unidentifiable_tail",
            "reason": "Need at least min_tail observations and three distinct positive sizes above a cutoff.",
        }
    chosen = min(candidates, key=lambda row: (row["ks"], row["xmin"]))
    return {**base, **chosen, "status": "fitted", "cutoff_candidates": len(candidates)}


def power_law_gof(
    values, *, upper: int, min_tail: int = 50, n_boot: int = 500, seed: int = 0
) -> dict:
    """Semiparametric body/tail bootstrap, refitting xmin and exponent every time.

    Body includes zeros. Synthetic tail draws come from the fitted discrete
    support. Ineligible refits are recorded; a p value is withheld if any
    refit cannot be defined, rather than silently discarding that draw.
    """
    sample = _sample(values, upper)
    n_boot = validate_count(n_boot, "n_boot", minimum=1)
    seed = validate_count(seed, "seed")
    fit = fit_discrete_tail(sample, upper=upper, min_tail=min_tail)
    fit.update(
        bootstrap_seed=seed,
        bootstrap_replicates=n_boot,
        p_value=None,
        source="https://arxiv.org/abs/0706.1062",
        scope="Bounded hypothesis only; p>=0.1 does not prove a power law or establish it against alternatives.",
    )
    if fit["status"] != "fitted":
        return fit
    rng = np.random.default_rng(seed)
    body = sample[sample < fit["xmin"]]
    support, probability = discrete_pmf(fit["exponent"], fit["xmin"], upper)
    distances, invalid = [], 0
    for _ in range(n_boot):
        tail_mask = rng.random(len(sample)) < fit["n_tail"] / len(sample)
        synthetic = np.empty(len(sample), dtype=int)
        synthetic[tail_mask] = rng.choice(
            support, size=int(tail_mask.sum()), p=probability
        )
        if len(body):
            synthetic[~tail_mask] = rng.choice(body, size=int((~tail_mask).sum()))
        refit = fit_discrete_tail(synthetic, upper=upper, min_tail=min_tail)
        if refit["status"] != "fitted":
            invalid += 1
        else:
            distances.append(refit["ks"])
    fit["invalid_refits"] = invalid
    fit["bootstrap_ks"] = distances
    if invalid:
        fit["status"] = "bootstrap_incomplete"
        fit["reason"] = "An ineligible synthetic tail prevents a calibrated p value."
    else:
        fit["p_value"] = (1 + sum(distance >= fit["ks"] for distance in distances)) / (
            n_boot + 1
        )
        fit["status"] = "rejected" if fit["p_value"] < 0.1 else "not_rejected"
        fit["p_mc_standard_error"] = float(
            np.sqrt(fit["p_value"] * (1 - fit["p_value"]) / (n_boot + 1))
        )
    return fit
