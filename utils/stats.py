"""Seeded percentile bootstrap confidence intervals."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from numbers import Integral
from statistics import mean

import numpy as np


def validate_count(value: int, name: str, *, minimum: int = 0) -> int:
    """Validate an integer count or RNG seed, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def bootstrap_ci(
    values: Iterable[float],
    statistic: Callable = mean,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Return a percentile interval for independent observations.

    Samples are resampled with replacement with a local generator. This does
    not alter NumPy's global RNG; it is not a spatial/time-series bootstrap.
    """
    samples = np.asarray(list(values), dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("values must be a non-empty finite one-dimensional sample")
    n_boot = validate_count(n_boot, "n_boot", minimum=1)
    seed = validate_count(seed, "seed")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(seed)
    estimates = np.array(
        [
            float(statistic(rng.choice(samples, size=len(samples), replace=True)))
            for _ in range(n_boot)
        ]
    )
    if not np.isfinite(estimates).all():
        raise ValueError("statistic must return a finite scalar")
    tail = (1 - confidence) / 2
    low, high = np.quantile(estimates, [tail, 1 - tail])
    return float(low), float(high)
