"""Statistical edge cases and local RNG reproducibility."""

import numpy as np
import pytest

from utils.stats import bootstrap_ci


def test_constant_and_singleton():
    assert bootstrap_ci([7] * 5, n_boot=20) == (7, 7)
    assert bootstrap_ci([2], n_boot=20) == (2, 2)


def test_seeded_bootstrap_is_reproducible_and_local():
    np.random.seed(13)
    before = np.random.get_state()
    first = bootstrap_ci([1, 2, 4, 8], n_boot=300, seed=31)
    assert first == bootstrap_ci([1, 2, 4, 8], n_boot=300, seed=31)
    assert 1 <= first[0] <= first[1] <= 8
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])


@pytest.mark.parametrize(
    "values,kwargs",
    [
        ([], {}),
        ([float("nan")], {}),
        ([[1, 2]], {}),
        ([1], {"confidence": 1}),
        ([1], {"confidence": float("nan")}),
        ([1], {"n_boot": 0}),
        ([1], {"n_boot": True}),
        ([1], {"seed": -1}),
        ([1], {"statistic": lambda x: float("inf")}),
    ],
)
def test_invalid_bootstrap(values, kwargs):
    with pytest.raises(ValueError):
        bootstrap_ci(values, **kwargs)
