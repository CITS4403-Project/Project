"""Tests for the yard-sale prototype (CITS4403 idea-1 investigation).

Run with either
    .venv/bin/python investigations/idea1-yardsale/prototype/test_yardsale.py
    .venv/bin/python -m pytest investigations/idea1-yardsale/prototype/test_yardsale.py

The heavy analytic/invariant battery lives in yardsale.selftest(); this module
adds fast unit tests for the statistics and the network builders, then calls
the full battery.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yardsale import (  # noqa: E402
    TOPOLOGIES,
    YardSaleModel,
    build_network,
    gini,
    hill_alpha,
    max_share,
    selftest,
    simulate,
    top_share,
)


def test_gini_limits():
    assert gini(np.ones(100)) == 0.0 or abs(gini(np.ones(100))) < 1e-12
    w = np.zeros(99)
    w = np.append(w, 1.0)
    assert abs(gini(w) - 0.99) < 1e-12
    # two-agent split
    assert abs(gini(np.array([1.0, 3.0])) - 0.25) < 1e-12


def test_shares():
    w = np.arange(1.0, 11.0)  # total 55
    assert abs(top_share(w, 0.1) - 10.0 / 55.0) < 1e-12
    assert abs(max_share(w) - 10.0 / 55.0) < 1e-12


def test_network_builders():
    n, k = 100, 4
    for topo in ("complete", "ER", "BA", "WS", "lattice"):
        g = build_network(topo, n=n if topo != "lattice" else 100, k=k, seed=0)
        assert g.number_of_nodes() == 100
    g = build_network("ER", n, k=k, seed=0)
    assert g.number_of_edges() == round(n * k / 2)
    g = build_network("WS", n, k=k, seed=0)
    assert g.number_of_edges() == n * k // 2
    g = build_network("lattice", 100, k=k, seed=0)
    assert g.number_of_edges() == 2 * 10 * 9


def test_two_agent_deterministic():
    m = YardSaleModel(n=2, f=0.5, p=1.0, seed=0, w0=np.array([2.0, 1.0]))
    m._trade_batch(np.array([0]), np.array([1]))
    assert np.allclose(np.sort(m.w), [0.5, 2.5])


def test_conservation_across_topologies():
    for topo in TOPOLOGIES:
        r = simulate(n=64 if topo != "lattice" else 64, f=0.3, p=0.6, tax=0.01,
                     topology=topo, k=4, sweeps=100, seed=1, sample_every=10)
        assert r["max_rel_drift"] < 1e-10, (topo, r["max_rel_drift"])


def test_selftest_battery():
    checks = selftest(verbose=False)
    failed = [c for c in checks if not c[1]]
    assert not failed, failed


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                fails += 1
                print(f"[FAIL] {name}: {exc}")
    print(f"\n{'OK' if not fails else f'{fails} FAILED'}")
    sys.exit(1 if fails else 0)