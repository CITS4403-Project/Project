"""Unweighted rail metrics with explicit intact-network denominators."""

from __future__ import annotations

import math
from numbers import Integral

import networkx as nx

from transperth.config import MetricsBundle


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def compute_metrics(
    graph: nx.Graph,
    *,
    baseline_efficiency: float | None = None,
    served_pairs: int | None = None,
    total_pairs: int | None = None,
    n_initial: int | None = None,
) -> MetricsBundle:
    """Measure an undirected simple graph without modifying it.

    ``n_initial`` defaults to ``graph.graph['n_initial']`` then the current
    count. Removal callers must pass/preserve the intact count. Efficiency
    divides summed inverse hop distances by ``n_initial*(n_initial-1)``;
    removed and unreachable pairs contribute zero. ASPL uses only the GCC.
    Empty ASPL, undefined damage and fractions with a zero denominator are 0;
    an empty OD universe is fully served (1). Explicit OD counts must be
    supplied together, and can describe a terminal subset of a layered graph.
    """
    if graph.is_directed() or graph.is_multigraph():
        raise ValueError("metrics require an undirected simple graph")
    n = len(graph)
    original = _count(
        graph.graph.get("n_initial", n) if n_initial is None else n_initial,
        "n_initial",
    )
    if original < n:
        raise ValueError("n_initial cannot be smaller than the surviving node count")
    components = sorted(
        nx.connected_components(graph),
        key=lambda nodes: (-len(nodes), tuple(sorted(map(str, nodes)))),
    )
    gcc = len(components[0]) if components else 0
    lcc = len(components[1]) if len(components) > 1 else 0
    inverse = 0.0
    reachable = 0
    for source, lengths in nx.all_pairs_shortest_path_length(graph):
        inverse += math.fsum(
            1 / distance for target, distance in lengths.items() if target != source
        )
        reachable += len(lengths) - 1
    efficiency = inverse / (original * (original - 1)) if original > 1 else 0.0
    damage = 0.0
    if baseline_efficiency is not None:
        baseline = float(baseline_efficiency)
        if not math.isfinite(baseline) or baseline < 0:
            raise ValueError("baseline_efficiency must be finite and non-negative")
        if baseline == 0 and efficiency != 0:
            raise ValueError("positive efficiency cannot use a zero baseline")
        if baseline:
            damage = 1 - efficiency / baseline
    if (served_pairs is None) != (total_pairs is None):
        raise ValueError("served_pairs and total_pairs must be supplied together")
    if served_pairs is None:
        served, total = reachable // 2, original * (original - 1) // 2
    else:
        served = _count(served_pairs, "served_pairs")
        total = _count(total_pairs, "total_pairs")
    if served > total:
        raise ValueError("served_pairs cannot exceed total_pairs")
    return MetricsBundle(
        n_nodes=n,
        n_edges=graph.number_of_edges(),
        gcc_size=gcc,
        gcc_fraction=gcc / original if original else 0.0,
        lcc_size=lcc,
        lcc_fraction=lcc / original if original else 0.0,
        isolated=len(list(nx.isolates(graph))),
        aspl=float(nx.average_shortest_path_length(graph.subgraph(components[0])))
        if gcc > 1
        else 0.0,
        efficiency=efficiency,
        network_damage=damage,
        served_od_fraction=served / total if total else 1.0,
    )
