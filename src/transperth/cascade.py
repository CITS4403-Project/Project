"""Synchronous rail cascades with fixed capacities and explicit lost load.

Capacities follow ``C_i = (1 + alpha) * K_i``. ``K`` defaults to the initial
loads; the optional ``reference_capacities`` hook replaces it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from numbers import Integral

import networkx as nx

from transperth.config import CascadeConfig, CascadeResult
from transperth.failure import removal_batch
from transperth.loads import capacities, initial_loads


def _ordered_graph(graph: nx.Graph) -> nx.Graph:
    if graph.is_directed() or graph.is_multigraph() or nx.number_of_selfloops(graph):
        raise ValueError(
            "cascade requires an undirected simple graph without self-loops"
        )
    if any(not isinstance(node, str) for node in graph):
        raise ValueError("cascade station IDs must be strings")
    ordered = nx.Graph()
    ordered.graph.update(graph.graph)
    ordered.add_nodes_from((node, dict(graph.nodes[node])) for node in sorted(graph))
    ordered.add_edges_from(
        (a, b, dict(graph.edges[a, b]))
        for a, b in sorted(tuple(sorted(edge)) for edge in graph.edges)
    )
    return ordered


def _checked_loads(
    values: Mapping[str, float], graph: nx.Graph, *, label: str
) -> dict[str, float]:
    """Return ``values`` as finite non-negative numbers covering exactly ``graph``."""
    if not isinstance(values, Mapping):
        raise TypeError(f"{label} must be a mapping of station values")
    loads: dict[str, float] = {}
    for node, raw in values.items():
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(
                f"{label} returned a non-numeric load for {node!r}"
            ) from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must return finite non-negative loads")
        loads[node] = value
    if set(loads) != set(graph):
        raise ValueError(f"{label} must return exactly the graph station IDs")
    return loads


def simulate_cascade(
    graph: nx.Graph,
    config: CascadeConfig,
    *,
    baseline_loads: Mapping[str, float] | None = None,
    load_function: Callable[[nx.Graph], Mapping[str, float]] | None = None,
    initial_failed: Iterable[str] | None = None,
    reference_capacities: Mapping[str, float] | None = None,
) -> CascadeResult:
    """Remove a trigger then fail overloaded stations in synchronous rounds.

    Static mode moves load only to neighbours surviving the same round.
    ``remaining_load`` plus ``lost_load`` equals ``initial_total_load``.
    Dynamic mode recomputes the selected load mode after removals; these two
    conservation fields are None because routing betweenness is not conserved.
    ``load_function`` is an additive hook that replaces the load mode for the
    intact baseline and for every dynamic recomputation; it receives the
    current graph, must not modify it, and must return one finite non-negative
    load per current station. ``baseline_loads`` still overrides the initial
    loads.

    ``reference_capacities`` is the additive capacity hook: when given it
    supplies ``K_i`` for the general law ``C_i = (1 + alpha) * K_i`` while the
    loads keep their source. The default ``None`` uses ``K = L0``, so the
    frozen law ``C_i = (1 + alpha) * L0_i`` is reproduced exactly. Both
    mappings must cover exactly the graph station IDs, and capacities stay
    fixed from the intact graph.
    The input graph and the supplied mappings remain unchanged.
    ``initial_failed`` replaces the single trigger with one sorted synchronous
    initial batch. An explicit empty collection applies no external failure.
    These initial failures are excluded from subsequent avalanche sizes.
    """
    current = _ordered_graph(graph)
    if config.trigger not in ("load", "degree", "random") or config.rule not in (
        "equal",
        "capacity",
    ):
        raise ValueError("unknown cascade trigger or redistribution rule")
    if not isinstance(config.dynamic, bool):
        raise ValueError("dynamic must be a bool")
    if load_function is not None and not callable(load_function):
        raise TypeError("load_function must be callable")
    if (
        isinstance(config.seed, bool)
        or not isinstance(config.seed, Integral)
        or config.seed < 0
    ):
        raise ValueError("seed must be a non-negative integer")
    if config.load_mode not in (
        "betweenness",
        "betweenness_freq",
        "betweenness_plus_trips",
        "demand",
    ):
        raise ValueError("unknown cascade load mode")
    if (
        initial_failed is None
        and config.target is not None
        and config.target not in current
    ):
        raise ValueError(f"unknown target station {config.target!r}")
    initial_batch = None
    if initial_failed is not None:
        if isinstance(initial_failed, (str, bytes)):
            raise TypeError(
                "initial_failed must be a collection of station IDs, not a string"
            )
        initial_batch = list(initial_failed)
        if any(not isinstance(node, str) for node in initial_batch):
            raise ValueError("initial_failed IDs must be strings")
        if len(set(initial_batch)) != len(initial_batch):
            raise ValueError("initial_failed contains duplicate IDs")
        unknown = set(initial_batch) - set(current)
        if unknown:
            raise ValueError(f"unknown initial_failed stations: {sorted(unknown)}")
        initial_batch.sort()
    if baseline_loads is not None:
        base = dict(baseline_loads)
    elif load_function is None:
        base = initial_loads(current, config.load_mode)
    else:
        base = _checked_loads(load_function(current), current, label="load_function")
    if set(base) != set(current):
        raise ValueError("baseline_loads must contain exactly the graph station IDs")
    reference = (
        _checked_loads(reference_capacities, current, label="reference_capacities")
        if reference_capacities is not None
        else dict(base)
    )
    limits = capacities(reference, config.alpha)
    if not all(math.isfinite(value) for value in limits.values()):
        raise ValueError(
            "capacity overflow: loads and alpha must give finite capacities"
        )
    loads = {node: float(base[node]) for node in sorted(base)}
    total = math.fsum(loads.values())
    if not math.isfinite(total):
        raise ValueError("total initial load must be finite")
    failed: list[str] = []
    avalanches: list[int] = []
    lost = 0.0

    def remove_batch(batch: list[str]) -> None:
        nonlocal lost, loads
        removed = set(batch)
        recipients = {
            node: sorted(
                neighbour for neighbour in current[node] if neighbour not in removed
            )
            for node in batch
        }
        amounts = {node: loads[node] for node in batch}
        current.remove_nodes_from(batch)
        failed.extend(batch)
        if config.dynamic:
            loads = (
                initial_loads(current, config.load_mode)
                if load_function is None
                else _checked_loads(
                    load_function(current), current, label="load_function"
                )
            )
            return
        increments: dict[str, list[float]] = {node: [] for node in current}
        losses = []
        for node in batch:
            neighbours = recipients[node]
            amount = amounts[node]
            if not neighbours:
                losses.append(amount)
                continue
            capacity_sum = math.fsum(limits[other] for other in neighbours)
            for other in neighbours:
                share = (
                    limits[other] / capacity_sum
                    if config.rule == "capacity" and capacity_sum
                    else 1 / len(neighbours)
                )
                increments[other].append(amount * share)
        lost = math.fsum([lost, *losses])
        loads = {node: math.fsum([loads[node], *increments[node]]) for node in current}

    if current:
        if initial_batch is None and config.target is not None:
            trigger = config.target
        elif initial_batch is None:
            order = (
                sorted(loads, key=lambda node: (-loads[node], node))
                if config.trigger == "load"
                else None
            )
            trigger = removal_batch(
                current,
                attack="random" if config.trigger == "random" else "targeted",
                measure="degree",
                count=1,
                seed=config.seed,
                order=order,
            )[0]
        if initial_batch is None:
            initial_batch = [trigger]
        if initial_batch:
            remove_batch(initial_batch)
        while current:
            overloaded = sorted(
                node
                for node in current
                if loads[node] > limits[node] + config.tolerance
            )
            if not overloaded:
                break
            avalanches.append(len(overloaded))
            remove_batch(overloaded)
    gcc = max(
        (len(component) for component in nx.connected_components(current)), default=0
    )
    n = len(graph)
    return CascadeResult(
        n_initial=n,
        failed=tuple(failed),
        gcc=gcc,
        gcc_fraction=gcc / n if n else 0.0,
        failed_fraction=len(failed) / n if n else 0.0,
        avalanche_sizes=tuple(avalanches),
        rounds=len(avalanches),
        initial_total_load=total,
        remaining_load=None if config.dynamic else math.fsum(loads.values()),
        lost_load=None if config.dynamic else lost,
        initial_failed=tuple(initial_batch or ()),
    )
