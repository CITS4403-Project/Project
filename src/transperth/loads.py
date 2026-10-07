"""Initial loads and capacities for the rail network.

Implements the load modes of ``docs/model.md`` section 2 and the capacity law
of section 3:

    L0_i = unnormalised betweenness centrality of station i on the intact graph
    C_i  = (1 + alpha) * L0_i,   alpha >= 0

Load modes:

============== =============================================================
``betweenness`` ``networkx.betweenness_centrality(G, normalized=False)``
``betweenness_freq``
                betweenness with edge length ``1 / trips``, so frequent
                services carry shorter paths. The input graph is copied and
                never mutated; bad trip data raises instead of being floored.
``betweenness_plus_trips``
                unnormalised betweenness plus ``0.1 * trips_served``, a
                throughput proxy that gives leaf stations a positive baseline.
``demand``      AM-peak boardings proxy; implemented by P2.4, so it raises
                :class:`NotImplementedError` until then.
============== =============================================================

Ownership: P1.3. The public names are frozen; see ``docs/model.md`` section 8.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import networkx as nx

from transperth.config import LoadMode

__all__ = ["capacities", "initial_loads"]

_MODES: tuple[LoadMode, ...] = (
    "betweenness",
    "betweenness_freq",
    "betweenness_plus_trips",
    "demand",
)


def _edge_trips(graph: nx.Graph, u: str, v: str, data: Mapping[str, object]) -> float:
    """Return the finite, strictly positive ``trips`` of one edge.

    The frozen loader guarantees positive trips; anything else is bad data and
    raises :class:`ValueError` that names the offending edge rather than being
    silently floored to 1 as in the prototype.
    """
    raw = data.get("trips")
    try:
        trips = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"edge ({u!r}, {v!r}) has no numeric 'trips' attribute: {raw!r}"
        ) from None
    if not math.isfinite(trips) or trips <= 0.0:
        raise ValueError(
            f"edge ({u!r}, {v!r}) has non-positive or non-finite trips={trips!r}"
        )
    return trips


def _node_trips_served(graph: nx.Graph, node: str) -> float:
    """Return ``trips_served`` of one node, defaulting to 0.0 when absent.

    A missing attribute (or an explicit ``None``) counts as zero throughput;
    a present value must be finite and non-negative.
    """
    raw = graph.nodes[node].get("trips_served", 0.0)
    if raw is None:
        return 0.0
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"node {node!r} has a non-numeric 'trips_served' attribute: {raw!r}"
        ) from None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"node {node!r} has a negative or non-finite trips_served={value!r}"
        )
    return value


def initial_loads(graph: nx.Graph, mode: LoadMode = "betweenness") -> dict[str, float]:
    """Return the initial load ``L0_i`` of every station in ``graph``.

    Parameters
    ----------
    graph:
        Intact rail graph; it is never mutated. For ``betweenness_freq`` every
        edge must carry a finite ``trips > 0`` attribute.
    mode:
        One of the :data:`transperth.config.LoadMode` values. ``demand`` is
        reserved for P2.4 and raises :class:`NotImplementedError`.

    Returns
    -------
    dict[str, float]
        Load per node, in graph node order. Unnormalised values, because the
        node set shrinks during a cascade and normalised values would inflate
        as it does.
    """
    if mode == "betweenness":
        centrality = nx.betweenness_centrality(graph, normalized=False)
        return {node: float(value) for node, value in centrality.items()}

    if mode == "betweenness_freq":
        weighted = nx.Graph()
        weighted.add_nodes_from(graph.nodes)
        for u, v, data in graph.edges(data=True):
            weighted.add_edge(u, v, length=1.0 / _edge_trips(graph, u, v, data))
        centrality = nx.betweenness_centrality(weighted, weight="length", normalized=False)
        return {node: float(value) for node, value in centrality.items()}

    if mode == "betweenness_plus_trips":
        base = nx.betweenness_centrality(graph, normalized=False)
        return {
            node: float(base[node]) + 0.1 * _node_trips_served(graph, node)
            for node in graph.nodes
        }

    if mode == "demand":
        raise NotImplementedError(
            "load mode 'demand' is implemented by P2.4; use 'betweenness' until then"
        )

    raise ValueError(f"unknown load mode {mode!r}; expected one of {_MODES!r}")


def capacities(loads: Mapping[str, float], alpha: float) -> dict[str, float]:
    """Return ``C_i = (1 + alpha) * L0_i`` for every station in ``loads``.

    Parameters
    ----------
    loads:
        Initial load per station. The mapping is only read, never mutated.
    alpha:
        Tolerance, a finite ``float >= 0``.

    Returns
    -------
    dict[str, float]
        Capacity per station, in the input order. A zero load yields a zero
        capacity (``0.0``, never negative or NaN), so such a station can only
        fail as an explicit trigger.

    Raises
    ------
    ValueError
        If ``alpha`` is negative or non-finite, or if a load is negative or
        non-finite.
    """
    if not math.isfinite(alpha) or alpha < 0.0:
        raise ValueError(f"alpha must be finite and non-negative, got {alpha!r}")

    result: dict[str, float] = {}
    for node, load in loads.items():
        try:
            value = float(load)
        except (TypeError, ValueError):
            raise ValueError(f"load of station {node!r} is not numeric: {load!r}") from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                f"load of station {node!r} must be finite and non-negative, got {load!r}"
            )
        result[node] = (1.0 + alpha) * value
    return result