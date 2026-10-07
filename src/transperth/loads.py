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
``demand``      AM-peak boardings proxy: ``am_peak_stops``, the number of
                scheduled departure events at the station inside the frozen
                [07:00, 09:00) window (P2.4).
============== =============================================================

:func:`demand_reference_capacities` builds the frequency-scaled reference
capacity ``K`` for the demand experiment from ``trips_served``.

Ownership: P1.3. The public names are frozen; see ``docs/model.md`` section 8.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import networkx as nx

from transperth.config import LoadMode

__all__ = ["capacities", "demand_reference_capacities", "initial_loads"]

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


def _node_count(graph: nx.Graph, node: str, attribute: str) -> float:
    """Return ``attribute`` of one node, defaulting to 0.0 when absent.

    A missing attribute (or an explicit ``None``) counts as zero; a present
    value must be finite and non-negative. The error names the station and the
    attribute so a bad frozen column is traceable to its source.
    """
    raw = graph.nodes[node].get(attribute, 0.0)
    if raw is None:
        return 0.0
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(
            f"node {node!r} has a non-numeric {attribute!r} attribute: {raw!r}"
        ) from None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            f"node {node!r} has a negative or non-finite {attribute}={value!r}"
        )
    return value


def initial_loads(graph: nx.Graph, mode: LoadMode = "betweenness") -> dict[str, float]:
    """Return the initial load ``L0_i`` of every station in ``graph``.

    Parameters
    ----------
    graph:
        Intact rail graph; it is never mutated. For ``betweenness_freq`` every
        edge must carry a finite ``trips > 0`` attribute, and for ``demand``
        every station should carry ``am_peak_stops`` (missing counts as zero).
    mode:
        One of the :data:`transperth.config.LoadMode` values.

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
            node: float(base[node]) + 0.1 * _node_count(graph, node, "trips_served")
            for node in graph.nodes
        }

    if mode == "demand":
        return {
            node: _node_count(graph, node, "am_peak_stops") for node in graph.nodes
        }

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
        capacity (``0.0``, never negative or NaN). It is not exempt from
        overload checks: redistributed or rerouted load above the tolerance
        can fail it in a cascade.

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


def demand_reference_capacities(graph: nx.Graph) -> dict[str, float]:
    """Return the frequency-scaled reference capacity ``K_i`` per station.

    The demand experiment replaces the hidden default ``K = L0`` of
    :func:`capacities` with a capacity derived from the full-day service
    frequency ``f_i = trips_served``. The uniform scale ``c`` is the smallest
    peak share that covers every station's AM-peak load on the graph passed in::

        c   = max_i (L0_i / f_i) over stations with f_i > 0
        K_i = c * f_i

    so ``K_i >= L0_i`` for every station, with equality only at the station
    that attains the maximum. A station with ``f_i = 0`` gets ``K_i = 0``; if
    it also carries a positive demand load the calibration is undefined and the
    function raises instead of inventing capacity.

    This is the scenario assumption of the P2.4 experiment, not a measured
    boarding capacity: one scheduled service is a frequency unit, and the
    scale is calibrated on the frozen graph so ``alpha = 0`` cannot overload a
    station before the trigger.

    Parameters
    ----------
    graph:
        Rail graph carrying ``am_peak_stops`` (via
        :func:`initial_loads`) and ``trips_served``. Missing counts are zero;
        present values must be finite and non-negative. The graph is never
        mutated.

    Returns
    -------
    dict[str, float]
        Reference capacity per station, in graph node order.

    Raises
    ------
    ValueError
        If a station carries a positive demand load but has zero service
        frequency, or if a count is negative, non-finite or non-numeric.
    """
    loads = initial_loads(graph, "demand")
    frequency = {
        node: _node_count(graph, node, "trips_served") for node in graph.nodes
    }
    for node in graph.nodes:
        if frequency[node] == 0.0 and loads[node] > 0.0:
            raise ValueError(
                f"station {node!r} has demand load {loads[node]!r} "
                "but zero trips_served"
            )
    active = [node for node in graph.nodes if frequency[node] > 0.0]
    scale = max((loads[node] / frequency[node] for node in active), default=0.0)
    result: dict[str, float] = {}
    for node in graph.nodes:
        value = scale * frequency[node]
        if value < loads[node]:
            # Only floating-point rounding can break the calibration on a
            # graph where every demand station has positive frequency.
            value = loads[node]
        result[node] = value
    return result