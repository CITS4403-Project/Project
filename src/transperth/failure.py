"""Failure, attack and percolation engine for the rail network.

One deterministic code path selects removals for random and targeted node or
edge attacks: the ranking helpers :func:`rank_targets` and
:func:`rank_edge_targets`, the random-permutation helpers
:func:`random_target_order` and :func:`random_edge_order`, and the batch
selector :func:`removal_batch` that all three build on. A cascade trigger
(P1.4) imports the ranking helpers, so a targeted trigger and a targeted attack
pick the same station, with ties broken by sorted station id.

Measures
--------

Nodes:

============== =============================================================
``degree``     graph degree
``betweenness`` unnormalised node betweenness centrality
``flow``       initial load from :func:`transperth.loads.initial_loads`; the
               hook the P2.4 demand mode extends. It currently returns the
               same values as ``betweenness`` for nodes.
============== =============================================================

Edges:

============== =============================================================
``degree``     sum of the endpoint degrees (the incidence count; the
               line-graph degree is this value minus two)
``betweenness`` unnormalised edge betweenness centrality
``flow``       unnormalised edge betweenness centrality, the initial edge
               load of the same betweenness family
============== =============================================================

Attacks
-------

================ =========================================================
``random``        nodes drawn uniformly without replacement
``targeted``      nodes ranked by the measure, ties by sorted station id
``random_edge``   edges drawn uniformly without replacement
``targeted_edge`` edges ranked by the edge measure, ties by sorted endpoints
================ =========================================================

A targeted attack computes its ranking once and removes the targets in order
when ``static=True`` (the default). With ``static=False`` the ranking is
recomputed on the surviving graph before every single removal. Random attacks
draw from a permutation of the sorted target list and ignore the flag.

Ownership: P1.3. The public signatures are frozen; see ``docs/model.md``
section 8. ``percolation_curve`` gains one optional keyword argument
(``static``) that defaults to the frozen behaviour.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Union

import networkx as nx
import numpy as np
import pandas as pd

from transperth.loads import initial_loads

__all__ = [
    "ATTACKS",
    "CURVE_COLUMNS",
    "EDGE_MEASURES",
    "NODE_MEASURES",
    "critical_fraction",
    "percolation_curve",
    "random_edge_order",
    "random_target_order",
    "rank_edge_targets",
    "rank_targets",
    "removal_batch",
    "target_order",
]

NODE_MEASURES: tuple[str, ...] = ("degree", "betweenness", "flow")
"""Measures that rank or weight stations."""

EDGE_MEASURES: tuple[str, ...] = ("degree", "betweenness", "flow")
"""Measures that rank or weight edges."""

ATTACKS: tuple[str, ...] = ("random", "targeted", "random_edge", "targeted_edge")
"""Supported attack values; see the module docstring."""

CURVE_COLUMNS: tuple[str, ...] = (
    "fraction",
    "seed",
    "attack",
    "measure",
    "n_initial",
    "gcc_size",
    "gcc_fraction",
    "lcc_size",
    "lcc_fraction",
    "isolated",
)
"""Column names of the table returned by :func:`percolation_curve`."""

_EDGE_ATTACKS = frozenset({"random_edge", "targeted_edge"})
_TARGETED_ATTACKS = frozenset({"targeted", "targeted_edge"})

Target = Union[str, tuple[str, str]]


def _validate_attack(attack: str) -> None:
    if attack not in ATTACKS:
        raise ValueError(f"unknown attack {attack!r}; expected one of {ATTACKS!r}")


def _validate_measure(measure: str, measures: tuple[str, ...]) -> None:
    if measure not in measures:
        raise ValueError(f"unknown measure {measure!r}; expected one of {measures!r}")


def _validate_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError(f"seed must be an integer, got {seed!r}")
    value = int(seed)
    if value < 0:
        raise ValueError(f"seed must be non-negative, got {value}")
    return value


def _validate_count(count: int) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"count must be an integer, got {count!r}")
    value = int(count)
    if value < 0:
        raise ValueError(f"count must be non-negative, got {value}")
    return value


def _edge_key(u: str, v: str) -> tuple[str, str]:
    """Return an edge as a canonical ``(lower_id, higher_id)`` tuple."""
    return (u, v) if str(u) <= str(v) else (v, u)


def _node_scores(graph: nx.Graph, measure: str) -> dict[str, float]:
    """Score every node by ``measure``; see the module docstring."""
    _validate_measure(measure, NODE_MEASURES)
    if measure == "degree":
        return {node: float(graph.degree(node)) for node in graph.nodes}
    if measure == "betweenness":
        centrality = nx.betweenness_centrality(graph, normalized=False)
        return {node: float(value) for node, value in centrality.items()}
    # measure == "flow": the initial load, the hook P2.4 extends.
    return {node: float(value) for node, value in initial_loads(graph).items()}


def _edge_scores(graph: nx.Graph, measure: str) -> dict[tuple[str, str], float]:
    """Score every edge by ``measure``; see the module docstring."""
    _validate_measure(measure, EDGE_MEASURES)
    if measure == "degree":
        return {
            _edge_key(u, v): float(graph.degree(u) + graph.degree(v))
            for u, v in graph.edges
        }
    # "betweenness" and "flow" both use the unnormalised edge betweenness.
    centrality = nx.edge_betweenness_centrality(graph, normalized=False)
    return {_edge_key(u, v): float(value) for (u, v), value in centrality.items()}


def _random_order(items: Sequence, seed: int) -> list:
    """Return a deterministic random permutation of ``items`` for ``seed``."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    return [items[int(index)] for index in order]


def rank_targets(graph: nx.Graph, measure: str = "degree") -> list[str]:
    """Rank stations by descending ``measure``, ties by sorted station id.

    This is the shared ranking path: :func:`target_order`, :func:`removal_batch`
    and the P1.4 cascade triggers all use it, so a targeted attack and a
    targeted trigger never disagree on ties. The sort key is the explicit
    ``(-score, node)`` pair, not an argmax over an array.
    """
    scores = _node_scores(graph, measure)
    return [node for node, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))]


def rank_edge_targets(graph: nx.Graph, measure: str = "degree") -> list[tuple[str, str]]:
    """Rank edges by descending ``measure``, ties by sorted endpoint ids.

    Edge measures are defined in the module docstring. Every edge is reported
    with its endpoints in canonical ``(lower_id, higher_id)`` order.
    """
    scores = _edge_scores(graph, measure)
    return sorted(scores, key=lambda edge: (-scores[edge], edge[0], edge[1]))


def random_target_order(graph: nx.Graph, *, seed: int = 0) -> list[str]:
    """Return all stations in a deterministic random removal order.

    The order is a uniform permutation of the sorted station ids drawn from
    ``numpy.random.default_rng(seed)``; it contains every node exactly once,
    so callers take the first ``k`` targets.
    """
    seed = _validate_seed(seed)
    return _random_order(sorted(graph.nodes), seed)


def random_edge_order(graph: nx.Graph, *, seed: int = 0) -> list[tuple[str, str]]:
    """Return all edges in a deterministic random removal order.

    Endpoints are reported in canonical ``(lower_id, higher_id)`` order and the
    permutation is drawn from ``numpy.random.default_rng(seed)``.
    """
    seed = _validate_seed(seed)
    edges = sorted(_edge_key(u, v) for u, v in graph.edges)
    return _random_order(edges, seed)


def target_order(graph: nx.Graph, measure: str = "degree", *, static: bool = True) -> list[str]:
    """Return the station ranking used by a targeted node attack.

    Parameters
    ----------
    graph:
        Graph to rank; it is never mutated.
    measure:
        ``degree``, ``betweenness`` or ``flow``.
    static:
        Removal policy flag for callers. ``static=True`` (the default) means
        the caller removes the targets in this once-computed order.
        ``static=False`` means the caller recomputes this ranking on the
        surviving graph before every removal; :func:`percolation_curve` does
        that only when it is called with ``static=False``. The returned list is
        the ranking of the graph passed in either way.

    Returns
    -------
    list[str]
        Stations in descending measure order, ties by sorted station id.
    """
    if not isinstance(static, bool):
        raise TypeError(f"static must be a bool, got {static!r}")
    return rank_targets(graph, measure)


def removal_batch(
    graph: nx.Graph,
    *,
    attack: str = "random",
    measure: str = "degree",
    count: int = 1,
    seed: int = 0,
    static: bool = True,
    order: Sequence[Target] | None = None,
) -> list[Target]:
    """Select the next removal batch through the one shared code path.

    Parameters
    ----------
    graph:
        Current graph; it is never mutated.
    attack:
        ``random``, ``targeted``, ``random_edge`` or ``targeted_edge``.
    measure:
        Ranking measure; node and edge measures are listed in the module
        docstring. Ignored by the random attacks.
    count:
        Number of targets to return, at most the number of remaining targets.
    seed:
        Seed of the random permutation used by the random attacks, and recorded
        by :func:`percolation_curve` for every run.
    static:
        Targeted attacks only. ``True`` removes the top ``count`` of a ranking
        computed once (pass the precomputed list via ``order`` to avoid
        recomputing it per run). ``False`` recomputes the ranking on the graph
        passed in, which is the single step of a dynamic attack: call again
        after each removal.
    order:
        Precomputed ranking from :func:`rank_targets` or
        :func:`rank_edge_targets`; used verbatim when given.

    Returns
    -------
    list[str] or list[tuple[str, str]]
        Stations for the node attacks, endpoints for the edge attacks, in
        stable order. Ties are broken by sorted station id.

    Notes
    -----
    This is the shared removal generator: :func:`percolation_curve` calls it
    for every attack, and a P1.4 cascade trigger can call it with ``count=1``
    to pick its first station by the same rules.
    """
    _validate_attack(attack)
    edges = attack in _EDGE_ATTACKS
    _validate_measure(measure, EDGE_MEASURES if edges else NODE_MEASURES)
    count = _validate_count(count)
    seed = _validate_seed(seed)
    if not isinstance(static, bool):
        raise TypeError(f"static must be a bool, got {static!r}")

    if attack.startswith("random"):
        full: list[Target] = (
            random_edge_order(graph, seed=seed)
            if edges
            else random_target_order(graph, seed=seed)
        )
        return full[:count]

    if order is not None:
        return list(order)[:count]
    full = (
        rank_edge_targets(graph, measure)
        if edges
        else rank_targets(graph, measure)
    )
    return full[:count]


def _validated_fractions(fractions: Sequence[float]) -> list[float]:
    if isinstance(fractions, (str, bytes)):
        raise TypeError("fractions must be a sequence of numbers in [0, 1]")
    values = [float(value) for value in fractions]
    if not values:
        raise ValueError("fractions must contain at least one value")
    for value in values:
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"fractions must lie in [0, 1], got {value!r}")
    return values


def _derived_seed(master: int, fraction_index: int, repetition: int) -> int:
    """Derive one run seed from the master seed, fraction and repetition index.

    ``numpy.random.SeedSequence([master, fraction_index, repetition])`` is
    stable for a fixed numpy version, so a fixed master seed and input produce
    identical curve rows and the returned integer is written to the ``seed``
    column of :func:`percolation_curve`.
    """
    sequence = np.random.SeedSequence([master, fraction_index, repetition])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _component_sizes(graph: nx.Graph) -> tuple[int, int, int]:
    """Return ``(gcc, lcc, isolated)`` sizes of the current graph."""
    sizes = sorted((len(component) for component in nx.connected_components(graph)), reverse=True)
    gcc = sizes[0] if sizes else 0
    lcc = sizes[1] if len(sizes) > 1 else 0
    isolated = sum(1 for node in graph.nodes if graph.degree(node) == 0)
    return gcc, lcc, isolated


def _remove_targets(graph: nx.Graph, batch: Sequence[Target], *, edges: bool) -> None:
    if edges:
        graph.remove_edges_from(batch)  # type: ignore[arg-type]
    else:
        graph.remove_nodes_from(batch)  # type: ignore[arg-type]


def percolation_curve(
    graph: nx.Graph,
    *,
    attack: str = "random",
    measure: str = "degree",
    fractions: Sequence[float],
    n_seeds: int = 100,
    seed: int = 0,
    static: bool = True,
) -> pd.DataFrame:
    """Sweep a percolation curve over removal fractions on ``graph``.

    For every fraction ``f`` the number of targets removed is
    ``min(total, int(round(f * total)))``, where ``total`` is the number of
    nodes for the node attacks and the number of edges for the edge attacks.
    Each fraction is repeated ``n_seeds`` times; run ``j`` of fraction ``i``
    uses the deterministic per-run seed
    ``SeedSequence([seed, i, j]).generate_state(1)[0]``, which is recorded in
    the ``seed`` column. Fraction 0 records the intact baseline (one row per
    seed), and fraction 1 removes every node or every edge.

    Parameters
    ----------
    graph:
        Intact graph. It is never mutated. Attach ``trips`` and
        ``trips_served`` attributes for the weighted load measures.
    attack:
        ``random``, ``targeted``, ``random_edge`` or ``targeted_edge``.
    measure:
        Ranking measure; node or edge measures per the module docstring.
    fractions:
        Removal fractions, each in ``[0, 1]``; at least one value.
    n_seeds:
        Number of independent runs per fraction, an integer ``>= 1``.
    seed:
        Master seed, a non-negative integer.
    static:
        ``True`` (frozen default) computes the targeted ranking once and
        removes it in order. ``False`` recomputes the ranking with
        :func:`target_order` after every single removal.

    Returns
    -------
    pandas.DataFrame
        One row per (fraction, seed) with the stable columns
        :data:`CURVE_COLUMNS`: ``fraction``, ``seed``, ``attack``, ``measure``,
        ``n_initial``, ``gcc_size``, ``gcc_fraction``, ``lcc_size``,
        ``lcc_fraction`` and ``isolated``. Fractions keep the order given.
        The second-largest component and the isolated count match the metric
        definitions of ``docs/model.md`` section 6. For an empty input graph
        ``gcc_fraction`` and ``lcc_fraction`` are 0.0.
    """
    _validate_attack(attack)
    edges = attack in _EDGE_ATTACKS
    _validate_measure(measure, EDGE_MEASURES if edges else NODE_MEASURES)
    if isinstance(n_seeds, bool) or not isinstance(n_seeds, (int, np.integer)):
        raise TypeError(f"n_seeds must be an integer, got {n_seeds!r}")
    if int(n_seeds) < 1:
        raise ValueError(f"n_seeds must be >= 1, got {n_seeds!r}")
    n_seeds = int(n_seeds)
    seed = _validate_seed(seed)
    if not isinstance(static, bool):
        raise TypeError(f"static must be a bool, got {static!r}")
    fraction_values = _validated_fractions(fractions)

    n_initial = graph.number_of_nodes()
    total = graph.number_of_edges() if edges else n_initial

    static_order: Sequence[Target] | None = None
    if attack in _TARGETED_ATTACKS and static:
        static_order = (
            rank_edge_targets(graph, measure)
            if edges
            else rank_targets(graph, measure)
        )

    rows: list[dict[str, object]] = []
    for index, fraction in enumerate(fraction_values):
        count = min(total, int(round(fraction * total)))
        for repetition in range(n_seeds):
            run_seed = _derived_seed(seed, index, repetition)
            working = graph.copy()
            if count > 0 and attack in _TARGETED_ATTACKS and not static:
                for _ in range(count):
                    batch = removal_batch(
                        working,
                        attack=attack,
                        measure=measure,
                        count=1,
                        seed=run_seed,
                        static=False,
                    )
                    _remove_targets(working, batch, edges=edges)
            elif count > 0:
                batch = removal_batch(
                    working,
                    attack=attack,
                    measure=measure,
                    count=count,
                    seed=run_seed,
                    static=static,
                    order=static_order,
                )
                _remove_targets(working, batch, edges=edges)
            gcc, lcc, isolated = _component_sizes(working)
            rows.append(
                {
                    "fraction": float(fraction),
                    "seed": run_seed,
                    "attack": attack,
                    "measure": measure,
                    "n_initial": n_initial,
                    "gcc_size": gcc,
                    "gcc_fraction": gcc / n_initial if n_initial else 0.0,
                    "lcc_size": lcc,
                    "lcc_fraction": lcc / n_initial if n_initial else 0.0,
                    "isolated": isolated,
                }
            )
    return pd.DataFrame(rows, columns=list(CURVE_COLUMNS))


def critical_fraction(
    curve: pd.DataFrame,
    *,
    threshold: float = 0.5,
    column: str = "gcc_fraction",
) -> float:
    """Estimate the removal fraction at which the curve collapses.

    The primary estimate is the susceptibility peak of the second-largest
    component. With ``n_initial`` from the curve and the mean of ``lcc_size``
    squared over the seeds at each fraction::

        chi(f) = mean_seeds(lcc_size ** 2) / n_initial

    The returned estimate is the smallest fraction whose ``chi`` is maximal
    (``numpy.argmax`` picks the first tie). The fallback, used when ``chi`` is
    degenerate (all zero or constant, so it has no peak), is the first fraction
    where the mean of ``column`` is at most ``threshold``, linearly
    interpolated between the surrounding two fractions. The result is clamped
    to ``[0, 1]``.

    Parameters
    ----------
    curve:
        Table from :func:`percolation_curve` (or the same columns by hand):
        ``fraction``, ``column``, ``lcc_size`` and ``n_initial``.
    threshold:
        Fallback crossing level in ``[0, 1]``.
    column:
        Column whose mean crossing defines the fallback, ``gcc_fraction`` by
        default.

    Returns
    -------
    float
        The estimate in ``[0, 1]``, or ``float("nan")`` when the curve never
        collapses: the mean of ``column`` stays above ``threshold`` at every
        fraction, so neither a crossing nor a collapse exists.

    Raises
    ------
    TypeError
        If ``curve`` is not a :class:`pandas.DataFrame`.
    ValueError
        If a required column is missing, a value is NaN, or ``threshold`` lies
        outside ``[0, 1]``.
    """
    if not isinstance(curve, pd.DataFrame):
        raise TypeError(f"curve must be a pandas.DataFrame, got {type(curve).__name__}")
    required = ("fraction", column, "lcc_size", "n_initial")
    missing = [name for name in required if name not in curve.columns]
    if missing:
        raise ValueError(f"curve is missing the columns {missing!r}")
    threshold = float(threshold)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must lie in [0, 1], got {threshold!r}")

    frame = curve[list(required)].astype(float)
    if frame.isna().any().any():
        raise ValueError("curve contains missing or non-numeric values")
    if not frame["fraction"].between(0.0, 1.0).all():
        raise ValueError("curve fractions must lie in [0, 1]")

    grouped = frame.groupby("fraction", sort=True).agg(
        value=(column, "mean"),
        lcc_square=(
            "lcc_size",
            lambda values: float(np.mean(np.square(values.to_numpy(dtype=float)))),
        ),
        n=("n_initial", "mean"),
    )
    fractions = grouped.index.to_numpy(dtype=float)
    values = grouped["value"].to_numpy(dtype=float)
    sizes = grouped["n"].to_numpy(dtype=float)
    chi = np.divide(
        grouped["lcc_square"].to_numpy(dtype=float),
        sizes,
        out=np.zeros_like(sizes),
        where=sizes > 0.0,
    )

    below = np.flatnonzero(values <= threshold)
    if below.size == 0:
        return float("nan")  # documented sentinel: the curve never collapses
    first = int(below[0])
    if first == 0:
        crossing = float(fractions[0])
    else:
        f0, f1 = float(fractions[first - 1]), float(fractions[first])
        v0, v1 = float(values[first - 1]), float(values[first])
        crossing = f1 if v1 == v0 else f0 + (threshold - v0) * (f1 - f0) / (v1 - v0)
        crossing = min(max(crossing, 0.0), 1.0)

    peak_usable = (
        chi.size > 0
        and np.isfinite(chi).all()
        and float(chi.max()) > 0.0
        and float(chi.max()) > float(chi.min())
    )
    if peak_usable:
        return float(fractions[int(np.argmax(chi))])
    return float(crossing)