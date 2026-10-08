"""Multilayer recovery scenarios, demand weights and the layered cascade model.

P2.3 (issue #33) builds the RQ3 experiment on the P1.5 terminal/facility
layers and the P1.6 budgeted strategies. This module owns three pieces:

* scenario assembly on the frozen rail graph: the exclusive stations of every
  line, the major interchanges, and seeded random failure sets;
* OD demand weights from the frozen ``am_peak_stops`` boarding proxy, used to
  turn the unweighted served-OD fraction into a served-demand fraction;
* :func:`layered_cascade`, the port of ``prototype/bus_backup_cascade.py``:
  terminal-subset loads with capacities fixed from the intact rail-only
  baseline, synchronous per-round rail closures after an initial failure batch,
  and buses that are already active when the first post-trigger loads are
  checked. Buses never prevent the initial closures; they only change the
  redistribution that follows.

The prototype starts from a single trigger. This port generalises the trigger
to an initial batch of station IDs, matching the shared line-closure scenario
of the plan (a set is removed synchronously and is excluded from the
avalanche sizes).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from numbers import Integral

import networkx as nx
import numpy as np
import pandas as pd

from transperth.config import DEFAULT_SEED, FAILURE_TOLERANCE
from transperth.experiments import child_seeds
from transperth.multilayer import (
    RAIL_PREFIX,
    TERMINAL_PREFIX,
    terminal_loads,
    terminal_pairs,
    travel_times,
)

__all__ = [
    "ENDPOINT_SUM_MODEL",
    "LayeredCascade",
    "endpoint_demand_weights",
    "exclusive_line_stations",
    "layered_cascade",
    "major_interchange_stations",
    "random_failure_sets",
    "recovery_metrics",
    "scenario_manifest",
]

Pair = tuple[str, str]

#: Label of the documented OD weighting rule; stored in the result sidecar.
ENDPOINT_SUM_MODEL = "endpoint-sum"


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------
def _count(value: int, name: str, *, minimum: int = 0) -> int:
    """Return ``value`` as an integer >= ``minimum`` or raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _station_text(value: object) -> str:
    """Normalise a station identifier, dropping a layer prefix if present."""
    text = str(value).strip()
    if text.startswith((TERMINAL_PREFIX, RAIL_PREFIX)):
        text = text[2:]
    if not text:
        raise ValueError("station ID cannot be empty")
    return text


def _station_pair(pair: Iterable[object]) -> Pair:
    """Return a canonical undirected pair of raw station IDs."""
    try:
        first, second = pair
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a station pair, got {pair!r}") from error
    canonical = tuple(sorted((_station_text(first), _station_text(second))))
    if canonical[0] == canonical[1]:
        raise ValueError("self-pairs are not allowed")
    return canonical


def station_lines(rail: nx.Graph, station: str) -> set[str]:
    """Return the corridor labels of ``station`` as a set of stripped names.

    The frozen loader stores ``lines`` as a list; fixtures and older tables may
    store a semicolon string. Both representations are accepted.
    """
    raw = rail.nodes[station].get("lines", "")
    if isinstance(raw, str):
        values = raw.split(";")
    else:
        try:
            values = list(raw)
        except TypeError as error:
            raise ValueError(
                f"station {station!r} lines must be a semicolon string or a list"
            ) from error
    return {str(value).strip() for value in values if str(value).strip()}


# ---------------------------------------------------------------------------
# scenario assembly
# ---------------------------------------------------------------------------
def exclusive_line_stations(rail: nx.Graph) -> dict[str, tuple[str, ...]]:
    """Return the stations served by exactly one line, per line.

    This is the prototype's line-closure definition
    (``prototype/cascade.py`` experiment C): a station is exclusive to a line
    when no other line also lists it. Lines with no exclusive station appear
    with an empty tuple, so the mapping always covers every frozen line.
    """
    if not isinstance(rail, nx.Graph) or rail.is_directed() or rail.is_multigraph():
        raise TypeError("rail must be an undirected simple networkx.Graph")
    membership: dict[str, set[str]] = {}
    for station in sorted(rail):
        for line in station_lines(rail, station):
            membership.setdefault(line, set()).add(station)
    exclusive: dict[str, tuple[str, ...]] = {}
    for line in sorted(membership):
        shared = set().union(
            *(membership[other] for other in membership if other != line), set()
        )
        exclusive[line] = tuple(sorted(membership[line] - shared))
    return exclusive


def major_interchange_stations(
    rail: nx.Graph, *, min_lines: int = 3
) -> tuple[str, ...]:
    """Return stations served by at least ``min_lines`` lines, sorted.

    ``min_lines`` must be at least two: a single line is not an interchange.
    """
    minimum = _count(min_lines, "min_lines", minimum=2)
    return tuple(
        station
        for station in sorted(rail)
        if len(station_lines(rail, station)) >= minimum
    )


def random_failure_sets(
    rail: nx.Graph,
    *,
    n_sets: int = 5,
    size: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[tuple[str, ...]]:
    """Return ``n_sets`` seeded failure sets of ``size`` distinct stations.

    Sets are drawn without replacement from the sorted station list with the
    P1.2 ``child_seeds`` scheme, so the output is reproducible from the master
    seed and independent across sets.
    """
    count = _count(n_sets, "n_sets")
    stations = sorted(rail)
    width = _count(size, "size", minimum=1)
    if width > len(stations):
        raise ValueError(
            f"size {width} exceeds the {len(stations)} available stations"
        )
    if count == 0:
        return []
    sets = []
    for child in child_seeds(seed, count):
        rng = np.random.default_rng(child)
        chosen = rng.choice(np.array(stations, dtype=object), size=width, replace=False)
        sets.append(tuple(sorted(str(station) for station in chosen)))
    return sets


SCENARIO_COLUMNS = ("scenario", "kind", "label", "n_failed", "failed_ids")


def scenario_manifest(
    rail: nx.Graph,
    *,
    n_random: int = 5,
    random_size: int = 5,
    min_interchange_lines: int = 3,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Return the frozen failure scenarios as one manifest table.

    One row per scenario with columns ``scenario``, ``kind`` (``line_closure``,
    ``interchange`` or ``random``), ``label``, ``n_failed`` and the
    semicolon-separated ``failed_ids``. Line closures skip lines with no
    exclusive station; the row order is deterministic (lines, interchanges,
    random sets).
    """
    rows = []
    for line, stations in exclusive_line_stations(rail).items():
        if not stations:
            continue
        rows.append(
            {
                "scenario": f"line_closure:{line}",
                "kind": "line_closure",
                "label": line,
                "n_failed": len(stations),
                "failed_ids": ";".join(stations),
            }
        )
    for station in major_interchange_stations(rail, min_lines=min_interchange_lines):
        rows.append(
            {
                "scenario": f"interchange:{station}",
                "kind": "interchange",
                "label": str(rail.nodes[station].get("name", station)),
                "n_failed": 1,
                "failed_ids": station,
            }
        )
    for index, stations in enumerate(
        random_failure_sets(rail, n_sets=n_random, size=random_size, seed=seed)
    ):
        rows.append(
            {
                "scenario": f"random:{index}",
                "kind": "random",
                "label": f"seeded set {index}",
                "n_failed": len(stations),
                "failed_ids": ";".join(stations),
            }
        )
    return pd.DataFrame(rows, columns=list(SCENARIO_COLUMNS))


# ---------------------------------------------------------------------------
# OD demand weights
# ---------------------------------------------------------------------------
def endpoint_demand_weights(
    pairs: Iterable[tuple[object, object]],
    demand: Mapping[str, float],
) -> dict[Pair, float]:
    """Weight each station pair by the sum of its endpoint demand proxies.

    ``demand`` maps raw station IDs to a finite non-negative proxy, normally
    the frozen ``am_peak_stops`` boardings count. The documented endpoint-sum
    model (:data:`ENDPOINT_SUM_MODEL`) makes a pair weight the sum of its
    endpoints' values, because each endpoint's scheduled departures generate
    trips to and from the pair, so the weight scales with either end. This is a
    scenario weighting, not an estimate of OD demand.

    Pair endpoints may carry a ``T:``/``R:`` prefix, which is stripped. Every
    endpoint must exist in ``demand``; missing stations are an error rather
    than a silent zero. Duplicate pairs are counted once.
    """
    weights: dict[Pair, float] = {}
    for pair in pairs:
        key = _station_pair(pair)
        if key in weights:
            continue
        values = []
        for station in key:
            if station not in demand:
                raise ValueError(f"unknown demand station {station!r}")
            try:
                value = float(demand[station])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"demand for {station!r} must be a finite number >= 0"
                ) from error
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"demand for {station!r} must be a finite number >= 0")
            values.append(value)
        weights[key] = values[0] + values[1]
    return weights


# ---------------------------------------------------------------------------
# layered cascade
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class LayeredCascade:
    """Outcome of one layered cascade on a terminal/facility graph.

    ``failed`` lists the raw station IDs in failure order: the initial batch
    first, then one group per synchronous round. ``avalanche_sizes`` holds the
    size of every secondary round, so ``n_secondary == sum(avalanche_sizes)``
    and ``rounds == len(avalanche_sizes)``. ``final_graph`` is the stable
    layered graph after all closures; it is excluded from equality so results
    can be compared on their fields alone. ``history`` and ``loads`` are set
    only for traced runs.
    """

    n_initial: int
    initial_failed: tuple[str, ...]
    failed: tuple[str, ...]
    avalanche_sizes: tuple[int, ...]
    rounds: int
    final_graph: nx.Graph = field(compare=False, repr=False)
    history: pd.DataFrame | None = field(default=None, compare=False, repr=False)
    loads: pd.DataFrame | None = field(default=None, compare=False, repr=False)

    @property
    def n_failed(self) -> int:
        """Number of failed rail facilities, including the initial batch."""
        return len(self.failed)

    @property
    def n_secondary(self) -> int:
        """Number of failures after the initial batch."""
        return self.n_failed - len(self.initial_failed)

    @property
    def failed_fraction(self) -> float:
        """Failed rail facilities divided by the intact rail facility count."""
        return self.n_failed / self.n_initial if self.n_initial else 0.0


def _checked_layered(graph: nx.Graph, label: str) -> nx.Graph:
    """Validate an undirected simple layered graph and return it unchanged."""
    if not isinstance(graph, nx.Graph) or graph.is_directed() or graph.is_multigraph():
        raise TypeError(f"{label} must be an undirected simple networkx.Graph")
    if any(not isinstance(node, str) for node in graph):
        raise ValueError(f"{label} node IDs must be strings")
    return graph


def layered_cascade(
    normal: nx.Graph,
    standby: nx.Graph,
    *,
    alpha: float,
    initial_failed: Iterable[object] = (),
    tolerance: float = FAILURE_TOLERANCE,
    overload_enabled: bool = True,
    trace: bool = False,
    baseline_pairs: Iterable[tuple[object, object]] | None = None,
    baseline_times: Mapping[tuple[str, str], float] | None = None,
) -> LayeredCascade:
    """Run the prototype layered cascade with an optional initial failure batch.

    ``normal`` is the intact rail-only layered graph that fixes the terminal
    subset loads and capacities ``C_i = (1 + alpha) * L0_i`` for the whole run.
    ``standby`` is the deployed graph the cascade actually evolves on; it must
    not contain nodes outside ``normal`` and may omit exactly the facilities in
    ``initial_failed``, so callers can pass either the intact graph or the
    already-failed baseline. It is never modified. The initial
    ``initial_failed`` station IDs are removed as one synchronous batch before
    the first load check, so the batch is excluded from ``avalanche_sizes``.
    Buses in ``standby`` are therefore active from the first post-trigger round
    on, including the round-0 check.

    Every round recomputes :func:`transperth.multilayer.terminal_loads` on the
    surviving graph and closes every rail facility strictly above its fixed
    capacity, together, before the next round. Rounds are algorithmic updates,
    not minutes. ``overload_enabled=False`` ports the prototype's control: the
    initial batch is removed, but no secondary closure can follow.

    With ``trace=True`` the returned ``history`` (one row per round) and
    ``loads`` (every surviving facility per round) tables follow the prototype
    logs. ``baseline_pairs`` and ``baseline_times`` only feed the traced
    service metrics; when omitted they are computed from ``normal``.
    """
    _checked_layered(normal, "normal")
    _checked_layered(standby, "standby")
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be a finite number >= 0") from error
    if not math.isfinite(alpha_value) or alpha_value < 0.0:
        raise ValueError("alpha must be a finite number >= 0")
    try:
        tolerance_value = float(tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError("tolerance must be a finite number >= 0") from error
    if not math.isfinite(tolerance_value) or tolerance_value < 0.0:
        raise ValueError("tolerance must be a finite number >= 0")
    if not isinstance(overload_enabled, bool) or not isinstance(trace, bool):
        raise ValueError("overload_enabled and trace must be bool")

    batch: list[str] = []
    for station in initial_failed:
        identifier = _station_text(station)
        node = RAIL_PREFIX + identifier
        if node not in normal:
            raise ValueError(f"unknown failed station {identifier!r}")
        if identifier not in batch:
            batch.append(identifier)
    batch.sort()

    extra = set(standby) - set(normal)
    if extra:
        raise ValueError("standby must not contain nodes outside normal")
    allowed_missing = {RAIL_PREFIX + station for station in batch}
    unexpected = sorted(set(normal) - set(standby) - allowed_missing)
    if unexpected:
        raise ValueError(
            "standby is missing nodes that are not in initial_failed: "
            + ", ".join(unexpected)
        )

    loads0 = terminal_loads(normal)
    limits = {node: (1.0 + alpha_value) * value for node, value in loads0.items()}
    current = standby.copy()
    current.remove_nodes_from(RAIL_PREFIX + station for station in batch)
    failed = [RAIL_PREFIX + station for station in batch]
    avalanches: list[int] = []
    history_rows: list[dict[str, object]] = []
    load_rows: list[dict[str, object]] = []

    if trace:
        if baseline_pairs is None:
            pairs = terminal_pairs(normal)
        else:
            pairs = {_station_pair(pair) for pair in baseline_pairs}
        baseline = {_terminal_pair(pair) for pair in pairs}
        if baseline_times is None:
            times0 = travel_times(normal, pairs=baseline)
        else:
            times0 = dict(baseline_times)
    round_no = 0
    removed = list(batch)
    while True:
        current_loads = terminal_loads(current)
        overloaded = (
            sorted(
                node
                for node, value in current_loads.items()
                if value > limits[node] + tolerance_value
            )
            if overload_enabled
            else []
        )
        if trace:
            if baseline:
                current_times = travel_times(current, pairs=baseline)
                changes = [
                    current_times[pair] - times0[pair]
                    for pair in current_times
                    if pair in times0
                ]
            else:
                current_times = {}
                changes = []
            history_rows.append(
                {
                    "round": round_no,
                    "n_failed": len(failed),
                    "n_secondary": len(failed) - len(batch),
                    "removed_ids": ";".join(removed),
                    "next_failed_ids": ";".join(node[2:] for node in overloaded),
                    "reachable_pairs": len(current_times),
                    "original_pairs": len(baseline),
                    "unmet_fraction": 1.0 - len(current_times) / len(baseline)
                    if baseline
                    else 0.0,
                    "mean_time_change_reachable_min": float(np.mean(changes))
                    if changes
                    else math.nan,
                }
            )
            for node, value in current_loads.items():
                load_rows.append(
                    {
                        "round": round_no,
                        "station_id": node[2:],
                        "name": str(current.nodes[node].get("name", node[2:])),
                        "initial_load": loads0[node],
                        "load": value,
                        "capacity": limits[node],
                        "load_ratio": value / limits[node] if limits[node] > 0 else math.nan,
                        "overloaded": node in overloaded,
                        "closes_next_round": node in overloaded,
                    }
                )
        if not overloaded:
            break
        avalanches.append(len(overloaded))
        failed.extend(overloaded)
        current.remove_nodes_from(overloaded)
        removed = [node[2:] for node in overloaded]
        round_no += 1

    history = pd.DataFrame(history_rows) if trace else None
    loads = pd.DataFrame(load_rows) if trace else None
    return LayeredCascade(
        n_initial=len(loads0),
        initial_failed=tuple(batch),
        failed=tuple(failed),
        avalanche_sizes=tuple(avalanches),
        rounds=round_no,
        final_graph=current,
        history=history,
        loads=loads,
    )


# ---------------------------------------------------------------------------
# service metrics
# ---------------------------------------------------------------------------
def _terminal_pair(pair: Iterable[object]) -> Pair:
    """Return the canonical ``T:`` pair of a raw or prefixed station pair."""
    first, second = _station_pair(pair)
    return (TERMINAL_PREFIX + first, TERMINAL_PREFIX + second)


def recovery_metrics(
    recovered: nx.Graph,
    intact: nx.Graph,
    *,
    baseline_pairs: Iterable[tuple[object, object]] | None = None,
    intact_times: Mapping[tuple[str, str], float] | None = None,
    demand: Mapping[Pair, float] | None = None,
) -> dict[str, float | int]:
    """Measure served OD, served demand and the travel-time penalty.

    ``baseline_pairs`` is the intact terminal-pair denominator; pairs that are
    unreachable in ``recovered`` count as unserved. ``intact_times`` is an
    optional cache of the intact shortest-path times. ``demand`` (raw station
    pair -> finite non-negative weight) must cover a subset of the baseline;
    the served fraction is the reachable weight over the total supplied weight.

    ``mean_travel_time_penalty_min`` averages ``recovered - intact`` minutes
    over the baseline pairs reachable in the recovered graph, so a positive
    value means slower retained service. The population is conditional, which
    is why ``reachable_pair_count`` is always reported next to it.
    """
    if baseline_pairs is None:
        pairs = {_station_pair(pair) for pair in terminal_pairs(intact)}
    else:
        pairs = {_station_pair(pair) for pair in baseline_pairs}
    baseline = {_terminal_pair(pair) for pair in pairs}
    times = travel_times(recovered, pairs=baseline)
    if intact_times is None:
        intact_times = travel_times(intact, pairs=baseline)
    result: dict[str, float | int] = {
        "total_pair_count": len(baseline),
        "reachable_pair_count": len(times),
        "served_od_fraction": len(times) / len(baseline) if baseline else 1.0,
    }
    changes = [times[pair] - intact_times[pair] for pair in times if pair in intact_times]
    result["mean_reachable_minutes"] = (
        float(np.mean(list(times.values()))) if times else math.nan
    )
    result["mean_travel_time_penalty_min"] = (
        float(np.mean(changes)) if changes else math.nan
    )
    if demand is not None:
        weights: dict[Pair, float] = {}
        for raw_pair, raw_weight in demand.items():
            key = _station_pair(raw_pair)
            if key not in pairs:
                raise ValueError(f"demand pair {key!r} is not in the baseline pairs")
            try:
                weight = float(raw_weight)
            except (TypeError, ValueError) as error:
                raise ValueError("demand weights must be finite numbers >= 0") from error
            if not math.isfinite(weight) or weight < 0.0:
                raise ValueError("demand weights must be finite numbers >= 0")
            weights[key] = weight
        total = math.fsum(weights.values())
        served = math.fsum(
            weight
            for pair, weight in weights.items()
            if _terminal_pair(pair) in times
        )
        result["total_demand_weight"] = total
        result["served_demand_weight"] = served
        result["served_demand_fraction"] = served / total if total else 1.0
    return result