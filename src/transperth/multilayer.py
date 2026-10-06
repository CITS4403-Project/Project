"""Layered terminal/facility model for the Transperth rail and bus network.

Every mapped rail station becomes two layer nodes:

* a terminal ``T:<station_id>``, which persists when railway infrastructure
  fails, and
* a rail facility ``R:<station_id>``, which can be closed by a cascade or an
  experiment.

The two are joined by an access edge of a fixed one minute (see
``docs/model.md`` section 5). Rail edges join ``R:`` facilities with a time
derived from the great-circle ``distance_m`` and ``speed_kmh``. Backup edges
join terminals: a passenger boards the substitute service at the station
entrance, so the recorded effective time already covers walking and waiting.

The four pairs in ``data/processed/backup_edges.csv`` are manual ground truth.
:func:`derive_gtfs_candidates` adds reproducible candidates from the frozen
snapshot (single bus trips that stop near two different mapped rail stations),
and :func:`compare_manual_pairs` reports every manual pair against them.
Disagreements are listed, never dropped. Standby buses activate immediately
after the trigger and have unlimited capacity in the baseline scenario;
activation scheduling itself belongs to P2.3.

The walking interchange between Perth (56) and Perth Underground (64) is not a
rail edge. :func:`add_walking_transfers` adds it as an explicit terminal edge
with :data:`WALKING_TRANSFER_MINUTES` (five minutes, the ``min_transfer_time``
of 300 seconds recorded for the pair in the snapshot's ``transfers.txt``).
:func:`build_layers` never adds it on its own.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from transperth.config import STATIONS_CSV

__all__ = [
    "BACKUP_KINDS",
    "DEFAULT_MATCH_TOLERANCE_MINUTES",
    "EARTH_RADIUS_M",
    "WALKING_TRANSFER_MINUTES",
    "add_backup_edges",
    "add_walking_transfers",
    "build_layers",
    "compare_manual_pairs",
    "derive_gtfs_candidates",
    "served_od_fraction",
    "terminal_loads",
    "terminal_pairs",
    "travel_times",
]

TERMINAL_PREFIX = "T:"
RAIL_PREFIX = "R:"
BACKUP_KINDS = frozenset({"existing_bus", "emergency_bus", "gtfs_candidate"})
WALKING_TRANSFER_MINUTES = 5.0
DEFAULT_MATCH_TOLERANCE_MINUTES = 5.0
EARTH_RADIUS_M = 6_371_000.0

_REQUIRED_SNAPSHOT_FILES = ("stops.txt", "stop_times.txt", "bus_trips.csv")
_RAIL_MINUTES_FLOOR = 0.1
_TIE_METRES = 1e-9
_CANDIDATE_COLUMNS = (
    "station_a",
    "station_b",
    "minutes",
    "kind",
    "route_ids",
    "trip_count",
    "evidence_trip_id",
    "evidence_from_stop",
    "evidence_to_stop",
    "evidence_from_sequence",
    "evidence_to_sequence",
)


# ---------------------------------------------------------------------------
# small validation helpers
# ---------------------------------------------------------------------------
def _positive_float(value: Any, *, label: str) -> float:
    """Return ``value`` as a finite float strictly greater than zero."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a finite number > 0") from error
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a finite number > 0")
    return number


def _station_id(value: Any) -> str:
    """Normalise a station identifier to its frozen string form."""
    return str(value).strip()


def _text(value: Any) -> str:
    """Return a provenance value as text, mapping missing values to ``""``."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """Return the undirected pair in the frozen lexical order."""
    return (a, b) if a <= b else (b, a)


def _terminal_nodes(layered: nx.Graph) -> list[str]:
    return sorted(
        node for node in layered if isinstance(node, str) and node.startswith(TERMINAL_PREFIX)
    )


def _facility_nodes(layered: nx.Graph) -> list[str]:
    return sorted(node for node in layered if isinstance(node, str) and node.startswith(RAIL_PREFIX))


def _set_fastest_edge(
    graph: nx.Graph,
    a: str,
    b: str,
    *,
    minutes: float,
    attributes: Mapping[str, Any],
) -> None:
    """Add or replace an undirected edge only when it is strictly faster.

    Keeping the first of two equal-time candidates makes the result depend
    only on the input order, which is deterministic for a fixed input.
    """
    if graph.has_edge(a, b) and graph[a][b].get("minutes", math.inf) <= minutes:
        return
    graph.add_edge(a, b, minutes=minutes, **dict(attributes))


# ---------------------------------------------------------------------------
# frozen public API
# ---------------------------------------------------------------------------
def build_layers(
    rail: nx.Graph,
    *,
    speed_kmh: float = 40.0,
    access_minutes: float = 1.0,
) -> nx.Graph:
    """Build the terminal/facility layer of ``rail``.

    Every rail node gets a persistent terminal ``T:<station_id>`` and a
    closable facility ``R:<station_id>`` joined by an access edge of
    ``access_minutes``. Rail edges join facilities with
    ``max(0.1, distance_m / 1000 / speed_kmh * 60)`` minutes and keep the
    source ``distance_m``. Nodes carry ``kind`` (``terminal``/``rail``),
    ``station_id`` and ``name``.

    The node order is the sorted station id, so the same inputs and parameters
    always produce the same graph. ``speed_kmh`` and ``access_minutes`` must
    be finite and positive; every rail edge needs a finite positive
    ``distance_m``. No walking interchange is added here.
    """
    speed = _positive_float(speed_kmh, label="speed_kmh")
    access = _positive_float(access_minutes, label="access_minutes")
    if not isinstance(rail, nx.Graph):
        raise TypeError("rail must be a networkx.Graph")

    layered = nx.Graph()
    nodes: dict[str, Any] = {}
    for node in rail.nodes:
        station_id = _station_id(node)
        if station_id in nodes:
            raise ValueError(f"duplicate station id after normalisation: {station_id!r}")
        nodes[station_id] = node
    for station_id in sorted(nodes):
        node = nodes[station_id]
        name = _text(rail.nodes[node].get("name")) or station_id
        layered.add_node(TERMINAL_PREFIX + station_id, kind="terminal", station_id=station_id, name=name)
        layered.add_node(RAIL_PREFIX + station_id, kind="rail", station_id=station_id, name=name)
        layered.add_edge(TERMINAL_PREFIX + station_id, RAIL_PREFIX + station_id, minutes=access, mode="access")

    for a, b, attributes in rail.edges(data=True):
        station_a, station_b = _station_id(a), _station_id(b)
        if station_a == station_b:
            raise ValueError(f"rail edge {station_a}-{station_b} is a self loop")
        distance = attributes.get("distance_m")
        try:
            distance = float(distance)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"rail edge {station_a}-{station_b}: distance_m must be a finite number > 0"
            ) from error
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError(
                f"rail edge {station_a}-{station_b}: distance_m must be a finite number > 0"
            )
        minutes = max(_RAIL_MINUTES_FLOOR, distance / 1000.0 / speed * 60.0)
        layered.add_edge(
            RAIL_PREFIX + station_a,
            RAIL_PREFIX + station_b,
            minutes=minutes,
            mode="rail",
            distance_m=distance,
        )
    return layered


def add_backup_edges(
    layered: nx.Graph,
    backup_table: pd.DataFrame,
    *,
    time_factor: float = 1.0,
) -> nx.Graph:
    """Return a copy of ``layered`` with the backup pairs of ``backup_table``.

    The table needs the columns ``station_a``, ``station_b``, ``minutes`` and
    ``kind``. Endpoints must name existing ``T:`` terminals, pairs must not be
    self loops, and the effective time ``minutes * time_factor`` must be
    finite and strictly positive. Allowed kinds are ``existing_bus``,
    ``emergency_bus`` and ``gtfs_candidate``; anything else raises
    ``ValueError`` naming the row. Duplicate endpoints keep the fastest
    effective time, whichever row order or endpoint orientation the table
    uses. Edges carry ``minutes``, ``mode="backup"``, the row ``kind`` and,
    when present, ``route_or_road`` (or ``route_ids``) and ``source``
    provenance. The input graph and table are never modified.
    """
    factor = _positive_float(time_factor, label="time_factor")
    if not isinstance(backup_table, pd.DataFrame):
        raise TypeError("backup_table must be a pandas.DataFrame")
    required = ("station_a", "station_b", "minutes", "kind")
    missing = [column for column in required if column not in backup_table.columns]
    if missing:
        raise ValueError("backup_table is missing required columns: " + ", ".join(missing))

    result = layered.copy()
    has_route_or_road = "route_or_road" in backup_table.columns
    has_route_ids = "route_ids" in backup_table.columns
    has_source = "source" in backup_table.columns
    for index, row in backup_table.iterrows():
        station_a = _station_id(row["station_a"])
        station_b = _station_id(row["station_b"])
        terminal_a, terminal_b = TERMINAL_PREFIX + station_a, TERMINAL_PREFIX + station_b
        if not result.has_node(terminal_a) or not result.has_node(terminal_b):
            raise ValueError(f"backup row {index!r}: unknown terminal {terminal_a!r} or {terminal_b!r}")
        if station_a == station_b:
            raise ValueError(f"backup row {index!r}: self pair {station_a!r}")
        kind = _text(row["kind"])
        if kind not in BACKUP_KINDS:
            raise ValueError(f"backup row {index!r}: unknown kind {row['kind']!r}")
        try:
            minutes = float(row["minutes"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"backup row {index!r}: minutes must be a finite number > 0") from error
        if not math.isfinite(minutes) or minutes <= 0.0:
            raise ValueError(f"backup row {index!r}: minutes must be a finite number > 0")
        effective = minutes * factor
        if not math.isfinite(effective) or effective <= 0.0:
            raise ValueError(f"backup row {index!r}: effective minutes must be a finite number > 0")
        route = _text(row["route_or_road"]) if has_route_or_road else ""
        if not route and has_route_ids:
            route = _text(row["route_ids"])
        _set_fastest_edge(
            result,
            terminal_a,
            terminal_b,
            minutes=effective,
            attributes={
                "mode": "backup",
                "kind": kind,
                "route_or_road": route,
                "source": _text(row["source"]) if has_source else "",
            },
        )
    return result


def add_walking_transfers(
    layered: nx.Graph,
    walking_transfers: pd.DataFrame | Mapping[str, Any] | Iterable[tuple[str, str]],
    *,
    minutes: float = WALKING_TRANSFER_MINUTES,
) -> nx.Graph:
    """Return a copy of ``layered`` with explicit walking terminal edges.

    ``walking_transfers`` may be a table with ``station_a``/``station_b``
    columns (an optional ``minutes`` column overrides the default), a mapping
    with ``station_a``/``station_b`` keys such as the ``walking_transfer``
    block of ``data/verified_topology.json``, or an iterable of station pairs.
    Edges join the two ``T:`` terminals with ``minutes``, ``mode="walk"`` and
    ``kind="walking_transfer"``. The default is
    :data:`WALKING_TRANSFER_MINUTES` (five minutes, the snapshot
    ``transfers.txt`` value for Perth/Perth Underground). The input graph is
    never modified.
    """
    default = _positive_float(minutes, label="minutes")
    result = layered.copy()
    for index, station_a, station_b, row_minutes in _iter_walking_transfers(walking_transfers):
        station_a, station_b = _station_id(station_a), _station_id(station_b)
        terminal_a, terminal_b = TERMINAL_PREFIX + station_a, TERMINAL_PREFIX + station_b
        if not result.has_node(terminal_a) or not result.has_node(terminal_b):
            raise ValueError(f"walking transfer {index!r}: unknown terminal {terminal_a!r} or {terminal_b!r}")
        if station_a == station_b:
            raise ValueError(f"walking transfer {index!r}: self pair {station_a!r}")
        walk_minutes = default if row_minutes is None else _positive_float(
            row_minutes, label=f"walking transfer {index!r} minutes"
        )
        _set_fastest_edge(
            result,
            terminal_a,
            terminal_b,
            minutes=walk_minutes,
            attributes={"mode": "walk", "kind": "walking_transfer"},
        )
    return result


def terminal_loads(layered: nx.Graph) -> dict[str, float]:
    """Return the terminal-subset weighted betweenness per ``R:`` facility.

    Sources and targets are all ``T:`` nodes; the shortest-path weight is the
    edge ``minutes`` attribute and the values are unnormalised
    (``normalized=False``), so they stay comparable as the graph shrinks.
    The keys are exactly the ``R:`` nodes of ``layered``.
    """
    terminals = _terminal_nodes(layered)
    values = nx.betweenness_centrality_subset(
        layered,
        sources=terminals,
        targets=terminals,
        normalized=False,
        weight="minutes",
    )
    return {node: float(values[node]) for node in _facility_nodes(layered)}


# ---------------------------------------------------------------------------
# OD helpers
# ---------------------------------------------------------------------------
def terminal_pairs(layered: nx.Graph) -> set[tuple[str, str]]:
    """Return the sorted undirected pairs of all ``T:`` nodes."""
    terminals = _terminal_nodes(layered)
    return {_canonical_pair(a, b) for i, a in enumerate(terminals) for b in terminals[i + 1 :]}


def travel_times(
    layered: nx.Graph,
    *,
    pairs: Iterable[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], float]:
    """Return shortest-path travel times in minutes for undirected node pairs.

    Pair endpoints are canonicalised to the frozen lexical order, so
    ``("T:C", "T:A")`` and ``("T:A", "T:C")`` address the same pair. Pairs
    that are unreachable, or that name a node missing from ``layered``, are
    omitted; callers that need the intact denominator should pass
    ``baseline_pairs`` to :func:`served_od_fraction` instead of recomputing
    pairs from the failed graph. With ``pairs=None`` every pair of ``T:``
    nodes is considered.
    """
    requested = terminal_pairs(layered) if pairs is None else {_canonical_pair(a, b) for a, b in pairs}
    by_source: dict[str, list[str]] = {}
    for source, target in requested:
        by_source.setdefault(source, []).append(target)
    times: dict[tuple[str, str], float] = {}
    for source in sorted(by_source):
        if source not in layered:
            continue
        lengths = nx.single_source_dijkstra_path_length(layered, source, weight="minutes")
        for target in sorted(by_source[source]):
            if target in lengths:
                times[(source, target)] = float(lengths[target])
    return times


def served_od_fraction(
    layered: nx.Graph,
    *,
    baseline_pairs: Iterable[tuple[str, str]] | None = None,
) -> float:
    """Return the reachable fraction of the baseline OD pairs.

    The denominator is the ``baseline_pairs`` supplied by the caller, so a
    failed graph keeps the intact-graph denominator: unreachable pairs count
    as unserved instead of dropping out. With ``baseline_pairs=None`` the
    undirected terminal pairs of the graph as-is are used. An empty baseline
    returns ``1.0`` (nothing can be unserved).
    """
    baseline = terminal_pairs(layered) if baseline_pairs is None else {
        _canonical_pair(a, b) for a, b in baseline_pairs
    }
    if not baseline:
        return 1.0
    reachable = travel_times(layered, pairs=baseline)
    return len(reachable) / len(baseline)


# ---------------------------------------------------------------------------
# GTFS candidate derivation
# ---------------------------------------------------------------------------
def derive_gtfs_candidates(
    snapshot_dir: str | Path,
    *,
    stations_csv: str | Path = STATIONS_CSV,
    radius_m: float = 400.0,
) -> pd.DataFrame:
    """Derive single-trip bus backup candidates from a frozen GTFS snapshot.

    Bus stops (``location_type=0`` with finite coordinates) are mapped to the
    nearest mapped rail station within ``radius_m`` of the great-circle
    distance (Earth radius 6,371,000 m); ties are broken by sorted station id.
    Only the selected bus trips of ``bus_trips.csv`` are inspected, and their
    full ``stop_times.txt`` sequences are read with GTFS service-day seconds
    (hours past 24 are valid). Every ordered stop pair that belongs to two
    different rail stations contributes the scheduled arrival at the
    destination stop minus the scheduled departure from the origin stop
    (falling back to the other timestamp on a row) when the difference is
    positive.

    Each undirected station pair appears once, sorted by
    ``(station_a, station_b)``, with the fastest observed time, the sorted
    semicolon route ids of all observations, the number of distinct
    contributing trips and one evidence trip/stop pair for provenance. The
    ``kind`` column is ``gtfs_candidate``. The result is deterministic for
    the same snapshot and parameters and drops non-positive or non-finite
    times.

    The snapshot tables are validated but never written. The large snapshot
    files are git-ignored, so callers should skip when they are absent.
    """
    radius = _positive_float(radius_m, label="radius_m")
    directory = Path(snapshot_dir)
    missing = [name for name in _REQUIRED_SNAPSHOT_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "snapshot tables missing: " + ", ".join(str(directory / name) for name in missing)
        )

    stations = _load_stations(stations_csv)
    stops = _read_gtfs_table(directory / "stops.txt", ("location_type", "stop_id", "stop_lat", "stop_lon"))
    if stops.stop_id.duplicated().any():
        raise ValueError("stops.txt has duplicate stop_id values")
    stops["stop_lat"] = pd.to_numeric(stops.stop_lat, errors="coerce")
    stops["stop_lon"] = pd.to_numeric(stops.stop_lon, errors="coerce")
    bus_stops = stops[
        stops.location_type.eq("0")
        & np.isfinite(stops.stop_lat.to_numpy(dtype=float))
        & np.isfinite(stops.stop_lon.to_numpy(dtype=float))
    ]
    stop_station = _nearest_station_lookup(bus_stops, stations, radius)

    trips = _read_gtfs_table(directory / "bus_trips.csv", ("trip_id", "route_id"))
    if trips.trip_id.duplicated().any():
        raise ValueError("bus_trips.csv has duplicate trip_id values")
    if trips.route_id.eq("").any():
        raise ValueError("bus_trips.csv has a trip without route_id")
    trip_route = dict(zip(trips.trip_id, trips.route_id))

    times = _read_gtfs_table(
        directory / "stop_times.txt",
        ("trip_id", "arrival_time", "departure_time", "stop_id", "stop_sequence"),
    )
    times = times[times.trip_id.isin(trip_route)].copy()
    times["stop_sequence"] = pd.to_numeric(times.stop_sequence, errors="coerce")
    if times.stop_sequence.isna().any():
        raise ValueError("stop_times.txt has a non-integer stop_sequence")
    times["stop_sequence"] = times.stop_sequence.astype(int)
    times = times.sort_values(["trip_id", "stop_sequence"], kind="stable")

    observations: dict[tuple[str, str], dict[str, Any]] = {}
    for trip_id, group in times.groupby("trip_id", sort=True):
        route_id = trip_route[trip_id]
        events: list[tuple[int, str, str | None, int | None, int | None]] = []
        for row in group.itertuples(index=False):
            events.append(
                (
                    row.stop_sequence,
                    row.stop_id,
                    stop_station.get(row.stop_id),
                    _parse_gtfs_time(row.arrival_time, what="arrival_time"),
                    _parse_gtfs_time(row.departure_time, what="departure_time"),
                )
            )
        for i, (sequence_i, stop_i, station_i, arrival_i, departure_i) in enumerate(events):
            if station_i is None:
                continue
            departure = departure_i if departure_i is not None else arrival_i
            if departure is None:
                continue
            for sequence_j, stop_j, station_j, arrival_j, departure_j in events[i + 1 :]:
                if station_j is None or station_j == station_i:
                    continue
                arrival = arrival_j if arrival_j is not None else departure_j
                if arrival is None:
                    continue
                minutes = (arrival - departure) / 60.0
                if not math.isfinite(minutes) or minutes <= 0.0:
                    continue
                pair = _canonical_pair(station_i, station_j)
                observation = observations.setdefault(
                    pair,
                    {"routes": set(), "trips": set(), "evidence": None},
                )
                observation["routes"].add(route_id)
                observation["trips"].add(trip_id)
                evidence = (minutes, trip_id, sequence_i, sequence_j, stop_i, stop_j)
                if observation["evidence"] is None or evidence[:4] < observation["evidence"][:4]:
                    observation["evidence"] = evidence

    rows = []
    for pair in sorted(observations):
        minutes, trip_id, sequence_i, sequence_j, stop_i, stop_j = observations[pair]["evidence"]
        rows.append(
            {
                "station_a": pair[0],
                "station_b": pair[1],
                "minutes": float(minutes),
                "kind": "gtfs_candidate",
                "route_ids": ";".join(sorted(observations[pair]["routes"])),
                "trip_count": len(observations[pair]["trips"]),
                "evidence_trip_id": trip_id,
                "evidence_from_stop": stop_i,
                "evidence_to_stop": stop_j,
                "evidence_from_sequence": int(sequence_i),
                "evidence_to_sequence": int(sequence_j),
            }
        )
    return pd.DataFrame(rows, columns=list(_CANDIDATE_COLUMNS))


def compare_manual_pairs(
    manual: pd.DataFrame,
    derived: pd.DataFrame,
    *,
    tolerance_minutes: float = DEFAULT_MATCH_TOLERANCE_MINUTES,
) -> pd.DataFrame:
    """Compare manual ground-truth pairs with derived candidates, row by row.

    ``manual`` needs ``station_a``, ``station_b`` and ``minutes`` columns; it
    is always ground truth. ``derived`` is matched per undirected station
    pair (fastest row wins if duplicates exist). Every manual pair appears in
    the result, sorted by ``(station_a, station_b)``, with columns
    ``station_a``, ``station_b``, ``manual_minutes``, ``derived_minutes``,
    ``delta_minutes`` (``derived_minutes - manual_minutes``), ``status`` and
    ``note``. Status is ``not_derived`` when no candidate pair exists,
    ``matched`` when ``abs(derived - manual) <= tolerance_minutes`` and
    ``time_differs`` otherwise. Nothing is silently dropped.
    """
    tolerance = float(tolerance_minutes)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance_minutes must be a finite number >= 0")
    for name, table in (("manual", manual), ("derived", derived)):
        required = {"station_a", "station_b", "minutes"}
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} table is missing required columns: {', '.join(missing)}")

    candidates: dict[tuple[str, str], float] = {}
    for row in derived.itertuples(index=False):
        minutes = float(row.minutes)
        if not math.isfinite(minutes) or minutes <= 0.0:
            raise ValueError(f"derived row {row.station_a!r}-{row.station_b!r}: minutes must be finite and positive")
        pair = _canonical_pair(_station_id(row.station_a), _station_id(row.station_b))
        if pair not in candidates or minutes < candidates[pair]:
            candidates[pair] = minutes

    rows = []
    for row in manual.itertuples(index=False):
        minutes = float(row.minutes)
        if not math.isfinite(minutes) or minutes <= 0.0:
            raise ValueError(f"manual row {row.station_a!r}-{row.station_b!r}: minutes must be finite and positive")
        pair = _canonical_pair(_station_id(row.station_a), _station_id(row.station_b))
        derived_minutes = candidates.get(pair)
        if derived_minutes is None:
            status = "not_derived"
            delta = math.nan
            note = (
                "manual ground truth; no single selected bus trip links both stations "
                "within radius_m in the frozen window"
            )
        else:
            delta = derived_minutes - minutes
            status = "matched" if abs(delta) <= tolerance else "time_differs"
            note = (
                "manual ground truth; derived minutes are scheduled ride time while "
                "the manual time includes walking, waiting and transfers"
            )
            if status == "matched":
                note = "manual ground truth; derived time within tolerance_minutes"
        record = {
            "station_a": pair[0],
            "station_b": pair[1],
            "manual_minutes": minutes,
            "derived_minutes": derived_minutes if derived_minutes is not None else math.nan,
            "delta_minutes": delta,
            "status": status,
            "note": note,
        }
        if "kind" in manual.columns:
            record["manual_kind"] = _text(row.kind)
        rows.append(record)
    frame = pd.DataFrame(rows)
    return frame.sort_values(["station_a", "station_b"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------
def _iter_walking_transfers(
    walking_transfers: pd.DataFrame | Mapping[str, Any] | Iterable[tuple[str, str]],
) -> Iterable[tuple[Any, Any, Any, Any]]:
    """Yield ``(index, station_a, station_b, minutes)`` for each transfer."""
    if isinstance(walking_transfers, pd.DataFrame):
        required = ("station_a", "station_b")
        missing = [column for column in required if column not in walking_transfers.columns]
        if missing:
            raise ValueError("walking table is missing required columns: " + ", ".join(missing))
        has_minutes = "minutes" in walking_transfers.columns
        for index, row in walking_transfers.iterrows():
            minutes = row["minutes"] if has_minutes else None
            if has_minutes and _text(minutes) == "":
                minutes = None
            yield index, row["station_a"], row["station_b"], minutes
        return
    if isinstance(walking_transfers, Mapping):
        if "station_a" not in walking_transfers or "station_b" not in walking_transfers:
            raise ValueError("walking mapping needs 'station_a' and 'station_b' keys")
        minutes = walking_transfers.get("minutes")
        if _text(minutes) == "":
            minutes = None
        yield None, walking_transfers["station_a"], walking_transfers["station_b"], minutes
        return
    try:
        iterator = iter(walking_transfers)
    except TypeError as error:
        raise TypeError(
            "walking_transfers must be a DataFrame, a mapping with station_a/station_b or an iterable of pairs"
        ) from error
    for index, pair in enumerate(iterator):
        try:
            station_a, station_b = pair
        except (TypeError, ValueError) as error:
            raise ValueError(f"walking transfer {index!r}: expected a station pair") from error
        yield index, station_a, station_b, None


def _read_gtfs_table(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    """Read a prepared GTFS table, stripping headers and string values."""
    try:
        table = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=list(columns))
    except ValueError as error:
        raise ValueError(f"{path.name}: {error}") from error
    table.columns = table.columns.str.strip()
    for column in table.columns:
        table[column] = table[column].str.strip()
    return table


def _load_stations(path: str | Path) -> pd.DataFrame:
    """Load the mapped stations with finite coordinates."""
    stations = pd.read_csv(path, dtype=str, keep_default_na=False)
    stations.columns = stations.columns.str.strip()
    missing = sorted({"station_id", "lat", "lon"} - set(stations.columns))
    if missing:
        raise ValueError(f"{Path(path).name} is missing required columns: {', '.join(missing)}")
    if stations.empty:
        raise ValueError(f"{Path(path).name} has no stations")
    for column in ("station_id", "lat", "lon"):
        stations[column] = stations[column].str.strip()
    if stations.station_id.eq("").any() or stations.station_id.duplicated().any():
        raise ValueError("stations must have unique non-empty station_id values")
    stations["lat"] = pd.to_numeric(stations.lat, errors="coerce")
    stations["lon"] = pd.to_numeric(stations.lon, errors="coerce")
    if not (np.isfinite(stations.lat.to_numpy(dtype=float)) & np.isfinite(stations.lon.to_numpy(dtype=float))).all():
        raise ValueError("station coordinates must be finite")
    return stations


def _great_circle_m(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    """Vectorised great-circle distance in metres, Earth radius 6,371,000 m."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    delta_lon = np.radians(lon2 - lon1)
    haversine = (
        np.sin((p2 - p1) / 2.0) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))


def _nearest_station_lookup(
    bus_stops: pd.DataFrame,
    stations: pd.DataFrame,
    radius_m: float,
) -> dict[str, str]:
    """Map each bus stop id to the nearest mapped station within ``radius_m``.

    Station ids are processed in sorted order, so a distance tie keeps the
    smallest id as required by the frozen derivation rule.
    """
    station_ids = stations.station_id.to_numpy()
    order = np.argsort(station_ids, kind="stable")
    sorted_ids = station_ids[order]
    station_lat = stations.lat.to_numpy(dtype=float)[order]
    station_lon = stations.lon.to_numpy(dtype=float)[order]
    lookup: dict[str, str] = {}
    for stop_id, lat, lon in zip(bus_stops.stop_id, bus_stops.stop_lat, bus_stops.stop_lon):
        distances = _great_circle_m(float(lat), float(lon), station_lat, station_lon)
        best = int(np.argmin(distances))
        if distances[best] > radius_m:
            continue
        tied = np.flatnonzero(distances <= distances[best] + _TIE_METRES)
        lookup[str(stop_id)] = str(sorted_ids[int(tied[0])])
    return lookup


def _parse_gtfs_time(value: Any, *, what: str) -> int | None:
    """Parse ``HH:MM:SS`` to service-day seconds; hours may exceed 24."""
    text = _text(value)
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(f"stop_times.txt has invalid {what} {value!r}")
    try:
        hours, minutes, seconds = (int(part) for part in parts)
    except ValueError as error:
        raise ValueError(f"stop_times.txt has invalid {what} {value!r}") from error
    if hours < 0 or minutes < 0 or seconds < 0 or minutes > 59 or seconds > 59:
        raise ValueError(f"stop_times.txt has invalid {what} {value!r}")
    return hours * 3600 + minutes * 60 + seconds