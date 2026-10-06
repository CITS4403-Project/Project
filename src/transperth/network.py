"""Graph builders for the frozen Transperth rail layer and bus coverage.

:func:`load_rail_graph` returns the undirected simple NetworkX graph defined in
``docs/model.md`` section 1. Node keys are ``station_id`` strings and every
node carries:

========================  =====================================================
Node attribute            Type
========================  =====================================================
``station_id`` (key)      ``str``, GTFS parent ``stop_id``
``name``                  ``str``
``lat``                   ``float`` WGS84 decimal degrees
``lon``                   ``float`` WGS84 decimal degrees
``modes``                 ``list[str]``; the CSV stores a semicolon list and a
                          blank value becomes ``[]``
``degree``                ``int`` edges in the frozen rail graph
``trips_served``          ``int`` full-sequence stop visits
``am_peak_stops``         ``int`` departures inside the morning window
``betweenness``           ``float`` in ``[0, 1]``
``lines``                 ``list[str]``; semicolon list, blank becomes ``[]``
========================  =====================================================

Every edge carries ``trips: int`` (scheduled traversals) and
``distance_m: float`` (great-circle metres). Nodes and edges are inserted in
sorted ``station_id`` order, so iteration over an unchanged input is identical
across runs.

:func:`load_bus_coverage` validates the 400 m coverage table and returns it with
one row per mapped station and its routes kept as a sorted, unique semicolon
string.

Both loaders validate the frozen processed tables before returning. Validation
errors are :class:`ValueError` messages that name the file, the 1-based data
row and, where it exists, the affected station or edge. The inputs are only
opened for reading.
"""
from __future__ import annotations

import math
from pathlib import Path

import networkx as nx
import pandas as pd

from transperth.config import BUS_COVERAGE_CSV, RAIL_EDGES_CSV, STATIONS_CSV

__all__ = ["load_bus_coverage", "load_rail_graph"]

_STATION_COLUMNS = (
    "station_id",
    "name",
    "lat",
    "lon",
    "modes",
    "degree",
    "trips_served",
    "am_peak_stops",
    "betweenness",
    "lines",
)
_RAIL_EDGE_COLUMNS = ("station_a", "station_b", "trips", "distance_m")
_BUS_COVERAGE_COLUMNS = (
    "station_id",
    "name",
    "n_bus_stops",
    "n_bus_routes",
    "bus_route_ids",
)


def _read_table(
    path: Path,
    dtype: dict[str, type[str]],
    expected: tuple[str, ...],
) -> pd.DataFrame:
    """Read one processed CSV and require its column set to match exactly.

    ``keep_default_na=False`` keeps every cell as its literal text, so blank
    cells reach the validators instead of being silently converted to NaN.
    """
    table = pd.read_csv(path, dtype=dtype, keep_default_na=False)
    missing = sorted(set(expected) - set(table.columns))
    extra = sorted(set(table.columns) - set(expected))
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing columns {missing}")
        if extra:
            problems.append(f"unexpected columns {extra}")
        raise ValueError(f"{path.name}: {'; '.join(problems)}")
    return table


def _location(path: Path, row: int, context: str | None = None) -> str:
    """Describe one 1-based data row for an error message."""
    where = f"{path.name}: data row {row}"
    if context is not None:
        where = f"{where} ({context})"
    return where


def _to_float(value: object) -> float:
    """Parse one CSV cell as a float, returning NaN when it is not numeric."""
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return math.nan


def _number(
    value: object,
    *,
    where: str,
    column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Parse a cell as a finite float inside the optional closed bounds."""
    if minimum is not None and maximum is not None:
        bounds = f" in [{minimum:g}, {maximum:g}]"
    elif minimum is not None:
        bounds = f" >= {minimum:g}"
    elif maximum is not None:
        bounds = f" <= {maximum:g}"
    else:
        bounds = ""
    number = _to_float(value)
    if not math.isfinite(number) or (
        (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise ValueError(
            f"{where}: {column} must be a finite number{bounds}, got {value!r}"
        )
    return number


def _integer(value: object, *, where: str, column: str) -> int:
    """Parse a cell as a non-negative integer."""
    number = _to_float(value)
    if not math.isfinite(number) or number < 0 or number != int(number):
        raise ValueError(
            f"{where}: {column} must be a non-negative integer, got {value!r}"
        )
    return int(number)


def _positive(value: object, *, where: str, column: str) -> float:
    """Parse a cell as a finite positive float."""
    number = _to_float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{where}: {column} must be a positive number, got {value!r}")
    return number


def _split_list(value: object, *, where: str, column: str) -> list[str]:
    """Split a semicolon cell; blank becomes an empty list."""
    text = str(value).strip()
    if not text:
        return []
    items = [item.strip() for item in text.split(";")]
    if any(not item for item in items):
        raise ValueError(f"{where}: {column} contains an empty entry: {text!r}")
    return items


def load_rail_graph(
    *,
    stations_csv: str | Path = STATIONS_CSV,
    edges_csv: str | Path = RAIL_EDGES_CSV,
) -> nx.Graph:
    """Load and validate the frozen rail layer as an undirected simple graph.

    Parameters
    ----------
    stations_csv
        ``stations.csv`` table; one row per mapped parent station.
    edges_csv
        ``rail_edges.csv`` table; one row per verified undirected adjacency.

    Returns
    -------
    networkx.Graph
        Nodes keyed by ``station_id`` with the attributes listed in the module
        docstring. Nodes and edges are inserted in sorted station ID order.

    Raises
    ------
    ValueError
        If a table has missing or extra columns, duplicate station IDs,
        unknown edge endpoints, self-loops, duplicate undirected edges,
        non-finite or out-of-range coordinates, or invalid numerics. The
        message names the file, the 1-based data row and the station or edge.
    """
    stations_path = Path(stations_csv)
    edges_path = Path(edges_csv)
    stations = _read_table(stations_path, {"station_id": str}, _STATION_COLUMNS)
    edges = _read_table(
        edges_path, {"station_a": str, "station_b": str}, _RAIL_EDGE_COLUMNS
    )

    attributes: dict[str, dict[str, object]] = {}
    first_row: dict[str, int] = {}
    for row_number, row in enumerate(stations.itertuples(index=False), start=1):
        station = str(row.station_id).strip()
        where = _location(
            stations_path, row_number, f"station_id {station!r}" if station else None
        )
        if not station:
            raise ValueError(f"{where}: station_id must not be blank")
        name = str(row.name).strip()
        if not name:
            raise ValueError(f"{where}: name must not be blank")
        if station in attributes:
            raise ValueError(
                f"{where}: duplicate station_id {station!r} "
                f"(first seen at data row {first_row[station]})"
            )
        first_row[station] = row_number
        attributes[station] = {
            "name": name,
            "lat": _number(
                row.lat, where=where, column="lat", minimum=-90.0, maximum=90.0
            ),
            "lon": _number(
                row.lon, where=where, column="lon", minimum=-180.0, maximum=180.0
            ),
            "modes": _split_list(row.modes, where=where, column="modes"),
            "degree": _integer(row.degree, where=where, column="degree"),
            "trips_served": _integer(
                row.trips_served, where=where, column="trips_served"
            ),
            "am_peak_stops": _integer(
                row.am_peak_stops, where=where, column="am_peak_stops"
            ),
            "betweenness": _number(
                row.betweenness,
                where=where,
                column="betweenness",
                minimum=0.0,
                maximum=1.0,
            ),
            "lines": _split_list(row.lines, where=where, column="lines"),
        }

    known = set(attributes)
    seen: dict[tuple[str, str], int] = {}
    parsed: list[tuple[str, str, int, float]] = []
    for row_number, row in enumerate(edges.itertuples(index=False), start=1):
        station_a = str(row.station_a).strip()
        station_b = str(row.station_b).strip()
        where = _location(
            edges_path, row_number, f"edge {station_a!r}-{station_b!r}"
        )
        if not station_a or not station_b:
            raise ValueError(f"{where}: station_a and station_b must not be blank")
        if station_a == station_b:
            raise ValueError(f"{where}: self-loop at station {station_a!r}")
        unknown = [
            endpoint
            for endpoint in (station_a, station_b)
            if endpoint not in known
        ]
        if unknown:
            raise ValueError(
                f"{where}: unknown station endpoint(s) {unknown} "
                f"not present in {stations_path.name}"
            )
        pair = (
            (station_a, station_b)
            if station_a <= station_b
            else (station_b, station_a)
        )
        if pair in seen:
            raise ValueError(
                f"{where}: duplicate undirected edge {pair} "
                f"(first seen at data row {seen[pair]})"
            )
        seen[pair] = row_number
        parsed.append(
            (
                station_a,
                station_b,
                _integer(row.trips, where=where, column="trips"),
                _positive(row.distance_m, where=where, column="distance_m"),
            )
        )

    graph = nx.Graph()
    for station in sorted(attributes):
        graph.add_node(station, **attributes[station])
    for station_a, station_b, trips, distance_m in sorted(parsed):
        graph.add_edge(station_a, station_b, trips=trips, distance_m=distance_m)
    return graph


def load_bus_coverage(*, coverage_csv: str | Path = BUS_COVERAGE_CSV) -> pd.DataFrame:
    """Load and validate the frozen 400 m bus coverage table.

    Returns
    -------
    pandas.DataFrame
        One row per mapped station with columns ``station_id`` and ``name`` as
        strings, ``n_bus_stops`` and ``n_bus_routes`` as non-negative integers
        and ``bus_route_ids`` kept as a semicolon-separated string. The route
        list is sorted, unique and consistent with ``n_bus_routes``: blank if
        and only if the count is zero.

    Raises
    ------
    ValueError
        If the table has missing or extra columns, a blank or duplicate
        station ID, invalid counts, or a route list that is unordered,
        duplicated or inconsistent with ``n_bus_routes``. The message names
        the file, the 1-based data row and the affected station.
    """
    path = Path(coverage_csv)
    table = _read_table(path, {"station_id": str}, _BUS_COVERAGE_COLUMNS)
    rows: list[dict[str, object]] = []
    first_row: dict[str, int] = {}
    for row_number, row in enumerate(table.itertuples(index=False), start=1):
        station = str(row.station_id).strip()
        where = _location(
            path, row_number, f"station_id {station!r}" if station else None
        )
        if not station:
            raise ValueError(f"{where}: station_id must not be blank")
        if station in first_row:
            raise ValueError(
                f"{where}: duplicate station_id {station!r} "
                f"(first seen at data row {first_row[station]})"
            )
        first_row[station] = row_number
        name = str(row.name).strip()
        if not name:
            raise ValueError(f"{where}: name must not be blank")
        n_bus_stops = _integer(row.n_bus_stops, where=where, column="n_bus_stops")
        n_bus_routes = _integer(row.n_bus_routes, where=where, column="n_bus_routes")
        routes = _split_list(row.bus_route_ids, where=where, column="bus_route_ids")
        if len(routes) != len(set(routes)):
            raise ValueError(
                f"{where}: bus_route_ids contains duplicate route IDs: "
                f"{row.bus_route_ids!r}"
            )
        if routes != sorted(routes):
            raise ValueError(
                f"{where}: bus_route_ids must be sorted: {row.bus_route_ids!r}"
            )
        if n_bus_routes == 0 and routes:
            raise ValueError(
                f"{where}: n_bus_routes is 0 but bus_route_ids lists "
                f"{len(routes)} route(s)"
            )
        if n_bus_routes > 0 and not routes:
            raise ValueError(
                f"{where}: bus_route_ids is blank but n_bus_routes is {n_bus_routes}"
            )
        if len(routes) != n_bus_routes:
            raise ValueError(
                f"{where}: n_bus_routes is {n_bus_routes} but bus_route_ids "
                f"lists {len(routes)} route(s)"
            )
        rows.append(
            {
                "station_id": station,
                "name": name,
                "n_bus_stops": n_bus_stops,
                "n_bus_routes": n_bus_routes,
                "bus_route_ids": ";".join(routes),
            }
        )
    return pd.DataFrame(rows, columns=list(_BUS_COVERAGE_COLUMNS))