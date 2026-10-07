"""Validation and frozen-input regression tests for :mod:`transperth.network`."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from transperth.config import BUS_COVERAGE_CSV, RAIL_EDGES_CSV, STATIONS_CSV
from transperth.network import load_bus_coverage, load_rail_graph

STATION_COLUMNS = (
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
EDGE_COLUMNS = ("station_a", "station_b", "trips", "distance_m")
COVERAGE_COLUMNS = ("station_id", "name", "n_bus_stops", "n_bus_routes", "bus_route_ids")
STATION_ATTRIBUTES = set(STATION_COLUMNS) - {"station_id"}
EDGE_ATTRIBUTES = set(EDGE_COLUMNS) - {"station_a", "station_b"}


def station_row(station_id: str = "1", **overrides: object) -> dict[str, object]:
    """Return one synthetic ``stations.csv`` row."""
    row: dict[str, object] = {
        "station_id": station_id,
        "name": f"Station {station_id}",
        "lat": -31.95,
        "lon": 115.86,
        "modes": "Bus;Rail",
        "degree": 1,
        "trips_served": 10,
        "am_peak_stops": 5,
        "betweenness": 0.25,
        "lines": "Fremantle Line",
    }
    row.update(overrides)
    return row


def edge_row(station_a: str = "1", station_b: str = "2", **overrides: object) -> dict[str, object]:
    """Return one synthetic ``rail_edges.csv`` row."""
    row: dict[str, object] = {
        "station_a": station_a,
        "station_b": station_b,
        "trips": 10,
        "distance_m": 1200.0,
    }
    row.update(overrides)
    return row


def coverage_row(station_id: str = "1", **overrides: object) -> dict[str, object]:
    """Return one synthetic ``bus_coverage.csv`` row."""
    row: dict[str, object] = {
        "station_id": station_id,
        "name": f"Station {station_id}",
        "n_bus_stops": 3,
        "n_bus_routes": 2,
        "bus_route_ids": "PAT-1;PAT-2",
    }
    row.update(overrides)
    return row


def write_table(
    directory: Path, name: str, columns: tuple[str, ...], rows: list[dict[str, object]]
) -> Path:
    """Write a synthetic CSV table and return its path."""
    path = directory / name
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)
    return path


def write_rail(
    directory: Path,
    stations: list[dict[str, object]] | None = None,
    edges: list[dict[str, object]] | None = None,
) -> tuple[Path, Path]:
    """Write the two synthetic rail tables and return their paths."""
    if stations is None:
        stations = [station_row("1"), station_row("2")]
    if edges is None:
        edges = [edge_row("1", "2")]
    return (
        write_table(directory, "stations.csv", STATION_COLUMNS, stations),
        write_table(directory, "rail_edges.csv", EDGE_COLUMNS, edges),
    )


def rewrite(path: Path, transform) -> None:
    """Read ``path`` as text cells, apply ``transform`` and write it back."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    transform(frame).to_csv(path, index=False)


def raised_message(action) -> str:
    """Run ``action`` and return the ``ValueError`` message it raises."""
    with pytest.raises(ValueError) as error:
        action()
    return str(error.value)


# ---------------------------------------------------------------------------
# Synthetic fixtures: valid loading
# ---------------------------------------------------------------------------
def test_synthetic_graph_keeps_types_and_sorted_order(tmp_path):
    stations = [
        station_row("2", name="B", modes="", lines=""),
        station_row("1", name="A", lines="Yanchep Line;Fremantle Line"),
    ]
    edges = [edge_row("2", "1", trips=7, distance_m=1234.5)]
    stations_path, edges_path = write_rail(tmp_path, stations, edges)

    graph = load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)

    assert isinstance(graph, nx.Graph)
    assert list(graph) == ["1", "2"]
    assert [tuple(edge) for edge in graph.edges] == [("1", "2")]
    assert graph.nodes["1"] == {
        "name": "A",
        "lat": -31.95,
        "lon": 115.86,
        "modes": ["Bus", "Rail"],
        "degree": 1,
        "trips_served": 10,
        "am_peak_stops": 5,
        "betweenness": 0.25,
        "lines": ["Yanchep Line", "Fremantle Line"],
    }
    assert graph.nodes["2"]["modes"] == []
    assert graph.nodes["2"]["lines"] == []
    assert graph.edges["1", "2"] == {"trips": 7, "distance_m": 1234.5}


# ---------------------------------------------------------------------------
# Synthetic fixtures: column sets
# ---------------------------------------------------------------------------
def test_stations_missing_column_names_file_and_column(tmp_path):
    stations_path, edges_path = write_rail(tmp_path)
    rewrite(stations_path, lambda frame: frame.drop(columns=["lines"]))
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "stations.csv" in message
    assert "missing columns" in message
    assert "lines" in message


def test_stations_extra_column_names_file_and_column(tmp_path):
    stations_path, edges_path = write_rail(tmp_path)
    rewrite(stations_path, lambda frame: frame.assign(note="x"))
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "stations.csv" in message
    assert "unexpected columns" in message
    assert "note" in message


def test_rail_edges_missing_column_names_file_and_column(tmp_path):
    stations_path, edges_path = write_rail(tmp_path)
    rewrite(edges_path, lambda frame: frame.drop(columns=["distance_m"]))
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "rail_edges.csv" in message
    assert "missing columns" in message
    assert "distance_m" in message


def test_rail_edges_extra_column_names_file_and_column(tmp_path):
    stations_path, edges_path = write_rail(tmp_path)
    rewrite(edges_path, lambda frame: frame.assign(note="x"))
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "rail_edges.csv" in message
    assert "unexpected columns" in message
    assert "note" in message


def test_bus_coverage_missing_column_names_file_and_column(tmp_path):
    path = write_table(
        tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, [coverage_row("1")]
    )
    rewrite(path, lambda frame: frame.drop(columns=["n_bus_stops"]))
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "bus_coverage.csv" in message
    assert "missing columns" in message
    assert "n_bus_stops" in message


def test_bus_coverage_extra_column_names_file_and_column(tmp_path):
    path = write_table(
        tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, [coverage_row("1")]
    )
    rewrite(path, lambda frame: frame.assign(note="x"))
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "bus_coverage.csv" in message
    assert "unexpected columns" in message
    assert "note" in message


# ---------------------------------------------------------------------------
# Synthetic fixtures: stations.csv row validation
# ---------------------------------------------------------------------------
def test_duplicate_station_ids_name_station_and_rows(tmp_path):
    stations = [station_row("1"), station_row("1")]
    stations_path, edges_path = write_rail(tmp_path, stations, [])
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "stations.csv" in message
    assert "duplicate station_id '1'" in message
    assert "data row 2" in message


def test_blank_station_id_names_row(tmp_path):
    stations_path, edges_path = write_rail(tmp_path, [station_row("")], [])
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "stations.csv" in message
    assert "data row 1" in message
    assert "station_id must not be blank" in message


def test_blank_name_names_station(tmp_path):
    stations_path, edges_path = write_rail(tmp_path, [station_row("5", name="")], [])
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "station_id '5'" in message
    assert "name must not be blank" in message


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("lat", math.nan),
        ("lat", math.inf),
        ("lat", 90.5),
        ("lat", "north"),
        ("lon", -math.inf),
        ("lon", -180.5),
    ],
)
def test_invalid_coordinates_name_station_and_column(tmp_path, column, value):
    stations = [station_row("7", **{column: value})]
    stations_path, edges_path = write_rail(tmp_path, stations, [])
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "station_id '7'" in message
    assert column in message
    assert "finite" in message


@pytest.mark.parametrize("column", ["degree", "trips_served", "am_peak_stops"])
@pytest.mark.parametrize("value", [-1, 1.5, "soon"])
def test_invalid_station_counts_name_station_and_column(tmp_path, column, value):
    stations = [station_row("3", **{column: value})]
    stations_path, edges_path = write_rail(tmp_path, stations, [])
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "station_id '3'" in message
    assert column in message
    assert "non-negative integer" in message


@pytest.mark.parametrize("value", [math.nan, math.inf, -0.1, 1.5, "high"])
def test_invalid_betweenness_names_station(tmp_path, value):
    stations = [station_row("4", betweenness=value)]
    stations_path, edges_path = write_rail(tmp_path, stations, [])
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "station_id '4'" in message
    assert "betweenness" in message


def test_modes_and_lines_blank_items_name_station(tmp_path):
    stations_path, edges_path = write_rail(
        tmp_path, [station_row("6", modes="Bus;;Rail")], []
    )
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "station_id '6'" in message
    assert "modes" in message
    assert "empty entry" in message


# ---------------------------------------------------------------------------
# Synthetic fixtures: rail_edges.csv row validation
# ---------------------------------------------------------------------------
def test_unknown_edge_endpoint_names_row_and_station(tmp_path):
    stations = [station_row("1")]
    edges = [edge_row("1", "9")]
    stations_path, edges_path = write_rail(tmp_path, stations, edges)
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "rail_edges.csv" in message
    assert "data row 1" in message
    assert "unknown station endpoint" in message
    assert "'9'" in message


def test_self_loop_names_station(tmp_path):
    stations = [station_row("1")]
    edges = [edge_row("1", "1")]
    stations_path, edges_path = write_rail(tmp_path, stations, edges)
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "self-loop" in message
    assert "'1'" in message


def test_duplicate_undirected_edge_names_pair_and_rows(tmp_path):
    stations = [station_row("1"), station_row("2")]
    edges = [edge_row("1", "2"), edge_row("2", "1")]
    stations_path, edges_path = write_rail(tmp_path, stations, edges)
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "duplicate undirected edge ('1', '2')" in message
    assert "data row 2" in message
    assert "first seen at data row 1" in message


@pytest.mark.parametrize(
    ("column", "value"),
    [("trips", -1), ("trips", 2.5), ("trips", "soon"), ("trips", math.nan)],
)
def test_invalid_edge_trips_names_row_and_column(tmp_path, column, value):
    stations = [station_row("1"), station_row("2")]
    edges = [edge_row("1", "2", **{column: value})]
    stations_path, edges_path = write_rail(tmp_path, stations, edges)
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "rail_edges.csv" in message
    assert "data row 1" in message
    assert column in message


@pytest.mark.parametrize(
    "value", [0.0, -5.0, math.inf, math.nan, "far"]
)
def test_invalid_edge_distance_names_row(tmp_path, value):
    stations = [station_row("1"), station_row("2")]
    edges = [edge_row("1", "2", distance_m=value)]
    stations_path, edges_path = write_rail(tmp_path, stations, edges)
    message = raised_message(
        lambda: load_rail_graph(stations_csv=stations_path, edges_csv=edges_path)
    )
    assert "rail_edges.csv" in message
    assert "data row 1" in message
    assert "distance_m" in message


# ---------------------------------------------------------------------------
# Synthetic fixtures: bus_coverage.csv row validation
# ---------------------------------------------------------------------------
def test_bus_coverage_keeps_semicolon_lists_and_blanks(tmp_path):
    rows = [
        coverage_row("2", n_bus_stops=0, n_bus_routes=0, bus_route_ids=""),
        coverage_row("1", bus_route_ids="PAT-1;PAT-2"),
    ]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    frame = load_bus_coverage(coverage_csv=path)
    assert list(frame.columns) == list(COVERAGE_COLUMNS)
    assert list(frame["station_id"]) == ["2", "1"]
    assert frame.loc[0, "bus_route_ids"] == ""
    assert frame.loc[1, "bus_route_ids"] == "PAT-1;PAT-2"
    assert frame.dtypes["n_bus_stops"] == "int64"
    assert frame.dtypes["n_bus_routes"] == "int64"


def test_bus_coverage_duplicate_station_names_station_and_rows(tmp_path):
    rows = [coverage_row("1"), coverage_row("1")]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "bus_coverage.csv" in message
    assert "duplicate station_id '1'" in message
    assert "data row 2" in message


@pytest.mark.parametrize("column", ["n_bus_stops", "n_bus_routes"])
@pytest.mark.parametrize("value", [-1, 2.5, "many", math.nan])
def test_invalid_bus_counts_name_station_and_column(tmp_path, column, value):
    rows = [coverage_row("8", **{column: value})]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "station_id '8'" in message
    assert column in message
    assert "non-negative integer" in message


def test_unsorted_bus_routes_name_station(tmp_path):
    rows = [coverage_row("1", n_bus_routes=2, bus_route_ids="PAT-2;PAT-1")]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "station_id '1'" in message
    assert "must be sorted" in message


def test_duplicate_bus_routes_name_station(tmp_path):
    rows = [coverage_row("1", n_bus_routes=2, bus_route_ids="PAT-1;PAT-1")]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "station_id '1'" in message
    assert "duplicate route IDs" in message


def test_bus_route_count_mismatch_names_station(tmp_path):
    rows = [coverage_row("1", n_bus_routes=3, bus_route_ids="PAT-1;PAT-2")]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "station_id '1'" in message
    assert "lists 2 route(s)" in message


def test_blank_bus_routes_with_nonzero_count_names_station(tmp_path):
    rows = [coverage_row("1", n_bus_routes=2, bus_route_ids="")]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "station_id '1'" in message
    assert "bus_route_ids is blank" in message


def test_nonblank_bus_routes_with_zero_count_names_station(tmp_path):
    rows = [coverage_row("1", n_bus_routes=0, bus_route_ids="PAT-1")]
    path = write_table(tmp_path, "bus_coverage.csv", COVERAGE_COLUMNS, rows)
    message = raised_message(lambda: load_bus_coverage(coverage_csv=path))
    assert "station_id '1'" in message
    assert "n_bus_routes is 0" in message


# ---------------------------------------------------------------------------
# Frozen dataset regressions
# ---------------------------------------------------------------------------
def test_frozen_rail_graph_is_a_connected_tree():
    graph = load_rail_graph()
    assert graph.number_of_nodes() == 86
    assert graph.number_of_edges() == 85
    assert len(set(graph)) == 86
    assert nx.is_connected(graph)
    assert nx.is_tree(graph)
    for station, data in graph.nodes(data=True):
        assert isinstance(station, str)
        assert set(data) == STATION_ATTRIBUTES
        assert isinstance(data["name"], str) and data["name"]
        assert isinstance(data["lat"], float) and math.isfinite(data["lat"])
        assert isinstance(data["lon"], float) and math.isfinite(data["lon"])
        assert -90.0 <= data["lat"] <= 90.0
        assert -180.0 <= data["lon"] <= 180.0
        assert isinstance(data["modes"], list)
        assert isinstance(data["lines"], list)
        assert isinstance(data["degree"], int)
        assert isinstance(data["trips_served"], int)
        assert isinstance(data["am_peak_stops"], int)
        assert isinstance(data["betweenness"], float)
        assert data["degree"] == graph.degree[station]
    for _, _, data in graph.edges(data=True):
        assert set(data) == EDGE_ATTRIBUTES
        assert isinstance(data["trips"], int)
        assert isinstance(data["distance_m"], float)
        assert data["distance_m"] > 0.0


def test_frozen_loading_is_deterministic():
    first = load_rail_graph()
    second = load_rail_graph()
    first_nodes = list(first)
    second_nodes = list(second)
    first_edges = [tuple(edge) for edge in first.edges]
    second_edges = [tuple(edge) for edge in second.edges]
    assert first_nodes == second_nodes == sorted(first_nodes)
    assert first_edges == second_edges == sorted(first_edges)


def test_frozen_bus_coverage_matches_rail_stations():
    graph = load_rail_graph()
    coverage = load_bus_coverage()
    assert len(coverage) == 86
    assert set(coverage["station_id"]) == set(graph)
    assert list(coverage["station_id"]) == sorted(coverage["station_id"])
    for row in coverage.itertuples(index=False):
        routes = [] if row.bus_route_ids == "" else row.bus_route_ids.split(";")
        assert routes == sorted(set(routes))
        assert len(routes) == row.n_bus_routes
        assert (row.n_bus_routes == 0) == (row.bus_route_ids == "")
        assert row.n_bus_stops >= 0


def test_loading_does_not_modify_inputs():
    paths = [STATIONS_CSV, RAIL_EDGES_CSV, BUS_COVERAGE_CSV]
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    load_rail_graph()
    load_bus_coverage()
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before