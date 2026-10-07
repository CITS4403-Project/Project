"""Tests for the load modes and the capacity law in :mod:`transperth.loads`."""

from __future__ import annotations

import math

import networkx as nx
import numpy as np
import pytest

from transperth.loads import capacities, demand_reference_capacities, initial_loads
from transperth.network import load_rail_graph


def _uniform_trips(graph: nx.Graph, trips: float = 2.0) -> nx.Graph:
    for u, v in graph.edges:
        graph.edges[u, v]["trips"] = trips
    return graph


def test_betweenness_matches_networkx_unnormalised():
    graph = nx.path_graph(4)
    expected = nx.betweenness_centrality(graph, normalized=False)
    assert initial_loads(graph, "betweenness") == pytest.approx(expected)


def test_betweenness_freq_uses_inverse_trips_as_edge_length():
    graph = nx.path_graph(3)
    graph.edges[0, 1]["trips"] = 1.0
    graph.edges[1, 2]["trips"] = 4.0
    weighted = nx.Graph()
    weighted.add_edge(0, 1, length=1.0)
    weighted.add_edge(1, 2, length=0.25)
    expected = nx.betweenness_centrality(weighted, weight="length", normalized=False)
    assert initial_loads(graph, "betweenness_freq") == pytest.approx(expected)


def test_betweenness_freq_concentrates_flow_on_the_shorter_weighted_path():
    # A 4-cycle carries no unweighted signal: every node scores 0.5 whatever
    # the edge weights, so only a genuinely weighted computation can move the
    # 0-1-2 flow onto node 1 and off node 3.
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])
    for edge, trips in (
        ((0, 1), 4.0),
        ((1, 2), 4.0),
        ((2, 3), 1.0),
        ((3, 0), 1.0),
    ):
        graph.edges[edge]["trips"] = trips
    weighted = nx.Graph()
    weighted.add_edge(0, 1, length=0.25)
    weighted.add_edge(1, 2, length=0.25)
    weighted.add_edge(2, 3, length=1.0)
    weighted.add_edge(3, 0, length=1.0)
    expected = nx.betweenness_centrality(weighted, weight="length", normalized=False)

    loads = initial_loads(graph, "betweenness_freq")
    assert loads == pytest.approx(expected)
    unweighted = initial_loads(graph, "betweenness")
    assert any(not math.isclose(loads[node], unweighted[node]) for node in graph)
    assert loads[1] > unweighted[1]
    assert loads[3] < unweighted[3]


def test_betweenness_freq_does_not_mutate_the_input_graph():
    graph = _uniform_trips(nx.path_graph(4))
    before = {(u, v): dict(data) for u, v, data in graph.edges(data=True)}
    initial_loads(graph, "betweenness_freq")
    after = {(u, v): dict(data) for u, v, data in graph.edges(data=True)}
    assert after == before
    assert all("length" not in data for data in after.values())


@pytest.mark.parametrize("trips", [0.0, -3.0, math.inf, math.nan])
def test_betweenness_freq_rejects_non_positive_or_non_finite_trips(trips):
    graph = nx.path_graph(3)
    graph.edges[0, 1]["trips"] = 2.0
    graph.edges[1, 2]["trips"] = trips
    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        initial_loads(graph, "betweenness_freq")


def test_betweenness_freq_rejects_missing_trips_and_names_the_edge():
    graph = nx.path_graph(3)
    graph.edges[0, 1]["trips"] = 2.0
    with pytest.raises(ValueError, match=r"\(1, 2\)"):
        initial_loads(graph, "betweenness_freq")


def test_betweenness_plus_trips_adds_the_throughput_share():
    graph = nx.path_graph(3)
    graph.nodes[1]["trips_served"] = 50.0
    base = nx.betweenness_centrality(graph, normalized=False)
    loads = initial_loads(graph, "betweenness_plus_trips")
    assert loads[1] == pytest.approx(base[1] + 5.0)
    # missing throughput is a zero baseline, not a KeyError
    assert loads[0] == pytest.approx(base[0])
    assert loads[2] == pytest.approx(base[2])


def test_betweenness_plus_trips_treats_none_as_zero():
    graph = nx.path_graph(3)
    graph.nodes[0]["trips_served"] = None
    graph.nodes[1]["trips_served"] = 20.0
    loads = initial_loads(graph, "betweenness_plus_trips")
    assert loads[0] == pytest.approx(0.0)
    assert loads[1] == pytest.approx(1.0 + 2.0)


@pytest.mark.parametrize("trips_served", [-1.0, math.inf, math.nan, "many"])
def test_betweenness_plus_trips_rejects_bad_throughput(trips_served):
    graph = nx.path_graph(3)
    graph.nodes[1]["trips_served"] = trips_served
    with pytest.raises(ValueError, match="trips_served"):
        initial_loads(graph, "betweenness_plus_trips")


def test_demand_mode_reads_am_peak_stops():
    graph = nx.path_graph(3)
    graph.nodes[0]["am_peak_stops"] = 12
    graph.nodes[1]["am_peak_stops"] = 5.0
    # node 2 has no attribute: a zero baseline, not a KeyError
    assert initial_loads(graph, "demand") == pytest.approx(
        {0: 12.0, 1: 5.0, 2: 0.0}
    )


def test_demand_mode_treats_none_as_zero():
    graph = nx.path_graph(2)
    graph.nodes[0]["am_peak_stops"] = None
    graph.nodes[1]["am_peak_stops"] = 7
    assert initial_loads(graph, "demand") == {0: 0.0, 1: 7.0}


@pytest.mark.parametrize("stops", [-1.0, math.inf, math.nan, "many"])
def test_demand_mode_rejects_bad_counts(stops):
    graph = nx.path_graph(3)
    graph.nodes[1]["am_peak_stops"] = stops
    with pytest.raises(ValueError, match="am_peak_stops"):
        initial_loads(graph, "demand")


def test_unknown_load_mode_is_rejected():
    with pytest.raises(ValueError, match="unknown load mode"):
        initial_loads(nx.path_graph(3), "made-up")  # type: ignore[arg-type]


def test_empty_graph_has_no_loads_and_no_capacities():
    assert initial_loads(nx.Graph()) == {}
    assert capacities({}, 0.2) == {}


def test_single_node_graph_has_zero_load_and_zero_capacity():
    graph = nx.Graph()
    graph.add_node("A")
    loads = initial_loads(graph, "betweenness")
    assert loads == {"A": 0.0}
    assert capacities(loads, 0.2) == {"A": 0.0}


def test_zero_load_stations_get_zero_capacity_without_nan():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C"])
    loads = initial_loads(graph, "betweenness")
    assert loads == {"A": 0.0, "B": 0.0, "C": 0.0}
    result = capacities(loads, 0.5)
    assert result == {"A": 0.0, "B": 0.0, "C": 0.0}
    assert all(not math.isnan(value) and value >= 0.0 for value in result.values())


def test_capacities_apply_the_tolerance_factor():
    loads = {"A": 2.0, "B": 0.0, "C": 4.0}
    assert capacities(loads, 0.0) == {"A": 2.0, "B": 0.0, "C": 4.0}
    assert capacities(loads, 0.25) == pytest.approx({"A": 2.5, "B": 0.0, "C": 5.0})


def test_capacities_do_not_mutate_the_input_mapping():
    loads = {"A": 2.0, "B": 3.5}
    snapshot = dict(loads)
    capacities(loads, 1.0)
    assert loads == snapshot


@pytest.mark.parametrize("alpha", [-1.0, -1e-9, math.inf, -math.inf, math.nan])
def test_capacities_reject_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        capacities({"A": 1.0}, alpha)


@pytest.mark.parametrize("load", [-1.0, math.inf, math.nan])
def test_capacities_reject_bad_loads(load):
    with pytest.raises(ValueError, match="station 'A'"):
        capacities({"A": load}, 0.2)


def test_capacities_reject_non_numeric_loads():
    with pytest.raises(ValueError, match="not numeric"):
        capacities({"A": "heavy"}, 0.2)  # type: ignore[dict-item]


def test_loads_are_numpy_free_python_floats():
    graph = _uniform_trips(nx.path_graph(4))
    loads = initial_loads(graph, "betweenness_freq")
    assert all(type(value) is float for value in loads.values())
    result = capacities(loads, 0.1)
    assert all(type(value) is float and np.isfinite(value) for value in result.values())


# ---------------------------------------------------------------------------
# frequency-scaled demand reference capacities (P2.4)
# ---------------------------------------------------------------------------
def _demand_graph(stops: list[float], frequency: list[float]) -> nx.Graph:
    graph = nx.path_graph(len(stops))
    for node, (load, trips) in enumerate(zip(stops, frequency)):
        graph.nodes[node]["am_peak_stops"] = load
        graph.nodes[node]["trips_served"] = trips
    return graph


def test_demand_reference_capacities_scale_frequency_to_cover_loads():
    graph = _demand_graph([4, 10, 0], [10, 20, 5])
    # c = max(4/10, 10/20, 0/5) = 0.5, so K = c * frequency
    assert demand_reference_capacities(graph) == pytest.approx(
        {0: 5.0, 1: 10.0, 2: 2.5}
    )
    loads = initial_loads(graph, "demand")
    reference = demand_reference_capacities(graph)
    assert all(reference[node] >= loads[node] for node in graph)


def test_demand_reference_capacities_are_zero_without_attributes():
    assert demand_reference_capacities(nx.path_graph(2)) == {0: 0.0, 1: 0.0}
    assert demand_reference_capacities(nx.Graph()) == {}


def test_demand_reference_capacities_reject_demand_without_frequency():
    graph = _demand_graph([3, 1], [0, 5])
    with pytest.raises(ValueError, match="zero trips_served"):
        demand_reference_capacities(graph)


@pytest.mark.parametrize("trips_served", [-2.0, math.inf, math.nan, "some"])
def test_demand_reference_capacities_reject_bad_frequency(trips_served):
    graph = _demand_graph([1, 1], [trips_served, 5])
    with pytest.raises(ValueError, match="trips_served"):
        demand_reference_capacities(graph)


def test_frozen_demand_capacities_cover_every_load():
    graph = load_rail_graph()
    loads = initial_loads(graph, "demand")
    reference = demand_reference_capacities(graph)
    assert set(reference) == set(graph)
    assert all(reference[node] >= loads[node] for node in graph)
    scale = max(
        loads[node] / float(graph.nodes[node]["trips_served"])
        for node in graph
        if graph.nodes[node]["trips_served"] > 0
    )
    for node in graph:
        expected = scale * float(graph.nodes[node]["trips_served"])
        assert reference[node] == pytest.approx(expected)