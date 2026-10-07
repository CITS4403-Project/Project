"""Closed-form and denominator checks for network metrics."""

import math

import networkx as nx
import pytest

from transperth.metrics import compute_metrics


def test_path_closed_form():
    graph = nx.path_graph(4)
    metrics = compute_metrics(graph)
    assert metrics.aspl == pytest.approx(5 / 3)
    assert metrics.efficiency == pytest.approx(13 / 18)
    assert metrics.gcc_fraction == 1
    assert metrics.served_od_fraction == 1


def test_complete_graph():
    assert compute_metrics(nx.complete_graph(5)).efficiency == 1


def test_fixed_denominator_and_damage():
    original = nx.complete_graph(4)
    survivors = original.subgraph([0, 1]).copy()
    survivors.graph["n_initial"] = 4
    metrics = compute_metrics(survivors, baseline_efficiency=1)
    assert metrics.gcc_fraction == 0.5
    assert metrics.efficiency == pytest.approx(1 / 6)
    assert metrics.network_damage == pytest.approx(5 / 6)
    assert metrics.served_od_fraction == pytest.approx(1 / 6)
    assert compute_metrics(nx.path_graph(30), n_initial=86).gcc_fraction == 30 / 86


def test_disconnected_and_explicit_od():
    graph = nx.Graph([(0, 1), (2, 3)])
    graph.add_node(4)
    metrics = compute_metrics(graph, served_pairs=1, total_pairs=6)
    assert (metrics.gcc_size, metrics.lcc_size, metrics.isolated) == (2, 2, 1)
    assert metrics.efficiency == 0.2
    assert metrics.aspl == 1
    assert metrics.served_od_fraction == 1 / 6


@pytest.mark.parametrize("graph", [nx.Graph(), nx.empty_graph(1)])
def test_degenerate_graphs(graph):
    metrics = compute_metrics(graph, baseline_efficiency=0)
    assert metrics.aspl == metrics.efficiency == metrics.network_damage == 0
    assert metrics.served_od_fraction == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_initial": 1},
        {"n_initial": True},
        {"served_pairs": 1},
        {"served_pairs": 2, "total_pairs": 1},
        {"served_pairs": -1, "total_pairs": 0},
        {"baseline_efficiency": math.nan},
        {"baseline_efficiency": 0},
    ],
)
def test_bad_denominators(kwargs):
    with pytest.raises(ValueError):
        compute_metrics(nx.path_graph(2), **kwargs)


@pytest.mark.parametrize("graph", [nx.DiGraph(), nx.MultiGraph()])
def test_graph_type(graph):
    with pytest.raises(ValueError):
        compute_metrics(graph)
