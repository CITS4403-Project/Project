"""Budget, persistent-terminal, evidence and marginal-demand regressions."""

import networkx as nx
import pandas as pd
import pytest

from transperth.config import Strategy
from transperth.multilayer import add_backup_edges, build_layers, served_od_fraction
from transperth.strategies import (
    CorridorReinforcement,
    DemandAdaptive,
    ExistingBus,
    NoBackup,
    ShuttleBridging,
)


def fixture():
    rail = nx.Graph([("A", "B"), ("B", "C")])
    nx.set_edge_attributes(rail, 1000, "distance_m")
    nx.set_node_attributes(rail, "red", "lines")
    nx.set_node_attributes(rail, 10, "trips_served")
    layered = build_layers(rail)
    table = pd.DataFrame(
        [
            {
                "station_a": "A",
                "station_b": "B",
                "minutes": 5,
                "kind": "existing_bus",
                "source": "manual",
            },
            {
                "station_a": "B",
                "station_b": "C",
                "minutes": 2,
                "kind": "gtfs_candidate",
                "source": "snapshot",
                "evidence_trip_id": "trip1",
            },
            {
                "station_a": "A",
                "station_b": "C",
                "minutes": 9,
                "kind": "emergency_bus",
                "source": "feasible-road",
            },
        ]
    )
    return rail, layered, table


def all_strategies(rail, table):
    return [
        NoBackup(),
        ExistingBus(table),
        ShuttleBridging(table),
        CorridorReinforcement(table, rail=rail),
        DemandAdaptive(table),
    ]


def test_budget_protocol_determinism_and_nonmutation():
    rail, graph, table = fixture()
    before_graph = nx.node_link_data(graph)
    before_table = table.copy(deep=True)
    reordered = nx.Graph()
    reordered.add_nodes_from(reversed(list(graph.nodes(data=True))))
    reordered.add_edges_from(reversed(list(graph.edges(data=True))))
    for first, second in zip(
        all_strategies(rail, table), all_strategies(rail, table.iloc[::-1])
    ):
        assert isinstance(first, Strategy)
        for budget in [0, 1, 2, 20]:
            pairs = first.deploy(graph, budget=budget, failed=["B"], seed=7)
            assert len(pairs) <= budget
            assert len(set(pairs)) == len(pairs)
            assert pairs == second.deploy(
                reordered, budget=budget, failed=["B"], seed=7
            )
        for bad in [-1, True, 1.5]:
            with pytest.raises(ValueError):
                first.deploy(graph, budget=bad)
    assert nx.node_link_data(graph) == before_graph
    pd.testing.assert_frame_equal(table, before_table)


def test_no_backup_matches_failed_baseline():
    _, graph, _ = fixture()
    graph.remove_node("R:B")
    strategy = NoBackup()
    selected = strategy.deploy(graph, budget=10, failed=["B"])
    restored = add_backup_edges(graph, strategy.selected_table(selected))
    assert nx.node_link_data(restored) == nx.node_link_data(graph)
    assert served_od_fraction(restored) == served_od_fraction(graph)


def test_shuttle_reconnects_persistent_failed_terminal():
    _, graph, table = fixture()
    graph.remove_node("R:B")
    strategy = ShuttleBridging(table)
    pairs = strategy.deploy(graph, budget=2, failed=["B"])
    restored = add_backup_edges(graph, strategy.selected_table(pairs))
    assert served_od_fraction(graph) == 0
    assert served_od_fraction(restored) == 1
    assert any("B" in pair for pair in pairs)


def test_demand_priority_and_no_double_counting():
    _, graph, table = fixture()
    strategy = DemandAdaptive(table)
    weights = {("A", "B"): 100, ("B", "C"): 1}
    assert strategy.deploy(graph, budget=1, failed=["B"], demand=weights) == [
        ("A", "B")
    ]
    assert strategy.deploy(graph, budget=10, failed=["B"], demand=weights) == [
        ("A", "B"),
        ("B", "C"),
    ]
    assert strategy.deploy(graph, budget=10, failed=["B"], demand={}) == []
    assert strategy.deploy(graph, budget=10, failed=["B"], demand={("A", "B"): 0}) == []
    assert strategy.deploy(graph, budget=1, failed=["B"], demand=None)


def test_active_existing_links_and_candidate_kinds():
    _, graph, table = fixture()
    strategy = ExistingBus(table)
    assert strategy.deploy(graph, budget=20) == [("B", "C"), ("A", "B")]
    active = add_backup_edges(graph, table.iloc[[1]])
    assert strategy.deploy(active, budget=20) == [("A", "B")]


def test_fastest_duplicate_retains_evidence():
    _, graph, table = fixture()
    duplicate = table.iloc[[1]].copy()
    duplicate["station_a"], duplicate["station_b"] = "C", "B"
    duplicate["minutes"] = 1
    duplicate["source"] = "faster-evidence"
    candidates = pd.concat([table, duplicate])
    strategy = ExistingBus(candidates)
    selected = strategy.deploy(graph, budget=1, failed=["B"])
    result_table = strategy.selected_table(selected)
    assert result_table.source.iloc[0] == "faster-evidence"
    assert result_table.evidence_trip_id.iloc[0] == "trip1"
    restored = add_backup_edges(graph, result_table)
    assert restored["T:B"]["T:C"]["minutes"] == 1
    assert restored["T:B"]["T:C"]["source"] == "faster-evidence"


@pytest.mark.parametrize("bad", [-1, float("nan"), float("inf"), 0])
def test_invalid_candidate_times(bad):
    _, _, table = fixture()
    table["minutes"] = table["minutes"].astype(float)
    table.loc[0, "minutes"] = bad
    with pytest.raises(ValueError):
        ExistingBus(table)


@pytest.mark.parametrize(
    "demand",
    [
        {("A", "B"): -1},
        {("A", "B"): float("nan")},
        {("missing", "A"): 1},
        {("A", "B"): 1, ("B", "A"): 1},
    ],
)
def test_invalid_demand(demand):
    _, graph, table = fixture()
    with pytest.raises(ValueError):
        DemandAdaptive(table).deploy(graph, budget=1, demand=demand)


def test_empty_unknown_and_corridor_context():
    rail, graph, table = fixture()
    for strategy in all_strategies(rail, table.iloc[:0]):
        assert strategy.deploy(graph, budget=10, failed=["B"]) == []
    with pytest.raises(ValueError):
        ExistingBus(table).deploy(graph, budget=1, failed=["missing"])
    table.loc[0, "station_a"] = "missing"
    with pytest.raises(ValueError):
        ExistingBus(table).deploy(graph, budget=1)
    with pytest.raises(ValueError):
        CorridorReinforcement(fixture()[2]).deploy(graph, budget=1, failed=["B"])


def test_corridor_uses_disrupted_flow_proxy():
    rail = nx.Graph()
    for station, line, trips in [
        ("A", "red", 100),
        ("B", "red", 100),
        ("C", "blue", 1),
        ("D", "blue", 1),
    ]:
        rail.add_node(station, lines=line, trips_served=trips)
    graph = build_layers(rail)
    table = pd.DataFrame(
        [
            {
                "station_a": "A",
                "station_b": "B",
                "minutes": 10,
                "kind": "emergency_bus",
            },
            {"station_a": "C", "station_b": "D", "minutes": 1, "kind": "emergency_bus"},
        ]
    )
    assert CorridorReinforcement(table, rail=rail).deploy(
        graph, budget=1, failed=["A", "C"]
    ) == [("A", "B")]


def test_corridor_accepts_the_frozen_loader_lines_list():
    rail = nx.Graph()
    for station, lines, trips in [
        ("A", ["red", "blue"], 100),
        ("B", ["red"], 50),
        ("C", ["red"], 50),
    ]:
        rail.add_node(station, lines=lines, trips_served=trips)
    graph = build_layers(rail)
    table = pd.DataFrame(
        [{"station_a": "B", "station_b": "C", "minutes": 5, "kind": "emergency_bus"}]
    )
    assert CorridorReinforcement(table, rail=rail).deploy(
        graph, budget=1, failed=["A"]
    ) == [("B", "C")]
