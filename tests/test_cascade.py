"""Cascade fixtures for synchronous updates, zero capacities and conservation."""

import math

import networkx as nx
import pytest

from transperth.cascade import simulate_cascade
from transperth.config import CascadeConfig
from transperth.failure import random_target_order, rank_targets


def path(n=3):
    return nx.relabel_nodes(nx.path_graph(n), lambda node: str(node))


def test_zero_capacity_receives_load_and_fails():
    result = simulate_cascade(path(), CascadeConfig(target="1", rule="capacity"))
    assert result.failed == ("1", "0", "2")
    assert result.avalanche_sizes == (2,)
    assert result.rounds == 1
    assert result.initial_total_load == result.lost_load == 1
    assert result.remaining_load == 0


@pytest.mark.parametrize("graph", [path(5), nx.relabel_nodes(nx.cycle_graph(6), str)])
@pytest.mark.parametrize("rule", ["equal", "capacity"])
def test_high_alpha_containment_with_positive_leaf_loads(graph, rule):
    nx.set_node_attributes(graph, 10, "trips_served")
    result = simulate_cascade(
        graph, CascadeConfig(alpha=100, rule=rule, load_mode="betweenness_plus_trips")
    )
    assert result.n_failed == 1
    assert result.rounds == 0
    assert result.avalanche_sizes == ()
    assert result.remaining_load + result.lost_load == pytest.approx(
        result.initial_total_load
    )


def test_isolated_and_zero_load_leaf_triggers():
    for graph, target in [(path(), "0"), (nx.empty_graph(["a"]), "a")]:
        result = simulate_cascade(graph, CascadeConfig(target=target))
        assert result.failed == (target,)
        assert result.rounds == 0


def test_simultaneous_failures_do_not_forward_to_each_other():
    graph = nx.Graph([("t", "a"), ("t", "b"), ("a", "b"), ("b", "c")])
    baseline = {"t": 2, "a": 1, "b": 1, "c": 1}
    result = simulate_cascade(
        graph,
        CascadeConfig(target="t", alpha=0.5, rule="equal"),
        baseline_loads=baseline,
    )
    assert result.failed == ("t", "a", "b", "c")
    assert result.avalanche_sizes == (2, 1)
    assert result.remaining_load + result.lost_load == pytest.approx(5)
    reversed_graph = nx.Graph()
    reversed_graph.add_nodes_from(reversed(list(graph.nodes)))
    reversed_graph.add_edges_from(reversed(list(graph.edges)))
    assert (
        simulate_cascade(
            reversed_graph,
            CascadeConfig(target="t", alpha=0.5, rule="equal"),
            baseline_loads=baseline,
        )
        == result
    )
    assert baseline == {"t": 2, "a": 1, "b": 1, "c": 1}
    assert len(graph) == 4


def test_dynamic_does_not_redistribute_and_does_not_claim_conservation():
    result = simulate_cascade(path(), CascadeConfig(target="1", dynamic=True))
    assert result.failed == ("1",)
    assert result.rounds == 0
    assert result.remaining_load is result.lost_load is None


def test_dynamic_capacities_are_fixed_and_mode_is_preserved():
    graph = path()
    nx.set_node_attributes(graph, 10, "trips_served")
    result = simulate_cascade(
        graph,
        CascadeConfig(
            target="1", dynamic=True, alpha=0, load_mode="betweenness_plus_trips"
        ),
        baseline_loads={"0": 0, "1": 2, "2": 0},
    )
    assert result.failed == ("1", "0", "2")


def test_trigger_rules_match_shared_engine():
    graph = path(6)
    assert (
        simulate_cascade(graph, CascadeConfig(trigger="degree")).failed[0]
        == rank_targets(graph)[0]
    )
    for seed in [0, 7, 31]:
        config = CascadeConfig(trigger="random", seed=seed)
        result = simulate_cascade(graph, config)
        assert result.failed[0] == random_target_order(graph, seed=seed)[0]
        assert result == simulate_cascade(graph, config)


@pytest.mark.parametrize(
    "baseline", [{"0": 1}, {"0": math.nan, "1": 1, "2": 0}, {"0": -1, "1": 1, "2": 0}]
)
def test_invalid_baseline(baseline):
    with pytest.raises(ValueError):
        simulate_cascade(path(), CascadeConfig(), baseline_loads=baseline)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"target": "missing"},
        {"seed": -1},
        {"trigger": "bad"},
        {"rule": "bad"},
        {"dynamic": 1},
        {"load_mode": "bad"},
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        simulate_cascade(path(), CascadeConfig(**kwargs))


def test_demand_is_not_silently_substituted():
    with pytest.raises(NotImplementedError):
        simulate_cascade(path(), CascadeConfig(load_mode="demand"))


def test_empty_graph():
    result = simulate_cascade(nx.Graph(), CascadeConfig())
    assert result.failed == ()
    assert result.gcc_fraction == result.failed_fraction == result.rounds == 0


def test_weighted_mode_and_tolerance():
    graph = path()
    nx.set_edge_attributes(graph, 5, "trips")
    result = simulate_cascade(
        graph, CascadeConfig(target="1", load_mode="betweenness_freq", tolerance=0.6)
    )
    assert result.failed == ("1",)
    assert result.remaining_load == 1


def test_load_function_drives_baseline_and_dynamic_recompute():
    graph = path()
    calls: list[list[str]] = []

    def loader(current):
        calls.append(sorted(current))
        return {node: 0.0 for node in current}

    result = simulate_cascade(
        graph, CascadeConfig(target="1", dynamic=True), load_function=loader
    )
    assert result.failed == ("1",)
    assert calls == [["0", "1", "2"], ["0", "2"]]


def test_baseline_loads_override_the_initial_load_function_call():
    graph = path()
    calls: list[list[str]] = []

    def loader(current):
        calls.append(sorted(current))
        return {node: 0.0 for node in current}

    result = simulate_cascade(
        graph,
        CascadeConfig(target="1", dynamic=True),
        baseline_loads={"0": 0.0, "1": 1.0, "2": 0.0},
        load_function=loader,
    )
    assert result.failed == ("1",)
    assert calls == [["0", "2"]]


def test_load_function_allows_demand_mode():
    demand = lambda current: {node: 1.0 for node in current}
    result = simulate_cascade(
        path(),
        CascadeConfig(target="1", load_mode="demand", alpha=1.0),
        load_function=demand,
    )
    assert result.failed == ("1",)
    assert result.remaining_load + result.lost_load == pytest.approx(
        result.initial_total_load
    )


@pytest.mark.parametrize(
    "bad",
    [
        {"0": 1.0},
        {"0": math.nan, "1": 1.0, "2": 0.0},
        {"0": -1.0, "1": 1.0, "2": 0.0},
    ],
)
def test_invalid_load_function(bad):
    with pytest.raises(ValueError):
        simulate_cascade(path(), CascadeConfig(), load_function=lambda current: bad)


def test_load_function_must_be_callable_and_return_a_mapping():
    with pytest.raises(TypeError):
        simulate_cascade(path(), CascadeConfig(), load_function=object())
    with pytest.raises(TypeError):
        simulate_cascade(
            path(), CascadeConfig(), load_function=lambda current: [1.0, 2.0, 3.0]
        )
