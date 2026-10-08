"""Scenario assembly, demand weights and the ported layered cascade (P2.3)."""

import copy
import math

import networkx as nx
import pandas as pd
import pytest

from transperth.multilayer import (
    add_backup_edges,
    add_walking_transfers,
    build_layers,
    terminal_pairs,
    travel_times,
)
from transperth.recovery import (
    ENDPOINT_SUM_MODEL,
    endpoint_demand_weights,
    exclusive_line_stations,
    layered_cascade,
    major_interchange_stations,
    random_failure_sets,
    recovery_metrics,
    scenario_manifest,
    station_lines,
)


def path_rail() -> nx.Graph:
    """Return a three-station path with lines, trips and demand attributes."""
    rail = nx.path_graph(["A", "B", "C"])
    nx.set_edge_attributes(rail, 1000, "distance_m")
    for node, lines, demand in (
        ("A", ["red"], 10.0),
        ("B", ["red", "blue"], 20.0),
        ("C", ["blue"], 30.0),
    ):
        rail.nodes[node].update(
            name=f"{node} Stn",
            lines=lines,
            trips_served=int(demand),
            am_peak_stops=demand,
        )
    return rail


def path_layers() -> nx.Graph:
    """Return the layered path graph with a five-minute bus between A and C."""
    rail = path_rail()
    return build_layers(rail)


def bus_table(minutes: float = 9.0) -> pd.DataFrame:
    """Return one terminal-to-terminal bus candidate."""
    return pd.DataFrame(
        [
            {
                "station_a": "A",
                "station_b": "C",
                "minutes": minutes,
                "kind": "emergency_bus",
            }
        ]
    )


class TestScenarioAssembly:
    def test_exclusive_line_stations_skip_shared_stations(self):
        assert exclusive_line_stations(path_rail()) == {"blue": ("C",), "red": ("A",)}

    def test_line_without_exclusive_station_keeps_an_empty_tuple(self):
        rail = nx.path_graph(["A", "B"])
        rail.nodes["A"].update(lines=["red", "blue"])
        rail.nodes["B"].update(lines=["red"])
        assert exclusive_line_stations(rail) == {"blue": (), "red": ("B",)}

    def test_station_lines_accepts_lists_strings_and_shares_the_fix(self):
        rail = path_rail()
        assert station_lines(rail, "B") == {"red", "blue"}
        rail.nodes["B"]["lines"] = "red; blue; "
        assert station_lines(rail, "B") == {"red", "blue"}
        rail.nodes["B"]["lines"] = 7
        with pytest.raises(ValueError, match="lines must be"):
            station_lines(rail, "B")

    def test_major_interchange_stations_needs_two_lines(self):
        rail = path_rail()
        assert major_interchange_stations(rail, min_lines=2) == ("B",)
        assert major_interchange_stations(rail, min_lines=3) == ()
        with pytest.raises(ValueError, match="min_lines"):
            major_interchange_stations(rail, min_lines=1)

    def test_random_failure_sets_are_seeded_distinct_and_sized(self):
        rail = path_rail()
        first = random_failure_sets(rail, n_sets=3, size=2, seed=0)
        second = random_failure_sets(rail, n_sets=3, size=2, seed=0)
        other = random_failure_sets(rail, n_sets=3, size=2, seed=1)
        assert first == second
        assert len(first) == 3
        for stations in first:
            assert len(stations) == 2
            assert len(set(stations)) == 2
        assert first != other
        assert random_failure_sets(rail, n_sets=0, size=1, seed=0) == []
        with pytest.raises(ValueError, match="exceeds"):
            random_failure_sets(rail, n_sets=1, size=4, seed=0)

    def test_scenario_manifest_covers_all_three_kinds(self):
        rail = path_rail()
        manifest = scenario_manifest(
            rail, n_random=2, random_size=1, min_interchange_lines=2, seed=0
        )
        assert list(manifest.columns) == [
            "scenario",
            "kind",
            "label",
            "n_failed",
            "failed_ids",
        ]
        assert manifest["kind"].tolist() == [
            "line_closure",
            "line_closure",
            "interchange",
            "random",
            "random",
        ]
        assert manifest["scenario"].tolist() == [
            "line_closure:blue",
            "line_closure:red",
            "interchange:B",
            "random:0",
            "random:1",
        ]
        assert manifest.loc[0, "n_failed"] == 1
        assert manifest.loc[0, "failed_ids"] == "C"


class TestDemandWeights:
    def test_endpoint_weights_sum_both_sides_and_strip_prefixes(self):
        pairs = [("T:A", "T:B"), ("B", "C")]
        weights = endpoint_demand_weights(pairs, {"A": 10.0, "B": 20.0, "C": 30.0})
        assert weights == {("A", "B"): 30.0, ("B", "C"): 50.0}
        assert ENDPOINT_SUM_MODEL in endpoint_demand_weights.__doc__

    def test_duplicate_pairs_count_once_and_unknown_stations_fail(self):
        weights = endpoint_demand_weights(
            [("A", "B"), ("B", "A")], {"A": 1.0, "B": 2.0}
        )
        assert weights == {("A", "B"): 3.0}
        with pytest.raises(ValueError, match="unknown demand station"):
            endpoint_demand_weights([("A", "Z")], {"A": 1.0})
        with pytest.raises(ValueError, match="finite"):
            endpoint_demand_weights([("A", "B")], {"A": -1.0, "B": 2.0})
        with pytest.raises(ValueError, match="self-pairs"):
            endpoint_demand_weights([("A", "A")], {"A": 1.0})


class TestLayeredCascade:
    def test_path_closure_disconnects_every_pair_without_buses(self):
        normal = path_layers()
        pairs = terminal_pairs(normal)
        result = layered_cascade(normal, normal, alpha=100.0, initial_failed=["B"])
        metrics = recovery_metrics(result.final_graph, normal, baseline_pairs=pairs)
        assert result.n_failed == 1
        assert result.rounds == 0
        assert metrics["reachable_pair_count"] == 0
        assert metrics["served_od_fraction"] == 0.0

    def test_buses_on_both_sides_reconnect_the_persistent_terminals(self):
        normal = path_layers()
        buses = pd.DataFrame(
            [
                {
                    "station_a": "A",
                    "station_b": "B",
                    "minutes": 5.0,
                    "kind": "existing_bus",
                },
                {
                    "station_a": "B",
                    "station_b": "C",
                    "minutes": 5.0,
                    "kind": "existing_bus",
                },
            ]
        )
        standby = add_backup_edges(normal, buses)
        pairs = terminal_pairs(normal)
        result = layered_cascade(normal, standby, alpha=100.0, initial_failed=["B"])
        metrics = recovery_metrics(result.final_graph, normal, baseline_pairs=pairs)
        assert metrics["reachable_pair_count"] == 3
        assert metrics["served_od_fraction"] == 1.0

    def test_batch_failures_are_removed_together_and_excluded(self):
        normal = path_layers()
        result = layered_cascade(
            normal, normal, alpha=100.0, initial_failed=["A", "C"]
        )
        assert result.initial_failed == ("A", "C")
        assert result.n_failed == 2
        assert result.n_secondary == 0
        assert result.rounds == 0
        assert result.avalanche_sizes == ()
        assert result.failed_fraction == pytest.approx(2 / 3)

    def test_duplicate_batch_entries_are_deduplicated(self):
        normal = path_layers()
        result = layered_cascade(normal, normal, alpha=100.0, initial_failed=["A", "A"])
        assert result.n_failed == 1

    def test_ring_rerouting_produces_a_synchronous_secondary_batch(self):
        ring = nx.cycle_graph([str(index) for index in range(10)])
        nx.set_edge_attributes(ring, 1000, "distance_m")
        normal = build_layers(ring)
        result = layered_cascade(
            normal, normal, alpha=0.0, initial_failed=["0"], trace=True
        )
        assert result.n_secondary > 0
        assert max(result.avalanche_sizes) > 1
        assert result.rounds == len(result.avalanche_sizes)
        assert result.history.iloc[-1]["next_failed_ids"] == ""
        assert len(result.history) == result.rounds + 1
        assert set(result.loads.columns) >= {
            "round",
            "station_id",
            "load",
            "capacity",
            "load_ratio",
            "overloaded",
        }

    def test_layered_cascade_matches_the_merged_engine_batch(self):
        """The port agrees with cascade.simulate_cascade after the P2.2 merge."""
        from transperth.cascade import simulate_cascade
        from transperth.config import CascadeConfig
        from transperth.multilayer import terminal_loads

        ring = nx.cycle_graph([str(index) for index in range(10)])
        nx.set_edge_attributes(ring, 1000, "distance_m")
        normal = build_layers(ring)
        loads0 = terminal_loads(normal)
        baseline = {node: float(loads0.get(node, 0.0)) for node in normal}

        def engine_loads(graph: nx.Graph) -> dict[str, float]:
            current = terminal_loads(graph)
            return {node: float(current.get(node, 0.0)) for node in graph}

        port = layered_cascade(
            normal, normal, alpha=0.0, initial_failed=["0"], trace=True
        )
        engine = simulate_cascade(
            normal,
            CascadeConfig(alpha=0.0, dynamic=True),
            baseline_loads=baseline,
            load_function=engine_loads,
            reference_capacities=baseline,
            initial_failed=["R:0"],
        )
        assert port.failed == engine.failed
        assert port.avalanche_sizes == engine.avalanche_sizes
        assert port.rounds == engine.rounds

    def test_overload_control_ports_the_no_cascade_case(self):
        ring = nx.cycle_graph([str(index) for index in range(10)])
        nx.set_edge_attributes(ring, 1000, "distance_m")
        normal = build_layers(ring)
        control = layered_cascade(
            normal,
            normal,
            alpha=0.0,
            initial_failed=["0"],
            overload_enabled=False,
        )
        assert control.n_failed == 1
        assert control.rounds == 0

    def test_standby_may_omit_exactly_the_initial_batch(self):
        normal = path_layers()
        failed_baseline = normal.copy()
        failed_baseline.remove_node("R:B")
        result = layered_cascade(
            normal, failed_baseline, alpha=100.0, initial_failed=["B"]
        )
        assert result.failed == ("R:B",)
        with pytest.raises(ValueError, match="missing"):
            layered_cascade(normal, failed_baseline, alpha=100.0)
        with pytest.raises(ValueError, match="outside"):
            extra = normal.copy()
            extra.add_node("T:Z", kind="terminal")
            layered_cascade(normal, extra, alpha=100.0)

    def test_inputs_are_not_modified_and_runs_are_deterministic(self):
        normal = path_layers()
        standby = add_backup_edges(normal, bus_table())
        before_normal = copy.deepcopy(normal)
        before_standby = copy.deepcopy(standby)
        first = layered_cascade(normal, standby, alpha=0.0, initial_failed=["B"])
        second = layered_cascade(normal, standby, alpha=0.0, initial_failed=["B"])
        assert first.failed == second.failed
        assert first.avalanche_sizes == second.avalanche_sizes
        assert nx.utils.graphs_equal(normal, before_normal)
        assert nx.utils.graphs_equal(standby, before_standby)

    def test_validation_rejects_bad_inputs(self):
        normal = path_layers()
        with pytest.raises(ValueError, match="unknown failed station"):
            layered_cascade(normal, normal, alpha=1.0, initial_failed=["Z"])
        for alpha in (-1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="alpha"):
                layered_cascade(normal, normal, alpha=alpha)
        with pytest.raises(ValueError, match="tolerance"):
            layered_cascade(normal, normal, alpha=1.0, tolerance=-1.0)
        with pytest.raises(ValueError, match="overload_enabled"):
            layered_cascade(normal, normal, alpha=1.0, overload_enabled="yes")
        with pytest.raises(TypeError, match="undirected"):
            layered_cascade(normal, nx.DiGraph(normal), alpha=1.0)


class TestRecoveryMetrics:
    def test_served_demand_and_travel_time_penalty(self):
        normal = path_layers()
        standby = add_backup_edges(normal, bus_table(minutes=9.0))
        pairs = terminal_pairs(normal)
        intact_times = travel_times(normal, pairs=pairs)
        result = layered_cascade(normal, standby, alpha=100.0, initial_failed=["B"])
        demand = endpoint_demand_weights(
            pairs, {"A": 10.0, "B": 20.0, "C": 30.0}
        )
        metrics = recovery_metrics(
            result.final_graph,
            normal,
            baseline_pairs=pairs,
            intact_times=intact_times,
            demand=demand,
        )
        assert metrics["reachable_pair_count"] == 1
        assert metrics["served_od_fraction"] == pytest.approx(1 / 3)
        assert metrics["total_demand_weight"] == pytest.approx(120.0)
        assert metrics["served_demand_weight"] == pytest.approx(40.0)
        assert metrics["served_demand_fraction"] == pytest.approx(1 / 3)
        # The nine-minute bus replaces an intact five-minute rail path.
        assert metrics["mean_travel_time_penalty_min"] == pytest.approx(4.0)

    def test_penalty_is_missing_when_nothing_is_reachable(self):
        normal = path_layers()
        pairs = terminal_pairs(normal)
        result = layered_cascade(normal, normal, alpha=100.0, initial_failed=["B"])
        metrics = recovery_metrics(result.final_graph, normal, baseline_pairs=pairs)
        assert math.isnan(metrics["mean_travel_time_penalty_min"])
        assert math.isnan(metrics["mean_reachable_minutes"])

    def test_demand_outside_the_baseline_is_rejected(self):
        normal = path_layers()
        pairs = terminal_pairs(normal)
        with pytest.raises(ValueError, match="not in the baseline"):
            recovery_metrics(
                normal, normal, baseline_pairs=pairs, demand={("A", "Z"): 1.0}
            )

    def test_walking_transfer_counts_in_the_baseline_denominator(self):
        normal = path_layers()
        walked = add_walking_transfers(normal, {"station_a": "A", "station_b": "C"})
        result = layered_cascade(normal, walked, alpha=100.0, initial_failed=["B"])
        pairs = terminal_pairs(walked)
        metrics = recovery_metrics(result.final_graph, walked, baseline_pairs=pairs)
        assert metrics["total_pair_count"] == 3
        assert metrics["reachable_pair_count"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))