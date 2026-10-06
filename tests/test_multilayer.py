"""Tests for the layered terminal/facility model in :mod:`transperth.multilayer`."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import sys

import networkx as nx
import pandas as pd
import pytest

from transperth.multilayer import (
    BACKUP_KINDS,
    WALKING_TRANSFER_MINUTES,
    add_backup_edges,
    add_walking_transfers,
    build_layers,
    compare_manual_pairs,
    derive_gtfs_candidates,
    served_od_fraction,
    terminal_loads,
    terminal_pairs,
    travel_times,
)

ROOT = Path(__file__).resolve().parents[1]
BACKUP_CSV = ROOT / "data" / "processed" / "backup_edges.csv"
RAIL_EDGES_CSV = ROOT / "data" / "processed" / "rail_edges.csv"
STATIONS_CSV = ROOT / "data" / "processed" / "stations.csv"
TOPOLOGY_JSON = ROOT / "data" / "verified_topology.json"
SNAPSHOT_DIR = ROOT / "data" / "snapshots" / "2026-10-05_0700-0900"
SNAPSHOT_TABLES = ("stops.txt", "stop_times.txt", "bus_trips.csv")
SNAPSHOT_PRESENT = all((SNAPSHOT_DIR / name).is_file() for name in SNAPSHOT_TABLES)

sys.path.insert(0, str(ROOT / "scripts"))
from run_backup_candidates import load_rail_graph, run as run_backup_candidates  # noqa: E402

# Small path graph with one kilometre segments: 1000 m at 40 km/h is 1.5 min.
RAIL_MINUTES = 1.5
ACCESS_MINUTES = 1.0


def path_rail() -> nx.Graph:
    """Return the A-B-C rail path used by the unit tests."""
    rail = nx.path_graph(["A", "B", "C"])
    nx.set_edge_attributes(rail, 1000.0, "distance_m")
    for node in rail:
        rail.nodes[node]["name"] = f"{node} Stn"
    return rail


def path_layers() -> nx.Graph:
    return build_layers(path_rail(), speed_kmh=40.0, access_minutes=ACCESS_MINUTES)


def deadline_backup_table() -> pd.DataFrame:
    return pd.DataFrame(
        [{"station_a": "A", "station_b": "C", "minutes": 8.0, "kind": "existing_bus"}]
    )


def network_from_csv(stations_csv: Path = STATIONS_CSV, edges_csv: Path = RAIL_EDGES_CSV) -> nx.Graph:
    """Tiny local loader for the frozen rail tables (P1.1 has network.py)."""
    stations = pd.read_csv(stations_csv, dtype={"station_id": str})
    edges = pd.read_csv(edges_csv, dtype={"station_a": str, "station_b": str})
    rail = nx.Graph()
    for row in stations.itertuples(index=False):
        rail.add_node(row.station_id, name=row.name)
    for row in edges.itertuples(index=False):
        rail.add_edge(row.station_a, row.station_b, distance_m=float(row.distance_m))
    return rail


# ---------------------------------------------------------------------------
# build_layers
# ---------------------------------------------------------------------------
class TestBuildLayers:
    def test_path_layer_counts_and_node_attributes(self):
        layered = path_layers()
        assert layered.number_of_nodes() == 6
        assert layered.number_of_edges() == 5
        assert set(layered) == {"T:A", "T:B", "T:C", "R:A", "R:B", "R:C"}
        for node in ("T:A", "R:A"):
            assert layered.nodes[node]["station_id"] == "A"
            assert layered.nodes[node]["name"] == "A Stn"
        assert layered.nodes["T:A"]["kind"] == "terminal"
        assert layered.nodes["R:A"]["kind"] == "rail"
        assert layered["T:A"]["R:A"] == {"minutes": ACCESS_MINUTES, "mode": "access"}
        assert layered["R:A"]["R:B"]["minutes"] == pytest.approx(RAIL_MINUTES)
        assert layered["R:A"]["R:B"]["mode"] == "rail"
        assert layered["R:A"]["R:B"]["distance_m"] == pytest.approx(1000.0)

    def test_parameters_must_be_finite_and_positive(self):
        rail = path_rail()
        for kwargs in (
            {"speed_kmh": 0.0},
            {"speed_kmh": -1.0},
            {"speed_kmh": math.nan},
            {"access_minutes": 0.0},
            {"access_minutes": math.inf},
        ):
            with pytest.raises(ValueError):
                build_layers(rail, **kwargs)

    def test_rail_distance_validation_names_the_edge(self):
        for bad in (0.0, -5.0, math.nan, math.inf):
            rail = path_rail()
            rail["A"]["B"]["distance_m"] = bad
            with pytest.raises(ValueError, match="A-B"):
                build_layers(rail)

    def test_speed_scales_rail_time(self):
        layered = build_layers(path_rail(), speed_kmh=80.0)
        assert layered["R:A"]["R:B"]["minutes"] == pytest.approx(0.75)

    def test_same_inputs_give_the_same_graph(self):
        first = build_layers(path_rail())
        second = build_layers(path_rail())
        assert list(first.nodes) == list(second.nodes)
        assert nx.utils.graphs_equal(first, second)


# ---------------------------------------------------------------------------
# backup edges
# ---------------------------------------------------------------------------
class TestBackupEdges:
    def test_removing_middle_facility_splits_and_backup_reconnects(self):
        intact = path_layers()
        baseline = terminal_pairs(intact)
        assert len(baseline) == 3
        assert travel_times(intact)[("T:A", "T:C")] == pytest.approx(5.0)

        failed = intact.copy()
        failed.remove_node("R:B")
        assert failed.has_node("T:B")
        assert travel_times(failed) == {}
        assert served_od_fraction(failed, baseline_pairs=baseline) == 0.0

        reconnected = add_backup_edges(failed, deadline_backup_table())
        times = travel_times(reconnected)
        # The backup edge joins the two terminals, so the terminal-to-terminal
        # time is the recorded 8.0 minutes. The access+backup+access route
        # (1 + 8 + 1 = 10) is the facility-to-facility journey R:A -> R:C.
        assert times[("T:A", "T:C")] == pytest.approx(8.0)
        assert travel_times(reconnected, pairs=[("R:A", "R:C")]) == {
            ("R:A", "R:C"): pytest.approx(10.0)
        }
        assert served_od_fraction(reconnected, baseline_pairs=baseline) == pytest.approx(1 / 3)
        # The original graph is untouched and still disconnected.
        assert not intact.has_edge("T:A", "T:C")
        assert travel_times(intact)[("T:A", "T:C")] == pytest.approx(5.0)

    def test_duplicate_endpoints_keep_fastest_time_in_either_row_order(self):
        slow_first = pd.DataFrame(
            [
                {"station_a": "A", "station_b": "C", "minutes": 8.0, "kind": "existing_bus"},
                {"station_a": "C", "station_b": "A", "minutes": 6.0, "kind": "emergency_bus"},
            ]
        )
        fast_first = slow_first.iloc[::-1].reset_index(drop=True)
        for table in (slow_first, fast_first):
            result = add_backup_edges(path_layers(), table)
            assert result["T:A"]["T:C"]["minutes"] == pytest.approx(6.0)
            assert result["T:A"]["T:C"]["kind"] == "emergency_bus"
            assert result["T:A"]["T:C"]["mode"] == "backup"

    @pytest.mark.parametrize("bad", [-1.0, 0.0, math.nan, math.inf, -math.inf, "soon"])
    def test_invalid_minutes_are_rejected(self, bad):
        table = pd.DataFrame([{"station_a": "A", "station_b": "C", "minutes": bad, "kind": "existing_bus"}])
        with pytest.raises(ValueError, match="row 0"):
            add_backup_edges(path_layers(), table)

    def test_invalid_minutes_are_rejected_even_when_a_faster_row_exists(self):
        table = pd.DataFrame(
            [
                {"station_a": "A", "station_b": "C", "minutes": 6.0, "kind": "existing_bus"},
                {"station_a": "A", "station_b": "C", "minutes": -1.0, "kind": "existing_bus"},
            ]
        )
        with pytest.raises(ValueError, match="row 1"):
            add_backup_edges(path_layers(), table)

    def test_time_factor_must_be_finite_and_positive(self):
        table = deadline_backup_table()
        for factor in (0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="time_factor"):
                add_backup_edges(path_layers(), table, time_factor=factor)

    def test_time_factor_scales(self):
        result = add_backup_edges(path_layers(), deadline_backup_table(), time_factor=1.5)
        assert result["T:A"]["T:C"]["minutes"] == pytest.approx(12.0)

    def test_unknown_endpoints_and_self_pairs_are_rejected(self):
        unknown = pd.DataFrame([{"station_a": "A", "station_b": "Z", "minutes": 5.0, "kind": "existing_bus"}])
        with pytest.raises(ValueError, match="unknown terminal"):
            add_backup_edges(path_layers(), unknown)
        self_pair = pd.DataFrame([{"station_a": "A", "station_b": "A", "minutes": 5.0, "kind": "existing_bus"}])
        with pytest.raises(ValueError, match="self pair"):
            add_backup_edges(path_layers(), self_pair)

    def test_unknown_kind_names_the_row(self):
        table = pd.DataFrame([{"station_a": "A", "station_b": "C", "minutes": 5.0, "kind": "shuttle"}])
        with pytest.raises(ValueError, match="row 0.*shuttle"):
            add_backup_edges(path_layers(), table)
        assert "gtfs_candidate" in BACKUP_KINDS

    def test_missing_columns_are_rejected(self):
        table = pd.DataFrame([{"station_a": "A", "station_b": "C", "minutes": 5.0}])
        with pytest.raises(ValueError, match="kind"):
            add_backup_edges(path_layers(), table)

    def test_derived_shape_is_accepted_with_route_provenance(self):
        derived = pd.DataFrame(
            [
                {
                    "station_a": "A",
                    "station_b": "C",
                    "minutes": 9.0,
                    "kind": "gtfs_candidate",
                    "route_ids": "R1;R2",
                    "route_or_road": "",
                    "source": "",
                }
            ]
        )
        result = add_backup_edges(path_layers(), derived)
        assert result["T:A"]["T:C"]["kind"] == "gtfs_candidate"
        assert result["T:A"]["T:C"]["route_or_road"] == "R1;R2"


# ---------------------------------------------------------------------------
# terminal loads
# ---------------------------------------------------------------------------
class TestTerminalLoads:
    def test_keys_are_exactly_the_rail_facilities(self):
        loads = terminal_loads(path_layers())
        assert set(loads) == {"R:A", "R:B", "R:C"}

    def test_middle_facility_has_the_highest_load(self):
        loads = terminal_loads(path_layers())
        assert loads["R:B"] == pytest.approx(3.0)
        assert loads["R:A"] == pytest.approx(2.0)
        assert loads["R:C"] == pytest.approx(2.0)

    def test_deterministic_and_does_not_modify_the_input(self):
        layered = path_layers()
        before = copy.deepcopy(layered)
        first = terminal_loads(layered)
        second = terminal_loads(layered)
        assert first == second
        assert nx.utils.graphs_equal(layered, before)


# ---------------------------------------------------------------------------
# walking interchange
# ---------------------------------------------------------------------------
class TestWalkingTransfers:
    def test_link_is_absent_by_default(self):
        assert not path_layers().has_edge("T:A", "T:C")

    def test_helper_adds_documented_walk_edge(self):
        layered = path_layers()
        walked = add_walking_transfers(layered, {"station_a": "A", "station_b": "C"})
        assert walked["T:A"]["T:C"]["minutes"] == pytest.approx(WALKING_TRANSFER_MINUTES)
        assert walked["T:A"]["T:C"]["mode"] == "walk"
        assert walked["T:A"]["T:C"]["kind"] == "walking_transfer"
        assert not layered.has_edge("T:A", "T:C")

    def test_custom_minutes_and_iterable_input(self):
        walked = add_walking_transfers(path_layers(), [("A", "C")], minutes=7.5)
        assert walked["T:A"]["T:C"]["minutes"] == pytest.approx(7.5)

    def test_dataframe_input_with_per_row_minutes(self):
        table = pd.DataFrame(
            [
                {"station_a": "A", "station_b": "C", "minutes": 4.0},
                {"station_a": "A", "station_b": "B", "minutes": math.nan},
            ]
        )
        walked = add_walking_transfers(path_layers(), table, minutes=5.0)
        assert walked["T:A"]["T:C"]["minutes"] == pytest.approx(4.0)
        assert walked["T:A"]["T:B"]["minutes"] == pytest.approx(5.0)

    def test_unknown_endpoint_rejected(self):
        with pytest.raises(ValueError, match="unknown terminal"):
            add_walking_transfers(path_layers(), {"station_a": "A", "station_b": "Z"})

    def test_perth_perth_underground_walk_is_not_a_rail_edge(self):
        edges = pd.read_csv(RAIL_EDGES_CSV, dtype={"station_a": str, "station_b": str})
        pairs = {tuple(sorted((a, b))) for a, b in zip(edges.station_a, edges.station_b)}
        assert ("56", "64") not in pairs
        transfer = json.loads(TOPOLOGY_JSON.read_text(encoding="utf-8"))["walking_transfer"]
        assert tuple(sorted((transfer["station_a"], transfer["station_b"]))) == ("56", "64")
        layered = build_layers(network_from_csv())
        assert not layered.has_edge("T:56", "T:64")
        assert not layered.has_edge("R:56", "R:64")
        walked = add_walking_transfers(layered, transfer)
        assert walked["T:56"]["T:64"]["minutes"] == pytest.approx(WALKING_TRANSFER_MINUTES)
        assert walked["T:56"]["T:64"]["mode"] == "walk"
        assert not layered.has_edge("T:56", "T:64")

    @pytest.mark.skipif(
        not (SNAPSHOT_DIR / "transfers.txt").is_file(),
        reason="the snapshot transfers table is git-ignored",
    )
    def test_default_walk_time_matches_the_snapshot_transfer(self):
        transfers = pd.read_csv(SNAPSHOT_DIR / "transfers.txt", dtype=str)
        pair = transfers[
            transfers.from_stop_id.isin(["56", "64"]) & transfers.to_stop_id.isin(["56", "64"])
        ]
        assert len(pair) == 2
        assert pair.transfer_type.unique().tolist() == ["2"]
        assert pair.min_transfer_time.unique().tolist() == ["300"]
        assert WALKING_TRANSFER_MINUTES == 5.0


# ---------------------------------------------------------------------------
# OD helpers
# ---------------------------------------------------------------------------
class TestOdHelpers:
    def test_travel_times_uses_minutes_and_canonical_pairs(self):
        intact = path_layers()
        times = travel_times(intact, pairs=[("T:C", "T:A"), ("T:B", "T:A")])
        assert times[("T:A", "T:B")] == pytest.approx(3.5)
        assert times[("T:A", "T:C")] == pytest.approx(5.0)

    def test_terminal_pairs_are_sorted_undirected_tuples(self):
        assert terminal_pairs(path_layers()) == {("T:A", "T:B"), ("T:A", "T:C"), ("T:B", "T:C")}

    def test_served_od_fraction_keeps_the_explicit_baseline(self):
        intact = path_layers()
        baseline = terminal_pairs(intact)
        failed = intact.copy()
        failed.remove_node("R:B")
        reconnected = add_backup_edges(failed, deadline_backup_table())

        # 3 intact pairs; only A-C is restored, so 1/3 of the baseline is served.
        assert served_od_fraction(reconnected, baseline_pairs=baseline) == pytest.approx(1 / 3)
        # Without an explicit baseline the current graph's terminal pairs are used.
        assert served_od_fraction(reconnected) == pytest.approx(1 / 3)

        # Once a terminal disappears a current-graph denominator would overstate
        # service; passing the intact baseline keeps the original denominator.
        without_b = reconnected.copy()
        without_b.remove_node("T:B")
        assert served_od_fraction(without_b) == pytest.approx(1.0)
        assert served_od_fraction(without_b, baseline_pairs=baseline) == pytest.approx(1 / 3)

    def test_intact_graph_serves_every_pair(self):
        intact = path_layers()
        assert served_od_fraction(intact, baseline_pairs=terminal_pairs(intact)) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# manual pair comparison
# ---------------------------------------------------------------------------
class TestCompareManualPairs:
    def test_statuses_and_no_silent_drops(self):
        manual = pd.DataFrame(
            [
                {"station_a": "A", "station_b": "B", "minutes": 10.0, "kind": "existing_bus"},
                {"station_a": "A", "station_b": "C", "minutes": 20.0, "kind": "existing_bus"},
                {"station_a": "B", "station_b": "C", "minutes": 5.0, "kind": "existing_bus"},
            ]
        )
        derived = pd.DataFrame(
            [
                {"station_a": "A", "station_b": "B", "minutes": 8.0, "kind": "gtfs_candidate"},
                {"station_a": "C", "station_b": "A", "minutes": 40.0, "kind": "gtfs_candidate"},
            ]
        )
        check = compare_manual_pairs(manual, derived, tolerance_minutes=5.0)
        assert len(check) == len(manual) == 3
        assert check.columns.tolist()[:7] == [
            "station_a", "station_b", "manual_minutes", "derived_minutes",
            "delta_minutes", "status", "note",
        ]
        statuses = dict(zip(check.station_a + "-" + check.station_b, check.status))
        assert statuses == {"A-B": "matched", "A-C": "time_differs", "B-C": "not_derived"}
        assert check.loc[check.status == "time_differs", "delta_minutes"].iloc[0] == pytest.approx(20.0)
        assert check.loc[check.status == "not_derived", "derived_minutes"].isna().all()
        assert (check.loc[check.status == "not_derived", "manual_minutes"] == 5.0).all()
        assert check.manual_kind.eq("existing_bus").all()

    def test_duplicate_derived_rows_keep_the_fastest(self):
        manual = pd.DataFrame([{"station_a": "A", "station_b": "C", "minutes": 10.0}])
        derived = pd.DataFrame(
            [
                {"station_a": "A", "station_b": "C", "minutes": 40.0},
                {"station_a": "A", "station_b": "C", "minutes": 8.0},
            ]
        )
        check = compare_manual_pairs(manual, derived)
        assert check.derived_minutes.iloc[0] == pytest.approx(8.0)
        assert check.status.iloc[0] == "matched"


# ---------------------------------------------------------------------------
# GTFS candidate derivation
# ---------------------------------------------------------------------------
def write_mini_snapshot(root: Path) -> tuple[Path, Path]:
    """Write a three-station synthetic snapshot; return (snapshot_dir, stations_csv)."""
    snapshot = root / "snapshot"
    snapshot.mkdir()
    stations_csv = root / "stations.csv"
    pd.DataFrame(
        [
            {"station_id": "10", "name": "A Stn", "lat": 0.0, "lon": 0.0},
            {"station_id": "2", "name": "B Stn", "lat": 0.0, "lon": 0.003},
            {"station_id": "30", "name": "C Stn", "lat": 0.0, "lon": 0.005},
            # 20 and 40 share a location to exercise the sorted-id tie-break.
            {"station_id": "20", "name": "D Stn", "lat": 0.0, "lon": 0.007},
            {"station_id": "40", "name": "E Stn", "lat": 0.0, "lon": 0.007},
        ]
    ).to_csv(stations_csv, index=False)
    pd.DataFrame(
        [
            {"location_type": "0", "stop_id": "b10", "stop_lat": 0.0, "stop_lon": 0.0002},
            {"location_type": "0", "stop_id": "b2", "stop_lat": 0.0, "stop_lon": 0.0028},
            {"location_type": "0", "stop_id": "b30", "stop_lat": 0.0, "stop_lon": 0.0052},
            {"location_type": "0", "stop_id": "btie", "stop_lat": 0.0, "stop_lon": 0.007},
            # Far outside the radius and a stop without usable coordinates.
            {"location_type": "0", "stop_id": "bfar", "stop_lat": 0.0, "stop_lon": 0.02},
            {"location_type": "0", "stop_id": "bnan", "stop_lat": "", "stop_lon": ""},
        ]
    ).to_csv(snapshot / "stops.txt", index=False)
    pd.DataFrame(
        [
            {"trip_id": "t1", "route_id": "R1"},
            {"trip_id": "t2", "route_id": "R2"},
            {"trip_id": "t3", "route_id": "R3"},
            {"trip_id": "t4", "route_id": "R4"},
            {"trip_id": "t5", "route_id": "R5"},
            {"trip_id": "t6", "route_id": "R6"},
            {"trip_id": "t7", "route_id": "R7"},
        ]
    ).to_csv(snapshot / "bus_trips.csv", index=False)
    rows = [
        # t1: A -> B in 10 minutes, then an unmapped stop.
        {"trip_id": "t1", "stop_id": "b10", "stop_sequence": "1", "arrival_time": "07:00:00", "departure_time": "07:00:00"},
        {"trip_id": "t1", "stop_id": "b2", "stop_sequence": "2", "arrival_time": "07:10:00", "departure_time": "07:10:00"},
        {"trip_id": "t1", "stop_id": "bfar", "stop_sequence": "3", "arrival_time": "07:30:00", "departure_time": "07:30:00"},
        # t2: slower A -> B direction.
        {"trip_id": "t2", "stop_id": "b10", "stop_sequence": "1", "arrival_time": "06:00:00", "departure_time": "06:00:00"},
        {"trip_id": "t2", "stop_id": "b2", "stop_sequence": "2", "arrival_time": "06:25:00", "departure_time": "06:25:00"},
        # t3: the reverse direction of the same undirected pair.
        {"trip_id": "t3", "stop_id": "b2", "stop_sequence": "1", "arrival_time": "07:00:00", "departure_time": "07:00:00"},
        {"trip_id": "t3", "stop_id": "b10", "stop_sequence": "2", "arrival_time": "07:40:00", "departure_time": "07:40:00"},
        # t4: service-day hours past 24.
        {"trip_id": "t4", "stop_id": "b2", "stop_sequence": "1", "arrival_time": "25:00:00", "departure_time": "25:00:00"},
        {"trip_id": "t4", "stop_id": "b30", "stop_sequence": "2", "arrival_time": "25:30:00", "departure_time": "25:30:00"},
        # t5: a zero-minute difference must not become a candidate.
        {"trip_id": "t5", "stop_id": "b10", "stop_sequence": "1", "arrival_time": "07:00:00", "departure_time": "07:00:00"},
        {"trip_id": "t5", "stop_id": "b30", "stop_sequence": "2", "arrival_time": "07:00:00", "departure_time": "07:00:00"},
        # t6: an unusable stop is skipped but the pair still derives.
        {"trip_id": "t6", "stop_id": "b10", "stop_sequence": "1", "arrival_time": "08:00:00", "departure_time": "08:00:00"},
        {"trip_id": "t6", "stop_id": "bnan", "stop_sequence": "2", "arrival_time": "08:05:00", "departure_time": "08:05:00"},
        {"trip_id": "t6", "stop_id": "b30", "stop_sequence": "3", "arrival_time": "08:18:00", "departure_time": "08:18:00"},
        # t7: btie is equidistant from stations 20 and 40; sorted id 20 wins.
        {"trip_id": "t7", "stop_id": "btie", "stop_sequence": "1", "arrival_time": "09:00:00", "departure_time": "09:00:00"},
        {"trip_id": "t7", "stop_id": "b30", "stop_sequence": "2", "arrival_time": "09:10:00", "departure_time": "09:10:00"},
    ]
    pd.DataFrame(rows).to_csv(snapshot / "stop_times.txt", index=False)
    return snapshot, stations_csv


class TestDeriveGtfsCandidates:
    def test_mini_snapshot_pairs(self, tmp_path):
        snapshot, stations_csv = write_mini_snapshot(tmp_path)
        derived = derive_gtfs_candidates(snapshot, stations_csv=stations_csv)
        assert derived.columns.tolist() == [
            "station_a", "station_b", "minutes", "kind", "route_ids", "trip_count",
            "evidence_trip_id", "evidence_from_stop", "evidence_to_stop",
            "evidence_from_sequence", "evidence_to_sequence",
        ]
        assert list(zip(derived.station_a, derived.station_b)) == [
            ("10", "2"), ("10", "30"), ("2", "30"), ("20", "30"),
        ]
        assert derived.kind.eq("gtfs_candidate").all()
        by_pair = {(a, b): row for (a, b), row in zip(zip(derived.station_a, derived.station_b), derived.itertuples())}
        fastest = by_pair[("10", "2")]
        assert fastest.minutes == pytest.approx(10.0)
        assert fastest.route_ids == "R1;R2;R3"
        assert fastest.trip_count == 3
        assert fastest.evidence_trip_id == "t1"
        assert (fastest.evidence_from_stop, fastest.evidence_to_stop) == ("b10", "b2")
        assert by_pair[("10", "30")].minutes == pytest.approx(18.0)
        assert by_pair[("10", "30")].route_ids == "R6"
        assert by_pair[("2", "30")].minutes == pytest.approx(30.0)
        assert by_pair[("2", "30")].route_ids == "R4"
        # btie maps to the smaller sorted station id (20), never to 40.
        assert by_pair[("20", "30")].minutes == pytest.approx(10.0)
        assert ("30", "40") not in by_pair

    def test_mini_snapshot_is_deterministic(self, tmp_path):
        snapshot, stations_csv = write_mini_snapshot(tmp_path)
        first = derive_gtfs_candidates(snapshot, stations_csv=stations_csv)
        second = derive_gtfs_candidates(snapshot, stations_csv=stations_csv)
        pd.testing.assert_frame_equal(first, second)

    def test_missing_snapshot_tables_raise_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="stops.txt"):
            derive_gtfs_candidates(tmp_path, stations_csv=STATIONS_CSV)

    def test_radius_must_be_positive(self, tmp_path):
        snapshot, stations_csv = write_mini_snapshot(tmp_path)
        with pytest.raises(ValueError, match="radius_m"):
            derive_gtfs_candidates(snapshot, stations_csv=stations_csv, radius_m=0.0)

    def test_derived_table_feeds_the_layer(self, tmp_path):
        snapshot, stations_csv = write_mini_snapshot(tmp_path)
        derived = derive_gtfs_candidates(snapshot, stations_csv=stations_csv)
        rail = nx.Graph()
        rail.add_edges_from([("10", "2"), ("2", "30"), ("30", "20")])
        nx.set_edge_attributes(rail, 1000.0, "distance_m")
        layered = add_backup_edges(build_layers(rail), derived)
        assert layered["T:10"]["T:2"]["kind"] == "gtfs_candidate"
        assert layered["T:10"]["T:2"]["route_or_road"] == "R1;R2;R3"
        assert layered["T:10"]["T:2"]["minutes"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# real frozen inputs
# ---------------------------------------------------------------------------
class TestFrozenInputs:
    def test_loading_does_not_modify_inputs(self):
        paths = (BACKUP_CSV, STATIONS_CSV, RAIL_EDGES_CSV, TOPOLOGY_JSON)
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        rail = network_from_csv()
        rail_before = copy.deepcopy(rail)
        manual = pd.read_csv(BACKUP_CSV, dtype={"station_a": str, "station_b": str})
        manual_before = manual.copy(deep=True)
        layered = build_layers(rail)
        layered_before = copy.deepcopy(layered)
        transfer = json.loads(TOPOLOGY_JSON.read_text(encoding="utf-8"))["walking_transfer"]
        walked = add_walking_transfers(layered, transfer)
        standby = add_backup_edges(walked, manual)
        assert standby.number_of_edges() == walked.number_of_edges() + 4

        assert nx.utils.graphs_equal(rail, rail_before)
        assert nx.utils.graphs_equal(layered, layered_before)
        pd.testing.assert_frame_equal(manual, manual_before)
        after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
        assert after == before

    @pytest.mark.skipif(not SNAPSHOT_PRESENT, reason="the large frozen snapshot tables are git-ignored")
    def test_frozen_snapshot_candidates_and_manual_comparison(self):
        derived = derive_gtfs_candidates(SNAPSHOT_DIR)
        # Frozen-snapshot regression: 263 unique single-trip pairs in the window.
        assert len(derived) == 263
        assert derived.minutes.gt(0).all()
        keys = list(zip(derived.station_a, derived.station_b))
        assert keys == sorted(keys)
        assert len(keys) == len(set(keys))

        manual = pd.read_csv(BACKUP_CSV, dtype={"station_a": str, "station_b": str})
        check = compare_manual_pairs(manual, derived)
        assert len(check) == len(manual) == 4
        assert (check.station_a + "-" + check.station_b).tolist() == ["23-56", "48-56", "56-88", "73-83"]
        statuses = dict(zip(check.station_a + "-" + check.station_b, check.status))
        assert statuses == {
            "23-56": "time_differs",   # derived 17 min ride vs 38 min door-to-door
            "48-56": "not_derived",    # nearest bus stop is 415 m outside the 400 m rule
            "56-88": "not_derived",    # Perth Busport maps to station 64, not 56
            "73-83": "not_derived",    # the manual evidence is a two-bus transfer
        }
        derived_minutes = dict(zip(check.station_a + "-" + check.station_b, check.derived_minutes))
        assert derived_minutes["23-56"] == pytest.approx(17.0)
        assert check.manual_kind.eq("existing_bus").all()
        assert check.note.str.contains("manual ground truth").all()
        assert check.loc[check.status == "not_derived", "derived_minutes"].isna().all()

    def test_runner_loader_matches_the_frozen_counts(self):
        graph = load_rail_graph(STATIONS_CSV.parent)
        assert (graph.number_of_nodes(), graph.number_of_edges()) == (86, 85)
        assert nx.is_tree(graph)

    @pytest.mark.skipif(not SNAPSHOT_PRESENT, reason="the large frozen snapshot tables are git-ignored")
    def test_runner_writes_both_tables_with_input_hashes(self, tmp_path):
        outcome = run_backup_candidates(snapshot_dir=SNAPSHOT_DIR, output_dir=tmp_path / "multilayer")
        assert outcome["intact_served_od_fraction"] == pytest.approx(1.0)
        candidates = pd.read_csv(tmp_path / "multilayer" / "gtfs_backup_candidates.csv")
        check = pd.read_csv(tmp_path / "multilayer" / "manual_pair_check.csv")
        assert len(candidates) == 263
        assert len(check) == 4
        meta = json.loads((tmp_path / "multilayer" / "manual_pair_check.meta.json").read_text(encoding="utf-8"))
        assert meta["experiment"] == "multilayer"
        assert meta["params"]["derived_candidates"] == 263
        assert meta["params"]["walking_transfer_minutes"] == pytest.approx(WALKING_TRANSFER_MINUTES)
        assert any("stop_times.txt" in key for key in meta["inputs"])
        assert any("backup_edges.csv" in key for key in meta["inputs"])
        assert any("stations.csv" in key for key in meta["inputs"])