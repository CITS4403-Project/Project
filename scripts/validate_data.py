"""Independently validate snapshot references, topology, metrics and bus evidence."""

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from build_network import bus_coverage, edge_key, haversine_m, mapped_graph
from gtfs_common import ROOT, rail_parents, read_gtfs, sha256, verify_files, write_json
from prepare_gtfs_7_1 import active_services, time_seconds
from datetime import date


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(
    gtfs_dir,
    output_dir,
    topology_path=ROOT / "data/verified_topology.json",
    manual_dir=ROOT / "data/manual",
):
    gtfs_dir, output_dir = Path(gtfs_dir), Path(output_dir)
    snapshot = json.loads(
        (gtfs_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    # Standalone validation of the canonical directory also checks its reviewed
    # frozen hashes; staging directories are checked by run_data before publish.
    if output_dir.resolve() == (ROOT / "data/processed").resolve():
        expected = json.loads(
            (ROOT / "data/frozen_checksums.json").read_text(encoding="utf-8")
        )
        for relative, digest in expected.items():
            if (
                relative.startswith("processed/")
                and relative != "processed/validation.json"
            ):
                require(
                    sha256(ROOT / "data" / relative) == digest,
                    f"Frozen checksum mismatch: {relative}",
                )
    verify_files(gtfs_dir, snapshot["prepared_files"])
    verify_files(output_dir, manifest["files"])
    source = json.loads(
        (ROOT / "data/source_manifest.json").read_text(encoding="utf-8")
    )
    require(
        snapshot["download"] == source,
        "Snapshot source differs from frozen acquisition",
    )
    require(
        manifest["inputs"]["snapshot_manifest"]
        == sha256(gtfs_dir / "snapshot_manifest.json"),
        "Snapshot provenance mismatch",
    )
    require(
        manifest["inputs"]["source_manifest"]
        == sha256(ROOT / "data/source_manifest.json"),
        "Source provenance mismatch",
    )
    require(
        manifest["inputs"]["verified_topology"] == sha256(topology_path),
        "Map provenance mismatch",
    )
    trips, routes, stops = (
        read_gtfs(gtfs_dir / name) for name in ["trips.txt", "routes.txt", "stops.txt"]
    )
    times, window = (
        read_gtfs(gtfs_dir / "stop_times.txt"),
        read_gtfs(gtfs_dir / "window_departures.csv"),
    )
    for table, key in [(trips, "trip_id"), (routes, "route_id"), (stops, "stop_id")]:
        require(not table[key].duplicated().any(), f"Duplicate {key}")
    calendar = (
        read_gtfs(gtfs_dir / "calendar.txt")
        if (gtfs_dir / "calendar.txt").exists()
        else pd.DataFrame()
    )
    exceptions = (
        read_gtfs(gtfs_dir / "calendar_dates.txt")
        if (gtfs_dir / "calendar_dates.txt").exists()
        else pd.DataFrame()
    )
    active, _, _, _ = active_services(
        calendar, exceptions, date.fromisoformat(snapshot["service_date"])
    )
    require(set(trips.service_id) <= active, "Inactive service in snapshot")
    require(set(trips.route_id) <= set(routes.route_id), "Unknown trip route")
    require(
        set(times.trip_id) == set(trips.trip_id), "Stop-time trip references incomplete"
    )
    require(set(times.stop_id) <= set(stops.stop_id), "Unknown stop-time stop")
    require(
        not times[["trip_id", "stop_sequence"]].duplicated().any(),
        "Duplicate trip/sequence",
    )
    start, end = [
        int(time_seconds(pd.Series([snapshot[key]])).iloc[0])
        for key in ["window_start", "window_end_exclusive"]
    ]
    seconds = time_seconds(times.departure_time)
    time_seconds(times.arrival_time)
    expected_events = times[seconds.ge(start) & seconds.lt(end)].copy()
    keys = ["trip_id", "stop_sequence", "stop_id", "departure_time"]
    require(
        set(map(tuple, expected_events[keys].to_numpy()))
        == set(map(tuple, window[keys].to_numpy())),
        "Window events do not match full sequences",
    )
    require(len(expected_events) == len(window), "Duplicated window events")
    require(
        set(window.trip_id) == set(trips.trip_id),
        "Selected trip has no window departure",
    )
    require(
        (
            time_seconds(window.departure_time).to_numpy()
            == window.departure_seconds.astype(int).to_numpy()
        ).all(),
        "Window seconds disagree with departure time",
    )
    for _, group in times.groupby("trip_id"):
        require(
            pd.to_numeric(group.stop_sequence).is_monotonic_increasing,
            "Unordered stop sequence",
        )
        known = time_seconds(group.departure_time).dropna()
        require(known.is_monotonic_increasing, "Decreasing GTFS departure times")
    route_modes = routes.set_index("route_id").route_type.map({"2": "rail", "3": "bus"})
    trip_modes = trips.set_index("trip_id").route_id.map(route_modes)
    require(
        window.trip_id.map(trip_modes).eq(window["mode"]).all(),
        "Window mode disagrees with GTFS route",
    )
    if not snapshot["event_variant"]:
        require(
            "WES-RAI-4313" not in set(routes.route_id),
            "Stadium event route leaked into regular baseline",
        )
    rail_times = times[times.trip_id.isin(trip_modes[trip_modes.eq("rail")].index)]
    rail_parents(rail_times, stops)

    topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    graph = mapped_graph(topology)
    stations, edges, coverage = (
        read_gtfs(output_dir / name)
        for name in ["stations.csv", "rail_edges.csv", "bus_coverage.csv"]
    )
    require(not stations.station_id.duplicated().any(), "Duplicate processed station")
    require(
        set(stations.station_id) == set(graph), "Processed station set differs from map"
    )
    actual_pairs = [edge_key(a, b) for a, b in zip(edges.station_a, edges.station_b)]
    require(len(actual_pairs) == len(set(actual_pairs)), "Duplicate undirected edge")
    require(
        set(actual_pairs) == {edge_key(a, b) for a, b in graph.edges},
        "Processed rail edges differ from mapped adjacency",
    )
    require(nx.is_connected(graph), "Mapped rail graph disconnected")
    indexed = stations.set_index("station_id")
    require(
        all(int(indexed.loc[sid, "degree"]) == graph.degree(sid) for sid in graph),
        "Station degree stale",
    )
    centrality = nx.betweenness_centrality(graph, normalized=True)
    require(
        all(
            np.isclose(
                float(indexed.loc[sid, "betweenness"]),
                centrality[sid],
                rtol=1e-10,
                atol=1e-12,
            )
            for sid in graph
        ),
        "Station centrality stale",
    )
    for row in edges.itertuples(index=False):
        a, b = indexed.loc[row.station_a], indexed.loc[row.station_b]
        distance = haversine_m(float(a.lat), float(a.lon), float(b.lat), float(b.lon))
        require(
            np.isclose(float(row.distance_m), distance, rtol=1e-9, atol=0.001),
            "Edge distance mismatch",
        )
        require(int(row.trips) >= 0, "Negative traversal count")
    numeric_stations = stations.copy()
    numeric_stations[["lat", "lon"]] = numeric_stations[["lat", "lon"]].astype(float)
    expected_coverage = bus_coverage(
        numeric_stations, window[window["mode"].eq("bus")], trips, stops
    )
    require(
        coverage.to_dict("records") == expected_coverage.astype(str).to_dict("records"),
        "Bus coverage not generated from window events",
    )
    audit = read_gtfs(output_dir / "topology_audit.csv")
    require(
        len(audit) == 96 and not audit[["station_a", "station_b"]].duplicated().any(),
        "Legacy audit must cover all 96 edges once",
    )
    require(
        audit.status.isin(
            [
                "verified_adjacent",
                "expand_skipped_stations",
                "exclude_off_map_connection",
            ]
        ).all(),
        "Unreviewed legacy edge",
    )
    backup = read_gtfs(output_dir / "backup_edges.csv")
    evidence = json.loads(
        (output_dir / "backup_edges_google_maps_evidence.json").read_text(
            encoding="utf-8"
        )
    )
    for name in manifest["manual_provenance"]:
        require(
            sha256(output_dir / name)
            == sha256(Path(manual_dir) / name)
            == manifest["manual_provenance"][name],
            "Manual bus provenance changed",
        )
    for row in backup.itertuples(index=False):
        require(
            row.station_a in graph
            and row.station_b in graph
            and row.station_a != row.station_b,
            "Invalid backup endpoints",
        )
        directions = [
            item
            for item in evidence["directions"]
            if {item["from"], item["to"]} == {row.station_a, row.station_b}
        ]
        require(
            len(directions) == 2
            and {(item["from"], item["to"]) for item in directions}
            == {(row.station_a, row.station_b), (row.station_b, row.station_a)},
            "Missing bidirectional Maps evidence",
        )
        require(
            float(row.minutes) == max(item["effective_minutes"] for item in directions),
            "Backup minutes differ from Maps evidence",
        )
    checks = {
        "status": "passed",
        "source_sha256": "passed",
        "snapshot_sha256": "passed",
        "processed_sha256": "passed",
        "calendar_exceptions_and_foreign_keys": "passed",
        "rail_parents": "passed",
        "service_day_times_and_half_open_window": "passed",
        "mapped_station_adjacency_and_metrics": "passed",
        "window_bus_coverage": "passed",
        "legacy_edges_audited": len(audit),
        "manual_backup_evidence": "passed",
        "stations": len(stations),
        "rail_edges": len(edges),
    }
    write_json(checks, output_dir / "validation.json")
    print(
        f"Validation passed: {len(stations)} stations, {len(edges)} edges, {len(audit)} audited legacy edges"
    )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtfs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topology", type=Path, default=ROOT / "data/verified_topology.json"
    )
    parser.add_argument("--manual-dir", type=Path, default=ROOT / "data/manual")
    args = parser.parse_args()
    validate(args.gtfs_dir, args.output_dir, args.topology, args.manual_dir)


if __name__ == "__main__":
    main()
