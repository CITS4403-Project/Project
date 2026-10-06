"""Build the map-verified rail graph and dated 400 m bus coverage."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil

import networkx as nx
import numpy as np
import pandas as pd

from gtfs_common import (
    ROOT,
    EARTH_RADIUS_M,
    file_records,
    rail_parents,
    read_gtfs,
    sha256,
    verify_files,
    write_csv,
    write_json,
)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def edge_key(a, b):
    return tuple(sorted((a, b)))


def mapped_graph(topology):
    graph = nx.Graph()
    for line, sequence in topology["lines"].items():
        if len(sequence) != len(set(sequence)):
            raise ValueError(f"Repeated station in mapped line: {line}")
        graph.add_nodes_from(sequence)
        graph.add_edges_from(zip(sequence, sequence[1:]))
    return graph


def expand_pair(a, b, graph, topology):
    """Expand stop-skipping trips through mapped stations, not shortcut edges."""
    if a == b:
        return [a]
    if edge_key(a, b) in {edge_key(*pair) for pair in topology["off_map_legacy_pairs"]}:
        raise ValueError(f"Service pair {a}-{b} is not represented on the regular map")
    if a not in graph or b not in graph:
        raise ValueError(f"Station outside verified topology: {a}-{b}")
    # Prefer the actual mapped corridor over a cross-line shortest path.
    candidates = []
    for sequence in topology["lines"].values():
        if a in sequence and b in sequence:
            i, j = sequence.index(a), sequence.index(b)
            path = sequence[min(i, j) : max(i, j) + 1]
            candidates.append(path if i < j else path[::-1])
    if candidates:
        return min(candidates, key=lambda path: (len(path), path))
    # The verified graph is a tree; cross-corridor paths are unique.
    if not nx.is_tree(graph):
        raise ValueError(f"Ambiguous cross-corridor service pair: {a}-{b}")
    return nx.shortest_path(graph, a, b)


def audit_legacy(legacy_edges, graph, topology, names):
    rows = []
    for row in legacy_edges.itertuples(index=False):
        a, b = row.station_a, row.station_b
        if graph.has_edge(a, b):
            status, path = "verified_adjacent", [a, b]
            note = "Retained as mapped station adjacency; service counts regenerated for the window."
        else:
            try:
                path = expand_pair(a, b, graph, topology)
                status = "expand_skipped_stations"
                note = "Consecutive stopping points are not adjacent physical stations; replace shortcut by mapped path."
            except ValueError:
                status, path = "exclude_off_map_connection", []
                note = "Not drawn as a regular rail connection on the official map; excluded, not assumed physically impossible."
        rows.append(
            {
                "station_a": a,
                "station_b": b,
                "name_a": names[a],
                "name_b": names[b],
                "status": status,
                "replacement_station_ids": ";".join(path),
                "note": note,
            }
        )
    return pd.DataFrame(rows).sort_values(["station_a", "station_b"])


def bus_coverage(stations, bus_events, trips, stops, radius_m=400.0):
    route_lookup = trips.set_index("trip_id").route_id
    events = bus_events.copy()
    events["route_id"] = events.trip_id.map(route_lookup)
    if events.route_id.isna().any():
        raise ValueError("Bus event trip not present in trips.txt")
    stop_routes = events.groupby("stop_id").route_id.agg(
        lambda values: sorted(set(values))
    )
    bus_stops = stops[
        stops.stop_id.isin(stop_routes.index) & stops.location_type.eq("0")
    ].copy()
    bus_stops["routes"] = bus_stops.stop_id.map(stop_routes)
    rows = []
    for station in stations.itertuples(index=False):
        # Great-circle distance avoids the old degrees-based longitude distortion.
        distances = haversine_m(
            station.lat,
            station.lon,
            bus_stops.stop_lat.astype(float).to_numpy(),
            bus_stops.stop_lon.astype(float).to_numpy(),
        )
        near = bus_stops.loc[distances <= radius_m]
        routes = set(route for route_list in near.routes for route in route_list)
        rows.append(
            {
                "station_id": station.station_id,
                "name": station.name,
                "n_bus_stops": len(near),
                "n_bus_routes": len(routes),
                "bus_route_ids": ";".join(sorted(routes)),
            }
        )
    return pd.DataFrame(rows).sort_values("station_id")


def build(gtfs_dir, output_dir, topology_path, manual_dir, baseline_dir):
    gtfs_dir, output_dir = Path(gtfs_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            "Build into an empty output directory; make data handles repeat runs"
        )
    snapshot = json.loads(
        (gtfs_dir / "snapshot_manifest.json").read_text(encoding="utf-8")
    )
    verify_files(gtfs_dir, snapshot["prepared_files"])
    topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    graph = mapped_graph(topology)
    stops, trips = read_gtfs(gtfs_dir / "stops.txt"), read_gtfs(gtfs_dir / "trips.txt")
    rail_trips = read_gtfs(gtfs_dir / "rail_trips.csv")
    times = read_gtfs(gtfs_dir / "stop_times.txt")
    rail_times = times[times.trip_id.isin(rail_trips.trip_id)].copy()
    rail_times["station"] = rail_parents(rail_times, stops)
    rail_times["stop_sequence"] = rail_times.stop_sequence.astype(int)
    rail_times = rail_times.sort_values(["trip_id", "stop_sequence"])
    indexed = stops.set_index("stop_id")
    if (
        not set(graph) <= set(indexed.index)
        or not indexed.loc[list(graph), "location_type"].eq("1").all()
    ):
        raise ValueError("Verified topology station absent or not a GTFS station")

    traversals, unmapped = Counter(), []
    for trip_id, group in rail_times.groupby("trip_id", sort=True):
        station_ids = group.station.to_list()
        trip_edges = []
        try:
            for a, b in zip(station_ids, station_ids[1:]):
                path = expand_pair(a, b, graph, topology)
                trip_edges.extend(edge_key(x, y) for x, y in zip(path, path[1:]))
        except ValueError as error:
            if not snapshot["event_variant"]:
                raise ValueError(f"Unmapped regular trip {trip_id}: {error}") from error
            unmapped.append({"trip_id": trip_id, "reason": str(error)})
            continue
        traversals.update(trip_edges)

    window = read_gtfs(gtfs_dir / "window_departures.csv")
    rail_window = window[window["mode"].eq("rail")].copy()
    rail_window["station"] = rail_parents(rail_window, stops)
    frequency = rail_times.groupby("station").size()
    peak_frequency = rail_window.groupby("station").size()
    centrality = nx.betweenness_centrality(graph, normalized=True)
    station_rows = []
    for sid in sorted(graph):
        stop = indexed.loc[sid]
        station_rows.append(
            {
                "station_id": sid,
                "name": stop.stop_name,
                "lat": float(stop.stop_lat),
                "lon": float(stop.stop_lon),
                "modes": stop.supported_modes,
                "degree": graph.degree(sid),
                "trips_served": int(frequency.get(sid, 0)),
                "am_peak_stops": int(peak_frequency.get(sid, 0)),
                "betweenness": centrality[sid],
                "lines": ";".join(
                    sorted(
                        line for line, seq in topology["lines"].items() if sid in seq
                    )
                ),
            }
        )
    stations = pd.DataFrame(station_rows)
    edges = pd.DataFrame(
        [
            {
                "station_a": a,
                "station_b": b,
                "trips": traversals[(a, b)],
                "distance_m": float(
                    haversine_m(
                        float(indexed.loc[a, "stop_lat"]),
                        float(indexed.loc[a, "stop_lon"]),
                        float(indexed.loc[b, "stop_lat"]),
                        float(indexed.loc[b, "stop_lon"]),
                    )
                ),
            }
            for a, b in sorted(edge_key(a, b) for a, b in graph.edges)
        ]
    )
    coverage = bus_coverage(stations, window[window["mode"].eq("bus")], trips, stops)
    legacy = read_gtfs(Path(baseline_dir) / "rail_edges.csv")
    names = indexed.stop_name.to_dict()
    audit = audit_legacy(legacy, graph, topology, names)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("stations.csv", stations),
        ("rail_edges.csv", edges),
        ("bus_coverage.csv", coverage),
        ("topology_audit.csv", audit),
        (
            "unmapped_event_trips.csv",
            pd.DataFrame(unmapped, columns=["trip_id", "reason"]),
        ),
    ]:
        write_csv(table, output_dir / name)
    for name in ["backup_edges.csv", "backup_edges_google_maps_evidence.json"]:
        shutil.copyfile(Path(manual_dir) / name, output_dir / name)
    inputs = {
        "source_manifest": sha256(ROOT / "data/source_manifest.json"),
        "verified_topology": sha256(topology_path),
        "snapshot_manifest": sha256(gtfs_dir / "snapshot_manifest.json"),
        "legacy_stations": sha256(Path(baseline_dir) / "stations.csv"),
        "legacy_rail_edges": sha256(Path(baseline_dir) / "rail_edges.csv"),
    }
    manifest = {
        "dataset": f"transperth-{snapshot['service_date']}-{snapshot['window_start'][:5].replace(':', '')}-{snapshot['window_end_exclusive'][:5].replace(':', '')}-map-v1"
        + ("-events" if snapshot["event_variant"] else ""),
        "service_date": snapshot["service_date"],
        "window_start": snapshot["window_start"],
        "window_end_exclusive": snapshot["window_end_exclusive"],
        "timezone": snapshot["timezone"],
        "event_variant": snapshot["event_variant"],
        "stations": len(stations),
        "rail_edges": len(edges),
        "rail_trips": len(rail_trips),
        "bus_trips": snapshot["bus_trips"],
        "bus_radius_m": 400,
        "topology_source": topology["source_url"],
        "topology_source_sha256": topology["source_sha256"],
        "inputs": inputs,
        "legacy_edge_audit_counts": audit.status.value_counts().sort_index().to_dict(),
        "unmapped_event_trips": len(unmapped),
        "manual_provenance": {
            name: sha256(Path(manual_dir) / name)
            for name in ["backup_edges.csv", "backup_edges_google_maps_evidence.json"]
        },
        "files": file_records(output_dir.iterdir()),
    }
    write_json(manifest, output_dir / "manifest.json")
    print(
        f"Built {len(stations)} mapped stations, {len(edges)} rail edges; audited {len(audit)} legacy edges"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gtfs-dir", type=Path, required=True, help="Prepared service-day snapshot"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topology", type=Path, default=ROOT / "data/verified_topology.json"
    )
    parser.add_argument("--manual-dir", type=Path, default=ROOT / "data/manual")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=ROOT / "investigations/idea2-perth-transport/data/processed",
    )
    args = parser.parse_args()
    build(
        args.gtfs_dir,
        args.output_dir,
        args.topology,
        args.manual_dir,
        args.baseline_dir,
    )


if __name__ == "__main__":
    main()
