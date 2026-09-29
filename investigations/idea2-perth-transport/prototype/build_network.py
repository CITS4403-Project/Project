"""Build Perth rail + bus network from the Transperth static GTFS feed.

Outputs (written to evidence/ and data/processed/):
  - stations.csv              one row per physical rail station with graph metrics
  - rail_edges.csv            undirected aggregated edges (trips traversing)
  - bus_coverage.csv          bus stops / routes within 400 m of each rail station
  - build_report.txt          summary counts

Run:  python prototype/build_network.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
GTFS = ROOT / "data" / "raw" / "google_transit"
OUT = ROOT / "evidence"
PROC = ROOT / "data" / "processed"
OUT.mkdir(exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

PERTH_BBOX = dict(lat_min=-32.80, lat_max=-31.30, lon_min=115.30, lon_max=116.40)
RADIUS_M = 400.0


def read_gtfs(name: str, **kw) -> pd.DataFrame:
    """Read a GTFS table, normalising the feed's stray whitespace."""
    df = pd.read_csv(GTFS / name, dtype=str, **kw)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def haversine_m(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in metres."""
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def in_perth(df: pd.DataFrame) -> pd.Series:
    lat, lon = df["stop_lat"].astype(float), df["stop_lon"].astype(float)
    return (lat.between(PERTH_BBOX["lat_min"], PERTH_BBOX["lat_max"])
            & lon.between(PERTH_BBOX["lon_min"], PERTH_BBOX["lon_max"]))


def main() -> None:
    routes = read_gtfs("routes.txt")
    trips = read_gtfs("trips.txt")
    stops = read_gtfs("stops.txt")
    st = read_gtfs("stop_times.txt",
                   usecols=lambda c: c.strip() in ("trip_id", "stop_id", "stop_sequence", "arrival_time"))
    stops["stop_lat"] = stops["stop_lat"].astype(float)
    stops["stop_lon"] = stops["stop_lon"].astype(float)

    report = []

    # ---------- rail network (Perth suburban lines only) ----------
    # The Stadium Special (route WES-RAI-4313, 17 event-only trips) is excluded
    # from the regular baseline: it links Perth Stadium directly to Thornlie,
    # Leederville and City West, creating shortcuts that do not exist on a
    # normal weekday. It is kept as an optional 'event day' variant.
    EVENT_ROUTES = {"WES-RAI-4313"}
    rail_routes = routes[routes["route_id"].str.startswith("WES-RAI")
                         & (routes["route_type"] == "2")
                         & ~routes["route_id"].isin(EVENT_ROUTES)]
    rail_trips = trips[trips["route_id"].isin(rail_routes["route_id"])]
    tripid2route = dict(zip(rail_trips["trip_id"], rail_trips["route_id"]))
    rst = st[st["trip_id"].isin(tripid2route)].copy()
    rst["route_id"] = rst["trip_id"].map(tripid2route)
    rst["stop_sequence"] = rst["stop_sequence"].astype(int)
    rst = rst.sort_values(["trip_id", "stop_sequence"])

    stop2parent = dict(zip(stops["stop_id"], stops["parent_station"]))
    rst["station"] = rst["stop_id"].map(stop2parent)
    assert rst["station"].notna().all(), "rail platform without parent_station"

    # edges: consecutive station pairs per trip, counted
    edges: dict[tuple[str, str], int] = {}
    for _, g in rst.groupby("trip_id"):
        seq = g["station"].to_numpy()
        for a, b in zip(seq[:-1], seq[1:]):
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edges[key] = edges.get(key, 0) + 1

    stations = stops[stops["stop_id"].isin(set(rst["station"]))].copy()
    stations = stations[["stop_id", "stop_name", "stop_lat", "stop_lon", "supported_modes"]]
    stations = stations.rename(columns={"stop_id": "station_id", "stop_name": "name"})

    G = nx.Graph()
    for _, r in stations.iterrows():
        G.add_node(r["station_id"], name=r["name"], lat=r["stop_lat"], lon=r["stop_lon"],
                   modes=r["supported_modes"])
    for (a, b), w in edges.items():
        G.add_edge(a, b, trips=w, distance_m=float(haversine_m(
            G.nodes[a]["lat"], G.nodes[a]["lon"], G.nodes[b]["lat"], G.nodes[b]["lon"])))

    report.append(f"Perth suburban rail routes: {len(rail_routes)} ({', '.join(rail_routes['route_long_name'])})")
    report.append(f"rail stations: {G.number_of_nodes()}, edges: {G.number_of_edges()}, "
                  f"components: {nx.number_connected_components(G)}")
    report.append(f"total rail trips: {rail_trips['trip_id'].nunique():,}, "
                  f"total edge-traversals: {sum(edges.values()):,}")

    # station-to-line mapping (lines are routes; a station may serve several)
    line_stations = rst.groupby("station")["route_id"].apply(lambda s: ";".join(sorted(set(s))))
    line_names = dict(zip(rail_routes["route_id"], rail_routes["route_long_name"]))
    station_lines = line_stations.map(lambda s: ";".join(line_names[r] for r in s.split(";")))

    # service frequency per station: number of trip stops at the station
    freq = rst.groupby("station").size()
    # boardings proxy: departures (stop_sequence < max) in AM peak 06:00-09:00
    rst["arr_h"] = rst["arrival_time"].str.slice(0, 2).astype(int)
    am = rst[(rst["arr_h"] >= 6) & (rst["arr_h"] < 9)]
    am_freq = am.groupby("station").size()

    # graph metrics
    btw = nx.betweenness_centrality(G)
    deg = dict(G.degree())
    for sid in G.nodes:
        G.nodes[sid]["degree"] = deg[sid]
        G.nodes[sid]["trips_served"] = int(freq.get(sid, 0))
        G.nodes[sid]["am_peak_stops"] = int(am_freq.get(sid, 0))
        G.nodes[sid]["betweenness"] = btw[sid]
        G.nodes[sid]["lines"] = station_lines.get(sid, "")

    # ---------- bus layer ----------
    bus_routes = routes[routes["route_type"] == "3"]
    bus_trips = trips[trips["route_id"].isin(bus_routes["route_id"])]
    bus_stops = stops[(stops["location_type"] == "0") & in_perth(stops)].copy()
    bus_stops = bus_stops[bus_stops["supported_modes"].str.contains("Bus")]
    # which routes visit each bus stop
    bst = st[st["trip_id"].isin(set(bus_trips["trip_id"]))].copy()
    bst["route_id"] = bst["trip_id"].map(dict(zip(bus_trips["trip_id"], bus_trips["route_id"])))
    stop_routes = bst.groupby("stop_id")["route_id"].apply(lambda s: sorted(set(s)))
    bus_stops["route_ids"] = bus_stops["stop_id"].map(stop_routes)
    bus_stops = bus_stops[bus_stops["route_ids"].notna()]
    report.append(f"bus routes: {len(bus_routes):,}, bus trips: {bus_trips['trip_id'].nunique():,}")
    report.append(f"bus stops in Perth bbox used by trips: {len(bus_stops):,}")

    tree = cKDTree(bus_stops[["stop_lat", "stop_lon"]].to_numpy())
    rows = []
    rail_xy = stations[["stop_lat", "stop_lon"]].to_numpy()
    for i, (_, s) in enumerate(stations.iterrows()):
        idx = tree.query_ball_point(rail_xy[i], r=RADIUS_M / 6371000.0 * 180 / np.pi)
        near = bus_stops.iloc[idx]
        rts = set()
        for rl in near["route_ids"]:
            rts.update(rl)
        rows.append(dict(station_id=s["station_id"], name=s["name"],
                         n_bus_stops=int(len(near)), n_bus_routes=int(len(rts)),
                         bus_route_ids=";".join(sorted(rts))[:2000]))
    coverage = pd.DataFrame(rows)
    coverage.to_csv(PROC / "bus_coverage.csv", index=False)

    n_with = (coverage["n_bus_stops"] > 0).mean()
    n_10 = (coverage["n_bus_routes"] >= 10).mean()
    report.append(f"rail stations with >=1 bus stop within {RADIUS_M:.0f} m: {n_with:.1%} "
                  f"({(coverage['n_bus_stops'] > 0).sum()}/{len(coverage)})")
    report.append(f"rail stations with >=10 bus routes within {RADIUS_M:.0f} m: {n_10:.1%}")

    # island stations (no bus stops nearby) - policy-relevant
    islands = coverage[coverage["n_bus_stops"] == 0]
    if len(islands):
        report.append("stations with NO bus stop within 400 m: " +
                      ", ".join(islands["name"].tolist()))

    # ---------- write outputs ----------
    node_df = pd.DataFrame([dict(sid=sid, **G.nodes[sid]) for sid in G.nodes])
    node_df = node_df.rename(columns={"sid": "station_id"})
    node_df = node_df.sort_values("betweenness", ascending=False)
    node_df.to_csv(PROC / "stations.csv", index=False)
    edf = pd.DataFrame([dict(station_a=a, station_b=b, **d) for a, b, d in G.edges(data=True)])
    edf.to_csv(PROC / "rail_edges.csv", index=False)

    report.append("\nTop-10 stations by betweenness:")
    for _, r in node_df.head(10).iterrows():
        report.append(f"  {r['name']:28s} btw={r['betweenness']:.3f} degree={r['degree']} "
                      f"trips={r['trips_served']} lines={r['lines']}")
    report.append("\nTop-10 by trips served:")
    for _, r in node_df.sort_values("trips_served", ascending=False).head(10).iterrows():
        report.append(f"  {r['name']:28s} trips={r['trips_served']} degree={r['degree']}")

    txt = "\n".join(report)
    (OUT / "build_report.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()