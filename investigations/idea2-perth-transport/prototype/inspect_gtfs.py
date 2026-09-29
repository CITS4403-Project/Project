"""Inspect the Transperth static GTFS feed.

Prints a table inventory and key counts, and writes small samples to evidence/.
Run:  python prototype/inspect_gtfs.py
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GTFS = ROOT / "data" / "raw" / "google_transit"
EVID = ROOT / "evidence"
EVID.mkdir(exist_ok=True)

ROUTE_TYPES = {"0": "tram", "1": "subway", "2": "rail", "3": "bus",
               "4": "ferry", "5": "cable tram", "6": "aerial lift",
               "7": "funicular", "11": "trolleybus", "12": "monorail"}


def read(name: str) -> pd.DataFrame:
    """Read a GTFS table, normalising the nonstandard whitespace in this feed."""
    df = pd.read_csv(GTFS / name, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df


def main() -> None:
    routes = read("routes.txt")
    stops = read("stops.txt")
    trips = read("trips.txt")
    agency = read("agency.txt")
    cal = read("calendar.txt")
    caldates = read("calendar_dates.txt")
    transfers = read("transfers.txt")

    print("=== FEED INVENTORY ===")
    for name, df in [("agency", agency), ("routes", routes), ("trips", trips),
                     ("stops", stops), ("calendar", cal),
                     ("calendar_dates", caldates), ("transfers", transfers)]:
        print(f"{name:16s} rows={len(df):>8,d} cols={list(df.columns)}")

    print("\n=== AGENCIES ===")
    print(agency[["agency_id", "agency_name"]].to_string(index=False))

    print("\n=== ROUTE TYPES ===")
    routes["mode"] = routes["route_type"].map(ROUTE_TYPES)
    print(routes.groupby(["route_type", "mode"]).size().to_string())

    print("\n=== AGENCY x MODE route counts ===")
    print(pd.crosstab(routes["agency_id"], routes["mode"]).to_string())

    rail = routes[routes["route_type"] == "2"]
    print(f"\n=== RAIL ROUTES ({len(rail)}) ===")
    cols = ["route_id", "agency_id", "route_short_name", "route_long_name", "route_desc"]
    print(rail[cols].to_string(index=False))

    rail_trips = trips[trips["route_id"].isin(rail["route_id"])]
    print(f"\nrail trips: {len(rail_trips):,}")

    print("\n=== STOPS ===")
    print("location_type counts:")
    print(stops["location_type"].value_counts(dropna=False).to_string())
    print("supported_modes counts:")
    print(stops["supported_modes"].value_counts(dropna=False).head(20).to_string())
    n_parent = stops.loc[stops["location_type"] == "1", "stop_id"].nunique()
    n_child = stops.loc[stops["location_type"] == "0", "stop_id"].nunique()
    print(f"stations (location_type=1): {n_parent:,}")
    print(f"stops/platforms (location_type=0): {n_child:,}")
    has_latlon = stops[["stop_lat", "stop_lon"]].notna().all(axis=1).mean()
    print(f"fraction with lat/lon: {has_latlon:.4f}")

    # Which stops are served by rail trips?
    st = pd.read_csv(GTFS / "stop_times.txt", dtype=str,
                     usecols=lambda c: c.strip() in ("trip_id", "stop_id", "stop_sequence"))
    st.columns = [c.strip() for c in st.columns]
    rail_st = st[st["trip_id"].isin(rail_trips["trip_id"])]
    rail_stop_ids = set(rail_st["stop_id"])
    rail_stops = stops[stops["stop_id"].isin(rail_stop_ids)]
    print(f"\nstops appearing in rail stop_times: {len(rail_stops):,}")
    print("their location_type counts:")
    print(rail_stops["location_type"].value_counts(dropna=False).to_string())
    print("their supported_modes counts:")
    print(rail_stops["supported_modes"].value_counts(dropna=False).to_string())

    # Unique physical rail stations via parent_station
    parents = rail_stops["parent_station"].fillna("")
    n_station = parents[parents != ""].nunique() + (parents == "").sum()
    print(f"unique physical rail stations (parent_station, or stop itself if none): {n_station:,}")

    # If location_type=1 stations exist among rail stops, print their names
    rail_parent_ids = set(parents[parents != ""])
    named = stops[stops["stop_id"].isin(rail_parent_ids)][["stop_id", "stop_name", "stop_lat", "stop_lon"]]
    named = named.sort_values("stop_name")
    print(f"\nrail parent stations ({len(named)}):")
    print(named.to_string(index=False))
    named.to_csv(EVID / "rail_stations.csv", index=False)

    # Samples
    routes.head(30).to_csv(EVID / "sample_routes.csv", index=False)
    stops.head(30).to_csv(EVID / "sample_stops.csv", index=False)
    trips.head(30).to_csv(EVID / "sample_trips.csv", index=False)
    st.head(30).to_csv(EVID / "sample_stop_times.csv", index=False)
    cal.head(10).to_csv(EVID / "sample_calendar.csv", index=False)
    transfers.head(20).to_csv(EVID / "sample_transfers.csv", index=False)

    # Bus vs rail basic counts
    bus_routes = routes[routes["route_type"] == "3"]
    bus_trips = trips[trips["route_id"].isin(bus_routes["route_id"])]
    bus_st = st[st["trip_id"].isin(bus_trips["trip_id"])]
    print(f"\nbus routes: {len(bus_routes):,}, bus trips: {len(bus_trips):,}, "
          f"bus stop_times rows: {len(bus_st):,}")
    print(f"bus stops used: {bus_st['stop_id'].nunique():,}")

    # Feed period
    print("\n=== FEED PERIOD ===")
    print("calendar dates:", cal["start_date"].min(), "->", cal["end_date"].max())

    # stop_times file size
    for f in ["stop_times.txt", "shapes.txt", "trips.txt", "stops.txt"]:
        p = GTFS / f
        print(f"{f}: {p.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()