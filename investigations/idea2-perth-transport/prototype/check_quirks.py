"""Quick checks: feed quirks before building the network."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GTFS = ROOT / "data" / "raw" / "google_transit"

def read(name, **kw):
    df = pd.read_csv(GTFS / name, dtype=str, **kw)
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        df[c] = df[c].str.strip()
    return df

routes = read("routes.txt")
trips = read("trips.txt")
stops = read("stops.txt")
st = read("stop_times.txt", usecols=lambda c: c.strip() in ("trip_id", "stop_id", "stop_sequence"))

perth_rail = sorted(routes.loc[routes.route_id.str.startswith("WES-RAI"), "route_id"])
transwa = sorted(routes.loc[routes.agency_id == "TRA-TRA", "route_id"])
print("Perth suburban rail route_ids:", perth_rail)
print("Transwa route_ids:", transwa)

# Which stops used by Perth suburban rail have supported_modes Bus?
rail_trips = trips[trips.route_id.isin(perth_rail)]
rst = st[st.trip_id.isin(set(rail_trips.trip_id))]
rstops = stops[stops.stop_id.isin(set(rst.stop_id))]
print(f"\nPerth rail stop_times rows: {len(rst):,}; distinct platform stops: {len(rstops):,}")
print(rstops.supported_modes.value_counts().to_string())
odd = rstops[rstops.supported_modes != "Rail"]
print("\nNon-'Rail' stops in Perth rail stop_times:")
print(odd[["stop_id","stop_name","supported_modes","parent_station","location_type"]].to_string(index=False))

# Do all Perth rail platforms have parent_station?
print("\nPerth rail platform parent coverage:", (rstops.parent_station != "").mean())

# parent stations of Perth rail
parents = sorted(set(rstops.parent_station[rstops.parent_station != ""]))
pstops = stops[stops.stop_id.isin(parents)]
print(f"\nPerth rail parent stations: {len(parents)}")
print("shared with Bus-supported? ", (pstops.supported_modes != "Rail").sum(), "parents of other kinds")
print(pstops.supported_modes.value_counts().to_string())

# stations per line
tt = rail_trips.merge(routes[["route_id","route_long_name"]], on="route_id")
tt = tt.merge(rst.groupby(["trip_id","stop_id"]).stop_sequence.min().reset_index(), on="trip_id")
print("\nStops per route (distinct platforms in stop_times):")
for rid, g in tt.groupby("route_long_name"):
    print(f"  {rid:28s} trips={g.trip_id.nunique():5d} platforms={g.stop_id.nunique():3d}")

# location_type=1 stations with Rail
l1 = stops[(stops.location_type == "1")]
print("\nlocation_type=1 stations:", len(l1), "| supported_modes:", l1.supported_modes.value_counts().to_dict())
print("how many have iptiscode populated:", (l1.iptiscode != "").mean().round(3))

# sg2tpn
sg = read("sg2tpn.txt")
print("\nsg2tpn.txt:")
print(sg.head().to_string(index=False))
print("cols:", list(sg.columns), "rows:", len(sg))

# transfers sample types
tr = read("transfers.txt")
print("\ntransfers transfer_type counts:", tr.transfer_type.value_counts().to_dict())