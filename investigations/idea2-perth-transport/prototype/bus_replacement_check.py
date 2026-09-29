"""Bus-replacement feasibility check for rail stations and corridors.

Quantifies, from the real GTFS feed:
  * bus stop / bus route coverage within 400 m of every rail station;
  * which rail stations have no bus alternative within walking distance;
  * parallel bus corridors: for each rail line, the fraction of its stations
    served by bus routes that also pass another station of the same line
    (a route is a 'parallel' candidate if it touches >=2 stations of the line);
  * a rough replacement travel-time proxy: for stations on a line, the distance
    to the nearest *other* rail station plus the nearest bus-stop distance.

Outputs: data/processed/bus_replacement.csv, per-line summary CSVs and
figures/fig4_bus_coverage.png

Run:  python prototype/bus_replacement_check.py
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
RADIUS_M = 400.0


def main() -> None:
    stations = pd.read_csv(PROC / "stations.csv", dtype={"station_id": str})
    coverage = pd.read_csv(PROC / "bus_coverage.csv", dtype={"station_id": str})
    edges = pd.read_csv(PROC / "rail_edges.csv", dtype={"station_a": str, "station_b": str})

    df = stations.merge(coverage, on=["station_id", "name"], how="left")
    df["n_bus_stops"] = df["n_bus_stops"].fillna(0).astype(int)
    df["n_bus_routes"] = df["n_bus_routes"].fillna(0).astype(int)

    # ---- coverage summary ----
    summary = []
    for k in (0, 1, 5, 10, 20):
        summary.append(dict(threshold=f">={k} bus routes within {RADIUS_M:.0f} m",
                            n_stations=int((df.n_bus_routes >= k).sum()),
                            fraction=float((df.n_bus_routes >= k).mean())))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(PROC / "bus_coverage_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    # ---- parallel bus corridors per rail line ----
    # A bus route is counted as 'parallel' to a rail line if it stops within
    # 400 m of at least two stations of that line.
    line_rows = []
    for _, r in df.iterrows():
        for ln in str(r["lines"]).split(";"):
            line_rows.append(dict(line=ln, station_id=r["station_id"], name=r["name"],
                                  routes=set(str(r["bus_route_ids"]).split(";")) if r["n_bus_routes"] else set()))
    ld = pd.DataFrame(line_rows)

    corridor_rows = []
    for ln, g in ld.groupby("line"):
        route_hits: dict[str, set[str]] = {}
        for _, r in g.iterrows():
            for rt in r["routes"]:
                if rt:
                    route_hits.setdefault(rt, set()).add(r["station_id"])
        parallel = {rt: ss for rt, ss in route_hits.items() if len(ss) >= 2}
        # stations of the line touched by at least one parallel bus route
        served = set().union(*parallel.values()) if parallel else set()
        corridor_rows.append(dict(
            line=ln, n_stations=len(g),
            n_parallel_bus_routes=len(parallel),
            frac_stations_on_parallel_corridor=len(served) / len(g),
            max_stations_one_bus_route_on_line=max((len(s) for s in parallel.values()), default=0)))
    corridor_df = pd.DataFrame(corridor_rows).sort_values("n_stations", ascending=False)
    corridor_df.to_csv(PROC / "bus_parallel_corridors.csv", index=False)
    print("\nParallel bus corridors per line:")
    print(corridor_df.to_string(index=False))

    # ---- rough replacement-time proxy ----
    # extra distance to reach any other rail station (interchange) if one
    # station is closed: straight-line nearest-neighbour distance.
    xy = df[["station_id", "name", "lat", "lon"]].copy()
    D = np.sqrt(((xy.lat.to_numpy()[:, None] - xy.lat.to_numpy()[None, :]) * 111.2) ** 2 +
                ((xy.lon.to_numpy()[:, None] - xy.lon.to_numpy()[None, :]) * 95.0) ** 2)
    np.fill_diagonal(D, np.inf)
    df["nearest_station_km"] = D.min(axis=1)
    df["has_bus_alt"] = df.n_bus_stops > 0
    df[["station_id", "name", "lat", "lon", "lines", "n_bus_stops", "n_bus_routes",
        "nearest_station_km", "has_bus_alt"]].to_csv(PROC / "bus_replacement.csv", index=False)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    order = df.sort_values("n_bus_routes", ascending=False)
    ax.barh(order["name"], order["n_bus_routes"], color="steelblue")
    ax.set_xlabel(f"bus routes within {RADIUS_M:.0f} m")
    ax.set_yticks([])
    ax.set_title(f"Bus-route coverage of the 86 rail stations\n"
                 f"({int((df.n_bus_routes == 0).sum())} stations with no bus route within 400 m)")
    ax.grid(alpha=0.3, axis="x")

    ax = axes[1]
    for _, r in edges.iterrows():
        a = df[df.station_id == r.station_a].iloc[0]
        b = df[df.station_id == r.station_b].iloc[0]
        ax.plot([a.lon, b.lon], [a.lat, b.lat], "-", color="0.75", lw=0.8, zorder=1)
    sc = ax.scatter(df.lon, df.lat, c=df.n_bus_routes, s=40, cmap="viridis",
                    norm=matplotlib.colors.LogNorm(vmin=1, vmax=max(df.n_bus_routes.max(), 2)),
                    edgecolors="k", linewidths=0.4, zorder=2)
    for _, r in df.iterrows():
        if r.n_bus_routes == 0 or r.name in ("Perth Stn", "Mandurah Stn", "Yanchep Stn",
                                             "Clarkson Stn", "Joondalup Stn"):
            ax.annotate(r["name"].replace(" Stn", ""), (r.lon, r.lat), fontsize=6,
                        xytext=(3, 2), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="bus routes within 400 m")
    ax.set_aspect(1.15)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("Transperth bus-route coverage around rail stations")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_bus_coverage.png", dpi=150)
    plt.close(fig)

    print("\nStations with zero bus routes within 400 m:",
          ", ".join(df[df.n_bus_routes == 0]["name"].tolist()))
    print("wrote fig4_bus_coverage.png, bus_replacement.csv, bus_parallel_corridors.csv")


if __name__ == "__main__":
    main()