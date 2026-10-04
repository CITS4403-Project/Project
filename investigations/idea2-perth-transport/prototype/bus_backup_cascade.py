"""Experiment 2: can standby buses aggravate a rail overload cascade?

Compare no backup with all verified existing_bus edges, using the notebook's
terminal/rail model. Terminals retain the original equal-demand OD pairs;
rail facilities close synchronously when shortest-path load exceeds capacity.
Capacities come from the SAME intact rail-only baseline in both scenarios.
Buses activate immediately after the trigger. Bus capacity is unlimited.

Usage (in the project's Python environment, from any working directory):
    python prototype/bus_backup_cascade.py
    python prototype/bus_backup_cascade.py --quick
    python prototype/bus_backup_cascade.py --targets "Perth Stn" "Bayswater Stn"
    python prototype/bus_backup_cascade.py --bus-time-factor 1.5
    python prototype/bus_backup_cascade.py --selftest

Writes ../results/bus_cascade_*.csv, bus_cascade_summary.json and four figures
per target to ../figures/fig_bus_cascade_*.png. --output-dir overrides the root.
The model uses provisional rail topology and distance/40 km/h rail-time proxies.
This script implements the notebook model directly; it never executes a notebook.
"""
from __future__ import annotations

import argparse
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


INV = Path(__file__).resolve().parents[1]
LABELS = {"rail_only": "No standby buses", "existing_bus": "Existing buses as standby"}
COLORS = {"rail_only": "C0", "existing_bus": "C1"}
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": .3})


# --------------------------------------------------------------------------
# inputs and model
# --------------------------------------------------------------------------
def load_inputs(data_dir):
    stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
    edges = pd.read_csv(data_dir / "rail_edges.csv", dtype={"station_a": str, "station_b": str})
    buses = pd.read_csv(data_dir / "backup_edges.csv", dtype={"station_a": str, "station_b": str})
    if stations.station_id.isna().any() or not stations.station_id.is_unique:
        raise ValueError("station IDs must be present and unique")
    G = nx.Graph()
    for row in stations.itertuples(index=False):
        G.add_node(row.station_id, name=row.name)
    for row in edges.itertuples(index=False):
        if row.station_a not in G or row.station_b not in G or row.station_a == row.station_b:
            raise ValueError("invalid rail edge endpoints")
        if not np.isfinite(row.distance_m) or row.distance_m <= 0:
            raise ValueError("rail distances must be finite and positive")
        G.add_edge(row.station_a, row.station_b, distance_m=float(row.distance_m))
    if G.number_of_nodes() < 2 or not nx.is_connected(G):
        raise ValueError("the baseline rail graph must be connected with at least two stations")
    required = ["station_a", "station_b", "minutes", "kind", "route_or_road", "source"]
    if not set(required) <= set(buses.columns) or buses[required].isna().any().any():
        raise ValueError("backup_edges.csv needs complete endpoints, times, kinds and provenance")
    if not buses.kind.isin(["existing_bus", "emergency_bus"]).all():
        raise ValueError("unknown backup kind")
    buses = buses[buses.kind == "existing_bus"].copy()
    if buses.empty:
        raise ValueError("no existing_bus records; this experiment needs verified existing bus paths")
    return G, buses


def layered_graph(rail, speed_kmh=40.):
    """One persistent terminal and one removable rail facility per station."""
    H = nx.Graph()
    for sid, attrs in rail.nodes(data=True):
        for prefix, kind in (("T:", "terminal"), ("R:", "rail")):
            H.add_node(prefix + sid, kind=kind, station_id=sid, name=attrs.get("name", sid))
        H.add_edge("T:" + sid, "R:" + sid, minutes=1., mode="access")
    for a, b, attrs in rail.edges(data=True):
        minutes = max(.1, attrs["distance_m"] / 1000 / speed_kmh * 60)
        H.add_edge("R:" + a, "R:" + b, minutes=minutes, mode="rail")
    return H


def add_backups(normal, table, time_factor=1.):
    """Undirected effective paths; duplicate endpoints retain the fastest path.

    The input evidence must establish availability in both directions. These
    effective paths include walking/waiting/transfers and are not road segments.
    """
    H = normal.copy()
    for row in table.itertuples(index=False):
        a, b = "T:" + str(row.station_a), "T:" + str(row.station_b)
        minutes = float(row.minutes) * time_factor
        if a not in H or b not in H or a == b:
            raise ValueError("invalid backup station IDs")
        if not np.isfinite(minutes) or minutes <= 0:
            raise ValueError("backup travel times must be finite and positive")
        if not H.has_edge(a, b) or minutes < H[a][b]["minutes"]:
            H.add_edge(a, b, minutes=minutes, mode="existing_bus")
    return H


def terminal_loads(H, terminals):
    values = nx.betweenness_centrality_subset(H, sources=terminals, targets=terminals,
                                             normalized=False, weight="minutes")
    return {n: values[n] for n in H if H.nodes[n]["kind"] == "rail"}


def pair_times(H, terminals):
    distances = {n: nx.single_source_dijkstra_path_length(H, n, weight="minutes") for n in terminals}
    return {(a, b): distances[a][b] for a, b in combinations(terminals, 2) if b in distances[a]}


def demand_metrics(times, baseline_times):
    """Unreachable OD pairs remain in the denominator; time is conditional."""
    changes = [minutes - baseline_times[pair] for pair, minutes in times.items()]
    return {"reachable_pairs": len(times), "original_pairs": len(baseline_times),
            "unmet_fraction": 1 - len(times) / len(baseline_times),
            "mean_time_change_reachable_min": float(np.mean(changes)) if changes else np.nan}


def simulate_cascade(normal, standby, alpha, target, baseline_loads=None,
                     baseline_times=None, trace=False, overload_enabled=True):
    """Trace states after each removal batch; round 0 follows the trigger.

    In each state, all current overloads are identified before simultaneous
    closure in the next round. The final stable state is included in traces.
    A round is an algorithmic update, not a minute of operating time.
    """
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and nonnegative")
    terminals = sorted(n for n in normal if normal.nodes[n]["kind"] == "terminal")
    if baseline_loads is None:
        baseline_loads = terminal_loads(normal, terminals)
    if baseline_times is None:
        baseline_times = pair_times(normal, terminals)
    capacity = {n: (1 + alpha) * value for n, value in baseline_loads.items()}
    trigger = "R:" + target
    if trigger not in capacity:
        raise ValueError("unknown trigger station")
    H = standby.copy()
    H.remove_node(trigger)
    failed, removed = [trigger], [trigger]
    history, load_rows = [], []
    round_no = 0
    while True:
        current = terminal_loads(H, terminals)
        overloaded = sorted(n for n in current if current[n] > capacity[n] + 1e-10)
        next_failed = overloaded if overload_enabled else []
        if trace or not next_failed:
            times = pair_times(H, terminals)
            metrics = demand_metrics(times, baseline_times)
        if trace:
            history.append({"round": round_no, "n_failed": len(failed),
                            "n_secondary": len(failed)-1, "removed_ids": ";".join(n[2:] for n in removed),
                            "next_failed_ids": ";".join(n[2:] for n in next_failed), **metrics})
            for node, load in current.items():
                load_rows.append({"round": round_no, "station_id": node[2:],
                                  "name": H.nodes[node]["name"], "initial_load": baseline_loads[node],
                                  "load": load, "capacity": capacity[node],
                                  "load_ratio": load / capacity[node] if capacity[node] > 0 else np.nan,
                                  "overloaded": node in overloaded,
                                  "closes_next_round": node in next_failed})
        if not next_failed:
            break
        failed.extend(next_failed)
        H.remove_nodes_from(next_failed)
        removed = next_failed
        round_no += 1
    return {"n_failed": len(failed), "n_secondary": len(failed)-1,
            "failed_fraction": len(failed)/len(capacity), "cascade_rounds": round_no,
            "failed_ids": ";".join(n[2:] for n in failed), **metrics,
            "history": history, "loads": load_rows, "pair_times": times}


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def expt_bus_cascade(normal, standby, targets, alphas, focus_alphas, progress=True):
    terminals = sorted(n for n in normal if normal.nodes[n]["kind"] == "terminal")
    baseline_loads = terminal_loads(normal, terminals)
    baseline_times = pair_times(normal, terminals)
    scenarios = {"rail_only": normal, "existing_bus": standby}
    rows, comparisons, history, loads, controls = [], [], [], [], []
    for target in targets:
        name = normal.nodes["R:"+target]["name"]
        for scenario, graph in scenarios.items():
            r = simulate_cascade(normal, graph, 0., target, baseline_loads, baseline_times,
                                 overload_enabled=False)
            controls.append({"target": target, "target_name": name, "scenario": scenario,
                             **{k: v for k, v in r.items() if k not in ("history", "loads", "pair_times")}})
        for alpha in alphas:
            case_results = {}
            for scenario, graph in scenarios.items():
                r = simulate_cascade(normal, graph, float(alpha), target, baseline_loads, baseline_times,
                                     trace=alpha in focus_alphas)
                tags = {"target": target, "target_name": name, "alpha": float(alpha), "scenario": scenario}
                rows.append({**tags, **{k: v for k, v in r.items() if k not in ("history", "loads", "pair_times")}})
                history.extend({**tags, **row} for row in r["history"])
                loads.extend({**tags, **row} for row in r["loads"])
                case_results[scenario] = r
            rail, bus = case_results["rail_only"], case_results["existing_bus"]
            common = rail["pair_times"].keys() & bus["pair_times"].keys()
            time_difference = [bus["pair_times"][p]-rail["pair_times"][p] for p in common]
            comparisons.append({"target": target, "target_name": name, "alpha": float(alpha),
                                "delta_unmet_fraction": bus["unmet_fraction"]-rail["unmet_fraction"],
                                "delta_n_failed": bus["n_failed"]-rail["n_failed"],
                                "common_reachable_fraction": len(common)/len(baseline_times),
                                "bus_minus_rail_time_common_min": float(np.mean(time_difference)) if common else np.nan})
        if progress:
            print(f"    {name}: {len(alphas)*2} cascades and 2 no-overload controls completed", flush=True)
    return {"scan": pd.DataFrame(rows), "comparison": pd.DataFrame(comparisons),
            "history": pd.DataFrame(history), "loads": pd.DataFrame(loads), "controls": pd.DataFrame(controls)}


def _save(fig, figures, name, title, bottom=.065):
    fig.suptitle(title, fontsize=12)
    fig.text(.5, .015, "Provisional rail/time proxies; fixed original OD pairs and capacities; unlimited standby bus capacity.",
             ha="center", fontsize=7, color="0.35")
    fig.tight_layout(rect=(0, bottom, 1, .93))
    path = figures / name
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# visualisations
# --------------------------------------------------------------------------
def plot_scan(tables, target, figures):
    scan = tables["scan"].query("target == @target")
    delta = tables["comparison"].query("target == @target")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for scenario, sub in scan.groupby("scenario", sort=False):
        for ax, column, scale in ((axes[0], "n_failed", 1), (axes[1], "unmet_fraction", 100)):
            ax.plot(sub.alpha, scale*sub[column], color=COLORS[scenario], label=LABELS[scenario], lw=1.5)
    axes[0].set(xlabel="Tolerance alpha", ylabel="Failed rail facilities (including trigger)", title="Rail damage")
    axes[0].legend(fontsize=7)
    axes[1].set(xlabel="Tolerance alpha", ylabel="Unserved original OD pairs (%)", title="Service loss", ylim=(0, 100))
    x, y = delta.alpha.to_numpy(), 100*delta.delta_unmet_fraction.to_numpy()
    axes[2].plot(x, y, color="0.25", lw=1)
    axes[2].fill_between(x, 0, y, where=y > 0, interpolate=True, color="C3", alpha=.4, label="Buses worsen service")
    axes[2].fill_between(x, 0, y, where=y < 0, interpolate=True, color="C2", alpha=.4, label="Buses improve service")
    axes[2].axhline(0, color="k", ls="--", lw=.7)
    axes[2].set(xlabel="Tolerance alpha", ylabel="Bus minus no-bus service loss (pp)", title="Net bus effect")
    axes[2].legend(fontsize=7)
    return _save(fig, figures, f"fig_bus_cascade_scan_{target}.png",
                 f"Can standby buses aggravate cascading failure?  Trigger: {scan.target_name.iloc[0]}")


def plot_control(tables, target, focus_alphas, figures):
    scan = tables["scan"].query("target == @target")
    control = tables["controls"].query("target == @target")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7))
    labels = ["Overload disabled"] + [f"Cascade, alpha={a:g}" for a in focus_alphas]
    x = np.arange(len(labels))
    for scenario in LABELS:
        direct = control[control.scenario == scenario].iloc[0]
        chosen = [direct] + [scan[(scan.scenario == scenario) & (scan.alpha == a)].iloc[0] for a in focus_alphas]
        offset = -.18 if scenario == "rail_only" else .18
        for ax, key, scale in ((axes[0], "unmet_fraction", 100), (axes[1], "n_failed", 1)):
            ax.bar(x+offset, [scale*r[key] for r in chosen], width=.34,
                   label=LABELS[scenario], color=COLORS[scenario])
            ax.set_xticks(x, labels, rotation=15, ha="right")
    axes[0].set(ylabel="Unserved original OD pairs (%)", title="Connectivity restoration versus overload cascade", ylim=(0, 100))
    axes[1].set(ylabel="Failed rail facilities", title="Rail losses under the same trigger")
    axes[1].legend(fontsize=7)
    return _save(fig, figures, f"fig_bus_cascade_control_{target}.png",
                 f"Control experiment: disable secondary overload closures  |  {scan.target_name.iloc[0]}")


def plot_history(tables, target, focus_alphas, figures):
    history = tables["history"].query("target == @target")
    fig, axes = plt.subplots(2, len(focus_alphas), figsize=(5*len(focus_alphas), 6), squeeze=False)
    for col, alpha in enumerate(focus_alphas):
        sub = history[history.alpha == alpha]
        for scenario, case in sub.groupby("scenario", sort=False):
            for row, key, scale in ((0, "n_failed", 1), (1, "unmet_fraction", 100)):
                axes[row, col].plot(case["round"], scale*case[key], "o-", color=COLORS[scenario],
                                    label=LABELS[scenario], ms=4)
        axes[0, col].set(title=f"alpha={alpha:g}", ylabel="Total failed rail facilities")
        axes[1, col].set(xlabel="Closure round (0 = initial trigger)", ylabel="Unserved original OD pairs (%)", ylim=(0, 100))
        for row in range(2):
            axes[row, col].set_xticks(range(int(sub["round"].max())+1))
        axes[0, col].legend(fontsize=7)
    return _save(fig, figures, f"fig_bus_cascade_history_{target}.png",
                 f"How the cascade develops after {history.target_name.iloc[0]} closes")


def plot_loads(tables, target, focus_alphas, figures, top=12):
    loads = tables["loads"].query("target == @target")
    order = loads.groupby("station_id").load_ratio.max().sort_values(ascending=False).head(top).index.tolist()
    names = loads.drop_duplicates("station_id").set_index("station_id")["name"]
    fig, axes = plt.subplots(len(focus_alphas), 2, figsize=(10, 3.7*len(focus_alphas)), squeeze=False)
    cmap = matplotlib.colormaps["YlOrRd"].copy()
    cmap.set_bad("0.8")
    for row, alpha in enumerate(focus_alphas):
        for col, scenario in enumerate(LABELS):
            case = loads[(loads.alpha == alpha) & (loads.scenario == scenario)]
            final_round = int(tables["history"].query("target == @target and alpha == @alpha and scenario == @scenario")["round"].max())
            grid = case.pivot(index="station_id", columns="round", values="load_ratio").reindex(index=order, columns=range(final_round+1))
            im = axes[row, col].imshow(grid.to_numpy(), aspect="auto", vmin=0, vmax=3, cmap=cmap, interpolation="nearest")
            axes[row, col].set_xticks(range(final_round+1))
            axes[row, col].set_yticks(range(len(order)), [names.loc[sid].removesuffix(" Stn") for sid in order])
            axes[row, col].set(xlabel="Closure round", title=f"{LABELS[scenario]}  |  alpha={alpha:g}")
            axes[row, col].grid(False)
            for yi in range(len(order)):
                for xi in range(final_round+1):
                    if np.isfinite(grid.iloc[yi, xi]) and grid.iloc[yi, xi] > 1 + 1e-10:
                        axes[row, col].text(xi, yi, "x", ha="center", va="center", fontsize=8, color="k")
            fig.colorbar(im, ax=axes[row, col], pad=.02, label="Load / fixed capacity (colour clipped at 3)")
    return _save(fig, figures, f"fig_bus_cascade_loads_{target}.png",
                 f"Top {len(order)} observed load ratios: x = overloaded; grey = closed or undefined")


# --------------------------------------------------------------------------
# checks and command-line entry point
# --------------------------------------------------------------------------
def selftest():
    rail = nx.path_graph(["A", "B", "C"])
    nx.set_edge_attributes(rail, 1000., "distance_m")
    normal = layered_graph(rail)
    table = pd.DataFrame([{"station_a": "A", "station_b": "B", "minutes": 8.},
                          {"station_a": "B", "station_b": "C", "minutes": 8.}])
    standby = add_backups(normal, table)
    no_bus = simulate_cascade(normal, normal, 100., "B", trace=True)
    buses = simulate_cascade(normal, standby, 100., "B", trace=True)
    ring = nx.cycle_graph([str(i) for i in range(10)])
    nx.set_edge_attributes(ring, 1000., "distance_m")
    ring_normal = layered_graph(ring)
    overload = simulate_cascade(ring_normal, ring_normal, 0., "0", trace=True)
    direct = simulate_cascade(ring_normal, ring_normal, 0., "0", overload_enabled=False)
    checks = [
        ("closing the middle rail facility disconnects all three original pairs", no_bus["unmet_fraction"] == 1),
        ("bus terminals survive rail facility closure", buses["unmet_fraction"] == 0),
        ("large capacity contains the secondary cascade", buses["n_failed"] == 1),
        ("all original OD pairs remain in the denominator", no_bus["original_pairs"] == buses["original_pairs"] == 3),
        ("baseline graph is unchanged", normal.number_of_nodes() == 6 and normal.number_of_edges() == 5),
        ("ring rerouting produces secondary overload closures", overload["n_secondary"] > 0),
        ("control disables secondary closures", direct["n_failed"] == 1),
        ("trace ends in a stable state", overload["history"][-1]["next_failed_ids"] == ""),
        ("overload batches contain simultaneous failures", len(overload["history"][0]["next_failed_ids"].split(";")) > 1),
    ]
    bad_time = table.copy()
    bad_time.loc[0, "minutes"] = -1
    try:
        add_backups(normal, bad_time)
        checks.append(("negative bus time rejected", False))
    except ValueError:
        checks.append(("negative bus time rejected", True))
    for label, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}: {label}")
    return all(ok for _, ok in checks)


def resolve_targets(rail, requested):
    resolved = []
    for text in requested:
        matches = [sid for sid in rail if sid == text or rail.nodes[sid]["name"].casefold() == text.casefold()
                   or rail.nodes[sid]["name"].removesuffix(" Stn").casefold() == text.casefold()]
        if len(matches) != 1:
            raise ValueError(f"target {text!r} does not identify exactly one station; use its ID or full name")
        if matches[0] not in resolved:
            resolved.append(matches[0])
    return resolved


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="five tolerances instead of the default 0..2 step .05")
    ap.add_argument("--targets", nargs="+", default=["Perth Stn"], help="trigger station names or IDs")
    ap.add_argument("--alphas", type=float, nargs="+", help="custom tolerance grid")
    ap.add_argument("--focus-alphas", type=float, nargs="+", default=[.2, 1.], help="tolerances with round/load plots; always included in grid")
    ap.add_argument("--bus-time-factor", type=float, default=1., help="multiply bus effective times, keeping original rail capacities")
    ap.add_argument("--speed-kmh", type=float, default=40., help="rail time proxy speed (default 40 km/h)")
    ap.add_argument("--data-dir", type=Path, default=INV / "data" / "processed")
    ap.add_argument("--output-dir", type=Path, help="write results/ and figures/ below this directory")
    ap.add_argument("--selftest", action="store_true", help="run small-network model checks and exit")
    args = ap.parse_args()
    if args.selftest:
        return 0 if selftest() else 1
    focus = sorted(set(args.focus_alphas))
    if not 1 <= len(focus) <= 4:
        ap.error("choose one to four focus tolerances for readable round/load plots")
    grid = args.alphas if args.alphas is not None else ([0, .2, .5, 1, 2] if args.quick else np.round(np.linspace(0, 2, 41), 8))
    alphas = np.unique(np.append(grid, focus))
    if not np.isfinite(alphas).all() or (alphas < 0).any():
        ap.error("tolerances must be finite and nonnegative")
    if not all(np.isfinite(v) and v > 0 for v in (args.speed_kmh, args.bus_time_factor)):
        ap.error("rail speed and bus time factor must be finite and positive")
    try:
        rail, buses = load_inputs(args.data_dir)
        targets = resolve_targets(rail, args.targets)
        normal = layered_graph(rail, args.speed_kmh)
        standby = add_backups(normal, buses, args.bus_time_factor)
    except (ValueError, OSError, KeyError) as exc:
        ap.error(str(exc))
    output = args.output_dir.resolve() if args.output_dir else INV
    results, figures = output / "results", output / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"--- bus cascade: {len(targets)} trigger(s), {len(alphas)} tolerances, {len(buses)} existing bus records ---", flush=True)
    tables = expt_bus_cascade(normal, standby, targets, alphas, focus)
    for name, table in tables.items():
        table.to_csv(results / f"bus_cascade_{name}.csv", index=False)
    paths = []
    for target in targets:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", target):
            raise ValueError("station ID cannot be used safely as an output filename")
        paths.extend([plot_scan(tables, target, figures), plot_control(tables, target, focus, figures),
                      plot_history(tables, target, focus, figures), plot_loads(tables, target, focus, figures)])
    summary = {"_meta": {
        "n_stations": rail.number_of_nodes(), "n_rail_edges": rail.number_of_edges(),
        "n_bus_records": len(buses), "n_unique_bus_edges": standby.number_of_edges()-normal.number_of_edges(),
        "alphas": alphas.tolist(), "focus_alphas": focus, "targets": targets,
        "bus_time_factor": args.bus_time_factor, "rail_speed_kmh": args.speed_kmh,
        "access_minutes": 1., "quick": args.quick,
        "model": "persistent terminals; terminal-subset weighted betweenness; synchronous irreversible rail closures",
        "capacity_baseline": "same intact rail-only load in both scenarios; C=(1+alpha)*L0",
        "assumptions": ["equal demand per original undirected terminal pair", "provisional service topology",
                        "rail time estimated from straight-line distance and speed", "fixed bidirectional effective bus times",
                        "standby activated at trigger", "unlimited bus capacity", "rounds are not elapsed minutes"],
        "travel_time_metrics": "scenario mean conditional on reachable pairs; paired difference uses only pairs reachable in BOTH scenarios",
        "n_cascade_runs": len(tables["scan"]), "n_controls": len(tables["controls"]),
        "wall_seconds": time.perf_counter()-started,
        "sources": {name: {"path": str((args.data_dir/name).resolve()),
                           "sha256": hashlib.sha256((args.data_dir/name).read_bytes()).hexdigest()}
                    for name in ("stations.csv", "rail_edges.csv", "backup_edges.csv")},
        "versions": {"networkx": nx.__version__, "numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__}}, "targets": {}, "figures": [str(p) for p in paths]}
    for target in targets:
        sub = tables["comparison"].query("target == @target")
        summary["targets"][target] = {
            "name": rail.nodes[target]["name"],
            "sampled_alphas_bus_worsens_service": sub.loc[sub.delta_unmet_fraction > 1e-12, "alpha"].tolist(),
            "sampled_alphas_bus_improves_service": sub.loc[sub.delta_unmet_fraction < -1e-12, "alpha"].tolist()}
    (results / "bus_cascade_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print("\nFocus results:")
    print(tables["scan"][tables["scan"].alpha.isin(focus)][["target_name", "alpha", "scenario", "n_failed", "unmet_fraction"]].to_string(index=False))
    print(f"\nCompleted {len(tables['scan'])} cascades and {len(tables['controls'])} controls in {time.perf_counter()-started:.1f}s")
    print(f"Results: {results}\nFigures: {figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
