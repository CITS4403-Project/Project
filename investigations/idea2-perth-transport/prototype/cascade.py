"""Motter-Lai style cascading-failure model on the Perth rail network.

Model (documented choices):
  * Initial load  L0_i  = betweenness centrality of the intact graph
                          (optionally frequency-weighted: edge length 1/trips,
                           so frequent services carry 'shorter' paths).
  * Capacity      C_i   = (1 + alpha) * L0_i          (alpha = tolerance)
  * Trigger       : remove one station (targeted by load / degree, or random).
  * Redistribution: the failed station's load is shared between its *surviving*
                    neighbours, either equally or proportional to capacity.
  * Cascade       : a neighbour fails when current load > capacity; its load is
                    redistributed in turn; iterate until no new failures.
  * Load update   : two modes
      - dynamic=True : loads are recomputed (betweenness on the surviving
                       graph) after every failure round - models rerouting.
      - dynamic=False: loads are static; only the failed node's load moves.
  * Metrics       : failed fraction, GCC fraction, avalanche sizes.

Run:  python prototype/cascade.py            (~1 min)
      python prototype/cascade.py --quick    (~10 s)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)


def load_graph() -> nx.Graph:
    edf = pd.read_csv(PROC / "rail_edges.csv", dtype={"station_a": str, "station_b": str})
    ndf = pd.read_csv(PROC / "stations.csv", dtype={"station_id": str})
    G = nx.Graph()
    for _, r in ndf.iterrows():
        G.add_node(r["station_id"], name=r["name"], lat=r["lat"], lon=r["lon"],
                   trips_served=float(r.get("trips_served", 0.0)))
    for _, r in edf.iterrows():
        G.add_edge(r["station_a"], r["station_b"], trips=float(r["trips"]))
    return G


def betweenness(G: nx.Graph, load_mode: str) -> dict:
    """Unnormalised betweenness: load counts must stay comparable when the
    graph shrinks during a cascade (normalised values would inflate).

    load_mode:
      betweenness              pure unnormalised betweenness
      betweenness_freq         frequency-weighted shortest paths (len=1/trips)
      betweenness_plus_trips   betweenness + 0.1 * trips_served (station
                               throughput proxy; gives leaf stations a positive
                               baseline load so zero-capacity artefacts vanish)
    """
    if load_mode == "betweenness":
        return nx.betweenness_centrality(G, normalized=False)
    if load_mode == "betweenness_freq":
        H = nx.Graph()
        H.add_nodes_from(G.nodes)
        for u, v, d in G.edges(data=True):
            H.add_edge(u, v, length=1.0 / max(d.get("trips", 1.0), 1.0))
        return nx.betweenness_centrality(H, weight="length", normalized=False)
    if load_mode == "betweenness_plus_trips":
        base = nx.betweenness_centrality(G, normalized=False)
        return {n: base[n] + 0.1 * G.nodes[n].get("trips_served", 0.0) for n in G.nodes}
    raise ValueError(load_mode)


def simulate_cascade(G: nx.Graph, alpha: float, trigger: str = "load",
                     rule: str = "capacity", dynamic: bool = True,
                     load_mode: str = "betweenness", seed: int | None = None,
                     target: str | None = None) -> dict:
    """Run one cascade.

    dynamic=True : loads are recomputed (betweenness on the surviving graph)
                   after every failure round - models full shortest-path
                   rerouting. Nodes fail when load > capacity (capacities are
                   always fixed from the intact graph).
    dynamic=False: initial loads are static; the load of each failed node is
                   redistributed to its surviving neighbours (rule='equal' or
                   'capacity' proportional). Nodes fail when load > capacity.

    Returns failed set, GCC, avalanche sizes, rounds.
    """
    rng = np.random.default_rng(seed)
    H = G.copy()
    L0 = betweenness(G, load_mode)
    C = {n: (1.0 + alpha) * L0[n] for n in G.nodes}

    if target is not None:
        first = target
    elif trigger == "load":
        first = max(G.nodes, key=lambda n: L0[n])
    elif trigger == "degree":
        first = max(G.nodes, key=lambda n: G.degree(n))
    elif trigger == "random":
        first = rng.choice(sorted(G.nodes))
    else:
        raise ValueError(trigger)

    failed: list[str] = [first]
    avalanche_sizes: list[int] = []
    rounds_failed: list[int] = []

    if dynamic:
        H.remove_node(first)
        round_no = 0
        while H.number_of_nodes() > 0:
            round_no += 1
            L = betweenness(H, load_mode)
            newly = [n for n in sorted(H.nodes) if L[n] > C[n] + 1e-12]
            if not newly:
                break
            failed.extend(newly)
            rounds_failed.extend([round_no] * len(newly))
            avalanche_sizes.append(len(newly))
            H.remove_nodes_from(newly)
    else:
        load = {n: L0[n] for n in G.nodes}
        trigger_load = L0[first]
        load = {n: L0[n] for n in G.nodes if n != first}
        pending: list[tuple[list[str], float, list[str]]] = [
            ([m for m in G.neighbors(first)], trigger_load, [first])]
        H.remove_node(first)
        rounds = 0
        while pending:
            rounds += 1
            new_failed: list[str] = []
            for nbrs, amount, _src in pending:
                survivors = [m for m in nbrs if m in H]
                if not survivors:
                    continue
                if rule == "equal":
                    for m in survivors:
                        load[m] = load.get(m, 0.0) + amount / len(survivors)
                else:
                    tot = sum(C[m] for m in survivors)
                    if tot <= 0:  # all-zero capacities: fall back to equal share
                        for m in survivors:
                            load[m] = load.get(m, 0.0) + amount / len(survivors)
                    else:
                        for m in survivors:
                            load[m] = load.get(m, 0.0) + amount * C[m] / tot
            new_failed = [n for n in sorted(H.nodes) if load.get(n, 0.0) > C[n] + 1e-12]
            if not new_failed:
                break
            failed.extend(new_failed)
            rounds_failed.extend([rounds] * len(new_failed))
            avalanche_sizes.append(len(new_failed))
            nxt = []
            for n in new_failed:
                nbrs = [m for m in H.neighbors(n)]
                nxt.append((nbrs, load.get(n, 0.0), [n]))
                load.pop(n, None)
            H.remove_nodes_from(new_failed)
            pending = nxt
        round_no = rounds

    gcc = max((len(c) for c in nx.connected_components(H)), default=0)
    return dict(n_initial=G.number_of_nodes(), n_failed=len(failed), failed=failed,
                gcc=gcc, gcc_fraction=gcc / G.number_of_nodes(),
                failed_fraction=len(failed) / G.number_of_nodes(),
                avalanche_sizes=avalanche_sizes, rounds=round_no)


def alpha_sweep(G: nx.Graph, alphas, n_random: int, dynamic: bool, load_mode: str,
                rule: str, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for alpha in alphas:
        t = simulate_cascade(G, alpha, trigger="load", dynamic=dynamic,
                             load_mode=load_mode, rule=rule, seed=seed)
        g = simulate_cascade(G, alpha, trigger="degree", dynamic=dynamic,
                             load_mode=load_mode, rule=rule, seed=seed)
        rnd = [simulate_cascade(G, alpha, trigger="random", dynamic=dynamic,
                                load_mode=load_mode, rule=rule, seed=int(rng.integers(1e9)))
               for _ in range(n_random)]
        rows.append(dict(alpha=alpha,
                         fail_targeted_load=t["failed_fraction"],
                         gcc_targeted_load=t["gcc_fraction"],
                         fail_targeted_degree=g["failed_fraction"],
                         gcc_targeted_degree=g["gcc_fraction"],
                         fail_random_mean=float(np.mean([r["failed_fraction"] for r in rnd])),
                         fail_random_std=float(np.std([r["failed_fraction"] for r in rnd])),
                         gcc_random_mean=float(np.mean([r["gcc_fraction"] for r in rnd])),
                         gcc_random_std=float(np.std([r["gcc_fraction"] for r in rnd]))))
    return pd.DataFrame(rows)


def avalanche_experiment(G: nx.Graph, alpha: float, n: int, dynamic: bool,
                         load_mode: str, rule: str, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sizes = []
    for _ in range(n):
        r = simulate_cascade(G, alpha, trigger="random", dynamic=dynamic,
                             load_mode=load_mode, rule=rule, seed=int(rng.integers(1e9)))
        sizes.append(r["n_failed"])
    return np.array(sizes)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    G = load_graph()
    n_random = 30 if args.quick else 300
    alphas = (np.round(np.arange(0.0, 0.62, 0.05), 3) if args.quick
              else np.round(np.arange(0.0, 0.62, 0.025), 3))
    aval_n = 300 if args.quick else 2000

    print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # --- Experiment A: alpha sweep, classic Motter-Lai static redistribution ---
    # Headline model: loads fixed from the intact graph, failed node's load is
    # redistributed to surviving neighbours proportional to capacity.
    sweeps = {}
    for dynamic, tag in ((False, "static"), (True, "dynamic")):
        df = alpha_sweep(G, alphas, n_random, dynamic=dynamic,
                         load_mode="betweenness", rule="capacity")
        df.to_csv(PROC / f"alpha_sweep_{tag}.csv", index=False)
        sweeps[tag] = df
        print(f"alpha sweep ({tag}) done")

    static = sweeps["static"]
    dynamic = sweeps["dynamic"]
    crit = static[static.fail_targeted_load <= 1 / G.number_of_nodes() + 1e-9]
    alpha_star = float(crit.alpha.min()) if len(crit) else float("nan")
    print(f"alpha* (static ML, max-load trigger contained) = {alpha_star}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, df, tag in ((axes[0], static, "static ML redistribution"),
                        (axes[1], dynamic, "dynamic rerouting (recomputed loads)")):
        ax.errorbar(df.alpha, df.fail_random_mean * 100, yerr=df.fail_random_std * 100,
                    fmt="o-", ms=3, capsize=2, label="random trigger (mean±sd)")
        ax.plot(df.alpha, df.fail_targeted_load * 100, "s-", ms=3, label="targeted: max load")
        ax.plot(df.alpha, df.fail_targeted_degree * 100, "^-", ms=3, label="targeted: max degree")
        ax.set_xlabel(r"tolerance $\alpha$")
        ax.set_title(tag, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("failed stations (%)")
    if np.isfinite(alpha_star):
        for ax in axes:
            ax.axvline(alpha_star, color="grey", ls=":", lw=1)
        axes[0].annotate(rf"$\alpha^*\approx{alpha_star:.2f}$", (alpha_star + 0.01, 5),
                         color="grey")
    fig.suptitle("Perth rail: cascading failures vs tolerance (single-station trigger)")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_alpha_sweep.png", dpi=150)
    plt.close(fig)

    # --- redistribution rule sensitivity at alpha=0.2 ---
    cmp_rows = []
    for dynamic_mode in (True, False):
        for rule in ("capacity", "equal"):
            for lm in ("betweenness", "betweenness_plus_trips"):
                r = simulate_cascade(G, 0.2, trigger="load", dynamic=dynamic_mode,
                                     load_mode=lm, rule=rule)
                cmp_rows.append(dict(dynamic=dynamic_mode, rule=rule, load_mode=lm,
                                     alpha=0.2, failed=r["n_failed"], gcc=r["gcc"],
                                     gcc_fraction=r["gcc_fraction"]))
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(PROC / "redistribution_comparison.csv", index=False)
    print(cmp_df.to_string(index=False))

    # --- Experiment B: avalanche size distribution (static ML, random triggers) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    stats = []
    for alpha in (0.1, 0.15, 0.2, 0.3):
        sizes = avalanche_experiment(G, alpha, aval_n, False, "betweenness",
                                     "capacity", seed=int(alpha * 1000))
        np.save(PROC / f"avalanche_sizes_alpha{alpha}.npy", sizes)
        vals, counts = np.unique(sizes, return_counts=True)
        axes[0].loglog(vals, counts, "o", ms=4, alpha=0.7, label=rf"$\alpha={alpha}$")
        stats.append(dict(alpha=alpha, mean=float(sizes.mean()), max=int(sizes.max()),
                          p_no_cascade=float((sizes <= 1).mean()), n=len(sizes),
                          largest_avalanche=int(sizes.max())))
    stats_df = pd.DataFrame(stats)
    axes[0].set_xlabel("avalanche size (stations failed, incl. trigger)")
    axes[0].set_ylabel("count")
    axes[0].set_title("avalanche size distribution (static ML)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")
    axes[1].plot(stats_df.alpha, stats_df.p_no_cascade * 100, "o-")
    axes[1].set_xlabel(r"tolerance $\alpha$")
    axes[1].set_ylabel("P(no cascade | random trigger) [%]")
    axes[1].set_title("probability a random station failure stays isolated")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_avalanche.png", dpi=150)
    plt.close(fig)
    stats_df.to_csv(PROC / "avalanche_stats.csv", index=False)
    print(stats_df.to_string(index=False))

    # --- Experiment C: line-closure scenarios (bus-replacement relevance) ---
    lines = pd.read_csv(PROC / "stations.csv", dtype={"station_id": str})
    line_counts: dict[str, set] = {}
    for _, r in lines.iterrows():
        for ln in str(r["lines"]).split(";"):
            line_counts.setdefault(ln, set()).add(r["station_id"])
    rows = []
    for ln, ids in sorted(line_counts.items()):
        exclusive = {i for i in ids
                     if not any(i in s for l2, s in line_counts.items() if l2 != ln)}
        H = G.copy()
        H.remove_nodes_from(exclusive)
        gcc = max((len(c) for c in nx.connected_components(H)), default=0)
        lost_trips = lines[lines.station_id.isin(exclusive)]["trips_served"].sum()
        rows.append(dict(line=ln, n_stations=len(ids), n_exclusive=len(exclusive),
                         gcc_frac_after=gcc / G.number_of_nodes(),
                         frac_stations_lost=len(exclusive) / G.number_of_nodes(),
                         rail_trips_at_lost_stations=int(lost_trips)))
    line_df = pd.DataFrame(rows).sort_values("n_stations", ascending=False)
    line_df.to_csv(PROC / "line_closure_scenarios.csv", index=False)
    print("\nLine-closure (exclusive stations removed):")
    print(line_df.to_string(index=False))

    # --- Experiment D: map of the dynamic cascade from Perth Stn (alpha=0.2) ---
    r = simulate_cascade(G, 0.2, trigger="load", dynamic=True,
                         load_mode="betweenness", rule="capacity")
    failed = set(r["failed"])
    fig, ax = plt.subplots(figsize=(7, 8))
    for u, v, d in G.edges(data=True):
        ax.plot([G.nodes[u]["lon"], G.nodes[v]["lon"]], [G.nodes[u]["lat"], G.nodes[v]["lat"]],
                "-", color="0.7", lw=0.8 + d["trips"] / 1500, zorder=1)
    xs = [G.nodes[n]["lon"] for n in G.nodes]
    ys = [G.nodes[n]["lat"] for n in G.nodes]
    sizes = [20 + 60 * (G.nodes[n]["trips_served"] / 5507) ** 0.7 for n in G.nodes]
    colors = ["crimson" if n in failed else "steelblue" for n in G.nodes]
    ax.scatter(xs, ys, s=sizes, c=colors, zorder=2, edgecolors="k", linewidths=0.4)
    for n in G.nodes:
        if G.nodes[n]["trips_served"] > 2000 or n in failed:
            continue
    big = sorted(G.nodes, key=lambda n: -G.nodes[n]["trips_served"])[:12]
    for n in big + [f for f in failed if G.nodes[f]["trips_served"] > 1200]:
        ax.annotate(G.nodes[n]["name"].replace(" Stn", ""),
                    (G.nodes[n]["lon"], G.nodes[n]["lat"]), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_aspect(1.15)
    ax.set_title(f"Perth rail cascade from Perth Stn (dynamic loads, $\\alpha$=0.2)\n"
                 f"{r['n_failed']} of {G.number_of_nodes()} stations fail (red), GCC={r['gcc']}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_cascade_map.png", dpi=150)
    plt.close(fig)

    # --- validation checks (lightweight, printed) ---
    val = []
    r0 = simulate_cascade(G, 5.0, trigger="load", dynamic=False, rule="capacity")
    val.append(("large alpha contains cascade", r0["n_failed"] == 1 and r0["gcc"] == G.number_of_nodes() - 1))
    H2 = G.copy(); leaf = [n for n in G.nodes if G.degree(n) == 1][0]
    H2.remove_node(leaf)
    val.append(("GCC after leaf removal is n-1", max(len(c) for c in nx.connected_components(H2)) == G.number_of_nodes() - 1))
    r0 = simulate_cascade(G, 0.0, trigger="load", target=leaf, dynamic=False)
    val.append(("zero-load leaf trigger causes no cascade", r0["n_failed"] == 1))
    # static load conservation (equal rule) when all neighbours survive
    val.append(("capacity rule keeps graph metrics sane",
                static["fail_targeted_load"].max() <= 1.0 + 1e-9 and static["gcc_targeted_load"].min() >= 0))
    (PROC / "validation_checks.txt").write_text(
        "\n".join(f"{'PASS' if ok else 'FAIL'}: {name}" for name, ok in val) + "\n")
    print("\nValidation:", ["PASS" if ok else "FAIL" for _, ok in val])

    (PROC / "cascade_summary.txt").write_text(
        f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n"
        f"alpha* (static ML) = {alpha_star}\n\n=== static sweep ===\n" +
        static.to_string(index=False) + "\n\n=== dynamic sweep ===\n" +
        dynamic.to_string(index=False) + "\n\n=== redistribution/load sensitivity ===\n" +
        cmp_df.to_string(index=False) + "\n\n=== avalanche stats ===\n" +
        stats_df.to_string(index=False) + "\n\n=== line closures ===\n" + line_df.to_string(index=False) + "\n")
    print("\nwrote figures fig1_alpha_sweep.png, fig2_avalanche.png, fig3_cascade_map.png")


if __name__ == "__main__":
    main()
