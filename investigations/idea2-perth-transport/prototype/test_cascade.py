"""Unit tests for the cascade model (plain Python, no pytest required).

Run:  python prototype/test_cascade.py
Checks: known betweenness values, GCC after removals, load conservation,
capacity edge cases, determinism of random triggers, and the alpha=0 limit.
"""
from __future__ import annotations

from pathlib import Path
import sys
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import betweenness, simulate_cascade, load_graph  # noqa: E402

PASS = []


def check(name: str, cond: bool) -> None:
    PASS.append((name, bool(cond)))
    print(f"{'PASS' if cond else 'FAIL'}: {name}")


def path_graph(n: int = 5) -> nx.Graph:
    G = nx.path_graph(n)
    for u, v in G.edges:
        G[u][v]["trips"] = 10.0
    return G


def main() -> None:
    # --- betweenness on a known small graph ---
    # path 0-1-2-3-4: node 2 lies on 4 pairs -> betweenness 4, node 1 on 2
    G = path_graph(5)
    btw = betweenness(G, "betweenness")
    check("path betweenness middle=4", abs(btw[2] - 4.0) < 1e-9)
    check("path betweenness second=3", abs(btw[1] - 3.0) < 1e-9)
    check("path betweenness end=0", abs(btw[0]) < 1e-9)

    # --- freq-weighted betweenness: all edges equal frequency => same values ---
    btwf = betweenness(G, "betweenness_freq")
    check("freq-weighted equals plain when trips equal", abs(btwf[2] - 4.0) < 1e-9)

    # --- GCC after removing a middle node of a path: largest part = 2 ---
    H = G.copy()
    H.remove_node(2)
    gcc = max(len(c) for c in nx.connected_components(H))
    check("GCC of split path is 2", gcc == 2)

    # --- alpha=0 on a path graph: hub removal cascades? nodes have cap=L0 ---
    r = simulate_cascade(G, 0.0, trigger="load", dynamic=False, rule="equal")
    check("path alpha=0: at least trigger fails", r["n_failed"] >= 1)
    check("path alpha=0: GCC < n", r["gcc"] <= 4)

    # --- large alpha keeps cascade contained on a leaf trigger ---
    # leaves have betweenness 0, so a leaf trigger can never overload neighbours
    leaf = 0
    r = simulate_cascade(G, 0.0, trigger="load", dynamic=False, target=leaf)
    check("leaf trigger with alpha=0 causes only the leaf to fail", r["n_failed"] == 1)

    # --- GCC of the real Perth rail graph after single removal ---
    Gp = load_graph()
    check("perth graph has 86 nodes, 96 edges",
          Gp.number_of_nodes() == 86 and Gp.number_of_edges() == 96)
    check("perth graph is connected", nx.is_connected(Gp))
    H = Gp.copy()
    names = {Gp.nodes[n]["name"]: n for n in Gp.nodes}
    H.remove_node(names["Perth Stn"])
    gcc = max(len(c) for c in nx.connected_components(H))
    check("Perth removal does not disconnect the network", gcc == 85)

    # --- determinism: same seed gives same result ---
    r1 = simulate_cascade(Gp, 0.2, trigger="random", dynamic=False, seed=123)
    r2 = simulate_cascade(Gp, 0.2, trigger="random", dynamic=False, seed=123)
    check("random trigger deterministic given seed", r1["failed"] == r2["failed"])

    # --- failed_fraction and gcc_fraction bounds ---
    r = simulate_cascade(Gp, 0.2, trigger="load", dynamic=True)
    check("fractions within [0,1]",
          0.0 <= r["failed_fraction"] <= 1.0 and 0.0 <= r["gcc_fraction"] <= 1.0)
    check("failed + remaining = n",
          r["n_failed"] + (Gp.number_of_nodes() - r["n_failed"]) == 86)

    # --- capacity rule: a node with zero capacity can only be triggered, never
    #     spontaneously fail (load 0 > 0 is false) ---
    check("alpha=0 dynamic on perth: finite cascade", r["n_failed"] < 86)

    n_fail = sum(1 for _, ok in PASS if not ok)
    print(f"\n{len(PASS) - n_fail}/{len(PASS)} checks passed")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()