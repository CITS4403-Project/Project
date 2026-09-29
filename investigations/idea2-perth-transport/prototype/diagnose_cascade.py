"""Diagnostics: trace targeted-load cascades; check zero-capacity node artefact."""
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade import load_graph, betweenness, simulate_cascade

G = load_graph()
L0 = betweenness(G, "betweenness")
print("top loads:")
for n, v in sorted(L0.items(), key=lambda kv: -kv[1])[:8]:
    print(f"  {G.nodes[n]['name']:28s} L0={v:8.1f} deg={G.degree(n)} cap(a=0.2)={1.2*v:8.1f}")
print("nodes with L0 == 0:", sum(v == 0 for v in L0.values()))

for alpha in (0.0, 0.1, 0.2, 0.4, 0.6, 1.0):
    r = simulate_cascade(G, alpha, trigger="load", dynamic=True)
    names = [G.nodes[n]["name"] for n in r["failed"]]
    print(f"\nalpha={alpha}: failed={r['n_failed']} gcc={r['gcc']} rounds={r['rounds']}")
    print("  sequence:", ", ".join(names[:20]))

# how much do loads change after removing the top hub?
first = max(G.nodes, key=lambda n: L0[n])
H = G.copy(); H.remove_node(first)
L1 = betweenness(H, "betweenness")
df = pd.DataFrame({
    "name": [G.nodes[n]["name"] for n in G.nodes if n != first],
    "L0": [L0[n] for n in G.nodes if n != first],
    "L_after": [L1[n] for n in G.nodes if n != first],
})
df["ratio"] = df["L_after"] / df["L0"].replace(0, np.nan)
print("\nloads INCREASED after removing", G.nodes[first]["name"], ":",
      (df["ratio"] > 1).sum(), "of", len(df))
print(df.sort_values("ratio", ascending=False).head(10).to_string(index=False))