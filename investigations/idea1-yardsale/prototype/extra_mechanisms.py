"""Two focused mechanism experiments for the investigation.

(1) Tax mechanism (fig10): why a tax on trade gains cannot stop condensation
    while a proportional wealth tax can.  We record, per sweep,
      * revenue rate = tax collected / total wealth,
      * Gini and the richest agent's share,
    for both tax modes at the same nominal rate (p=0.5, f=0.2, N=400).
    Prediction: wealth-tax revenue stays ~tau, gain-tax revenue decays as
    trades start to involve pauperised agents (min(w_i,w_j) -> 0).

(2) Rewiring mechanism (fig11): random rewiring accelerates condensation.
    We test whether this is a topology effect (hub creation) or a
    partner-access effect, by recording the degree distribution and the number
    of distinct trading partners met by the agent that ends up richest.

Run:  .venv/bin/python investigations/idea1-yardsale/prototype/extra_mechanisms.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yardsale import YardSaleModel, gini, max_share  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.dirname(HERE)
RESULTS = os.path.join(INV, "results")
FIGURES = os.path.join(INV, "figures")

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})


def tax_mechanism(sweeps=12000, seeds=4, tau=0.1):
    rows = []
    for mode in ("wealth", "gain"):
        for seed in range(seeds):
            m = YardSaleModel(n=400, f=0.2, p=0.5, tax=tau, tax_mode=mode,
                              topology="complete", seed=seed)
            prev = 0.0
            W = m.total_wealth
            for t in range(1, sweeps + 1):
                m.sweep()
                rev = (m.pool_collected - prev) / W
                prev = m.pool_collected
                if t % 20 == 0 or t == sweeps:
                    rows.append({"mode": mode, "seed": seed, "t": t, "revenue_rate": rev,
                                 "gini": gini(m.w), "max_share": max_share(m.w)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "tax_mechanism.csv.gz"), index=False, compression="gzip")

    g = df.groupby(["mode", "t"])[["gini", "max_share", "revenue_rate"]].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    for mode, sub in g.groupby("mode"):
        axes[0].plot(sub.t, sub.gini, label=mode)
        axes[1].plot(sub.t, sub.max_share, label=mode)
        axes[2].plot(sub.t, sub.revenue_rate, label=mode)
    axes[0].set(xlabel="sweeps", ylabel="Gini", title="p=0.5, tau=0.1: inequality vs time")
    axes[1].set(xlabel="sweeps", ylabel="richest agent's share", title="Condensation continues under a gain tax")
    axes[2].set(xlabel="sweeps", ylabel="tax revenue / total wealth per sweep",
                title="Revenue: fixed tau$\\,W$ vs decaying tax base", yscale="log")
    for ax in axes:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig10_tax_mechanism.png"))
    plt.close(fig)

    out = {}
    for mode, sub in g.groupby("mode"):
        out[mode] = {
            "gini_t1000": float(sub[sub.t == 1000].gini.iloc[0]),
            "gini_tend": float(sub[sub.t == sub.t.max()].gini.iloc[0]),
            "max_tend": float(sub[sub.t == sub.t.max()].max_share.iloc[0]),
            "revenue_t1000": float(sub[sub.t == 1000].revenue_rate.iloc[0]),
            "revenue_tend": float(sub[sub.t == sub.t.max()].revenue_rate.iloc[0]),
        }
    return out


def rewiring_mechanism(sweeps=1500, seeds=4, rs=(0.0, 0.05, 0.2)):
    out = {}
    ws_rows = []
    for r in rs:
        for seed in range(seeds):
            m = YardSaleModel(n=400, f=0.2, p=0.6, topology="WS", k=4, rewiring=r, seed=seed)
            pairs = []

            orig = m._trade_batch

            def wrapped(u, v, _orig=orig, _pairs=pairs):
                if len(_pairs) < 4_000_000:
                    _pairs.append((u.copy(), v.copy()))
                _orig(u, v)

            m._trade_batch = wrapped
            for _ in range(sweeps):
                m.sweep()
            rich = int(np.argmax(m.w))
            if pairs:
                allu = np.concatenate([p[0] for p in pairs])
                allv = np.concatenate([p[1] for p in pairs])
                partners = np.unique(np.concatenate([allu[allv == rich], allv[allu == rich]]))
                n_partners = int(partners.size)
            else:
                n_partners = 0
            deg = np.array([d for _, d in m.graph.degree()]) if m.graph is not None else None
            ws_rows.append({"r": r, "seed": seed, "max_share": max_share(m.w), "gini": gini(m.w),
                            "rich_degree": int(m.adj[rich].__len__()), "rich_partners": n_partners,
                            "max_degree": int(deg.max()) if deg is not None else None,
                            "mean_degree": float(deg.mean()) if deg is not None else None})
    df = pd.DataFrame(ws_rows)
    df.to_csv(os.path.join(RESULTS, "rewiring_mechanism.csv.gz"), index=False, compression="gzip")

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
    keys = ["max_share", "rich_partners", "max_degree"]
    titles = ["richest agent's share", "distinct partners met by that agent", "max node degree"]
    for ax, key, title in zip(axes, keys, titles):
        stat = df.groupby("r")[key].agg(["mean", "std"])
        ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], fmt="o-", capsize=3)
        ax.set(xlabel="rewiring rate r", ylabel=title, xscale="symlog")
    axes[2].axhline(df.mean_degree.mean(), color="k", ls="--", lw=0.8, label="mean degree")
    axes[2].legend(fontsize=8)
    fig.suptitle("Rewiring does not create hubs - it multiplies trading partners")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(os.path.join(FIGURES, "fig11_rewiring_mechanism.png"))
    plt.close(fig)

    for r, sub in df.groupby("r"):
        out[str(r)] = {
            "max_share_mean": float(sub.max_share.mean()),
            "rich_partners_mean": float(sub.rich_partners.mean()),
            "max_degree_mean": float(sub.max_degree.mean()),
            "mean_degree_mean": float(sub.mean_degree.mean()),
        }
    return out


def main():
    summary = {"tax_mechanism": tax_mechanism(), "rewiring_mechanism": rewiring_mechanism()}
    path = os.path.join(RESULTS, "summary_extra.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()