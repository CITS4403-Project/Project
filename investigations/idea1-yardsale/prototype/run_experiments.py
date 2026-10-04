"""Experiment driver for the yard-sale prototype (CITS4403 idea-1 investigation).

Runs all experiments, writes raw results to ../results/*.csv, a machine-readable
summary to ../results/summary.json and figures to ../figures/*.png.

Usage (from the worktree root):
    .venv/bin/python investigations/idea1-yardsale/prototype/run_experiments.py
    .venv/bin/python investigations/idea1-yardsale/prototype/run_experiments.py --quick
    .venv/bin/python investigations/idea1-yardsale/prototype/run_experiments.py --only topology
    .venv/bin/python investigations/idea1-yardsale/prototype/run_experiments.py --selftest

All runs are seeded; ensemble statistics are means +/- std over seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from yardsale import TOPOLOGIES, simulate, selftest  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INV = os.path.dirname(HERE)
RESULTS = os.path.join(INV, "results")
FIGURES = os.path.join(INV, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

SUMMARY: dict = {}


# --------------------------------------------------------------------------
def _series_df(name, configs, seeds, **sim_kwargs):
    """Run one simulation per (config, seed); return long-format DataFrame."""
    rows = []
    for cfg in configs:
        for seed in range(seeds):
            r = simulate(seed=seed, **cfg, **sim_kwargs)
            for k, t in enumerate(r["times"]):
                rows.append(
                    {
                        **cfg,
                        "seed": seed,
                        "t": int(t),
                        "gini": r["gini"][k],
                        "top1": r["top1"][k],
                        "top10": r["top10"][k],
                        "max_share": r["max_share"][k],
                        "t_cond": r["t_cond"],
                        "wall_s": r["wall_seconds"],
                    }
                )
    return pd.DataFrame(rows)


def _group(df, keys, cols=("gini", "top1", "top10", "max_share")):
    g = df.groupby(["t"] + keys)[list(cols)]
    mean = g.mean().add_suffix("_mean").reset_index()
    std = g.std().add_suffix("_std").reset_index()
    return mean.merge(std, on=["t"] + keys)


def _final_stats(df, keys, horizon):
    """Mean +/- std of Gini and max share at the last sampled time per config."""
    last = df[df.t == df.t.max()]
    g = last.groupby(keys)[["gini", "max_share", "top10"]].agg(["mean", "std"])
    return g


def _cond_stats(df, keys, horizon):
    """Fraction of seeds that condensed by `horizon` (t_cond > 0)."""
    one = df.drop_duplicates(subset=[c for c in df.columns if c not in ("t", "gini", "top1", "top10", "max_share", "wall_s")])
    sub = one.copy()
    sub["condensed"] = sub["t_cond"].notna()
    out = sub.groupby(keys)["condensed"].mean()
    return out


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------
def expt_bias_time(quick=False):
    """(a) inequality vs time, unbiased vs biased (mean field)."""
    seeds = 5 if quick else 8
    sweeps = 1500 if quick else 3000
    configs = [{"n": 400, "f": 0.2, "p": p, "topology": "complete"} for p in (0.5, 0.55, 0.6, 0.7, 0.9)]
    df = _series_df("bias_time", configs, seeds, sweeps=sweeps, sample_every=10)
    df.to_csv(os.path.join(RESULTS, "bias_time.csv.gz"), index=False, compression="gzip")
    g = _group(df, ["p"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for p, sub in g.groupby("p"):
        axes[0].plot(sub.t, sub.gini_mean, label=f"p={p}")
        axes[0].fill_between(sub.t, sub.gini_mean - sub.gini_std, sub.gini_mean + sub.gini_std, alpha=0.15)
        axes[1].plot(sub.t, sub.max_share_mean, label=f"p={p}")
        axes[1].fill_between(
            sub.t, sub.max_share_mean - sub.max_share_std, sub.max_share_mean + sub.max_share_std, alpha=0.15
        )
    axes[0].set(xlabel="sweeps", ylabel="Gini coefficient", title="Inequality vs time (mean field, f=0.2)")
    axes[1].set(xlabel="sweeps", ylabel="richest agent's wealth share", title="Condensation onset")
    axes[1].axhline(0.5, color="k", lw=0.6, ls="--")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig1_inequality_vs_time.png"))
    plt.close(fig)

    SUMMARY["bias_time"] = {}
    for p, sub in df.groupby("p"):
        last = sub[sub.t == sub.t.max()]
        SUMMARY["bias_time"][str(p)] = {
            "gini_final_mean": float(last.gini.mean()),
            "gini_final_std": float(last.gini.std()),
            "max_share_final_mean": float(last.max_share.mean()),
        }
    return df


def expt_p_scan(quick=False):
    """(b) bias scan: condensation time and early growth rate vs p."""
    seeds = 5 if quick else 8
    sweeps = 2500 if quick else 5000
    ps = (0.50, 0.51, 0.52, 0.55, 0.60, 0.70, 0.85, 0.95)
    rows = []
    for p in ps:
        for seed in range(seeds):
            r = simulate(n=400, f=0.2, p=p, topology="complete", sweeps=sweeps, seed=seed,
                         sample_every=5, stop_on_cond=True)
            # early exponential growth rate of the richest agent's share
            ms = r["max_share"]
            tt = r["times"]
            mask = (ms > 0.01) & (ms < 0.15) & (tt > 0)
            lam = float(np.nan)
            if mask.sum() >= 3:
                lam = float(np.polyfit(tt[mask], np.log(ms[mask]), 1)[0])
            rows.append(
                {
                    "p": p,
                    "seed": seed,
                    "t_cond": r["t_cond"],
                    "censored": r["t_cond"] is None,
                    "lambda": lam,
                    "final_gini": r["final_gini"],
                    "final_max_share": r["final_max_share"],
                    "wall_s": r["wall_seconds"],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "p_scan.csv.gz"), index=False, compression="gzip")

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for p, sub in df.groupby("p"):
        cond = sub.t_cond.dropna()
        if len(cond):
            axes[0].errorbar(p, cond.mean(), yerr=cond.std(), fmt="o", capsize=3, color="C0")
        else:
            axes[0].plot(p, sweeps, "x", color="C3")
    axes[0].set(xlabel="bias p (richer wins w.p. p)", ylabel="sweeps to 50% share",
                title="Condensation time vs bias", yscale="log")
    for p, sub in df.groupby("p"):
        lam = sub["lambda"].dropna()
        if len(lam):
            axes[1].errorbar(p, lam.mean(), yerr=lam.std(), fmt="o", capsize=3, color="C1")
    axes[1].set(xlabel="bias p", ylabel="early growth rate $\\lambda$ of max share",
                title="Early exponential growth of the oligarch")
    axes[0].annotate("x = censored\n(no condensation in 5000 sweeps)", xy=(0.5, sweeps * 0.75),
                     fontsize=7, ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_bias_scan.png"))
    plt.close(fig)

    SUMMARY["p_scan"] = {}
    for p, sub in df.groupby("p"):
        cond = sub.t_cond.dropna()
        SUMMARY["p_scan"][str(p)] = {
            "t_cond_mean": float(cond.mean()) if len(cond) else None,
            "t_cond_std": float(cond.std()) if len(cond) else None,
            "condensed_fraction": float((sub.t_cond.notna()).mean()),
            "lambda_mean": float(sub["lambda"].dropna().mean()) if sub["lambda"].notna().any() else None,
        }
    return df


def expt_topology(quick=False):
    """(c) topology comparison at fixed parameters."""
    seeds = 5 if quick else 8
    sweeps = 1000 if quick else 1500
    configs = [{"n": 400, "f": 0.2, "p": 0.6, "topology": t, "k": 4} for t in TOPOLOGIES]
    df = _series_df("topology", configs, seeds, sweeps=sweeps, sample_every=5)
    df.to_csv(os.path.join(RESULTS, "topology.csv.gz"), index=False, compression="gzip")
    g = _group(df, ["topology"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for topo in TOPOLOGIES:
        sub = g[g.topology == topo]
        axes[0].plot(sub.t, sub.max_share_mean, label=topo)
        axes[0].fill_between(sub.t, sub.max_share_mean - sub.max_share_std,
                             sub.max_share_mean + sub.max_share_std, alpha=0.15)
    axes[0].set(xlabel="sweeps", ylabel="richest agent's share",
                title="Wealth condensation by topology (f=0.2, p=0.6, $\\langle k\\rangle\\approx4$)")
    axes[0].legend(fontsize=7)

    stat = df[df.t == df.t.max()].groupby("topology")["gini"].agg(["mean", "std"]).reindex(TOPOLOGIES)
    axes[1].bar(range(len(stat)), stat["mean"], yerr=stat["std"], capsize=3, color="C2")
    axes[1].set_xticks(range(len(stat)))
    axes[1].set_xticklabels(stat.index, rotation=30, ha="right")
    axes[1].set(ylabel=f"Gini at t={sweeps}", title="Inequality at fixed horizon")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig3_topology.png"))
    plt.close(fig)

    SUMMARY["topology"] = {}
    for topo, sub in df.groupby("topology"):
        last = sub[sub.t == sub.t.max()]
        cond = sub.drop_duplicates(["seed"]).t_cond.dropna()
        SUMMARY["topology"][topo] = {
            "gini_final_mean": float(last.gini.mean()),
            "gini_final_std": float(last.gini.std()),
            "max_share_final_mean": float(last.max_share.mean()),
            "t_cond_mean": float(cond.mean()) if len(cond) else None,
            "condensed_fraction": float(sub.drop_duplicates(["seed"]).t_cond.notna().mean()),
        }
    return df


def expt_f_scan(quick=False):
    """(d) transaction-size scan and diffusive time rescaling."""
    seeds = 4 if quick else 6
    sweeps = 1500 if quick else 3000
    fs = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
    configs = [{"n": 400, "f": f, "p": 0.6, "topology": "complete"} for f in fs]
    df = _series_df("f_scan", configs, seeds, sweeps=sweeps, sample_every=10)
    df.to_csv(os.path.join(RESULTS, "f_scan.csv.gz"), index=False, compression="gzip")
    g = _group(df, ["f"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for f, sub in g.groupby("f"):
        axes[0].plot(sub.t, sub.max_share_mean, label=f"f={f}")
        axes[1].plot(sub.t * f**2, sub.max_share_mean, label=f"f={f}")
    axes[0].set(xlabel="sweeps", ylabel="richest agent's share", title="Condensation speed vs transaction size")
    axes[1].set(xlabel="$f^2 \\cdot$ sweeps (diffusive scaling)", ylabel="richest agent's share",
                title="Same curves vs rescaled time", xscale="log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig4_f_scan.png"))
    plt.close(fig)

    SUMMARY["f_scan"] = {}
    for f, sub in df.groupby("f"):
        one = sub.drop_duplicates(["seed"])
        cond = one.t_cond.dropna()
        SUMMARY["f_scan"][str(f)] = {
            "t_cond_mean": float(cond.mean()) if len(cond) else None,
            "condensed_fraction": float(one.t_cond.notna().mean()),
            "gini_final_mean": float(sub[sub.t == sub.t.max()].gini.mean()),
        }
    return df


def expt_tax(quick=False):
    """(e) proportional wealth tax with equal redistribution."""
    seeds = 4 if quick else 6
    sweeps = 3000 if quick else 8000
    taus = (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1)
    configs = [
        {"n": 300, "f": 0.2, "p": p, "tax": tau, "tax_mode": "wealth", "topology": "complete"}
        for p in (0.5, 0.6)
        for tau in taus
    ]
    df = _series_df("tax", configs, seeds, sweeps=sweeps, sample_every=20)
    df.to_csv(os.path.join(RESULTS, "tax.csv.gz"), index=False, compression="gzip")
    g = _group(df, ["p", "tax"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for p, sub in g.groupby("p"):
        last = sub[sub.t == sub.t.max()]
        axes[0].errorbar(last.tax, last.gini_mean, yerr=last.gini_std, fmt="o-", capsize=3, label=f"p={p}")
        axes[1].errorbar(last.tax, last.max_share_mean, yerr=last.max_share_std, fmt="o-", capsize=3, label=f"p={p}")
    for ax, ylab in ((axes[0], "Gini at horizon"), (axes[1], "richest agent's share at horizon")):
        ax.set(xlabel="wealth tax rate $\\tau$ per sweep", ylabel=ylab, xscale="symlog",
               title="Wealth tax vs inequality")
    axes[1].axhline(0.5, color="k", lw=0.6, ls="--")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig5_wealth_tax.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for tau in (0.0, 0.001, 0.01, 0.1):
        sub = g[(g.p == 0.6) & (g["tax"] == tau)]
        ax.plot(sub.t, sub.gini_mean, label=f"$\\tau$={tau}")
        ax.fill_between(sub.t, sub.gini_mean - sub.gini_std, sub.gini_mean + sub.gini_std, alpha=0.15)
    ax.set(xlabel="sweeps", ylabel="Gini", title="Biased model (p=0.6): Gini vs time for tax rates")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig5b_gini_time_tax.png"))
    plt.close(fig)

    SUMMARY["tax"] = {}
    for (p, tau), sub in df.groupby(["p", "tax"]):
        one = sub.drop_duplicates(["seed"])
        cond = one.t_cond.dropna()
        SUMMARY["tax"][f"p{p}_tau{tau:g}"] = {
            "gini_final_mean": float(sub[sub.t == sub.t.max()].gini.mean()),
            "gini_final_std": float(sub[sub.t == sub.t.max()].gini.std()),
            "max_share_final_mean": float(sub[sub.t == sub.t.max()].max_share.mean()),
            "condensed_fraction": float(one.t_cond.notna().mean()),
            "t_cond_mean": float(cond.mean()) if len(cond) else None,
        }
    return df


def expt_tax_mode(quick=False):
    """(f) tax on trade gains vs tax on wealth (key comparison)."""
    seeds = 4 if quick else 8
    sweeps = 2000 if quick else 4000
    configs = [
        {"n": 300, "f": 0.2, "p": 0.5, "tax": tau, "tax_mode": mode, "topology": "complete"}
        for mode in ("wealth", "gain")
        for tau in (0.0, 0.001, 0.01, 0.05, 0.2)
    ]
    df = _series_df("tax_mode", configs, seeds, sweeps=sweeps, sample_every=10)
    df.to_csv(os.path.join(RESULTS, "tax_mode.csv.gz"), index=False, compression="gzip")
    g = _group(df, ["tax_mode", "tax"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
    for (mode, tau), sub in g.groupby(["tax_mode", "tax"]):
        axes[0 if mode == "wealth" else 1].plot(sub.t, sub.gini_mean, label=f"$\\tau$={tau}")
    axes[0].set(xlabel="sweeps", ylabel="Gini", title="Wealth tax (Boghosian-style)")
    axes[1].set(xlabel="sweeps", title="Tax on trade gains only")
    axes[1].legend(fontsize=7)
    fig.suptitle("Fair model (p=0.5, f=0.2): which tax stops condensation?")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    fig.savefig(os.path.join(FIGURES, "fig6_tax_mode.png"))
    plt.close(fig)

    SUMMARY["tax_mode"] = {}
    for (mode, tau), sub in df.groupby(["tax_mode", "tax"]):
        last = sub[sub.t == sub.t.max()]
        SUMMARY["tax_mode"][f"{mode}_tau{tau:g}"] = {
            "gini_final_mean": float(last.gini.mean()),
            "gini_final_std": float(last.gini.std()),
            "max_share_final_mean": float(last.max_share.mean()),
        }
    return df


def expt_rewiring(quick=False):
    """(g) dynamic network: random rewiring probability."""
    seeds = 4 if quick else 6
    sweeps = 1000 if quick else 1500
    rs = (0.0, 1e-3, 1e-2, 5e-2, 0.2)
    configs = [{"n": 400, "f": 0.2, "p": 0.6, "topology": "WS", "k": 4, "rewiring": r} for r in rs]
    df = _series_df("rewiring", configs, seeds, sweeps=sweeps, sample_every=5)
    df.to_csv(os.path.join(RESULTS, "rewiring.csv.gz"), index=False, compression="gzip")
    g = _group(df, ["rewiring"])

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for r, sub in g.groupby("rewiring"):
        axes[0].plot(sub.t, sub.max_share_mean, label=f"r={r:g}")
        axes[1].plot(sub.t, sub.gini_mean, label=f"r={r:g}")
    axes[0].set(xlabel="sweeps", ylabel="richest agent's share", title="Rewiring speeds up condensation (WS, p=0.6)")
    axes[1].set(xlabel="sweeps", ylabel="Gini", title="Gini vs time")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig7_rewiring.png"))
    plt.close(fig)

    SUMMARY["rewiring"] = {}
    for r, sub in df.groupby("rewiring"):
        one = sub.drop_duplicates(["seed"])
        cond = one.t_cond.dropna()
        SUMMARY["rewiring"][str(r)] = {
            "t_cond_mean": float(cond.mean()) if len(cond) else None,
            "condensed_fraction": float(one.t_cond.notna().mean()),
            "gini_final_mean": float(sub[sub.t == sub.t.max()].gini.mean()),
        }
    return df


def expt_finite_size(quick=False):
    """(h) finite-size scaling of the condensation time at p=0.5 and p=0.55."""
    seeds = 6 if quick else 12
    sweeps = 10000 if quick else 30000
    ns = (50, 100, 200, 400)
    rows = []
    for p in (0.5, 0.55):
        for n in ns:
            for seed in range(seeds):
                r = simulate(n=n, f=0.3, p=p, topology="complete", sweeps=sweeps, seed=seed,
                             sample_every=100, stop_on_cond=True)
                rows.append({"p": p, "n": n, "seed": seed, "t_cond": r["t_cond"],
                             "final_gini": r["final_gini"], "wall_s": r["wall_seconds"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS, "finite_size.csv.gz"), index=False, compression="gzip")

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for p, sub in df.groupby("p"):
        stat = sub.groupby("n")["t_cond"].agg(["mean", "std", "count"])
        ax.errorbar(stat.index, stat["mean"], yerr=stat["std"], fmt="o-", capsize=3, label=f"p={p}")
        for n, row in stat.iterrows():
            if np.isnan(row["mean"]):
                ax.plot(n, sweeps, "x", color=plt.gca().lines[-1].get_color())
    ax.set(xlabel="number of agents N", ylabel="sweeps to 50% share", xscale="log", yscale="log",
           title="Fair model condensation is slow\nand N-dependent (f=0.3)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig8_finite_size.png"))
    plt.close(fig)

    SUMMARY["finite_size"] = {}
    for (p, n), sub in df.groupby(["p", "n"]):
        cond = sub.t_cond.dropna()
        SUMMARY["finite_size"][f"p{p}_N{n}"] = {
            "t_cond_mean": float(cond.mean()) if len(cond) else None,
            "t_cond_std": float(cond.std()) if len(cond) else None,
            "condensed_fraction": float(sub.t_cond.notna().mean()),
        }
    return df


def expt_distributions(quick=False):
    """(i) wealth distributions: Lorenz curves and rank-size tails."""
    from yardsale import YardSaleModel, gini, hill_alpha

    seeds = 2 if quick else 4
    sweeps = 1000 if quick else 2000
    n = 1000
    out = {}
    for p in (0.5, 0.7):
        ws = []
        for seed in range(seeds):
            m = YardSaleModel(n=n, f=0.2, p=p, topology="complete", seed=seed)
            for _ in range(sweeps):
                m.sweep()
            ws.append(np.sort(m.w)[::-1])
        out[p] = ws

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    for p, ws in out.items():
        for w in ws:
            frac = np.cumsum(np.sort(w)) / w.sum()
            axes[0].plot(np.linspace(0, 1, n), frac, lw=1, alpha=0.7, label=f"p={p}")
            axes[1].plot(np.arange(1, n + 1), w / w.sum(), lw=1, alpha=0.7, label=f"p={p}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="equality")
    axes[0].set(xlabel="cumulative population fraction", ylabel="cumulative wealth fraction",
                title="Lorenz curves after 2000 sweeps")
    axes[1].set(xlabel="rank", ylabel="wealth share", xscale="log", yscale="log",
                title=f"Rank-size (Zipf) plot, N={n}")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig9_distributions.png"))
    plt.close(fig)

    SUMMARY["distributions"] = {}
    for p, ws in out.items():
        SUMMARY["distributions"][str(p)] = {
            "gini_mean": float(np.mean([gini(w) for w in ws])),
            "hill_alpha_mean": float(np.nanmean([hill_alpha(w) for w in ws])),
            "top1_mean": float(np.mean([w[: max(1, n // 100)].sum() / w.sum() for w in ws])),
        }
    return out


EXPTS = {
    "bias_time": expt_bias_time,
    "p_scan": expt_p_scan,
    "topology": expt_topology,
    "f_scan": expt_f_scan,
    "tax": expt_tax,
    "tax_mode": expt_tax_mode,
    "rewiring": expt_rewiring,
    "finite_size": expt_finite_size,
    "distributions": expt_distributions,
}


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller ensembles for a fast smoke test")
    ap.add_argument("--only", nargs="*", default=None, help="subset of experiment names")
    ap.add_argument("--selftest", action="store_true", help="run sanity checks and exit")
    args = ap.parse_args()

    if args.selftest:
        checks = selftest()
        bad = [c for c in checks if not c[1]]
        sys.exit(1 if bad else 0)

    names = args.only or list(EXPTS)
    t_start = time.perf_counter()
    for name in names:
        t0 = time.perf_counter()
        print(f"--- {name} ---", flush=True)
        EXPTS[name](quick=args.quick)
        print(f"    done in {time.perf_counter() - t0:.1f}s", flush=True)
    SUMMARY["_meta"] = {"total_wall_seconds": time.perf_counter() - t_start, "quick": args.quick}
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(SUMMARY, fh, indent=2, sort_keys=True)
    print(f"\nAll experiments in {time.perf_counter() - t_start:.1f}s -> {RESULTS}")


if __name__ == "__main__":
    main()