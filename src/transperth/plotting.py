"""Shared figure style and plotting helpers, plus the ``make figures`` entry point.

This module is owned by P0.1. Experiment owners add their plot functions
through reviewed pull requests so every figure in the report shares one style.
Figures are always rendered headless and saved under ``figures/``.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # figures are always rendered without a display

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd

from transperth.config import FIGURES_DIR, RAIL_EDGES_CSV, STATIONS_CSV

__all__ = [
    "FIGURE_DPI",
    "STYLE",
    "apply_style",
    "main",
    "plot_bus_service_mechanism",
    "plot_network",
    "plot_robustness_cascade",
    "plot_series",
    "plot_uncertainty_avalanche",
    "save_figure",
]

FIGURE_DPI = 150
STYLE = {
    "figure.dpi": FIGURE_DPI,
    "savefig.dpi": FIGURE_DPI,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def apply_style() -> None:
    """Apply the shared matplotlib style."""
    plt.rcParams.update(STYLE)


def save_figure(
    fig,
    name: str,
    *,
    figures_dir: str | Path = FIGURES_DIR,
    dpi: int = FIGURE_DPI,
) -> Path:
    """Save ``fig`` under ``figures_dir`` and return the written path."""
    path = Path(figures_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_series(table: pd.DataFrame, x: str, y: str, *, ax=None, label: str | None = None, **style):
    """Plot one column pair of a result table and return the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=style.pop("figsize", (6.0, 4.0)))
    ax.plot(table[x], table[y], label=label, **style)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if label is not None:
        ax.legend(fontsize=8)
    return ax


def plot_network(
    stations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    ax=None,
    lat: str = "lat",
    lon: str = "lon",
    size_by: str = "trips_served",
):
    """Draw the rail graph from the frozen processed tables and return the axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6.0, 7.0))
    nodes = stations.set_index("station_id")
    for row in edges.itertuples(index=False):
        start, end = nodes.loc[row.station_a], nodes.loc[row.station_b]
        ax.plot([start[lon], end[lon]], [start[lat], end[lat]], "-", color="0.75", lw=0.8, zorder=1)
    if size_by in nodes:
        weights = nodes[size_by].astype(float)
        sizes = 12.0 + 40.0 * weights / max(weights.max(), 1.0)
    else:
        sizes = 18.0
    ax.scatter(
        nodes[lon],
        nodes[lat],
        s=sizes,
        color="steelblue",
        edgecolors="black",
        linewidths=0.3,
        zorder=2,
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def plot_robustness_cascade(removal, runs, ci):
    """Report RQ1/RQ2: removal curves and static secondary failure with intervals.

    ``removal`` maps curve names to their per-seed result tables, ``runs`` holds
    the deterministic triggers and ``ci`` the random-trigger confidence rows.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    for name, label in [
        ("random_degree", "Random (100 trials)"),
        ("targeted_degree", "Fixed degree rank"),
        ("targeted_betweenness", "Fixed betweenness rank"),
    ]:
        curve = removal[name].groupby("fraction").gcc_fraction.mean()
        axes[0].plot(curve.index, curve.values, label=label)
    axes[0].axhline(0.5, color="0.5", ls=":", lw=1)
    axes[0].set(
        xlim=(0, 0.2),
        ylim=(0, 1.02),
        xlabel="Removed station fraction",
        ylabel="GCC / 86",
        title="(a) Structural fragmentation",
    )
    axes[0].legend()
    for rule in ["capacity", "equal"]:
        rows = runs[(runs.rule == rule) & (runs.trigger == "load")].sort_values("alpha")
        axes[1].plot(rows.alpha, rows.failed_fraction, label=f"Max load: {rule}")
    random = ci[(ci.rule == "capacity") & (ci.metric == "failed_fraction")].sort_values(
        "alpha"
    )
    axes[1].plot(
        random.alpha, random["mean"], "--", color="C2", label="Random capacity mean"
    )
    axes[1].fill_between(random.alpha, random.low, random.high, color="C2", alpha=0.18)
    axes[1].set(
        xlim=(0, 0.8),
        ylim=(0, 1.02),
        xlabel="Capacity tolerance alpha",
        ylabel="Failed stations / 86",
        title="(b) Static secondary failure",
    )
    axes[1].legend()
    fig.tight_layout()
    return fig


def plot_bus_service_mechanism(bay, ratios):
    """Report RQ3: Bayswater service fraction and Fremantle load ratio.

    ``bay`` and ``ratios`` are the scenario-indexed slices prepared from the
    frozen recovery tables.
    """
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    labels = ["Rail only", "4 manual\npairs", "Full bus\npool"]
    colours = ["#64748b", "#08916b", "#2563eb"]
    served = 1 - bay.unmet_fraction
    bars = axes[0].bar(labels, served, color=colours)
    for bar, failures in zip(bars, bay.n_failed):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.025,
            f"{failures} rail failed",
            ha="center",
            fontsize=7.5,
        )
    axes[0].set(
        ylim=(0, 1.16),
        ylabel="Served terminal-pair fraction",
        title="(a) Bayswater closure, alpha=0.2",
    )
    axes[1].bar(labels, ratios.load_ratio, color=colours)
    axes[1].axhline(1, color="#dc2626", ls="--", label="Fixed-capacity limit")
    axes[1].set(
        ylim=(0, 1.55),
        ylabel="Load / fixed capacity",
        title="(b) Fremantle, first load check",
    )
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    return fig


def plot_uncertainty_avalanche(recommendations, avalanche):
    """Report RQ4: seed-count resolution and finite avalanche outcome masses."""
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0))
    evidence = recommendations["seed_candidates"]
    available = [e for e in evidence if e["all_condition_prefixes_available"]]
    axes[0].plot(
        [e["n"] for e in available],
        [e["worst_ci_half_width"] for e in available],
        "o-",
        label="Worst 95% CI half width",
    )
    axes[0].axhline(0.03, color="#dc2626", ls="--", label="Chosen error tolerance")
    axes[0].set(
        xlabel="Random-trigger trials",
        ylabel="Fraction scale",
        title="(a) Sampling resolution",
    )
    axes[0].legend()
    for alpha, group in avalanche[~avalanche.dynamic].groupby("alpha"):
        counts = group.post_trigger_size.value_counts().sort_index()
        axes[1].scatter(
            counts.index, counts / len(group), s=16, label=f"alpha={alpha:g}"
        )
    axes[1].set(
        xlabel="Secondary failures S",
        ylabel="Probability mass",
        xlim=(-2, 87),
        ylim=(0, 1.05),
        title="(b) Finite avalanche outcomes",
    )
    axes[1].legend(loc="upper left")
    fig.tight_layout()
    return fig


def main(argv: Sequence[str] | None = None) -> int:
    """Regenerate the figures that need only the frozen processed data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    apply_style()
    if not STATIONS_CSV.is_file() or not RAIL_EDGES_CSV.is_file():
        print(f"no processed data in {STATIONS_CSV.parent}; run `make data` first")
        return 0
    stations = pd.read_csv(STATIONS_CSV, dtype={"station_id": str})
    edges = pd.read_csv(RAIL_EDGES_CSV, dtype={"station_a": str, "station_b": str})
    fig, ax = plt.subplots(figsize=(6.0, 7.0))
    plot_network(stations, edges, ax=ax)
    path = save_figure(fig, "network_baseline.png", figures_dir=args.figures_dir)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
