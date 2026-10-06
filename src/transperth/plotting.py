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
    "plot_network",
    "plot_series",
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