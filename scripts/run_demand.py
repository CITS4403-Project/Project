"""Regenerate the demand-weighted load and capacity results (P2.4).

Usage (from the repository root; the Makefile exports ``PYTHONPATH=src``):

    make demand PYTHON=.venv/bin/python

or directly:

    PYTHONPATH=src .venv/bin/python scripts/run_demand.py

The runner loads the frozen rail graph through ``network.load_rail_graph`` and
writes four tables through :mod:`transperth.experiments`, each with its input
hashes and run parameters:

    results/demand/demand_rankings.csv
    results/demand/demand_ranking_summary.csv
    results/demand/demand_alpha_sweep.csv
    results/demand/demand_alpha_star.csv

and two figures:

    figures/fig_demand_rankings.png
    figures/fig_demand_alpha.png

Two scenarios share the frozen alpha grid, the max-load trigger, the
capacity-proportional static rule and the default tolerance:

topology
    betweenness loads with the frozen default ``K = L0``;
demand
    the AM-peak boardings proxy (``am_peak_stops``) with the frequency-scaled
    reference capacity from :func:`transperth.loads.demand_reference_capacities`,
    so ``C_i = (1 + alpha) * K_i``.

``rank_change`` in the ranking table is the topology rank minus the demand
rank, so a positive value means the station ranks higher under demand loads.
``alpha_star`` is the smallest grid tolerance at which the max-load trigger
stays contained, as defined in ``docs/model.md`` section 6.
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pandas as pd
from scipy import stats

from transperth.cascade import simulate_cascade
from transperth.config import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_SEED,
    FIGURES_DIR,
    RAIL_EDGES_CSV,
    STATIONS_CSV,
    CascadeConfig,
)
from transperth.experiments import RunMeta, results_dir, save_table
from transperth.failure import rank_targets
from transperth.loads import demand_reference_capacities, initial_loads
from transperth.network import load_rail_graph
from transperth.plotting import apply_style, save_figure

import matplotlib.pyplot as plt  # noqa: E402 - import after plotting sets Agg

TOP_K: tuple[int, ...] = (5, 10, 20)
CONTAINED_TOLERANCE = 1e-9
SCENARIOS: tuple[str, ...] = ("topology", "demand")
SCENARIO_LABELS = {
    "topology": "topology loads, K = L0",
    "demand": "demand loads, frequency K",
}
SCENARIO_COLORS = {"topology": "steelblue", "demand": "darkorange"}


def _rank_map(order: list[str]) -> dict[str, int]:
    """Return ``{station: 1-based rank}`` from a ranked station list."""
    return {station: index + 1 for index, station in enumerate(order)}


def ranking_table(graph: nx.Graph) -> pd.DataFrame:
    """Compare the topology and demand rankings station by station."""
    topology = initial_loads(graph, "betweenness")
    demand = initial_loads(graph, "demand")
    topology_rank = _rank_map(rank_targets(graph, "betweenness"))
    demand_rank = _rank_map(rank_targets(graph, "flow"))
    rows = []
    for station in sorted(graph):
        rows.append(
            {
                "station_id": station,
                "name": graph.nodes[station].get("name", station),
                "betweenness_load": topology[station],
                "betweenness_rank": topology_rank[station],
                "demand_load": demand[station],
                "demand_rank": demand_rank[station],
                "rank_change": topology_rank[station] - demand_rank[station],
            }
        )
    return pd.DataFrame(rows)


def ranking_summary(rankings: pd.DataFrame) -> pd.DataFrame:
    """Return the rank correlation and top-k overlap of the two rankings."""
    correlation = stats.spearmanr(
        rankings["betweenness_load"], rankings["demand_load"]
    )
    rows = [
        {"metric": "n_stations", "value": float(len(rankings))},
        {"metric": "spearman_rho", "value": float(correlation.statistic)},
        {"metric": "spearman_p", "value": float(correlation.pvalue)},
    ]
    for k in TOP_K:
        topology_top = set(
            rankings.nsmallest(k, "betweenness_rank")["station_id"]
        )
        demand_top = set(rankings.nsmallest(k, "demand_rank")["station_id"])
        overlap = len(topology_top & demand_top)
        rows.append({"metric": f"top_{k}_overlap", "value": float(overlap)})
        rows.append(
            {"metric": f"top_{k}_overlap_fraction", "value": overlap / k}
        )
    return pd.DataFrame(rows)


def _alpha_star(sweep: pd.DataFrame, scenario: str) -> float:
    """Smallest contained grid tolerance of one scenario, NaN when none is."""
    part = sweep[(sweep["scenario"] == scenario) & sweep["contained"]]
    if part.empty:
        return float("nan")
    return float(part["alpha"].min())


def alpha_comparison(
    graph: nx.Graph,
    *,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep the frozen alpha grid for the topology and demand scenarios."""
    reference = demand_reference_capacities(graph)
    n_initial = graph.number_of_nodes()
    scenario_inputs = [
        (
            "topology",
            CascadeConfig(
                trigger="load", load_mode="betweenness", rule="capacity", seed=seed
            ),
            None,
        ),
        (
            "demand",
            CascadeConfig(
                trigger="load", load_mode="demand", rule="capacity", seed=seed
            ),
            reference,
        ),
    ]
    rows = []
    for scenario, config, capacities in scenario_inputs:
        for alpha in DEFAULT_ALPHA_GRID:
            result = simulate_cascade(
                graph,
                replace(config, alpha=alpha),
                reference_capacities=capacities,
            )
            contained = (
                result.failed_fraction <= 1.0 / n_initial + CONTAINED_TOLERANCE
            )
            rows.append(
                {
                    "scenario": scenario,
                    "alpha": alpha,
                    "n_failed": result.n_failed,
                    "failed_fraction": result.failed_fraction,
                    "gcc_fraction": result.gcc_fraction,
                    "rounds": result.rounds,
                    "contained": contained,
                }
            )
    sweep = pd.DataFrame(rows)
    star_rows = []
    for scenario, config, capacities in scenario_inputs:
        star_rows.append(
            {
                "scenario": scenario,
                "alpha_star": _alpha_star(sweep, scenario),
                "trigger": config.trigger,
                "rule": config.rule,
                "dynamic": config.dynamic,
                "load_mode": config.load_mode,
                "reference_capacity": (
                    "L0" if capacities is None else "frequency_scaled"
                ),
                "n_initial": n_initial,
            }
        )
    return sweep, pd.DataFrame(star_rows)


def _ranking_figure(
    rankings: pd.DataFrame,
    rho: float,
    *,
    figures_dir: Path | str = FIGURES_DIR,
) -> Path:
    """Scatter the two load scales and the two rank orders."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.5, 4.3))
    left.scatter(
        rankings["betweenness_load"],
        rankings["demand_load"],
        s=16,
        color="steelblue",
        edgecolors="black",
        linewidths=0.3,
    )
    left.set_xlabel("betweenness load (topology)")
    left.set_ylabel("AM-peak stops (demand proxy)")
    left.set_title(f"load scale (Spearman rho = {rho:.3f})", fontsize=9)

    n_stations = len(rankings)
    right.plot([1, n_stations], [1, n_stations], "--", lw=1.0, color="0.6")
    right.scatter(
        rankings["betweenness_rank"],
        rankings["demand_rank"],
        s=16,
        color="darkorange",
        edgecolors="black",
        linewidths=0.3,
    )
    right.set_xlabel("betweenness rank (topology)")
    right.set_ylabel("demand rank")
    right.set_title("rank order (dashed line: unchanged)", fontsize=9)
    fig.tight_layout()
    return save_figure(fig, "fig_demand_rankings.png", figures_dir=figures_dir)


def _alpha_figure(
    sweep: pd.DataFrame,
    star: pd.DataFrame,
    *,
    figures_dir: Path | str = FIGURES_DIR,
) -> Path:
    """Plot the failed fraction against tolerance for both scenarios."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    star_by_scenario = dict(zip(star["scenario"], star["alpha_star"]))
    for scenario in SCENARIOS:
        part = sweep[sweep["scenario"] == scenario]
        estimate = star_by_scenario.get(scenario, float("nan"))
        label = SCENARIO_LABELS[scenario]
        if math.isfinite(estimate):
            label = rf"{label} ($\alpha^*$ = {estimate:g})"
        ax.plot(
            part["alpha"],
            part["failed_fraction"] * 100.0,
            marker="o",
            ms=2.5,
            lw=1.2,
            color=SCENARIO_COLORS[scenario],
            label=label,
        )
        if math.isfinite(estimate):
            ax.axvline(
                estimate,
                color=SCENARIO_COLORS[scenario],
                ls=":",
                lw=1.1,
            )
    ax.set_xlabel(r"tolerance $\alpha$")
    ax.set_ylabel("failed stations (%)")
    ax.set_title("max-load trigger, static capacity rule", fontsize=9)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "fig_demand_alpha.png", figures_dir=figures_dir)


def run(
    *,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, object]:
    """Build all P2.4 tables and figures and return the frames and paths."""
    graph = load_rail_graph()
    reference = demand_reference_capacities(graph)
    demand_loads = initial_loads(graph, "demand")
    frequency = {
        node: float(graph.nodes[node]["trips_served"]) for node in graph
    }
    active = [node for node in graph if frequency[node] > 0.0]
    scale = max(
        (demand_loads[node] / frequency[node] for node in active), default=0.0
    )

    rankings = ranking_table(graph)
    summary = ranking_summary(rankings)
    sweep, star = alpha_comparison(graph, seed=seed)
    rho = float(
        summary.loc[summary["metric"] == "spearman_rho", "value"].iloc[0]
    )

    meta = RunMeta.create(
        "demand",
        seed=seed,
        params={
            "demand_load": "am_peak_stops",
            "reference_capacity": "frequency-scaled trips_served",
            "capacity_scale": scale,
            "capacity_law": "C = (1 + alpha) * K",
            "default_reference_capacity": "K = L0",
            "trigger": "load",
            "rule": "capacity",
            "dynamic": False,
            "alpha_grid": [min(DEFAULT_ALPHA_GRID), max(DEFAULT_ALPHA_GRID)],
            "alpha_step": DEFAULT_ALPHA_GRID[1] - DEFAULT_ALPHA_GRID[0],
            "top_k": list(TOP_K),
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
        },
        inputs=[STATIONS_CSV, RAIL_EDGES_CSV],
    )
    target = results_dir("demand") if output_dir is None else Path(output_dir)
    figures_target = FIGURES_DIR if figures_dir is None else Path(figures_dir)
    table_paths = [
        save_table(rankings, target / "demand_rankings.csv", meta),
        save_table(summary, target / "demand_ranking_summary.csv", meta),
        save_table(sweep, target / "demand_alpha_sweep.csv", meta),
        save_table(star, target / "demand_alpha_star.csv", meta),
    ]
    figure_paths = [
        _ranking_figure(rankings, rho, figures_dir=figures_target),
        _alpha_figure(sweep, star, figures_dir=figures_target),
    ]
    return {
        "graph": graph,
        "rankings": rankings,
        "summary": summary,
        "alpha_sweep": sweep,
        "alpha_star": star,
        "reference_capacity": reference,
        "capacity_scale": scale,
        "tables": table_paths,
        "figures": figure_paths,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, help="default: results/demand")
    parser.add_argument("--figures-dir", type=Path, help="default: figures/")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)

    started = time.perf_counter()
    outcome = run(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        seed=args.seed,
    )
    graph = outcome["graph"]
    summary = outcome["summary"]
    star = outcome["alpha_star"]
    print(
        f"frozen graph: {graph.number_of_nodes()} stations, "
        f"{graph.number_of_edges()} edges"
    )
    print(
        "capacity calibration: scale = "
        f"{outcome['capacity_scale']:.6f} * trips_served "
        "(K_i >= L0_i on the frozen graph)"
    )
    print(summary.to_string(index=False, float_format="%.4g"))
    star_by_scenario = dict(zip(star["scenario"], star["alpha_star"]))
    topology_star = star_by_scenario.get("topology", float("nan"))
    demand_star = star_by_scenario.get("demand", float("nan"))
    if math.isfinite(topology_star) and math.isfinite(demand_star):
        print(
            f"alpha* topology = {topology_star:g}, demand = {demand_star:g} "
            f"(shift {demand_star - topology_star:+g})"
        )
    else:
        print(f"alpha* topology = {topology_star}, demand = {demand_star}")
    print(f"elapsed: {time.perf_counter() - started:.2f} s")
    for csv_path, meta_path in outcome["tables"]:
        print(f"wrote {csv_path} and {meta_path.name}")
    for figure_path in outcome["figures"]:
        print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())