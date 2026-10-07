"""Regenerate the P2.3 multilayer recovery and bus-strategy results.

Usage (from the repository root; the Makefile exports ``PYTHONPATH=src``):

    make recovery PYTHON=.venv/bin/python

or directly:

    PYTHONPATH=src .venv/bin/python scripts/run_recovery.py --budget 2 --seed 0

The runner loads the frozen rail graph through ``network.load_rail_graph``,
builds the P1.5 layered model with the documented Perth/Perth Underground
walking interchange, and pools the four manual bus pairs with the GTFS-derived
candidates from ``scripts/run_backup_candidates.py``. The candidates are
regenerated from the frozen snapshot so the committed sidecars hash the
snapshot tables; when the snapshot is missing the table written by the P1.5
runner, ``results/multilayer/gtfs_backup_candidates.csv``, is read instead.
The runner then writes the P2.3 tables through ``transperth.experiments``,
each with input hashes and run parameters:

    results/recovery/scenario_manifest.csv
    results/recovery/strategy_comparison.csv
    results/recovery/bus_cascade_scan.csv
    results/recovery/bus_cascade_comparison.csv
    results/recovery/bus_cascade_controls.csv
    results/recovery/bus_cascade_history.csv
    results/recovery/bus_cascade_loads.csv

and the figures:

    figures/fig_recovery_strategies.png
    figures/fig_recovery_bus_cascade.png
    figures/fig_recovery_mechanism.png

The strategy comparison applies all five P1.6 strategies to the same scenario
closures and budgets, runs the ported layered cascade with the strategy's buses
active, and measures served OD, served demand, recovery rounds and the
travel-time penalty on the stable graph. The bus-cascade tables follow the
prototype layout: a rail-only run, the manual pairs as standby, and the full
candidate pool as standby, over the prototype's 0 to 2 tolerance grid.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from transperth.config import (
    BACKUP_EDGES_CSV,
    DEFAULT_ALPHA,
    DEFAULT_SEED,
    FIGURES_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAIL_EDGES_CSV,
    STATIONS_CSV,
)
from transperth.experiments import RunMeta, results_dir, save_table
from transperth.multilayer import (
    WALKING_TRANSFER_MINUTES,
    add_backup_edges,
    add_walking_transfers,
    build_layers,
    derive_gtfs_candidates,
    terminal_pairs,
    travel_times,
)
from transperth.network import load_rail_graph
from transperth.plotting import apply_style, save_figure
from transperth.recovery import (
    ENDPOINT_SUM_MODEL,
    endpoint_demand_weights,
    layered_cascade,
    recovery_metrics,
    scenario_manifest,
)
from transperth.strategies import (
    CorridorReinforcement,
    DemandAdaptive,
    ExistingBus,
    NoBackup,
    ShuttleBridging,
)

import matplotlib.pyplot as plt  # noqa: E402 - import after plotting sets Agg

DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots" / "2026-10-05_0700-0900"
DEFAULT_TOPOLOGY_PATH = PROJECT_ROOT / "data" / "verified_topology.json"
DEFAULT_CANDIDATES_PATH = PROJECT_ROOT / "results" / "multilayer" / "gtfs_backup_candidates.csv"
SNAPSHOT_INPUTS = ("stops.txt", "stop_times.txt", "bus_trips.csv")
DEFAULT_TRIGGERS: tuple[str, ...] = ("56", "23", "28")
DEFAULT_CASCADE_ALPHAS: tuple[float, ...] = tuple(
    round(0.05 * index, 8) for index in range(41)
)
DEFAULT_FOCUS_ALPHAS: tuple[float, ...] = (0.0, 0.2)
BUS_SCENARIOS: tuple[str, ...] = ("rail_only", "manual_bus", "candidate_bus")
SCENARIO_KINDS: tuple[str, ...] = ("line_closure", "interchange", "random")
STRATEGY_ORDER: tuple[str, ...] = (
    "no_backup",
    "existing_bus",
    "shuttle_bridging",
    "corridor_reinforcement",
    "demand_adaptive",
)
STRATEGY_COLUMNS = (
    "scenario",
    "kind",
    "label",
    "n_closure",
    "strategy",
    "budget",
    "alpha",
    "selected_count",
    "recovery_rounds",
    "secondary_failures",
    "failed_after_cascade",
    "total_pair_count",
    "reachable_pair_count",
    "served_od_fraction",
    "total_demand_weight",
    "served_demand_weight",
    "served_demand_fraction",
    "mean_travel_time_penalty_min",
    "mean_reachable_minutes",
)
SCAN_COLUMNS = (
    "trigger",
    "trigger_name",
    "alpha",
    "scenario",
    "n_failed",
    "n_secondary",
    "failed_fraction",
    "rounds",
    "failed_ids",
    "reachable_pairs",
    "original_pairs",
    "unmet_fraction",
    "mean_time_change_reachable_min",
)
COMPARISON_COLUMNS = (
    "trigger",
    "trigger_name",
    "alpha",
    "scenario",
    "delta_n_failed",
    "delta_unmet_fraction",
    "common_reachable_fraction",
    "bus_minus_rail_time_common_min",
)
CONTROL_COLUMNS = (
    "trigger",
    "trigger_name",
    "scenario",
    "n_failed",
    "n_secondary",
    "rounds",
    "failed_ids",
    "reachable_pairs",
    "original_pairs",
    "unmet_fraction",
    "mean_time_change_reachable_min",
)
HISTORY_COLUMNS = (
    "trigger",
    "trigger_name",
    "alpha",
    "scenario",
    "round",
    "n_failed",
    "n_secondary",
    "removed_ids",
    "next_failed_ids",
    "reachable_pairs",
    "original_pairs",
    "unmet_fraction",
    "mean_time_change_reachable_min",
)
LOAD_COLUMNS = (
    "trigger",
    "trigger_name",
    "alpha",
    "scenario",
    "round",
    "station_id",
    "name",
    "initial_load",
    "load",
    "capacity",
    "load_ratio",
    "overloaded",
    "closes_next_round",
)


# ---------------------------------------------------------------------------
# input assembly
# ---------------------------------------------------------------------------
def candidate_pool(derived: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Pool derived candidates with manual ground truth; manual pairs win.

    The two tables need the P1.5 columns ``station_a``, ``station_b``,
    ``minutes`` and ``kind``. Where a pair appears in both sources the manual
    row is kept and the derived row is dropped, so the ground-truth effective
    time is never silently replaced by a scheduled ride time. All other derived
    rows keep their evidence columns.
    """
    required = {"station_a", "station_b", "minutes", "kind"}
    for name, table in (("derived", derived), ("manual", manual)):
        missing = sorted(required - set(table.columns))
        if missing:
            raise ValueError(f"{name} table is missing required columns: {', '.join(missing)}")

    def keys(table: pd.DataFrame) -> list[tuple[str, str]]:
        return [
            tuple(sorted((str(a).strip(), str(b).strip())))
            for a, b in zip(table["station_a"], table["station_b"])
        ]

    manual_keys = set(keys(manual))
    keep = [key not in manual_keys for key in keys(derived)]
    filtered = derived.loc[keep]
    return pd.concat([manual, filtered], ignore_index=True)


def frozen_candidate_pool(
    *,
    data_dir: Path = PROCESSED_DATA_DIR,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
) -> tuple[pd.DataFrame, Path]:
    """Load the frozen candidate pool and return it with its evidence path.

    The manual pairs always come from ``data/processed/backup_edges.csv``. The
    derived table is regenerated from the frozen snapshot because that keeps
    the committed sidecar's SHA-256 provenance complete; the snapshot is
    deterministic and faster than a minute. When the snapshot is unavailable,
    the table from ``results/multilayer/gtfs_backup_candidates.csv`` written by
    ``scripts/run_backup_candidates.py`` is read instead, and its hash is
    recorded. The returned path is the snapshot directory or the CSV file.
    """
    manual = pd.read_csv(
        Path(data_dir) / "backup_edges.csv", dtype={"station_a": str, "station_b": str}
    )
    snapshot_files = [Path(snapshot_dir) / name for name in SNAPSHOT_INPUTS]
    if all(path.is_file() for path in snapshot_files):
        derived = derive_gtfs_candidates(
            snapshot_dir, stations_csv=Path(data_dir) / "stations.csv"
        )
        return candidate_pool(derived, manual), Path(snapshot_dir)
    source = Path(candidates_path)
    if source.is_file():
        derived = pd.read_csv(source, dtype={"station_a": str, "station_b": str})
        return candidate_pool(derived, manual), source
    raise FileNotFoundError(
        "no frozen GTFS snapshot tables and no candidate table: run make data "
        f"(expected {', '.join(str(path) for path in snapshot_files)})"
    )


def load_frozen_transfer(topology_path: Path = DEFAULT_TOPOLOGY_PATH) -> Mapping[str, object]:
    """Return the documented walking interchange from the topology metadata."""
    topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    return topology["walking_transfer"]


def _portable_meta(
    experiment: str,
    *,
    seed: int,
    params: Mapping[str, object],
    inputs: Iterable[str | Path],
) -> RunMeta:
    """Create a sidecar with repository-relative input paths for portability."""
    meta = RunMeta.create(experiment, seed=seed, params=params, inputs=inputs)
    portable: dict[str, str] = {}
    for filename, digest in meta.inputs.items():
        path = Path(filename)
        try:
            key = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            key = str(path)
        portable[key] = digest
    return replace(meta, inputs=portable)


# ---------------------------------------------------------------------------
# experiment families
# ---------------------------------------------------------------------------
def _strategies(
    pool: pd.DataFrame, rail: nx.Graph
) -> list[object]:
    """Build the five P1.6 strategies on one candidate pool."""
    return [
        NoBackup(),
        ExistingBus(pool),
        ShuttleBridging(pool),
        CorridorReinforcement(pool, rail=rail),
        DemandAdaptive(pool),
    ]


def strategy_comparison(
    *,
    rail: nx.Graph,
    pool: pd.DataFrame,
    manifest: pd.DataFrame,
    intact: nx.Graph,
    baseline_pairs: Sequence[tuple[str, str]],
    intact_times: Mapping[tuple[str, str], float],
    demand: Mapping[tuple[str, str], float],
    budgets: Sequence[int],
    alpha: float,
    seed: int,
) -> pd.DataFrame:
    """Apply every strategy to every scenario/budget and measure the recovery.

    The cascade starts from the scenario's closed stations with capacities
    fixed from the intact rail-only baseline, and the strategy's buses are
    active from the first post-closure round. Service metrics describe the
    stable graph, so ``recovery_rounds == 0`` means the closure was contained.
    """
    strategies = _strategies(pool, rail)
    rows = []
    for scenario in manifest.to_dict("records"):
        failed = str(scenario["failed_ids"]).split(";")
        failed_baseline = intact.copy()
        failed_baseline.remove_nodes_from("R:" + station for station in failed)
        for budget in budgets:
            for strategy in strategies:
                selected = strategy.deploy(
                    intact, budget=budget, failed=failed, demand=demand, seed=seed
                )
                if len(selected) > budget:
                    raise RuntimeError(
                        f"{strategy.name} returned {len(selected)} links for budget {budget}"
                    )
                recovered = add_backup_edges(
                    failed_baseline, strategy.selected_table(selected)
                )
                cascade = layered_cascade(
                    intact, recovered, alpha=alpha, initial_failed=failed
                )
                metrics = recovery_metrics(
                    cascade.final_graph,
                    intact,
                    baseline_pairs=baseline_pairs,
                    intact_times=intact_times,
                    demand=demand,
                )
                rows.append(
                    {
                        "scenario": scenario["scenario"],
                        "kind": scenario["kind"],
                        "label": scenario["label"],
                        "n_closure": scenario["n_failed"],
                        "strategy": strategy.name,
                        "budget": budget,
                        "alpha": alpha,
                        "selected_count": len(selected),
                        "recovery_rounds": cascade.rounds,
                        "secondary_failures": cascade.n_secondary,
                        "failed_after_cascade": ";".join(
                            node[2:] for node in cascade.failed
                        ),
                        **metrics,
                    }
                )
    frame = pd.DataFrame(rows, columns=list(STRATEGY_COLUMNS))
    return frame.sort_values(
        ["kind", "scenario", "budget", "strategy"], ignore_index=True
    )


def bus_cascade_family(
    *,
    rail: nx.Graph,
    pool: pd.DataFrame,
    manual: pd.DataFrame,
    intact: nx.Graph,
    baseline_pairs: Sequence[tuple[str, str]],
    intact_times: Mapping[tuple[str, str], float],
    triggers: Sequence[str],
    alphas: Sequence[float],
    focus_alphas: Sequence[float],
) -> dict[str, pd.DataFrame]:
    """Run the ported layered cascade over triggers, tolerances and bus sets.

    The three scenarios are rail-only, the four manual ground-truth pairs as
    standby, and the whole candidate pool as standby. The comparison rows are
    paired bus-minus-rail deltas; the control disables secondary closures; the
    history and loads tables cover the focus tolerances only.
    """
    scenarios = {
        "rail_only": intact,
        "manual_bus": add_backup_edges(intact, manual),
        "candidate_bus": add_backup_edges(intact, pool),
    }
    focus = set(float(alpha) for alpha in focus_alphas)
    scan_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []
    load_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []

    def measure(
        scenario: str, standby: nx.Graph, trigger: str, name: str, alpha: float
    ) -> tuple[dict[str, object], dict[tuple[str, str], float]]:
        """Run one cascade and return its scan row plus reachable pair times."""
        traced = float(alpha) in focus
        cascade = layered_cascade(
            intact,
            standby,
            alpha=alpha,
            initial_failed=(trigger,),
            trace=traced,
            baseline_pairs=baseline_pairs,
            baseline_times=intact_times,
        )
        times = travel_times(cascade.final_graph, pairs=baseline_pairs)
        changes = [
            times[pair] - intact_times[pair]
            for pair in times
            if pair in intact_times
        ]
        row: dict[str, object] = {
            "trigger": trigger,
            "trigger_name": name,
            "alpha": float(alpha),
            "scenario": scenario,
            "n_failed": cascade.n_failed,
            "n_secondary": cascade.n_secondary,
            "failed_fraction": cascade.failed_fraction,
            "rounds": cascade.rounds,
            "failed_ids": ";".join(node[2:] for node in cascade.failed),
            "reachable_pairs": len(times),
            "original_pairs": len(baseline_pairs),
            "unmet_fraction": 1.0 - len(times) / len(baseline_pairs)
            if baseline_pairs
            else 0.0,
            "mean_time_change_reachable_min": float(np.mean(changes))
            if changes
            else math.nan,
        }
        if traced and cascade.history is not None and cascade.loads is not None:
            tags = {
                "trigger": trigger,
                "trigger_name": name,
                "alpha": float(alpha),
                "scenario": scenario,
            }
            history_rows.extend(
                {**tags, **history_row}
                for history_row in cascade.history.to_dict("records")
            )
            load_rows.extend(
                {**tags, **load_row} for load_row in cascade.loads.to_dict("records")
            )
        return row, times

    for trigger in triggers:
        name = str(rail.nodes[trigger].get("name", trigger))
        for alpha in alphas:
            rail_row, rail_times = measure(
                "rail_only", intact, trigger, name, float(alpha)
            )
            scan_rows.append(rail_row)
            for scenario in BUS_SCENARIOS[1:]:
                bus_row, bus_times = measure(
                    scenario, scenarios[scenario], trigger, name, float(alpha)
                )
                scan_rows.append(bus_row)
                common = set(rail_times) & set(bus_times)
                differences = [bus_times[pair] - rail_times[pair] for pair in common]
                comparison_rows.append(
                    {
                        "trigger": trigger,
                        "trigger_name": name,
                        "alpha": float(alpha),
                        "scenario": scenario,
                        "delta_n_failed": bus_row["n_failed"] - rail_row["n_failed"],
                        "delta_unmet_fraction": bus_row["unmet_fraction"]
                        - rail_row["unmet_fraction"],
                        "common_reachable_fraction": len(common) / len(baseline_pairs)
                        if baseline_pairs
                        else 0.0,
                        "bus_minus_rail_time_common_min": float(np.mean(differences))
                        if differences
                        else math.nan,
                    }
                )
        for scenario, standby in scenarios.items():
            control = layered_cascade(
                intact,
                standby,
                alpha=0.0,
                initial_failed=(trigger,),
                overload_enabled=False,
            )
            times = travel_times(control.final_graph, pairs=baseline_pairs)
            changes = [
                times[pair] - intact_times[pair]
                for pair in times
                if pair in intact_times
            ]
            control_rows.append(
                {
                    "trigger": trigger,
                    "trigger_name": name,
                    "scenario": scenario,
                    "n_failed": control.n_failed,
                    "n_secondary": control.n_secondary,
                    "rounds": control.rounds,
                    "failed_ids": ";".join(node[2:] for node in control.failed),
                    "reachable_pairs": len(times),
                    "original_pairs": len(baseline_pairs),
                    "unmet_fraction": 1.0 - len(times) / len(baseline_pairs)
                    if baseline_pairs
                    else 0.0,
                    "mean_time_change_reachable_min": float(np.mean(changes))
                    if changes
                    else math.nan,
                }
            )

    return {
        "scan": pd.DataFrame(scan_rows, columns=list(SCAN_COLUMNS)),
        "comparison": pd.DataFrame(
            comparison_rows, columns=list(COMPARISON_COLUMNS)
        ),
        "controls": pd.DataFrame(control_rows, columns=list(CONTROL_COLUMNS)),
        "history": pd.DataFrame(history_rows, columns=list(HISTORY_COLUMNS)),
        "loads": pd.DataFrame(load_rows, columns=list(LOAD_COLUMNS)),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _bar_chart(axis, table: pd.DataFrame, column: str, *, label: str) -> None:
    """Grouped bars of one metric per strategy and scenario kind."""
    kinds = [
        kind for kind in SCENARIO_KINDS if kind in set(table["kind"])
    ]
    x = np.arange(len(kinds))
    width = 0.8 / max(len(STRATEGY_ORDER), 1)
    for index, strategy in enumerate(STRATEGY_ORDER):
        part = table[table["strategy"] == strategy]
        values = [
            float(part.loc[part["kind"] == kind, column].mean())
            if not part.loc[part["kind"] == kind].empty
            else math.nan
            for kind in kinds
        ]
        axis.bar(
            x + (index + 0.5) * width - 0.4,
            np.nan_to_num(values, nan=0.0),
            width,
            label=strategy.replace("_", " "),
        )
    axis.set_xticks(x, [kind.replace("_", " ") for kind in kinds])
    axis.set_ylabel(label)


def strategy_figure(
    comparison: pd.DataFrame, *, figures_dir: str | Path = FIGURES_DIR
) -> Path:
    """Plot the served-OD and travel-time comparison per strategy."""
    apply_style()
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    _bar_chart(left, comparison, "served_od_fraction", label="served OD fraction")
    left.set_ylim(0.0, 1.05)
    left.set_title("service restored on the stable graph", fontsize=9)
    _bar_chart(
        right,
        comparison,
        "mean_travel_time_penalty_min",
        label="mean travel-time penalty (min)",
    )
    right.set_title("penalty over reachable intact pairs", fontsize=9)
    right.axhline(0.0, color="0.4", lw=0.8, ls="--")
    handles, labels = left.get_legend_handles_labels()
    left.legend(handles, labels, fontsize=8, ncols=2)
    fig.tight_layout()
    return save_figure(fig, "fig_recovery_strategies.png", figures_dir=figures_dir)


def bus_cascade_figure(
    comparison: pd.DataFrame,
    *,
    triggers: Sequence[str],
    figures_dir: str | Path = FIGURES_DIR,
) -> Path:
    """Plot the bus-minus-rail damage and service deltas against tolerance."""
    apply_style()
    rows = len(triggers)
    fig, axes = plt.subplots(rows, 2, figsize=(10.0, 2.7 * rows), squeeze=False)
    for row, trigger in enumerate(triggers):
        part = comparison[comparison["trigger"] == trigger]
        name = str(part["trigger_name"].iloc[0]) if not part.empty else trigger
        for scenario in BUS_SCENARIOS[1:]:
            scenario_rows = part[part["scenario"] == scenario].sort_values("alpha")
            label = scenario.replace("_", " ")
            axes[row, 0].plot(
                scenario_rows["alpha"], scenario_rows["delta_n_failed"], label=label
            )
            axes[row, 1].plot(
                scenario_rows["alpha"],
                100.0 * scenario_rows["delta_unmet_fraction"],
                label=label,
            )
        axes[row, 0].axhline(0.0, color="0.4", lw=0.8, ls="--")
        axes[row, 1].axhline(0.0, color="0.4", lw=0.8, ls="--")
        axes[row, 0].set_title(f"{name}: extra failed facilities (bus - rail)", fontsize=9)
        axes[row, 1].set_title(
            f"{name}: service-loss change (pp, bus - rail)", fontsize=9
        )
        for column in range(2):
            axes[row, column].set_xlabel(r"tolerance $\alpha$")
        if row == 0:
            axes[row, 0].legend(fontsize=8)
    fig.tight_layout()
    return save_figure(fig, "fig_recovery_bus_cascade.png", figures_dir=figures_dir)


def mechanism_figure(
    comparison: pd.DataFrame,
    loads: pd.DataFrame,
    *,
    figures_dir: str | Path = FIGURES_DIR,
    top: int = 12,
) -> Path | None:
    """Show the load ratios of the strongest recorded bus-aggravated cascade.

    The case is the traced comparison row with the largest positive service
    loss; when buses never worsen service, the largest positive failure-count
    delta is used, because a bus-aggravated physical cascade can still improve
    connectivity. When every traced row has buses improving or matching the
    rail-only run on both measures, there is no mechanism to show and the
    function returns ``None``.
    """
    if loads.empty or comparison.empty:
        return None
    traced = comparison[
        comparison.apply(
            lambda row: not loads[
                (loads["trigger"] == row["trigger"])
                & (loads["alpha"] == row["alpha"])
                & (loads["scenario"] == row["scenario"])
            ].empty,
            axis=1,
        )
    ]
    if traced.empty:
        return None
    positive = traced[traced["delta_unmet_fraction"] > 1e-12]
    if positive.empty:
        positive = traced[traced["delta_n_failed"] > 0]
    if positive.empty:
        return None
    case = positive.loc[positive["delta_unmet_fraction"].idxmax()]
    part = loads[
        (loads["trigger"] == case["trigger"])
        & (loads["alpha"] == case["alpha"])
        & (loads["scenario"].isin(("rail_only", case["scenario"])))
    ]
    if part.empty:
        return None
    order = (
        part.groupby("station_id")["load_ratio"]
        .max()
        .sort_values(ascending=False)
        .head(top)
        .index
        .tolist()
    )
    names = part.drop_duplicates("station_id").set_index("station_id")["name"]
    apply_style()
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 0.32 * len(order) + 2.2), squeeze=False
    )
    for column, scenario in enumerate(("rail_only", case["scenario"])):
        case_rows = part[part["scenario"] == scenario]
        rounds = int(case_rows["round"].max()) if not case_rows.empty else 0
        grid = (
            case_rows.pivot(index="station_id", columns="round", values="load_ratio")
            .reindex(index=order, columns=range(rounds + 1))
        )
        image = axes[0, column].imshow(
            grid.to_numpy(),
            aspect="auto",
            vmin=0.0,
            vmax=2.0,
            cmap="YlOrRd",
            interpolation="nearest",
        )
        axes[0, column].set_xticks(range(rounds + 1))
        axes[0, column].set_yticks(
            range(len(order)),
            [str(names.get(station, station)).removesuffix(" Stn") for station in order],
        )
        axes[0, column].set_xlabel("closure round")
        axes[0, column].set_title(
            scenario.replace("_", " ") + f"  |  alpha={case['alpha']:g}", fontsize=9
        )
        axes[0, column].grid(False)
        for y_index in range(len(order)):
            for x_index in range(rounds + 1):
                value = grid.iloc[y_index, x_index]
                if np.isfinite(value) and value > 1.0 + 1e-12:
                    axes[0, column].text(
                        x_index, y_index, "x", ha="center", va="center", fontsize=8
                    )
    fig.colorbar(
        image,
        ax=axes[0, 1],
        pad=0.02,
        label="load / fixed rail-only capacity",
    )
    fig.suptitle(
        f"Mechanism: {case['trigger_name']} closure, "
        f"alpha={case['alpha']:g}, {case['scenario'].replace('_', ' ')}"
        f"\n(extra facility closures: {case['delta_n_failed']:+.0f}; "
        f"served-service change: {100.0 * case['delta_unmet_fraction']:+.1f} pp)",
        fontsize=10,
    )
    fig.tight_layout()
    return save_figure(fig, "fig_recovery_mechanism.png", figures_dir=figures_dir)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def run(
    *,
    rail: nx.Graph | None = None,
    candidates: pd.DataFrame | None = None,
    manual: pd.DataFrame | None = None,
    transfer: Mapping[str, object] | None = None,
    data_dir: Path = PROCESSED_DATA_DIR,
    topology_path: Path = DEFAULT_TOPOLOGY_PATH,
    candidates_path: Path = DEFAULT_CANDIDATES_PATH,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    budgets: Sequence[int] = (2,),
    alpha: float = DEFAULT_ALPHA,
    cascade_alphas: Sequence[float] | None = None,
    focus_alphas: Sequence[float] = DEFAULT_FOCUS_ALPHAS,
    triggers: Sequence[str] = DEFAULT_TRIGGERS,
    n_random: int = 5,
    random_size: int = 5,
    min_interchange_lines: int = 3,
    seed: int = DEFAULT_SEED,
    output_dir: Path | None = None,
    figures_dir: Path | None = None,
    inputs: Iterable[str | Path] | None = None,
) -> dict[str, object]:
    """Build all P2.3 tables and figures and return the frames and paths.

    ``rail``, ``candidates`` and ``transfer`` default to the frozen inputs.
    Tests pass synthetic versions; pass ``transfer={}`` to skip the walking
    interchange. ``inputs=None`` records the frozen input files in the sidecar
    when the rail graph was loaded internally, and no inputs otherwise.
    """
    frozen = rail is None
    evidence_inputs: list[Path] = []
    candidate_source = "provided"
    if rail is None:
        rail = load_rail_graph()
    if candidates is None:
        candidates, evidence = frozen_candidate_pool(
            data_dir=Path(data_dir),
            candidates_path=Path(candidates_path),
            snapshot_dir=Path(snapshot_dir),
        )
        if evidence.is_file():
            evidence_inputs.append(evidence)
            candidate_source = f"file:{evidence.name}"
        else:
            evidence_inputs.extend(Path(snapshot_dir) / name for name in SNAPSHOT_INPUTS)
            candidate_source = "frozen snapshot derivation"
    if manual is None:
        manual = (
            pd.read_csv(
                Path(data_dir) / "backup_edges.csv",
                dtype={"station_a": str, "station_b": str},
            )
            if frozen
            else candidates.head(0)
        )
    if transfer is None:
        transfer = load_frozen_transfer(topology_path)
    if inputs is None:
        inputs = (
            [STATIONS_CSV, RAIL_EDGES_CSV, BACKUP_EDGES_CSV, Path(topology_path)]
            + evidence_inputs
            if frozen
            else []
        )

    intact = build_layers(rail)
    if transfer:
        intact = add_walking_transfers(intact, transfer)
    baseline_pairs = terminal_pairs(intact)
    intact_times = travel_times(intact, pairs=baseline_pairs)
    station_demand = {
        str(station): float(rail.nodes[station].get("am_peak_stops", 0.0))
        for station in rail
    }
    demand = endpoint_demand_weights(baseline_pairs, station_demand)
    manifest = scenario_manifest(
        rail,
        n_random=n_random,
        random_size=random_size,
        min_interchange_lines=min_interchange_lines,
        seed=seed,
    )
    grid = tuple(
        float(value)
        for value in (DEFAULT_CASCADE_ALPHAS if cascade_alphas is None else cascade_alphas)
    )
    if not grid:
        raise ValueError("the cascade alpha grid cannot be empty")
    focus = tuple(sorted(set(float(value) for value in focus_alphas)))
    missing_focus = sorted(set(focus) - set(grid))
    if missing_focus:
        raise ValueError(f"focus alphas not in the cascade grid: {missing_focus}")
    trigger_ids = []
    for trigger in triggers:
        identifier = str(trigger).strip()
        if identifier not in rail:
            raise ValueError(f"unknown trigger station {identifier!r}")
        if identifier not in trigger_ids:
            trigger_ids.append(identifier)
    if not trigger_ids:
        raise ValueError("at least one trigger station is required")
    budget_values = tuple(int(budget) for budget in budgets)
    if not budget_values:
        raise ValueError("at least one deployment budget is required")

    strategy_table = strategy_comparison(
        rail=rail,
        pool=candidates,
        manifest=manifest,
        intact=intact,
        baseline_pairs=sorted(baseline_pairs),
        intact_times=intact_times,
        demand=demand,
        budgets=budget_values,
        alpha=alpha,
        seed=seed,
    )
    bus_tables = bus_cascade_family(
        rail=rail,
        pool=candidates,
        manual=manual,
        intact=intact,
        baseline_pairs=sorted(baseline_pairs),
        intact_times=intact_times,
        triggers=trigger_ids,
        alphas=grid,
        focus_alphas=focus,
    )

    params = {
        "model": (
            "P1.5 terminal/facility layers; terminal-subset weighted betweenness; "
            "synchronous irreversible rail closures; capacities fixed from the intact "
            "rail-only baseline"
        ),
        "capacity_law": "C = (1 + alpha) * L0",
        "walking_transfer_minutes": float(
            transfer.get("minutes", WALKING_TRANSFER_MINUTES)
        )
        if isinstance(transfer, Mapping)
        else WALKING_TRANSFER_MINUTES,
        "candidate_count": int(len(candidates)),
        "candidate_source": candidate_source,
        "budgets": list(budget_values),
        "alpha": float(alpha),
        "cascade_alphas": (
            [min(grid), max(grid), grid[1] - grid[0]] if len(grid) > 1 else list(grid)
        ),
        "focus_alphas": list(focus),
        "triggers": trigger_ids,
        "scenarios": int(len(manifest)),
        "scenario_kinds": {
            kind: int((manifest["kind"] == kind).sum()) for kind in SCENARIO_KINDS
        },
        "n_random": int(n_random),
        "random_size": int(random_size),
        "min_interchange_lines": int(min_interchange_lines),
        "demand_weight": (
            f"{ENDPOINT_SUM_MODEL}: sum of endpoint am_peak_stops "
            "(P2.4 boarding proxy)"
        ),
        "travel_time_unit": "minutes; penalties and changes are conditional on reachable pairs",
        "assumptions": [
            "manual bus pairs are ground truth and win duplicate candidate pairs",
            "GTFS candidates are single-trip co-occurrence evidence within 400 m",
            "standby buses activate immediately after the initial closures",
            "standby bus capacity is unlimited in this scenario model",
            "rounds are algorithmic updates, not elapsed minutes",
        ],
    }
    meta = _portable_meta("recovery", seed=seed, params=params, inputs=inputs)
    target = results_dir("recovery") if output_dir is None else Path(output_dir)
    figures_target = FIGURES_DIR if figures_dir is None else Path(figures_dir)
    table_paths = [
        save_table(manifest, target / "scenario_manifest.csv", meta),
        save_table(strategy_table, target / "strategy_comparison.csv", meta),
        save_table(bus_tables["scan"], target / "bus_cascade_scan.csv", meta),
        save_table(
            bus_tables["comparison"], target / "bus_cascade_comparison.csv", meta
        ),
        save_table(bus_tables["controls"], target / "bus_cascade_controls.csv", meta),
        save_table(bus_tables["history"], target / "bus_cascade_history.csv", meta),
        save_table(bus_tables["loads"], target / "bus_cascade_loads.csv", meta),
    ]
    figure_paths = [
        strategy_figure(strategy_table, figures_dir=figures_target),
        bus_cascade_figure(
            bus_tables["comparison"],
            triggers=trigger_ids,
            figures_dir=figures_target,
        ),
    ]
    mechanism = mechanism_figure(
        bus_tables["comparison"], bus_tables["loads"], figures_dir=figures_target
    )
    if mechanism is not None:
        figure_paths.append(mechanism)
    return {
        "rail": rail,
        "intact": intact,
        "pool": candidates,
        "manifest": manifest,
        "strategy_comparison": strategy_table,
        **bus_tables,
        "meta": meta,
        "tables": table_paths,
        "figures": figure_paths,
        "output_dir": target,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--budget",
        type=int,
        nargs="+",
        default=[2],
        help="deployment-link budgets (default: 2)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="tolerance of the strategy-comparison cascade (default: 0.2)",
    )
    parser.add_argument(
        "--cascade-alphas",
        type=float,
        nargs="+",
        help="bus-cascade tolerance grid (default: 0 to 2 step 0.05)",
    )
    parser.add_argument(
        "--focus-alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_FOCUS_ALPHAS),
        help="traced tolerances for the history/loads tables (default: 0 0.2)",
    )
    parser.add_argument(
        "--triggers",
        nargs="+",
        default=list(DEFAULT_TRIGGERS),
        help="bus-cascade trigger stations (default: 56 23 28)",
    )
    parser.add_argument("--n-random", type=int, default=5)
    parser.add_argument("--random-size", type=int, default=5)
    parser.add_argument("--min-interchange-lines", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES_PATH)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--output-dir", type=Path, help="default: results/recovery")
    parser.add_argument("--figures-dir", type=Path, help="default: figures/")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    outcome = run(
        data_dir=args.data_dir,
        topology_path=args.topology,
        candidates_path=args.candidates,
        snapshot_dir=args.snapshot_dir,
        budgets=tuple(args.budget),
        alpha=args.alpha,
        cascade_alphas=tuple(args.cascade_alphas) if args.cascade_alphas else None,
        focus_alphas=tuple(args.focus_alphas),
        triggers=tuple(args.triggers),
        n_random=args.n_random,
        random_size=args.random_size,
        min_interchange_lines=args.min_interchange_lines,
        seed=args.seed,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
    table = outcome["strategy_comparison"]
    scan = outcome["scan"]
    comparison = outcome["comparison"]
    print(
        f"scenarios: {len(outcome['manifest'])} "
        f"({outcome['manifest']['kind'].value_counts().to_dict()})"
    )
    print(f"candidates: {len(outcome['pool'])}")
    print(f"strategy rows: {len(table)}; budgets: {sorted(set(table['budget']))}")
    print(
        "budget exceeded: "
        f"{(table['selected_count'] > table['budget']).any()}"
    )
    focus_cases = comparison[
        comparison["alpha"].isin(args.focus_alphas)
        & (comparison["delta_unmet_fraction"] > 1e-12)
    ]
    damage_cases = comparison[
        comparison["alpha"].isin(args.focus_alphas)
        & (comparison["delta_n_failed"] > 0)
    ]
    print(
        f"bus-cascade rows: {len(scan)}; traced worse-service cases: "
        f"{len(focus_cases)}; traced bus-aggravated cascades: {len(damage_cases)}"
    )
    print(f"elapsed: {time.perf_counter() - started:.2f} s")
    for csv_path, meta_path in outcome["tables"]:
        print(f"wrote {csv_path} and {meta_path.name}")
    for figure_path in outcome["figures"]:
        print(f"wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())