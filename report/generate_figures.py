"""Build the report's selected figures and numeric macros from frozen results.

Run from the repository root: make figures
No experiment rerun or raw GTFS archive is needed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from transperth.experiments import file_sha256, load_meta
from transperth.plotting import (
    apply_style,
    plot_bus_service_mechanism,
    plot_robustness_cascade,
    plot_uncertainty_avalanche,
    save_figure,
)


def build_report_figures(root: Path = ROOT) -> None:
    report = root / "report"
    output = report / "figures"
    output.mkdir(exist_ok=True)
    sources = {}

    def read(relative: str):
        path = root / relative
        sidecar = path.with_suffix(".meta.json")
        meta = load_meta(sidecar)
        for name, digest in meta["inputs"].items():
            if (
                name.startswith("data/processed/")
                and file_sha256(root / name) != digest
            ):
                raise ValueError(f"Frozen graph differs from {relative}: {name}")
        family = path.parent / "experiment_manifest.json"
        if family.exists():
            expected = json.loads(family.read_text())["artifact_sha256"][path.name]
            if file_sha256(path) != expected:
                raise ValueError(f"Changed result: {relative}")
        sources[relative] = file_sha256(path)
        sources[str(sidecar.relative_to(root)).replace("\\", "/")] = file_sha256(
            sidecar
        )
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        return pd.read_csv(path)

    critical = read("results/percolation/percolation_critical_fractions.csv").set_index(
        "slug"
    )
    removal = {
        name: read(f"results/percolation/percolation_{name}.csv")
        for name in ["random_degree", "targeted_degree", "targeted_betweenness"]
    }
    runs = read("results/cascade/runs.csv")
    ci = read("results/uncertainty/cascade_ci.csv")
    scan = read("results/recovery/bus_cascade_scan.csv")
    loads = read("results/recovery/bus_cascade_loads.csv")
    strategies = read("results/recovery/strategy_comparison.csv")
    avalanche = read("results/cascade/avalanche_samples.csv")
    recommendations = read("results/uncertainty/recommendations.json")
    fits = read("results/uncertainty/power_law_fits.json")
    demand = read("results/demand/demand_alpha_star.csv").set_index("scenario")

    # Report scenario slices: Bayswater closure and the Fremantle first load check.
    bay = (
        scan[(scan.trigger == 23) & (scan.alpha == 0.2)]
        .set_index("scenario")
        .loc[["rail_only", "manual_bus", "candidate_bus"]]
    )
    ratios = (
        loads[
            (loads.trigger == 23)
            & (loads.alpha == 0.2)
            & (loads["round"] == 0)
            & (loads.station_id == 87)
        ]
        .set_index("scenario")
        .loc[bay.index]
    )
    served = 1 - bay.unmet_fraction

    apply_style()
    plt.rcParams.update({"font.size": 9, "legend.fontsize": 7.5})
    products = []

    def save(fig, name):
        products.append(save_figure(fig, name, figures_dir=output, dpi=220))

    save(plot_robustness_cascade(removal, runs, ci), "robustness_cascade.png")
    save(plot_bus_service_mechanism(bay, ratios), "bus_service_mechanism.png")
    save(
        plot_uncertainty_avalanche(recommendations, avalanche),
        "uncertainty_avalanche.png",
    )

    airport = strategies[
        (strategies.scenario == "line_closure:Airport Line") & (strategies.budget == 2)
    ].set_index("strategy")
    maxload = runs[
        (runs.alpha == 0.2) & (runs.rule == "capacity") & (runs.trigger == "load")
    ].iloc[0]
    fit = next(f for f in fits["fits"] if not f["dynamic"] and f["alpha"] == 0.3)
    coarse = next(
        e
        for e in recommendations["alpha_grid_evidence"]
        if e["rule"] == "capacity" and e["step"] == 0.1
    )
    fine = next(
        e
        for e in recommendations["alpha_grid_evidence"]
        if e["rule"] == "capacity" and e["step"] == 0.025
    )
    equal = next(
        e
        for e in recommendations["alpha_grid_evidence"]
        if e["rule"] == "equal" and e["step"] == 0.025
    )
    evidence = recommendations["seed_candidates"]
    thousand = next(e for e in evidence if e["n"] == 1000)
    numbers = {
        "RandomCrossPercent": 100
        * critical.loc["random_degree", "gcc_threshold_fraction"],
        "DegreeCrossPercent": 100
        * critical.loc["targeted_degree", "gcc_threshold_fraction"],
        "BetweenCrossPercent": 100
        * critical.loc["targeted_betweenness", "gcc_threshold_fraction"],
        "CapacityAlphaStar": fine["alpha_star"],
        "EqualAlphaStar": equal["alpha_star"],
        "CoarseAlphaStar": coarse["alpha_star"],
        "DemandAlphaStar": demand.loc["demand", "alpha_star"],
        "MaxLoadFailed": int(maxload.n_failed),
        "MaxLoadRounds": int(maxload.rounds),
        "BayRailServicePercent": 100 * served.loc["rail_only"],
        "BayBusServicePercent": 100 * served.loc["candidate_bus"],
        "FremantleRailRatio": ratios.loc["rail_only", "load_ratio"],
        "FremantleBusRatio": ratios.loc["candidate_bus", "load_ratio"],
        "AirportRailServicePercent": 100
        * airport.loc["no_backup", "served_od_fraction"],
        "AirportBusServicePercent": 100
        * airport.loc["existing_bus", "served_od_fraction"],
        "AirportRailDemandPercent": 100
        * airport.loc["no_backup", "served_demand_fraction"],
        "AirportBusDemandPercent": 100
        * airport.loc["existing_bus", "served_demand_fraction"],
        "RecommendedSeeds": recommendations["recommended_random_seeds"],
        "WorstHalfWidth": thousand["worst_ci_half_width"],
        "WorstMeanChange": thousand["worst_mean_change"],
        "TailPValue": fit["p_value"],
    }
    precision = {
        "RandomCrossPercent": 2,
        "DegreeCrossPercent": 2,
        "BetweenCrossPercent": 2,
        "CapacityAlphaStar": 3,
        "EqualAlphaStar": 3,
        "CoarseAlphaStar": 3,
        "DemandAlphaStar": 3,
        "MaxLoadFailed": 0,
        "MaxLoadRounds": 0,
        "RecommendedSeeds": 0,
        "WorstHalfWidth": 4,
        "WorstMeanChange": 4,
        "TailPValue": 3,
    }
    macros = [
        "% Generated by report/generate_figures.py from frozen results; do not edit."
    ]
    macros.extend(
        f"\\newcommand{{\\{key}}}{{{value:.{precision.get(key, 2)}f}}}"
        for key, value in numbers.items()
    )
    values = report / "sections" / "00-results-values.tex"
    values.write_text("\n".join(macros) + "\n", encoding="utf-8", newline="\n")
    products.append(values)
    detail = output / "report_values.json"
    detail.write_text(
        json.dumps(numbers, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    products.append(detail)
    sources["report/generate_figures.py"] = file_sha256(Path(__file__))
    sources["src/transperth/plotting.py"] = file_sha256(
        ROOT / "src/transperth/plotting.py"
    )
    artifact = {
        str(p.relative_to(root)).replace("\\", "/"): file_sha256(p) for p in products
    }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "inputs": sources,
                "artifacts": artifact,
                "selection": {
                    "robustness_cascade.png": "RQ1 removal and RQ2 tolerance; 95% intervals for random cascade means",
                    "bus_service_mechanism.png": "RQ3 Bayswater example: service and facility load ratio",
                    "uncertainty_avalanche.png": "RQ4 seed-count resolution and finite avalanche support",
                },
                "assumptions": "Fixed 86-station graph; algorithmic rounds; scheduled-stop proxy, not passengers.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        "Built 3 report figures, evidence manifest and numeric macros from frozen tables."
    )


if __name__ == "__main__":
    build_report_figures()
