"""Compare five strategies: python -m transperth.strategy_example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from transperth.config import PROJECT_ROOT
from transperth.experiments import RunMeta, save_table
from transperth.multilayer import (
    add_backup_edges,
    build_layers,
    served_od_fraction,
    terminal_pairs,
    travel_times,
)
from transperth.strategies import (
    CorridorReinforcement,
    DemandAdaptive,
    ExistingBus,
    NoBackup,
    ShuttleBridging,
)


def main() -> None:
    """Write an evidenced synthetic comparison using identical failures/budgets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data/examples/recovery_fixture.json",
    )
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/recovery"
    )
    args = parser.parse_args()
    fixture = json.loads(args.input.read_text(encoding="utf-8"))
    rail = nx.Graph()
    rail.add_nodes_from(sorted(fixture["nodes"]))
    rail.add_edges_from(fixture["edges"])
    nx.set_edge_attributes(rail, 1000, "distance_m")
    nx.set_node_attributes(rail, "synthetic corridor", "lines")
    nx.set_node_attributes(rail, 10, "trips_served")
    candidates = pd.DataFrame(fixture["candidates"])
    demand = {(a, b): weight for a, b, weight in fixture["demand"]}
    intact = build_layers(rail)
    original_pairs = terminal_pairs(intact)
    failed = intact.copy()
    failed.remove_nodes_from("R:" + station for station in fixture["failed"])
    rows = []
    for strategy in [
        NoBackup(),
        ExistingBus(candidates),
        ShuttleBridging(candidates),
        CorridorReinforcement(candidates, rail=rail),
        DemandAdaptive(candidates),
    ]:
        pairs = strategy.deploy(
            failed,
            budget=args.budget,
            failed=fixture["failed"],
            demand=demand,
            seed=args.seed,
        )
        recovered = add_backup_edges(failed, strategy.selected_table(pairs))
        times = travel_times(recovered, pairs=original_pairs)
        served_weight = sum(
            weight
            for (a, b), weight in demand.items()
            if tuple(sorted(("T:" + a, "T:" + b))) in times
        )
        total_weight = sum(demand.values())
        rows.append(
            {
                "strategy": strategy.name,
                "budget": args.budget,
                "selected_count": len(pairs),
                "selected_pairs": json.dumps(pairs),
                "seed": args.seed,
                "served_od_fraction": served_od_fraction(
                    recovered, baseline_pairs=original_pairs
                ),
                "served_demand_fraction": served_weight / total_weight
                if total_weight
                else 1.0,
                "reachable_pair_count": len(times),
                "mean_reachable_minutes": sum(times.values()) / len(times)
                if times
                else None,
            }
        )
    metadata = RunMeta.create(
        "recovery",
        seed=args.seed,
        inputs=[args.input],
        params={
            "budget": args.budget,
            "budget_unit": "deployment links",
            "failed": fixture["failed"],
            "synthetic": True,
            "original_pair_count": len(original_pairs),
            "travel_time_unit": "minutes; mean over reachable original terminal pairs only",
        },
    )
    save_table(
        pd.DataFrame(rows), args.output_dir / "strategy_comparison.csv", metadata
    )
    print(f"Saved all five strategies to {args.output_dir}")


if __name__ == "__main__":
    main()
