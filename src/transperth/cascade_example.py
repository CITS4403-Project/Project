"""Run one frozen-rail cascade: python -m transperth.cascade_example."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from transperth.cascade import simulate_cascade
from transperth.config import PROJECT_ROOT, RAIL_EDGES_CSV, STATIONS_CSV, CascadeConfig
from transperth.experiments import RunMeta, save_table


def main() -> None:
    """Load via P1.1 and save one fully parameterised cascade result."""
    from transperth.network import load_rail_graph

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--rule", choices=["equal", "capacity"], default="capacity")
    parser.add_argument(
        "--load-mode",
        choices=["betweenness", "betweenness_freq", "betweenness_plus_trips"],
        default="betweenness",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/cascade"
    )
    args = parser.parse_args()
    config = CascadeConfig(
        alpha=args.alpha,
        seed=args.seed,
        target=args.target,
        dynamic=args.dynamic,
        rule=args.rule,
        load_mode=args.load_mode,
    )
    result = simulate_cascade(load_rail_graph(), config)
    metadata = RunMeta.create(
        "cascade",
        seed=args.seed,
        params=asdict(config),
        inputs=[STATIONS_CSV, RAIL_EDGES_CSV],
    )
    save_table(
        pd.DataFrame([result.to_row()]), args.output_dir / "single_run.csv", metadata
    )
    print(
        f"{result.n_failed} failed stations in {result.rounds} cascade rounds; saved to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
