"""Run a small seeded removal/metrics example: python -m transperth.metrics_example."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import networkx as nx
import numpy as np

from transperth.config import PROJECT_ROOT
from transperth.experiments import RunMeta, run_seeded, save_table, summarize_runs
from transperth.metrics import compute_metrics


def main() -> None:
    """Save repeat-run metrics and bootstrap summaries with input hashes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=PROJECT_ROOT / "data/examples/metrics_graph.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "results/metrics"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-runs", type=int, default=20)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    graph = nx.Graph()
    graph.add_nodes_from(sorted(data["nodes"]))
    graph.add_edges_from(data["edges"])
    if not graph:
        parser.error("example needs at least one node")
    baseline = compute_metrics(graph).efficiency

    def run(seed: int) -> dict:
        survivor = graph.copy()
        survivor.remove_node(np.random.default_rng(seed).choice(sorted(graph)))
        return asdict(
            compute_metrics(
                survivor, n_initial=len(graph), baseline_efficiency=baseline
            )
        )

    table = run_seeded(run, seed=args.seed, n_runs=args.n_runs)
    meta = RunMeta.create(
        "metrics",
        seed=args.seed,
        inputs=[args.input],
        params={
            "n_runs": args.n_runs,
            "n_initial": len(graph),
            "bootstrap_count": 1000,
            "confidence": 0.95,
        },
    )
    save_table(table, args.output_dir / "runs.csv", meta)
    summary = summarize_runs(
        table,
        ["gcc_fraction", "efficiency", "network_damage"],
        n_boot=1000,
        seed=args.seed,
    )
    save_table(summary, args.output_dir / "summary.csv", meta)
    print(f"Saved {len(table)} runs and bootstrap intervals to {args.output_dir}")


if __name__ == "__main__":
    main()
