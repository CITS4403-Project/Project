"""Write one percolation curve on the frozen rail graph.

The default sweep is a targeted node attack ranked by degree, with removal
fractions 0.00, 0.05, ..., 1.00 and 100 seeds per fraction (2,100 runs). The
curve is written with :func:`transperth.experiments.save_table`, so the CSV
gets a sidecar recording the seed, the run parameters, the package versions
and the SHA-256 hashes of ``stations.csv`` and ``rail_edges.csv``. A full
default sweep took 0.42 s on the development machine (Python 3.14, networkx
3.7); a static full sweep costs the same for any measure, while the
``--dynamic`` variant that recomputes a betweenness ranking after every
removal took about 61 s. ``--quick`` (six fractions, ten seeds) runs in
well under a second.

Usage (the Makefile exports ``PYTHONPATH=src``, direct runs need it):

    PYTHONPATH=src python scripts/run_percolation.py            # full sweep
    PYTHONPATH=src python scripts/run_percolation.py --quick    # quick sweep

Options: ``--attack`` (``random``, ``targeted``, ``random_edge``,
``targeted_edge``), ``--measure`` (``degree``, ``betweenness``, ``flow``),
``--fractions``, ``--n-seeds``, ``--seed``, ``--dynamic`` (recompute the
target ranking after every removal), ``--output``.

Results land in ``results/percolation/`` and figures in ``figures/``; both are
committed as report evidence, like the other Phase-2 experiment outputs.
"""
from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

import networkx as nx

from transperth.config import DEFAULT_SEED, RAIL_EDGES_CSV, STATIONS_CSV
from transperth.experiments import RunMeta, results_dir, save_table
from transperth.failure import (
    ATTACKS,
    EDGE_MEASURES,
    NODE_MEASURES,
    critical_fraction,
    percolation_curve,
)
from transperth.network import load_rail_graph

DEFAULT_FRACTIONS: tuple[float, ...] = tuple(round(0.05 * step, 2) for step in range(21))
QUICK_FRACTIONS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_SEEDS = 100
QUICK_SEEDS = 10


def load_frozen_graph() -> nx.Graph:
    """Load the frozen rail graph through the packaged P1.1 loader."""
    return load_rail_graph(stations_csv=STATIONS_CSV, edges_csv=RAIL_EDGES_CSV)


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attack", choices=ATTACKS, default="targeted")
    parser.add_argument("--measure", choices=sorted(set(NODE_MEASURES) | set(EDGE_MEASURES)), default="degree")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=None,
        help="removal fractions in [0, 1]; default 0.00 to 1.00 in steps of 0.05",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help=f"runs per fraction; default {DEFAULT_SEEDS} ({QUICK_SEEDS} with --quick)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="recompute the target ranking after every removal",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=f"quick sweep: fractions {QUICK_FRACTIONS} and {QUICK_SEEDS} seeds",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV path; default results/percolation/percolation_<attack>_<measure>.csv",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    fractions = (
        tuple(args.fractions)
        if args.fractions is not None
        else (QUICK_FRACTIONS if args.quick else DEFAULT_FRACTIONS)
    )
    n_seeds = (
        args.n_seeds
        if args.n_seeds is not None
        else (QUICK_SEEDS if args.quick else DEFAULT_SEEDS)
    )
    static = not args.dynamic

    graph = load_frozen_graph()
    print(f"frozen graph: {graph.number_of_nodes()} stations, {graph.number_of_edges()} edges")
    print(
        f"sweep: attack={args.attack} measure={args.measure} static={static} "
        f"fractions={len(fractions)} seeds={n_seeds} ({len(fractions) * n_seeds} runs)"
    )

    started = time.perf_counter()
    curve = percolation_curve(
        graph,
        attack=args.attack,
        measure=args.measure,
        fractions=fractions,
        n_seeds=n_seeds,
        seed=args.seed,
        static=static,
    )
    elapsed = time.perf_counter() - started
    estimate = critical_fraction(curve)

    output = args.output or (
        results_dir("percolation") / f"percolation_{args.attack}_{args.measure}.csv"
    )
    meta = RunMeta.create(
        "percolation",
        seed=args.seed,
        params={
            "attack": args.attack,
            "measure": args.measure,
            "static": static,
            "fractions": list(fractions),
            "n_seeds": n_seeds,
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
        },
        inputs=[STATIONS_CSV, RAIL_EDGES_CSV],
    )
    csv_path, meta_path = save_table(curve, output, meta)

    print(f"elapsed: {elapsed:.2f} s")
    print(f"critical fraction (threshold 0.5): {estimate}")
    print(f"wrote {csv_path} and {meta_path.name}")


if __name__ == "__main__":
    main()