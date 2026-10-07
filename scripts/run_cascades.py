"""Regenerate P2.2 numerical results and figures: PYTHONPATH=src python scripts/run_cascades.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from transperth.cascade_experiments import run_cascade_experiments
from transperth.config import PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small separate smoke-test outputs; never the report dataset",
    )
    parser.add_argument("--random-seeds", type=int)
    parser.add_argument("--avalanche-seeds", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args()
    suffix = "cascade_quick" if args.quick else "cascade"
    output = args.output_dir or PROJECT_ROOT / "results" / suffix
    figures = args.figures_dir or PROJECT_ROOT / "figures" / suffix
    n_random = (
        args.random_seeds
        if args.random_seeds is not None
        else (8 if args.quick else 300)
    )
    n_avalanche = (
        args.avalanche_seeds
        if args.avalanche_seeds is not None
        else (16 if args.quick else 2000)
    )
    if args.seed < 0 or min(n_random, n_avalanche) < (1 if args.quick else 300):
        parser.error(
            "seed must be non-negative; full experiments require at least 300 seeds per random family"
        )
    if args.quick and (
        output.resolve() == (PROJECT_ROOT / "results/cascade").resolve()
        or figures.resolve() == (PROJECT_ROOT / "figures/cascade").resolve()
    ):
        parser.error("quick runs must not overwrite report outputs")
    run_cascade_experiments(
        output_dir=output,
        figures_dir=figures,
        n_random=n_random,
        n_avalanche=n_avalanche,
        seed=args.seed,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
