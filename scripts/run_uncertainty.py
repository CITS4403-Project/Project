"""Regenerate P2.5: PYTHONPATH=src python scripts/run_uncertainty.py."""

from __future__ import annotations

import argparse
from pathlib import Path

from transperth.config import PROJECT_ROOT
from transperth.uncertainty import run_uncertainty


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-seeds", type=int)
    parser.add_argument("--bootstrap", type=int)
    parser.add_argument("--tail-bootstrap", type=int)
    parser.add_argument(
        "--cascade-dir", type=Path, default=PROJECT_ROOT / "results/cascade"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figures-dir", type=Path)
    args = parser.parse_args()
    suffix = "uncertainty_quick" if args.quick else "uncertainty"
    output = args.output_dir or PROJECT_ROOT / "results" / suffix
    figures = args.figures_dir or PROJECT_ROOT / "figures" / suffix
    counts = (
        args.random_seeds
        if args.random_seeds is not None
        else (16 if args.quick else 1000),
        args.bootstrap if args.bootstrap is not None else (30 if args.quick else 2000),
        args.tail_bootstrap
        if args.tail_bootstrap is not None
        else (10 if args.quick else 500),
    )
    if args.seed < 0 or min(counts) < 1 or (not args.quick and counts[0] < 300):
        parser.error(
            "nonnegative seed, positive bootstrap counts, and at least 300 full random seeds required"
        )
    if not args.quick and (counts[1] < 1000 or counts[2] < 500):
        parser.error(
            "full reports require at least 1000 bootstrap and 500 tail-bootstrap replicates"
        )
    if args.quick and (
        output.resolve() == (PROJECT_ROOT / "results/uncertainty").resolve()
        or figures.resolve() == (PROJECT_ROOT / "figures/uncertainty").resolve()
    ):
        parser.error("quick mode cannot overwrite report outputs")
    run_uncertainty(
        output_dir=output,
        figures_dir=figures,
        cascade_dir=args.cascade_dir,
        n_seeds=counts[0],
        n_boot=counts[1],
        n_tail_boot=counts[2],
        seed=args.seed,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
