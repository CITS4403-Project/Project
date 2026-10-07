"""Percolation, phase transition and critical fractions on the frozen rail graph.

This is P2.1's RQ1 experiment. The runner sweeps removal fractions on the
frozen rail graph with :func:`transperth.failure.percolation_curve` and
estimates two collapse points per curve: the susceptibility peak
(:func:`transperth.failure.critical_fraction`) and the fraction where the mean
GCC first reaches 0.5
(:func:`transperth.failure.gcc_threshold_crossing`). On the frozen tree the
susceptibility peak fires early, while the largest component is still large,
so RQ1 reports both and explains the difference.

The default batch (``--all``) sweeps removal fractions 0.00, 0.01, ..., 0.90
plus 1.00 with 100 seeds per fraction, for node and edge removal and for the
random, targeted-degree, targeted-betweenness and targeted-flow attacks. It
writes

    results/percolation/percolation_<attack>_<measure>.csv   (+ sidecars)
    results/percolation/percolation_critical_fractions.csv
    results/percolation/percolation_conclusions.csv
    figures/fig_percolation_curves.png
    figures/fig_percolation_critical.png

Every sidecar records the seed, the run parameters, the package versions and
the SHA-256 hashes of ``stations.csv`` and ``rail_edges.csv``. Results and
figures are committed as report evidence, like the other Phase-2 outputs.

Usage (the Makefile exports ``PYTHONPATH=src``; direct runs need it):

    PYTHONPATH=src python scripts/run_percolation.py                  # one targeted degree curve
    PYTHONPATH=src python scripts/run_percolation.py --all            # full RQ1 batch
    PYTHONPATH=src python scripts/run_percolation.py --all --quick    # six fractions, ten seeds
    PYTHONPATH=src python scripts/run_percolation.py --attack targeted_edge --measure betweenness

Options: ``--attack`` (``random``, ``targeted``, ``random_edge``,
``targeted_edge``), ``--measure`` (``degree``, ``betweenness``, ``flow``),
``--fractions``, ``--n-seeds``, ``--seed``, ``--threshold``, ``--dynamic``,
``--all``, ``--output``, ``--output-dir`` and ``--figures-dir``. ``--dynamic``
recomputes the target ranking after every removal; it is available for single
curves and is not part of the default batch.

The ``flow`` measure is the shared P2.4 demand definition: ``am_peak_stops``
for nodes and unnormalised edge betweenness for edges, as documented in
:mod:`transperth.failure`.

Runtime on the development machine (Python 3.14, networkx 3.7): the full
``--all`` batch takes about 17 s (73,600 runs; every static curve 2 to 3 s at
the default fractions of 0.01), and ``--quick`` runs in well under a second.
The dynamic betweenness curve is the slow variant (about 4 min for the default
grid) because every removal recomputes the ranking.
"""
from __future__ import annotations

import argparse
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

from transperth.config import DEFAULT_SEED, FIGURES_DIR, RAIL_EDGES_CSV, STATIONS_CSV
from transperth.experiments import RunMeta, results_dir, save_table
from transperth.failure import (
    ATTACKS,
    EDGE_MEASURES,
    NODE_MEASURES,
    critical_fraction,
    gcc_threshold_crossing,
    percolation_curve,
)
from transperth.network import load_rail_graph
from transperth.plotting import apply_style, save_figure

import matplotlib.pyplot as plt  # noqa: E402 - import after transperth.plotting sets Agg

DEFAULT_FRACTIONS: tuple[float, ...] = tuple(
    round(0.01 * step, 2) for step in range(91)
) + (1.0,)
"""0.00 to 0.90 in steps of 0.01 plus 1.00; the frozen P2.1 sweep."""

QUICK_FRACTIONS: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_SEEDS = 100
QUICK_SEEDS = 10
DEFAULT_THRESHOLD = 0.5


def load_frozen_graph() -> nx.Graph:
    """Load the frozen rail graph through the packaged P1.1 loader."""
    return load_rail_graph(stations_csv=STATIONS_CSV, edges_csv=RAIL_EDGES_CSV)


def curve_slug(attack: str, measure: str, *, dynamic: bool = False) -> str:
    """Return the stable slug of one curve, e.g. ``targeted_betweenness``."""
    return f"{attack}_{measure}" + ("_dynamic" if dynamic else "")


@dataclass(frozen=True, slots=True)
class CurveSpec:
    """One percolation curve: attack family, ranking measure and policy."""

    attack: str
    measure: str
    dynamic: bool = False

    @property
    def slug(self) -> str:
        """Stable file-name fragment of the curve."""
        return curve_slug(self.attack, self.measure, dynamic=self.dynamic)

    @property
    def label(self) -> str:
        """Human-readable curve name."""
        suffix = " (dynamic)" if self.dynamic else ""
        return f"{self.attack} {self.measure}{suffix}"


FULL_CURVES: tuple[CurveSpec, ...] = (
    CurveSpec("random", "degree"),
    CurveSpec("random_edge", "degree"),
    CurveSpec("targeted", "degree"),
    CurveSpec("targeted", "betweenness"),
    CurveSpec("targeted", "flow"),
    CurveSpec("targeted_edge", "degree"),
    CurveSpec("targeted_edge", "betweenness"),
    CurveSpec("targeted_edge", "flow"),
)
"""The eight static curves of the RQ1 batch; random attacks ignore ``measure``."""

SUMMARY_COLUMNS: tuple[str, ...] = (
    "slug",
    "attack",
    "measure",
    "dynamic",
    "critical_fraction",
    "gcc_threshold_fraction",
    "threshold",
    "n_seeds",
    "n_fractions",
    "fraction_min",
    "fraction_max",
    "n_nodes",
    "n_edges",
    "gcc_fraction_at_estimate",
    "isolated_at_estimate",
)
"""Columns of the per-curve collapse-point table."""

COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("node vs edge", "random_degree", "random_edge_degree"),
    ("node vs edge", "targeted_degree", "targeted_edge_degree"),
    ("node vs edge", "targeted_betweenness", "targeted_edge_betweenness"),
    ("node vs edge", "targeted_flow", "targeted_edge_flow"),
    ("random vs targeted, nodes", "random_degree", "targeted_degree"),
    ("random vs targeted, nodes", "random_degree", "targeted_betweenness"),
    ("random vs targeted, nodes", "random_degree", "targeted_flow"),
    ("random vs targeted, edges", "random_edge_degree", "targeted_edge_degree"),
    (
        "random vs targeted, edges",
        "random_edge_degree",
        "targeted_edge_betweenness",
    ),
    ("random vs targeted, edges", "random_edge_degree", "targeted_edge_flow"),
)
"""RQ1 pairs of curves, ``(question, first_slug, second_slug)``."""

CONCLUSION_COLUMNS: tuple[str, ...] = (
    "question",
    "first_curve",
    "second_curve",
    "first_critical_fraction",
    "second_critical_fraction",
    "critical_difference",
    "critical_ratio",
    "first_gcc_threshold_fraction",
    "second_gcc_threshold_fraction",
    "gcc_threshold_difference",
    "gcc_threshold_ratio",
    "first_gcc_fraction_at_estimate",
    "second_gcc_fraction_at_estimate",
)
"""Columns of the RQ1 conclusions table (differences are first - second)."""

CURVE_LABELS: Mapping[str, str] = {
    "random_degree": "random",
    "random_edge_degree": "random",
    "targeted_degree": "degree",
    "targeted_betweenness": "betweenness",
    "targeted_flow": "flow",
    "targeted_edge_degree": "degree",
    "targeted_edge_betweenness": "betweenness",
    "targeted_edge_flow": "flow",
}
"""Figure labels per curve slug; random has no ranking measure."""

LINE_STYLES: Mapping[str, dict[str, object]] = {
    "random": {"color": "0.3", "ls": "--"},
    "degree": {"color": "C0"},
    "betweenness": {"color": "C1"},
    "flow": {"color": "C2"},
}

FAMILY_PANELS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "node removal",
        (
            "random_degree",
            "targeted_degree",
            "targeted_betweenness",
            "targeted_flow",
        ),
    ),
    (
        "edge removal",
        (
            "random_edge_degree",
            "targeted_edge_degree",
            "targeted_edge_betweenness",
            "targeted_edge_flow",
        ),
    ),
)
"""Slug groups of the two figure rows; node slugs first, edge slugs second."""


def summarize_curve(
    curve: pd.DataFrame,
    spec: CurveSpec,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    n_edges: int | None = None,
) -> dict[str, object]:
    """Summarize one curve with both collapse-point estimates.

    ``critical_fraction`` is the susceptibility peak of
    :func:`transperth.failure.critical_fraction`; ``gcc_threshold_fraction`` is
    :func:`transperth.failure.gcc_threshold_crossing`, the fraction where the
    mean GCC first reaches ``threshold``. On the frozen tree the peak fires
    early, where the second-largest component is widest and the largest
    component is still large, so RQ1 reports both.
    ``gcc_fraction_at_estimate`` and ``isolated_at_estimate`` are the seed
    means at the grid fraction nearest the susceptibility estimate.
    """
    estimate = critical_fraction(curve, threshold=threshold)
    threshold_fraction = gcc_threshold_crossing(curve, threshold=threshold)
    grouped = curve.groupby("fraction", sort=True).agg(
        gcc_fraction=("gcc_fraction", "mean"),
        isolated=("isolated", "mean"),
    )
    nearest: float | None = None
    if math.isfinite(estimate):
        nearest = min(grouped.index, key=lambda value: abs(float(value) - estimate))
    return {
        "slug": spec.slug,
        "attack": spec.attack,
        "measure": spec.measure,
        "dynamic": spec.dynamic,
        "critical_fraction": float(estimate),
        "gcc_threshold_fraction": float(threshold_fraction),
        "threshold": float(threshold),
        "n_seeds": int(curve["seed"].nunique()),
        "n_fractions": int(curve["fraction"].nunique()),
        "fraction_min": float(curve["fraction"].min()),
        "fraction_max": float(curve["fraction"].max()),
        "n_nodes": int(curve["n_initial"].iloc[0]),
        "n_edges": None if n_edges is None else int(n_edges),
        "gcc_fraction_at_estimate": (
            float(grouped.loc[nearest, "gcc_fraction"])
            if nearest is not None
            else float("nan")
        ),
        "isolated_at_estimate": (
            float(grouped.loc[nearest, "isolated"])
            if nearest is not None
            else float("nan")
        ),
    }


def summary_frame(summaries: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    """Return the per-curve summaries with the stable column order."""
    return pd.DataFrame(summaries, columns=list(SUMMARY_COLUMNS))


def _ratio(first: float, second: float) -> float:
    """Return ``first / second``, or NaN when ``second`` is zero."""
    return first / second if second != 0.0 else float("nan")


def conclusions_table(summaries: pd.DataFrame) -> pd.DataFrame:
    """Compare node versus edge and random versus targeted collapse points.

    Both estimates of every curve are compared: the susceptibility
    ``critical_fraction`` and the ``gcc_threshold_fraction`` crossing. A
    positive difference means the first curve of the pair collapses at a
    higher removal fraction, so it is less vulnerable; ratios are
    ``first / second`` and NaN when the second estimate is zero. Every curve
    referenced by :data:`COMPARISONS` must be present.
    """
    table = summaries.set_index("slug")
    rows = []
    for question, first, second in COMPARISONS:
        missing = [slug for slug in (first, second) if slug not in table.index]
        if missing:
            raise ValueError(f"conclusions need the curves {missing!r}")
        first_value = float(table.loc[first, "critical_fraction"])
        second_value = float(table.loc[second, "critical_fraction"])
        first_threshold = float(table.loc[first, "gcc_threshold_fraction"])
        second_threshold = float(table.loc[second, "gcc_threshold_fraction"])
        rows.append(
            {
                "question": question,
                "first_curve": first,
                "second_curve": second,
                "first_critical_fraction": first_value,
                "second_critical_fraction": second_value,
                "critical_difference": first_value - second_value,
                "critical_ratio": _ratio(first_value, second_value),
                "first_gcc_threshold_fraction": first_threshold,
                "second_gcc_threshold_fraction": second_threshold,
                "gcc_threshold_difference": first_threshold - second_threshold,
                "gcc_threshold_ratio": _ratio(first_threshold, second_threshold),
                "first_gcc_fraction_at_estimate": float(
                    table.loc[first, "gcc_fraction_at_estimate"]
                ),
                "second_gcc_fraction_at_estimate": float(
                    table.loc[second, "gcc_fraction_at_estimate"]
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(CONCLUSION_COLUMNS))


def _seed_band(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Return the 25% and 75% per-fraction quantiles of ``column``."""
    grouped = frame.groupby("fraction", sort=True)[column].quantile([0.25, 0.75])
    low = grouped.xs(0.25, level=1)
    high = grouped.xs(0.75, level=1)
    return low, high


def plot_curves(
    curves: Mapping[str, pd.DataFrame],
    *,
    figures_dir: str | Path = FIGURES_DIR,
) -> Path:
    """Plot the mean collapse and isolation curves of the RQ1 batch."""
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 7.0), sharex=True)
    for row, (family, slugs) in enumerate(FAMILY_PANELS):
        for slug in slugs:
            frame = curves[slug]
            grouped = frame.groupby("fraction", sort=True)[
                ["gcc_fraction", "isolated"]
            ].mean()
            label = CURVE_LABELS[slug]
            style = LINE_STYLES[label]
            axes[row, 0].plot(
                grouped.index,
                100.0 * grouped["gcc_fraction"],
                lw=1.3,
                label=label if row == 0 else None,
                **style,
            )
            axes[row, 1].plot(grouped.index, grouped["isolated"], lw=1.3, **style)
            if slug.startswith("random"):
                gcc_low, gcc_high = _seed_band(frame, "gcc_fraction")
                isolated_low, isolated_high = _seed_band(frame, "isolated")
                axes[row, 0].fill_between(
                    gcc_low.index,
                    100.0 * gcc_low,
                    100.0 * gcc_high,
                    color="0.3",
                    alpha=0.15,
                    lw=0,
                )
                axes[row, 1].fill_between(
                    isolated_low.index,
                    isolated_low,
                    isolated_high,
                    color="0.3",
                    alpha=0.15,
                    lw=0,
                )
        axes[row, 0].set_ylabel("GCC fraction (%)")
        axes[row, 1].set_ylabel("isolated stations")
        axes[row, 0].set_title(family)
        axes[row, 1].set_title(family)
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].set_xlabel("removed nodes (fraction)")
    axes[1, 1].set_xlabel("removed edges (fraction)")
    axes[0, 0].set_ylim(bottom=0.0)
    axes[1, 0].set_ylim(bottom=0.0)
    fig.suptitle(
        "frozen rail graph, mean over seeds; band: random 25-75% spread",
        fontsize=11,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return save_figure(fig, "fig_percolation_curves.png", figures_dir=figures_dir)


def plot_critical_fractions(
    summaries: pd.DataFrame,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    figures_dir: str | Path = FIGURES_DIR,
) -> Path:
    """Bar-compare both collapse estimates of node and edge curves."""
    apply_style()
    lookup = summaries.set_index("slug")
    categories = ("random", "degree", "betweenness", "flow")
    node_slugs = (
        "random_degree",
        "targeted_degree",
        "targeted_betweenness",
        "targeted_flow",
    )
    edge_slugs = (
        "random_edge_degree",
        "targeted_edge_degree",
        "targeted_edge_betweenness",
        "targeted_edge_flow",
    )
    positions = list(range(len(categories)))
    width = 0.36
    panels = (
        ("critical_fraction", "susceptibility peak"),
        ("gcc_threshold_fraction", f"GCC <= {threshold:g} crossing"),
    )
    finite = [
        float(lookup.loc[slug, field])
        for field, _ in panels
        for slug in node_slugs + edge_slugs
        if math.isfinite(float(lookup.loc[slug, field]))
    ]
    ceiling = 1.2 * max(finite, default=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.4), sharey=True)
    for ax, (field, title) in zip(axes, panels):
        node_values = [float(lookup.loc[slug, field]) for slug in node_slugs]
        edge_values = [float(lookup.loc[slug, field]) for slug in edge_slugs]
        node_bars = ax.bar(
            [position - width / 2 for position in positions],
            node_values,
            width,
            label="node removal",
            color="steelblue",
        )
        edge_bars = ax.bar(
            [position + width / 2 for position in positions],
            edge_values,
            width,
            label="edge removal",
            color="darkorange",
        )
        for bars in (node_bars, edge_bars):
            for bar in bars:
                height = float(bar.get_height())
                if math.isfinite(height):
                    ax.annotate(
                        f"{height:.2f}",
                        (bar.get_x() + bar.get_width() / 2.0, height),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
        ax.set_xticks(positions, labels=categories)
        ax.set_xlabel("attack")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0.0, ceiling)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("removal fraction")
    fig.suptitle("collapse point by attack and removal type", fontsize=11)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    return save_figure(fig, "fig_percolation_critical.png", figures_dir=figures_dir)


def run_batch(
    graph: nx.Graph,
    *,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
    n_seeds: int = DEFAULT_SEEDS,
    seed: int = DEFAULT_SEED,
    threshold: float = DEFAULT_THRESHOLD,
    specs: Sequence[CurveSpec] = FULL_CURVES,
    output_dir: str | Path | None = None,
    figures_dir: str | Path | None = None,
    inputs: Sequence[str | Path] = (STATIONS_CSV, RAIL_EDGES_CSV),
) -> dict[str, object]:
    """Run every curve of the RQ1 batch and write its tables and figures."""
    target = results_dir("percolation") if output_dir is None else Path(output_dir)
    figures_target = FIGURES_DIR if figures_dir is None else Path(figures_dir)
    curve_inputs = [float(value) for value in fractions]
    curves: dict[str, pd.DataFrame] = {}
    summaries: list[dict[str, object]] = []
    timings: dict[str, float] = {}
    tables: list[tuple[Path, Path]] = []
    for spec in specs:
        started = time.perf_counter()
        curve = percolation_curve(
            graph,
            attack=spec.attack,
            measure=spec.measure,
            fractions=fractions,
            n_seeds=n_seeds,
            seed=seed,
            static=not spec.dynamic,
        )
        timings[spec.slug] = time.perf_counter() - started
        curves[spec.slug] = curve
        summaries.append(
            summarize_curve(
                curve,
                spec,
                threshold=threshold,
                n_edges=graph.number_of_edges(),
            )
        )
        meta = RunMeta.create(
            "percolation",
            seed=seed,
            params={
                "attack": spec.attack,
                "measure": spec.measure,
                "dynamic": spec.dynamic,
                "fractions": curve_inputs,
                "n_seeds": int(n_seeds),
                "threshold": float(threshold),
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
            },
            inputs=inputs,
        )
        tables.append(
            save_table(
                curve, target / f"percolation_{spec.slug}.csv", meta
            )
        )
    summary = summary_frame(summaries)
    batch_meta = RunMeta.create(
        "percolation",
        seed=seed,
        params={
            "batch": "full",
            "curves": [spec.slug for spec in specs],
            "fractions": curve_inputs,
            "n_seeds": int(n_seeds),
            "threshold": float(threshold),
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
        },
        inputs=inputs,
    )
    tables.append(
        save_table(summary, target / "percolation_critical_fractions.csv", batch_meta)
    )
    conclusions = conclusions_table(summary)
    tables.append(
        save_table(conclusions, target / "percolation_conclusions.csv", batch_meta)
    )
    figures = [
        plot_curves(curves, figures_dir=figures_target),
        plot_critical_fractions(
            summary, threshold=threshold, figures_dir=figures_target
        ),
    ]
    return {
        "graph": graph,
        "curves": curves,
        "summary": summary,
        "conclusions": conclusions,
        "timings": timings,
        "tables": tables,
        "figures": figures,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="run the full RQ1 batch; --attack and --measure are ignored",
    )
    parser.add_argument("--attack", choices=ATTACKS, default="targeted")
    parser.add_argument(
        "--measure",
        choices=sorted(set(NODE_MEASURES) | set(EDGE_MEASURES)),
        default="degree",
    )
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="+",
        default=None,
        help="removal fractions in [0, 1]; default 0.00 to 0.90 by 0.01 plus 1.00",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=None,
        help=f"runs per fraction; default {DEFAULT_SEEDS} ({QUICK_SEEDS} with --quick)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="master seed")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="fallback GCC threshold of critical_fraction",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="single curve only: recompute the target ranking after every removal",
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
        help="single-curve CSV path; default results/percolation/percolation_<slug>.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="batch table directory; default results/percolation",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="figure directory; default figures/",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not math.isfinite(args.threshold) or not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must lie in [0, 1]")
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
    graph = load_frozen_graph()
    print(
        f"frozen graph: {graph.number_of_nodes()} stations, "
        f"{graph.number_of_edges()} edges"
    )
    started = time.perf_counter()

    if args.all:
        print(
            f"batch: curves={len(FULL_CURVES)} fractions={len(fractions)} "
            f"seeds={n_seeds} ({len(FULL_CURVES) * len(fractions) * n_seeds} runs)"
        )
        outcome = run_batch(
            graph,
            fractions=fractions,
            n_seeds=n_seeds,
            seed=args.seed,
            threshold=args.threshold,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
        )
        for slug, elapsed in outcome["timings"].items():  # type: ignore[union-attr]
            print(f"  {slug}: {elapsed:.2f} s")
        summary = outcome["summary"]
        conclusions = outcome["conclusions"]
        print(summary.to_string(index=False, float_format="%.4g"))
        print(conclusions.to_string(index=False, float_format="%.4g"))
        for csv_path, meta_path in outcome["tables"]:  # type: ignore[union-attr]
            print(f"wrote {csv_path} and {meta_path.name}")
        for figure_path in outcome["figures"]:  # type: ignore[union-attr]
            print(f"wrote {figure_path}")
    else:
        spec = CurveSpec(args.attack, args.measure, dynamic=args.dynamic)
        static = not spec.dynamic
        print(
            f"sweep: attack={spec.attack} measure={spec.measure} static={static} "
            f"fractions={len(fractions)} seeds={n_seeds} "
            f"({len(fractions) * n_seeds} runs)"
        )
        curve = percolation_curve(
            graph,
            attack=spec.attack,
            measure=spec.measure,
            fractions=fractions,
            n_seeds=n_seeds,
            seed=args.seed,
            static=static,
        )
        summary = summarize_curve(
            curve,
            spec,
            threshold=args.threshold,
            n_edges=graph.number_of_edges(),
        )
        output = args.output or (
            results_dir("percolation") / f"percolation_{spec.slug}.csv"
        )
        meta = RunMeta.create(
            "percolation",
            seed=args.seed,
            params={
                "attack": spec.attack,
                "measure": spec.measure,
                "dynamic": spec.dynamic,
                "fractions": list(fractions),
                "n_seeds": int(n_seeds),
                "threshold": float(args.threshold),
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
            },
            inputs=[STATIONS_CSV, RAIL_EDGES_CSV],
        )
        csv_path, meta_path = save_table(curve, output, meta)
        estimate = summary["critical_fraction"]
        print(f"critical fraction (threshold {args.threshold:g}): {estimate}")
        print(f"wrote {csv_path} and {meta_path.name}")

    print(f"elapsed: {time.perf_counter() - started:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())