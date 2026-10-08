"""Monte Carlo intervals, paired sensitivity, convergence and bounded tail tests."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from transperth.cascade_experiments import artifact_meta, experiment_manifest
from transperth.config import PROJECT_ROOT, RAIL_EDGES_CSV, STATIONS_CSV
from transperth.experiments import child_seeds, file_sha256, save_json, save_table
from transperth.failure import percolation_curve
from transperth.network import load_rail_graph
from transperth.stats import bootstrap_ci, validate_count
from transperth.tail_fit import power_law_gof


def mean_interval(
    values: Iterable[float], *, seed: int, n_boot: int
) -> dict[str, int | float]:
    values = np.asarray(values, dtype=float)
    low, high = bootstrap_ci(values, statistic=np.mean, n_boot=n_boot, seed=seed)
    return {
        "n": len(values),
        "mean": float(values.mean()),
        "low": low,
        "high": high,
        "half_width": (high - low) / 2,
        "bootstrap_seed": seed,
    }


def grouped_intervals(
    table: pd.DataFrame,
    groups: Sequence[str],
    metrics: Sequence[str],
    *,
    seed: int = 0,
    n_boot: int = 2000,
) -> pd.DataFrame:
    """One interval per fixed condition; never resample dependent rounds."""
    rows = []
    collections = list(table.groupby(groups, sort=True))
    seeds = iter(child_seeds(seed, len(collections) * len(metrics)))
    for keys, group in collections:
        if not isinstance(keys, tuple):
            keys = (keys,)
        for metric in metrics:
            rows.append(
                {
                    **dict(zip(groups, keys)),
                    "metric": metric,
                    **mean_interval(group[metric], seed=next(seeds), n_boot=n_boot),
                }
            )
    return pd.DataFrame(rows)


def seed_convergence(
    table: pd.DataFrame,
    groups: Sequence[str],
    metrics: Sequence[str],
    *,
    seed: int = 0,
    n_boot: int = 2000,
    counts: Sequence[int] = (25, 50, 100, 200, 300, 500, 1000, 2000),
) -> pd.DataFrame:
    """Nested prefixes compared with the complete stored sample (not true means)."""
    counts = tuple(validate_count(count, "prefix count", minimum=1) for count in counts)
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("prefix counts must be a nonempty unique sequence")
    rows = []
    collections = list(table.groupby(groups, sort=True))
    seeds = iter(
        child_seeds(
            seed,
            sum(
                sum(count <= len(group) for count in counts) * len(metrics)
                for _, group in collections
            ),
        )
    )
    for keys, group in collections:
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = group.sort_values("replicate")
        if group.replicate.duplicated().any():
            raise ValueError(
                "convergence needs one independent observation per replicate"
            )
        for metric in metrics:
            reference = float(group[metric].mean())
            for count in counts:
                if count > len(group):
                    continue
                summary = mean_interval(
                    group[metric].iloc[:count], seed=next(seeds), n_boot=n_boot
                )
                rows.append(
                    {
                        **dict(zip(groups, keys)),
                        "metric": metric,
                        "reference_n": len(group),
                        "reference_mean": reference,
                        "absolute_change": abs(summary["mean"] - reference),
                        **summary,
                    }
                )
    return pd.DataFrame(rows)


def paired_difference(
    left: pd.DataFrame,
    right: pd.DataFrame,
    metric: str,
    *,
    seed: int = 0,
    n_boot: int = 2000,
) -> dict[str, int | float]:
    """Right minus left on exactly matched seed/target replicates."""
    keys = ["replicate", "seed", "target"]
    if left.replicate.duplicated().any() or right.replicate.duplicated().any():
        raise ValueError("paired samples need unique replicate IDs")
    matched = left[keys + [metric]].merge(
        right[keys + [metric]],
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not matched["_merge"].eq("both").all():
        raise ValueError("paired samples have unmatched seeds or targets")
    return mean_interval(
        matched[metric + "_y"] - matched[metric + "_x"], seed=seed, n_boot=n_boot
    )


def grid_sensitivity(full: pd.DataFrame) -> pd.DataFrame:
    """Coarsen an existing fine sweep; no interpolation of the trigger RNG."""
    rows = []
    for rule, group in full.groupby("rule"):
        targeted = group[group.trigger.eq("load")].sort_values("alpha")
        fine = group[group.trigger.eq("random")].groupby("alpha").failed_fraction.mean()
        for step in (0.025, 0.05, 0.1, 0.2, 0.25, 0.5):
            selector = np.isclose(
                targeted.alpha / step, np.round(targeted.alpha / step)
            )
            selected = targeted[selector]
            contained = selected[selected.post_trigger_size.eq(0)]
            grid = fine[
                np.isclose(
                    fine.index.to_numpy() / step, np.round(fine.index.to_numpy() / step)
                )
            ]
            interpolated = np.interp(fine.index, grid.index, grid.to_numpy())
            rows.append(
                {
                    "rule": rule,
                    "step": step,
                    "points": len(grid),
                    "alpha_star": float(contained.alpha.min())
                    if len(contained)
                    else None,
                    "status": "observed_on_grid"
                    if len(contained)
                    else "right_censored",
                    "max_interpolation_error": float(
                        np.max(np.abs(interpolated - fine.to_numpy()))
                    ),
                }
            )
    return pd.DataFrame(rows)


def verify_cascade_sources(directory: Path) -> list[Path]:
    """Reject tampered/incomplete source artifacts or a stale frozen graph."""
    manifest = directory / "experiment_manifest.json"
    provenance = json.loads(manifest.with_suffix(".meta.json").read_text())
    for name, digest in provenance["inputs"].items():
        if (
            name.startswith(("src/", "scripts/"))
            and file_sha256(PROJECT_ROOT / name) != digest
        ):
            raise ValueError(
                f"Cascade model source changed ({name}); rerun scripts/run_cascades.py"
            )
    for path in (STATIONS_CSV, RAIL_EDGES_CSV):
        key = path.relative_to(PROJECT_ROOT).as_posix()
        if provenance["inputs"].get(key) != file_sha256(path):
            raise ValueError(
                "Cascade graph inputs changed; rerun scripts/run_cascades.py"
            )
    info = json.loads(manifest.read_text())
    if info["params"].get("quick") or info["params"]["n_random"] < 300:
        raise ValueError("Uncertainty report requires the full P2.2 sample")
    inputs = [manifest, manifest.with_suffix(".meta.json")]
    for name, digest in info["artifact_sha256"].items():
        if name.startswith("figures/"):
            continue
        path = directory / name
        if file_sha256(path) != digest:
            raise ValueError(f"Cascade artifact hash mismatch: {name}")
        inputs.extend([path, path.with_suffix(".meta.json")])
    return inputs


def run_uncertainty(
    *,
    output_dir: Path,
    figures_dir: Path,
    cascade_dir: Path,
    n_seeds: int = 1000,
    n_boot: int = 2000,
    n_tail_boot: int = 500,
    seed: int = 0,
    quick: bool = False,
) -> dict:
    from transperth.uncertainty_figures import render_uncertainty_figures

    for name, value in [
        ("n_seeds", n_seeds),
        ("n_boot", n_boot),
        ("n_tail_boot", n_tail_boot),
    ]:
        validate_count(value, name, minimum=1)
    validate_count(seed, "seed")
    sources = verify_cascade_sources(cascade_dir)
    graph = load_rail_graph()
    params = {
        "seed": seed,
        "percolation_seeds": n_seeds,
        "bootstrap_replicates": n_boot,
        "tail_bootstrap_replicates": n_tail_boot,
        "confidence": 0.95,
        "quick": quick,
        "percolation_fractions": np.round(np.arange(21) * 0.05, 3).tolist(),
        "tail_alphas": [0.1, 0.15, 0.2, 0.3],
        "tail_min_observations": 50,
        "tail_min_distinct_sizes": 3,
        "tail_upper": len(graph) - 1,
        "seed_design": "P1.3 derived seed per fraction/replicate; nested prefixes; P2.2 matched child seeds across cascade settings",
        "ci_scope": "Monte Carlo trigger sampling conditional on the fixed graph/model; not GTFS or structural uncertainty",
        "recommendation_limits": {"mean_fraction_change": 0.02, "ci_half_width": 0.03},
    }
    inputs = [
        STATIONS_CSV,
        RAIL_EDGES_CSV,
        *sources,
        Path(__file__),
        PROJECT_ROOT / "src/transperth/tail_fit.py",
        PROJECT_ROOT / "src/transperth/stats.py",
        PROJECT_ROOT / "src/transperth/failure.py",
        PROJECT_ROOT / "src/transperth/network.py",
        PROJECT_ROOT / "src/transperth/cascade_experiments.py",
        PROJECT_ROOT / "src/transperth/uncertainty_figures.py",
        PROJECT_ROOT / "scripts/run_uncertainty.py",
    ]
    meta = artifact_meta("uncertainty", seed, params, inputs)
    output_dir.mkdir(parents=True, exist_ok=True)
    percolation = percolation_curve(
        graph,
        attack="random",
        fractions=params["percolation_fractions"],
        n_seeds=n_seeds,
        seed=seed,
    )
    percolation["replicate"] = percolation.groupby("fraction").cumcount()
    save_table(percolation, output_dir / "random_failure_samples.csv", meta)
    failure_ci = grouped_intervals(
        percolation, ["fraction"], ["gcc_fraction"], seed=seed, n_boot=n_boot
    )
    save_table(failure_ci, output_dir / "random_failure_ci.csv", meta)
    print("Random-failure samples and intervals saved", flush=True)
    full = pd.read_csv(
        cascade_dir / "runs.csv", dtype={"seed": "uint64", "target": str}
    )
    sensitivity = pd.read_csv(
        cascade_dir / "sensitivity_runs.csv", dtype={"seed": "uint64", "target": str}
    )
    avalanche = pd.read_csv(
        cascade_dir / "avalanche_samples.csv", dtype={"seed": "uint64", "target": str}
    )
    avalanche["size_fraction"] = avalanche.post_trigger_size / (len(graph) - 1)
    avalanche["p_cascade"] = avalanche.post_trigger_size.gt(0).astype(int)
    collapse_ci = grouped_intervals(
        full[full.trigger.eq("random")],
        ["rule", "alpha"],
        ["failed_fraction", "gcc_fraction"],
        seed=seed + 1,
        n_boot=n_boot,
    )
    save_table(collapse_ci, output_dir / "cascade_ci.csv", meta)
    avalanche_ci = grouped_intervals(
        avalanche,
        ["dynamic", "alpha"],
        ["post_trigger_size", "rounds", "geographic_span_km", "p_cascade"],
        seed=seed + 2,
        n_boot=n_boot,
    )
    save_table(avalanche_ci, output_dir / "avalanche_ci.csv", meta)
    bootstrap_rows = []
    diagnostics = [
        ("random_failure_f0.3", percolation[percolation.fraction.eq(0.3)].gcc_fraction),
        (
            "static_avalanche_a0.3",
            avalanche[~avalanche.dynamic & avalanche.alpha.eq(0.3)].size_fraction,
        ),
    ]
    for label, values in diagnostics:
        reference = mean_interval(values, seed=seed + 6, n_boot=n_boot)
        for count in sorted({min(n_boot, value) for value in (200, 500, 1000, 2000)}):
            result = mean_interval(values, seed=seed + 6, n_boot=count)
            bootstrap_rows.append(
                {
                    "scenario": label,
                    "resamples": count,
                    "reference_resamples": n_boot,
                    "max_endpoint_change": max(
                        abs(result["low"] - reference["low"]),
                        abs(result["high"] - reference["high"]),
                    ),
                    **result,
                }
            )
    save_table(
        pd.DataFrame(bootstrap_rows), output_dir / "bootstrap_resolution.csv", meta
    )
    convergence_parts = []
    for family, table, groups, metrics in [
        ("random_failure", percolation, ["fraction"], ["gcc_fraction"]),
        ("avalanche", avalanche, ["dynamic", "alpha"], ["size_fraction", "p_cascade"]),
    ]:
        curve = seed_convergence(
            table,
            groups,
            metrics,
            seed=seed + 3,
            n_boot=n_boot,
            counts=(4, 8, 16) if quick else (25, 50, 100, 200, 300, 500, 1000, 2000),
        )
        curve["family"] = family
        convergence_parts.append(curve)
    convergence = pd.concat(convergence_parts, ignore_index=True)
    save_table(convergence, output_dir / "seed_convergence.csv", meta)
    grid = grid_sensitivity(full)
    save_table(grid, output_dir / "alpha_grid_sensitivity.csv", meta)
    random = sensitivity[sensitivity.trigger.eq("random")]
    comparisons = []
    compare_seeds = iter(child_seeds(seed + 4, 1000))
    for (dynamic, alpha), group in random.groupby(["dynamic", "alpha"]):
        for mode, subset in group.groupby("load_mode"):
            left, right = (
                subset[subset.rule.eq("equal")],
                subset[subset.rule.eq("capacity")],
            )
            comparisons.append(
                {
                    "comparison": "capacity_minus_equal",
                    "dynamic": dynamic,
                    "alpha": alpha,
                    "load_mode": mode,
                    **paired_difference(
                        left,
                        right,
                        "failed_fraction",
                        seed=next(compare_seeds),
                        n_boot=n_boot,
                    ),
                }
            )
        baseline = group[group.load_mode.eq("betweenness") & group.rule.eq("capacity")]
        for mode in ("betweenness_freq", "betweenness_plus_trips", "demand"):
            other = group[group.load_mode.eq(mode) & group.rule.eq("capacity")]
            comparisons.append(
                {
                    "comparison": "load_minus_betweenness",
                    "dynamic": dynamic,
                    "alpha": alpha,
                    "load_mode": mode,
                    **paired_difference(
                        baseline,
                        other,
                        "failed_fraction",
                        seed=next(compare_seeds),
                        n_boot=n_boot,
                    ),
                }
            )
    paired = pd.DataFrame(comparisons)
    save_table(paired, output_dir / "paired_sensitivity.csv", meta)
    print(
        "Cascade, avalanche, convergence and paired sensitivity intervals saved",
        flush=True,
    )
    fits = []
    for (dynamic, alpha), group in avalanche.groupby(["dynamic", "alpha"]):
        fit = power_law_gof(
            group.post_trigger_size.to_numpy(),
            upper=len(graph) - 1,
            n_boot=n_tail_boot,
            seed=seed + 5,
        )
        fits.append({"dynamic": bool(dynamic), "alpha": float(alpha), **fit})
        print(f"Tail alpha={alpha}, dynamic={dynamic}: {fit['status']}", flush=True)
    save_json(
        {
            "fits": fits,
            "pooling": "Each alpha/dynamic setting tested separately; total secondary size per trigger; no rounds pooled.",
        },
        output_dir / "power_law_fits.json",
        meta,
    )
    candidates = []
    for count, subset in convergence.groupby("n"):
        candidates.append(
            {
                "n": int(count),
                "worst_mean_change": float(subset.absolute_change.max()),
                "worst_ci_half_width": float(subset.half_width.max()),
                "passes": bool(
                    subset.absolute_change.max() <= 0.02
                    and subset.half_width.max() <= 0.03
                ),
                "all_condition_prefixes_available": len(subset)
                == len(convergence[convergence.n.eq(convergence.n.min())]),
            }
        )
    eligible = [
        item["n"]
        for item in candidates
        if item["passes"] and item["all_condition_prefixes_available"]
    ]
    recommendation = {
        "seed_candidates": candidates,
        "recommended_random_seeds": min(eligible) if eligible and not quick else None,
        "recommendation_scope": "Fixed graph/model and tested fractions/alphas; larger full samples serve as estimates, not exact population means. Require all tested normalized means to change <=0.02 and all 95% CI half widths <=0.03. Data/model uncertainty is not measured.",
        "recommended_alpha_step": 0.025,
        "alpha_grid_evidence": grid.astype(object)
        .where(pd.notna(grid), None)
        .to_dict(orient="records"),
        "recommended_bootstrap_replicates": 2000,
        "tail_bootstrap_replicates": 500,
        "tail_p_resolution": 1 / (n_tail_boot + 1),
        "bootstrap_resolution_evidence": bootstrap_rows,
        "tail_conclusion": "Do not claim a power law: inspect per-setting identifiability and refitted goodness-of-fit; non-rejection alone is insufficient.",
        "demand_scope": "Includes P2.4 AM-peak-stop demand proxy and fixed frequency-scaled reference capacities alongside the three betweenness-based models. Differences jointly change load and capacity assumptions; stop counts are not passenger observations.",
        "quick": quick,
    }
    save_json(recommendation, output_dir / "recommendations.json", meta)
    paths = render_uncertainty_figures(
        failure_ci, collapse_ci, convergence, paired, avalanche, fits, grid, figures_dir
    )
    experiment_manifest(output_dir, paths, params, meta)
    return recommendation
