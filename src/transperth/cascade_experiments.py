"""Seeded cascade experiments, closed-line scenarios and deduplicated outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, replace
from itertools import product
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from transperth.cascade import simulate_cascade
from transperth.config import PROJECT_ROOT, RAIL_EDGES_CSV, STATIONS_CSV, CascadeConfig
from transperth.experiments import (
    RunMeta,
    child_seeds,
    file_sha256,
    save_json,
    save_table,
)
from transperth.failure import random_target_order
from transperth.loads import demand_reference_capacities, initial_loads

MODES = ("betweenness", "betweenness_freq", "betweenness_plus_trips", "demand")
RULES = ("equal", "capacity")
REDUCED_GRID = (0.0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0, 1.5, 2.0)


def geographic_span_km(graph: nx.Graph, failed: tuple[str, ...]) -> float:
    """Maximum great-circle distance between failed stations; singleton span is 0."""
    if len(failed) < 2:
        return 0.0
    coordinates = np.array(
        [(graph.nodes[node]["lat"], graph.nodes[node]["lon"]) for node in failed],
        dtype=float,
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("spatial extent needs finite station coordinates")
    latitude, longitude = np.radians(coordinates).T
    lat_diff = latitude[:, None] - latitude[None, :]
    lon_diff = longitude[:, None] - longitude[None, :]
    haversine = (
        np.sin(lat_diff / 2) ** 2
        + np.cos(latitude[:, None])
        * np.cos(latitude[None, :])
        * np.sin(lon_diff / 2) ** 2
    )
    return float(6371 * 2 * np.arcsin(np.sqrt(np.clip(haversine, 0, 1))).max())


def station_lines(graph: nx.Graph, node: str) -> set[str]:
    """The production loader exposes lists; CSV-like fixtures may use semicolons."""
    raw = graph.nodes[node].get("lines", [])
    labels = raw.split(";") if isinstance(raw, str) else raw
    return {line.strip() for line in labels if line.strip()}


def exclusive_line_stations(graph: nx.Graph) -> dict[str, tuple[str, ...]]:
    """Close only stations belonging to exactly one line, as in the prototype."""
    result: dict[str, list[str]] = {}
    for node in sorted(graph):
        lines = station_lines(graph, node)
        for line in lines:
            result.setdefault(line, [])
        if len(lines) == 1:
            result[next(iter(lines))].append(node)
    return {line: tuple(nodes) for line, nodes in sorted(result.items())}


def artifact_meta(
    experiment: str, seed: int, params: dict, inputs: list[Path]
) -> RunMeta:
    """Hash actual inputs while storing portable repository-relative path keys."""
    meta = RunMeta.create(experiment, seed=seed, params=params, inputs=inputs)
    portable = {}
    for filename, digest in meta.inputs.items():
        path = Path(filename).resolve()
        portable[
            path.relative_to(PROJECT_ROOT).as_posix()
            if path.is_relative_to(PROJECT_ROOT)
            else str(path)
        ] = digest
    return replace(meta, inputs=portable)


class CascadeCache:
    """Memoise deterministic outcomes after the seeded trigger has been resolved.

    Seeds remain in every sampled row. Repeated station selections are separate
    observations of the uniform trigger experiment, not extra graph randomness.
    """

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.baselines = {mode: initial_loads(graph, mode) for mode in MODES}
        self.demand_reference = demand_reference_capacities(graph)
        self.cache: dict[tuple, tuple[str, dict]] = {}
        self.outcomes: dict[str, dict] = {}

    def target(self, trigger: str, mode: str, seed: int) -> str:
        if trigger == "random":
            return random_target_order(self.graph, seed=seed)[0]
        scores = self.baselines[mode] if trigger == "load" else dict(self.graph.degree)
        return min(scores, key=lambda node: (-scores[node], node))

    def run(
        self,
        *,
        alpha: float,
        rule: str,
        dynamic: bool,
        mode: str,
        trigger: str = "load",
        seed: int = 0,
        target: str | None = None,
        initial_failed: tuple[str, ...] | None = None,
    ) -> dict:
        target = target or (
            self.target(trigger, mode, seed) if initial_failed is None else None
        )
        initial = (
            tuple(sorted(initial_failed)) if initial_failed is not None else (target,)
        )
        key = (float(alpha), rule, dynamic, mode, initial)
        if key not in self.cache:
            config = CascadeConfig(
                alpha=alpha,
                rule=rule,
                dynamic=dynamic,
                load_mode=mode,
                trigger=trigger if initial_failed is None else "load",
                target=target,
                seed=seed,
            )
            result = simulate_cascade(
                self.graph,
                config,
                baseline_loads=self.baselines[mode],
                reference_capacities=self.demand_reference
                if mode == "demand"
                else None,
                initial_failed=initial_failed,
            )
            outcome = {
                **asdict(result),
                "n_failed": result.n_failed,
                "n_initial_failed": result.n_initial_failed,
                "post_trigger_size": result.n_failed - result.n_initial_failed,
                "geographic_span_km": geographic_span_km(self.graph, result.failed),
            }
            encoded = json.dumps(outcome, sort_keys=True, allow_nan=False)
            identifier = hashlib.sha256(encoded.encode()).hexdigest()[:20]
            if identifier in self.outcomes and self.outcomes[identifier] != outcome:
                raise RuntimeError("outcome hash collision")
            self.outcomes[identifier] = outcome
            self.cache[key] = identifier, outcome
        identifier, outcome = self.cache[key]
        return {
            "alpha": float(alpha),
            "rule": rule,
            "dynamic": dynamic,
            "load_mode": mode,
            "trigger": trigger,
            "seed": seed,
            "target": target,
            "outcome_id": identifier,
            **{
                name: outcome[name]
                for name in [
                    "n_initial",
                    "n_initial_failed",
                    "n_failed",
                    "post_trigger_size",
                    "failed_fraction",
                    "gcc",
                    "gcc_fraction",
                    "rounds",
                    "geographic_span_km",
                ]
            },
        }


def sample_sweep(
    cache: CascadeCache,
    alphas: list[float],
    *,
    n_random: int,
    seed: int,
    rules=RULES,
    modes=("betweenness",),
    dynamics=(False,),
    label="sweep",
) -> pd.DataFrame:
    """Use common child seeds across every alpha/model setting for paired comparisons."""
    seeds = child_seeds(seed, n_random)
    # Resolve the random target once per seed instead of making a permutation in every row.
    targets = [cache.target("random", "betweenness", child) for child in seeds]
    rows = []
    for mode, rule, dynamic in product(modes, rules, dynamics):
        for alpha in alphas:
            for trigger in ("load", "degree"):
                rows.append(
                    cache.run(
                        alpha=alpha,
                        rule=rule,
                        dynamic=dynamic,
                        mode=mode,
                        trigger=trigger,
                        seed=seed,
                    )
                )
            for index, (child, target) in enumerate(zip(seeds, targets)):
                row = cache.run(
                    alpha=alpha,
                    rule=rule,
                    dynamic=dynamic,
                    mode=mode,
                    trigger="random",
                    seed=child,
                    target=target,
                )
                row["replicate"] = index
                rows.append(row)
        print(
            f"{label}: mode={mode}, rule={rule}, dynamic={dynamic}; {len(rows)} sampled rows",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table["replicate"] = table["replicate"].fillna(-1).astype(int)
    return table


def legacy_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Regenerate the prototype alpha-sweep column names on corrected frozen inputs."""
    rows = []
    for alpha, group in table.groupby("alpha", sort=True):
        random = group[group.trigger.eq("random")]
        row = {"alpha": alpha}
        for trigger in ("load", "degree"):
            targeted = group[group.trigger.eq(trigger)].iloc[0]
            row[f"fail_targeted_{trigger}"] = targeted.failed_fraction
            row[f"gcc_targeted_{trigger}"] = targeted.gcc_fraction
        for metric, short in [("failed_fraction", "fail"), ("gcc_fraction", "gcc")]:
            row[f"{short}_random_mean"] = float(random[metric].mean())
            row[f"{short}_random_std"] = float(random[metric].std(ddof=0))
        rows.append(row)
    return pd.DataFrame(rows)


def empirical_ccdf(values: np.ndarray) -> pd.DataFrame:
    """Unconditional P(S >= s) on positive integer sizes; zeros stay in the denominator."""
    values = np.asarray(values)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or np.any(values < 0)
        or np.any(values != np.floor(values))
    ):
        raise ValueError("CCDF needs a nonempty non-negative sample")
    return pd.DataFrame(
        [
            {
                "size": int(size),
                "ccdf": float(np.mean(values >= size)),
                "n": len(values),
                "p_zero": float(np.mean(values == 0)),
            }
            for size in np.unique(values[values > 0])
        ],
        columns=["size", "ccdf", "n", "p_zero"],
    )


def experiment_manifest(
    directory: Path, figures: list[Path], params: dict, meta: RunMeta
) -> None:
    """Record hashes of every numerical artifact and figure, excluding timestamp sidecars."""
    paths = sorted(
        path
        for path in directory.iterdir()
        if path.suffix in (".csv", ".json")
        and not path.name.endswith(".meta.json")
        and path.name != "experiment_manifest.json"
    )
    hashes = {path.name: file_sha256(path) for path in paths}
    hashes.update({f"figures/{path.name}": file_sha256(path) for path in figures})
    save_json(
        {"params": params, "artifact_sha256": hashes},
        directory / "experiment_manifest.json",
        meta,
    )


def run_cascade_experiments(
    *,
    output_dir: Path,
    figures_dir: Path,
    n_random: int = 300,
    n_avalanche: int = 2000,
    seed: int = 0,
    quick: bool = False,
) -> dict:
    """Regenerate prototype targets first, then the full tolerance and sensitivity families."""
    from transperth.cascade_figures import render_cascade_figures
    from transperth.network import load_rail_graph

    graph = load_rail_graph()
    cache = CascadeCache(graph)
    output_dir.mkdir(parents=True, exist_ok=True)
    alphas = (
        [0.0, 0.2, 1.0, 2.0] if quick else np.round(np.arange(81) * 0.025, 3).tolist()
    )
    legacy_grid = (
        [0.0, 0.2, 0.6] if quick else np.round(np.arange(25) * 0.025, 3).tolist()
    )
    reduced = [0.0, 0.2, 2.0] if quick else list(REDUCED_GRID)
    params = {
        "master_seed": seed,
        "n_random": n_random,
        "n_avalanche": n_avalanche,
        "alpha_grid": alphas,
        "reduced_grid": reduced,
        "legacy_grid": legacy_grid,
        "rules": list(RULES),
        "load_modes": list(MODES),
        "capacity_reference": "Demand uses P2.4 frequency-scaled K; betweenness modes use K=L0; capacities fixed from intact graph.",
        "quick": quick,
        "graph_nodes": len(graph),
        "graph_edges": graph.number_of_edges(),
        "initial_batch": "exclusive-line stations; synchronous; omitted from secondary avalanche sizes",
        "seed_design": "same SeedSequence child seeds across conditions; deterministic station outcomes cached",
        "spatial_unit": "maximum failed-station great-circle span, kilometres",
    }
    sources = [
        Path(__file__),
        PROJECT_ROOT / "src/transperth/cascade.py",
        PROJECT_ROOT / "src/transperth/config.py",
        PROJECT_ROOT / "src/transperth/loads.py",
        PROJECT_ROOT / "src/transperth/failure.py",
        PROJECT_ROOT / "src/transperth/network.py",
        PROJECT_ROOT / "src/transperth/experiments.py",
        PROJECT_ROOT / "src/transperth/plotting.py",
        PROJECT_ROOT / "src/transperth/cascade_figures.py",
        PROJECT_ROOT / "scripts/run_cascades.py",
    ]
    meta = artifact_meta(
        "cascade", seed, params, [STATIONS_CSV, RAIL_EDGES_CSV, *sources]
    )
    for dynamic, name in [(False, "static"), (True, "dynamic")]:
        target = sample_sweep(
            cache,
            legacy_grid,
            n_random=n_random,
            seed=seed,
            rules=("capacity",),
            dynamics=(dynamic,),
            label="prototype targets",
        )
        save_table(legacy_summary(target), output_dir / f"alpha_sweep_{name}.csv", meta)
    avalanche_seeds = child_seeds(seed + 1, n_avalanche)
    avalanche_targets = [
        cache.target("random", "betweenness", child) for child in avalanche_seeds
    ]
    avalanche_rows = []
    for dynamic, alpha in product((False, True), (0.1, 0.15, 0.2, 0.3)):
        for index, (child, target) in enumerate(
            zip(avalanche_seeds, avalanche_targets)
        ):
            row = cache.run(
                alpha=alpha,
                rule="capacity",
                dynamic=dynamic,
                mode="betweenness",
                trigger="random",
                seed=child,
                target=target,
            )
            row["replicate"] = index
            avalanche_rows.append(row)
    avalanche = pd.DataFrame(avalanche_rows)
    save_table(avalanche, output_dir / "avalanche_samples.csv", meta)
    stats = (
        avalanche.groupby(["dynamic", "alpha"])
        .agg(
            mean=("n_failed", "mean"),
            max=("n_failed", "max"),
            n=("n_failed", "size"),
            p_no_cascade=(
                "post_trigger_size",
                lambda values: float(np.mean(values == 0)),
            ),
        )
        .reset_index()
    )
    save_table(stats, output_dir / "avalanche_stats.csv", meta)
    save_json(
        {
            str(alpha): avalanche[avalanche.alpha.eq(alpha) & ~avalanche.dynamic]
            .n_failed.astype(int)
            .tolist()
            for alpha in (0.1, 0.15, 0.2, 0.3)
        },
        output_dir / "avalanche_sizes.json",
        meta,
    )
    full = sample_sweep(
        cache, alphas, n_random=n_random, seed=seed, label="full tolerance"
    )
    save_table(full, output_dir / "runs.csv", meta)
    sensitivity = sample_sweep(
        cache,
        reduced,
        n_random=n_random,
        seed=seed,
        modes=MODES,
        dynamics=(False, True),
        label="sensitivity",
    )
    save_table(sensitivity, output_dir / "sensitivity_runs.csv", meta)
    closure_rows, cascade_rows = [], []
    for line, closed in exclusive_line_stations(graph).items():
        remaining = graph.copy()
        remaining.remove_nodes_from(closed)
        members = [node for node in graph if line in station_lines(graph, node)]
        closure_rows.append(
            {
                "line": line,
                "n_stations": len(members),
                "n_exclusive": len(closed),
                "initial_failed": ";".join(closed),
                "gcc_frac_after": max(
                    map(len, nx.connected_components(remaining)), default=0
                )
                / len(graph),
                "frac_stations_lost": len(closed) / len(graph),
                "rail_trips_at_lost_stations": math.fsum(
                    graph.nodes[node]["trips_served"] for node in closed
                ),
            }
        )
        for alpha, rule, dynamic in product((0.0, 0.2, 2.0), RULES, (False, True)):
            cascade_rows.append(
                {
                    "line": line,
                    **cache.run(
                        alpha=alpha,
                        rule=rule,
                        dynamic=dynamic,
                        mode="betweenness",
                        initial_failed=closed,
                        seed=seed,
                        trigger="closed_set",
                    ),
                }
            )
    closures = pd.DataFrame(closure_rows)
    line_cascades = pd.DataFrame(cascade_rows)
    save_table(closures, output_dir / "line_closure_scenarios.csv", meta)
    save_table(line_cascades, output_dir / "line_cascade_runs.csv", meta)
    maps = [
        cache.run(
            alpha=0.2,
            rule="capacity",
            dynamic=dynamic,
            mode="betweenness",
            target="56",
            seed=seed,
        )
        for dynamic in (False, True)
    ]
    save_table(pd.DataFrame(maps), output_dir / "map_scenarios.csv", meta)
    distributions = []
    for (dynamic, alpha), group in avalanche.groupby(["dynamic", "alpha"]):
        curve = empirical_ccdf(group.post_trigger_size.to_numpy())
        if len(curve):
            curve["dynamic"], curve["alpha"] = dynamic, alpha
            distributions.append(curve)
    distribution = (
        pd.concat(distributions, ignore_index=True)
        if distributions
        else pd.DataFrame(columns=["size", "ccdf", "n", "p_zero", "dynamic", "alpha"])
    )
    save_table(distribution, output_dir / "avalanche_ccdf.csv", meta)
    save_json(cache.outcomes, output_dir / "outcomes.json", meta)
    targeted = full[full.trigger.eq("load") & full.rule.eq("capacity")]
    contained = targeted[targeted.post_trigger_size.eq(0)]
    conclusions = {
        "alpha_star_static_max_load": float(contained.alpha.min())
        if len(contained)
        else None,
        "alpha_star_status": "observed_on_grid"
        if len(contained)
        else "right_censored_above_grid_max",
        "random_seeds": n_random,
        "avalanche_seeds": n_avalanche,
        "unique_cached_runs": len(cache.cache),
        "unique_outcomes": len(cache.outcomes),
        "prototype_regression": "same named targets on 86/85 audited tree and corrected synchronous rounds; old 96-edge values are historical, not numerical equality targets",
        "dynamic_interpretation": "forest removal cannot create alternative paths; supported betweenness-based routing loads cannot increase on survivors, so containment does not establish tolerance in a redundant network",
    }
    save_json(conclusions, output_dir / "conclusions.json", meta)
    figure_paths = render_cascade_figures(
        graph,
        full,
        sensitivity,
        avalanche,
        line_cascades,
        maps,
        cache.outcomes,
        figures_dir,
    )
    experiment_manifest(output_dir, figure_paths, params, meta)
    print(
        f"Finished: {len(full)} tolerance rows; {len(sensitivity)} sensitivity rows; {len(cache.cache)} cached runs",
        flush=True,
    )
    return conclusions
