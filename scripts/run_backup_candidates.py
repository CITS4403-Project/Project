"""Derive GTFS backup candidates and check them against the manual ground truth.

Usage (from the repository root):

    PYTHONPATH=src .venv/bin/python scripts/run_backup_candidates.py

The runner builds the frozen rail graph from ``data/processed/``, layers it
with the Perth (56) to Perth Underground (64) walking interchange, derives
single-trip bus candidates from the frozen GTFS snapshot and writes two tables
through :mod:`transperth.experiments` with input hashes and run parameters:

    results/multilayer/gtfs_backup_candidates.csv
    results/multilayer/manual_pair_check.csv

The four pairs in ``data/processed/backup_edges.csv`` are manual ground truth
and are never replaced by derived rows; the comparison table lists every
manual pair with ``matched``, ``time_differs`` or ``not_derived``. Generated
results stay out of git.

The small local rail loader is used because P1.1's
``transperth.network.load_rail_graph`` is not on this branch; switch to it once
P1.1 merges. The large snapshot tables are git-ignored; without them the
runner writes nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import pandas as pd

from transperth.config import PROCESSED_DATA_DIR, PROJECT_ROOT
from transperth.experiments import RunMeta, results_dir, save_table
from transperth.multilayer import (
    DEFAULT_MATCH_TOLERANCE_MINUTES,
    WALKING_TRANSFER_MINUTES,
    add_backup_edges,
    add_walking_transfers,
    build_layers,
    compare_manual_pairs,
    derive_gtfs_candidates,
    served_od_fraction,
    terminal_pairs,
)

DEFAULT_SNAPSHOT_DIR = PROJECT_ROOT / "data" / "snapshots" / "2026-10-05_0700-0900"
DEFAULT_TOPOLOGY_PATH = PROJECT_ROOT / "data" / "verified_topology.json"
SNAPSHOT_INPUTS = ("stops.txt", "stop_times.txt", "bus_trips.csv")


def load_rail_graph(data_dir: Path) -> nx.Graph:
    """Load the frozen rail graph until P1.1's ``load_rail_graph`` lands."""
    stations = pd.read_csv(data_dir / "stations.csv", dtype={"station_id": str})
    edges = pd.read_csv(data_dir / "rail_edges.csv", dtype={"station_a": str, "station_b": str})
    if stations.station_id.isna().any() or not stations.station_id.is_unique:
        raise ValueError("station IDs must be present and unique")
    if stations.empty or edges.empty:
        raise ValueError("the frozen rail tables must not be empty")
    graph = nx.Graph()
    for row in stations.itertuples(index=False):
        graph.add_node(row.station_id, name=row.name)
    for row in edges.itertuples(index=False):
        if row.station_a not in graph or row.station_b not in graph or row.station_a == row.station_b:
            raise ValueError(f"invalid rail edge {row.station_a}-{row.station_b}")
        graph.add_edge(row.station_a, row.station_b, distance_m=float(row.distance_m))
    if not nx.is_connected(graph):
        raise ValueError("the frozen rail graph must be connected")
    return graph


def run(
    *,
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR,
    data_dir: Path = PROCESSED_DATA_DIR,
    topology_path: Path = DEFAULT_TOPOLOGY_PATH,
    output_dir: Path | None = None,
    radius_m: float = 400.0,
    walk_minutes: float = WALKING_TRANSFER_MINUTES,
    tolerance_minutes: float = DEFAULT_MATCH_TOLERANCE_MINUTES,
) -> dict[str, object]:
    """Build the layer, derive candidates, compare and write both tables."""
    rail = load_rail_graph(data_dir)
    topology = json.loads(Path(topology_path).read_text(encoding="utf-8"))
    transfer = topology["walking_transfer"]
    manual = pd.read_csv(data_dir / "backup_edges.csv", dtype={"station_a": str, "station_b": str})

    layered = build_layers(rail)
    walked = add_walking_transfers(layered, transfer, minutes=walk_minutes)
    standby = add_backup_edges(walked, manual)
    walk_edge = walked.get_edge_data(f"T:{transfer['station_a']}", f"T:{transfer['station_b']}", {})
    if walk_edge.get("mode") != "walk":
        raise ValueError("the walking interchange is not in the layered graph")
    baseline_pairs = terminal_pairs(standby)
    served = served_od_fraction(standby, baseline_pairs=baseline_pairs)

    derived = derive_gtfs_candidates(
        snapshot_dir, stations_csv=data_dir / "stations.csv", radius_m=radius_m
    )
    check = compare_manual_pairs(manual, derived, tolerance_minutes=tolerance_minutes)

    experiment = "multilayer"
    meta = RunMeta.create(
        experiment,
        seed=0,
        params={
            "snapshot_dir": str(Path(snapshot_dir).resolve()),
            "radius_m": radius_m,
            "walking_transfer_minutes": walk_minutes,
            "match_tolerance_minutes": tolerance_minutes,
            "rail_stations": rail.number_of_nodes(),
            "rail_edges": rail.number_of_edges(),
            "manual_pairs": len(manual),
            "derived_candidates": len(derived),
            "baseline_terminal_pairs": len(baseline_pairs),
            "intact_served_od_fraction": served,
        },
        inputs=[
            data_dir / "stations.csv",
            data_dir / "rail_edges.csv",
            data_dir / "backup_edges.csv",
            topology_path,
            *[Path(snapshot_dir) / name for name in SNAPSHOT_INPUTS],
        ],
    )
    target = Path(output_dir) if output_dir is not None else results_dir(experiment)
    save_table(derived, target / "gtfs_backup_candidates.csv", meta)
    save_table(check, target / "manual_pair_check.csv", meta)
    return {
        "candidates": derived,
        "manual_check": check,
        "layered": standby,
        "meta": meta,
        "output_dir": target,
        "intact_served_od_fraction": served,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROCESSED_DATA_DIR)
    parser.add_argument("--topology", type=Path, default=DEFAULT_TOPOLOGY_PATH)
    parser.add_argument("--output-dir", type=Path, help="default: results/multilayer")
    parser.add_argument("--radius-m", type=float, default=400.0)
    parser.add_argument("--walk-minutes", type=float, default=WALKING_TRANSFER_MINUTES)
    parser.add_argument("--tolerance-minutes", type=float, default=DEFAULT_MATCH_TOLERANCE_MINUTES)
    args = parser.parse_args(argv)

    missing = [name for name in SNAPSHOT_INPUTS if not (args.snapshot_dir / name).is_file()]
    if missing:
        parser.error(
            "snapshot tables are git-ignored and must exist for a real run: "
            + ", ".join(str(args.snapshot_dir / name) for name in missing)
        )
    outcome = run(
        snapshot_dir=args.snapshot_dir,
        data_dir=args.data_dir,
        topology_path=args.topology,
        output_dir=args.output_dir,
        radius_m=args.radius_m,
        walk_minutes=args.walk_minutes,
        tolerance_minutes=args.tolerance_minutes,
    )
    candidates = outcome["candidates"]
    check = outcome["manual_check"]
    target = outcome["output_dir"]
    print(
        f"Rail layer: {outcome['layered'].number_of_nodes()} nodes, "
        f"{outcome['layered'].number_of_edges()} edges"
    )
    print(f"Intact served OD fraction: {outcome['intact_served_od_fraction']:.6f}")
    print(f"Derived GTFS candidates: {len(candidates)}")
    print(f"Manual pair comparison ({len(check)} rows):")
    print(check.to_string(index=False))
    print(f"Tables: {target / 'gtfs_backup_candidates.csv'}")
    print(f"        {target / 'manual_pair_check.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())