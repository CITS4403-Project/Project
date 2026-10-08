"""Tests for the P2.3 recovery runner helpers in ``scripts/run_recovery``."""

import json
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_recovery import candidate_pool, run  # noqa: E402

TABLE_NAMES = (
    "scenario_manifest",
    "strategy_comparison",
    "bus_cascade_scan",
    "bus_cascade_comparison",
    "bus_cascade_controls",
    "bus_cascade_history",
    "bus_cascade_loads",
)


def _rail(nodes: int = 5) -> nx.Graph:
    """Return a path graph with the loader's node attribute schema."""
    rail = nx.path_graph([str(index) for index in range(nodes)])
    nx.set_edge_attributes(rail, 1000, "distance_m")
    for index, node in enumerate(rail):
        rail.nodes[node].update(
            name=f"Test {index} Stn",
            lines=["blue", "red"] if index >= 3 else ["red"],
            trips_served=10 + index,
            am_peak_stops=10.0 + index,
        )
    return rail


def _candidates() -> pd.DataFrame:
    """Return a small mixed pool of backup candidates."""
    return pd.DataFrame(
        [
            {
                "station_a": "0",
                "station_b": "2",
                "minutes": 4.0,
                "kind": "existing_bus",
            },
            {
                "station_a": "1",
                "station_b": "3",
                "minutes": 5.0,
                "kind": "gtfs_candidate",
            },
            {
                "station_a": "2",
                "station_b": "4",
                "minutes": 6.0,
                "kind": "emergency_bus",
            },
        ]
    )


class TestCandidatePool:
    def test_manual_rows_win_duplicate_pairs(self):
        derived = pd.DataFrame(
            [
                {
                    "station_a": "A",
                    "station_b": "B",
                    "minutes": 3.0,
                    "kind": "gtfs_candidate",
                },
                {
                    "station_a": "C",
                    "station_b": "D",
                    "minutes": 4.0,
                    "kind": "gtfs_candidate",
                },
            ]
        )
        manual = pd.DataFrame(
            [
                {
                    "station_a": "B",
                    "station_b": "A",
                    "minutes": 9.0,
                    "kind": "existing_bus",
                }
            ]
        )
        pool = candidate_pool(derived, manual)
        assert len(pool) == 2
        kept = {
            tuple(sorted((str(row.station_a), str(row.station_b)))): row
            for row in pool.itertuples(index=False)
        }
        assert kept[("A", "B")].minutes == 9.0
        assert kept[("A", "B")].kind == "existing_bus"
        assert kept[("C", "D")].minutes == 4.0

    def test_missing_columns_are_rejected(self):
        derived = pd.DataFrame([{"station_a": "A", "station_b": "B"}])
        manual = _candidates()
        with pytest.raises(ValueError, match="derived table"):
            candidate_pool(derived, manual)


class TestRun:
    def test_run_writes_tables_figures_and_respects_budget(self, tmp_path):
        candidates = _candidates()
        outcome = run(
            rail=_rail(),
            candidates=candidates,
            manual=candidates.iloc[:1],
            transfer={},
            budgets=(1, 2),
            alpha=0.0,
            cascade_alphas=(0.0, 1.0),
            focus_alphas=(0.0,),
            triggers=("2",),
            n_random=1,
            random_size=2,
            min_interchange_lines=2,
            seed=0,
            output_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
        )
        for name in TABLE_NAMES:
            assert (tmp_path / "results" / f"{name}.csv").is_file()
            assert (tmp_path / "results" / f"{name}.meta.json").is_file()
        meta = json.loads(
            (tmp_path / "results" / "strategy_comparison.meta.json").read_text(
                encoding="utf-8"
            )
        )
        assert meta["experiment"] == "recovery"
        assert meta["seed"] == 0
        assert meta["params"]["budgets"] == [1, 2]
        assert meta["params"]["alpha"] == 0.0
        assert meta["params"]["triggers"] == ["2"]
        assert meta["inputs"] == {}

        strategies = outcome["strategy_comparison"]
        assert set(strategies["strategy"]) == {
            "no_backup",
            "existing_bus",
            "shuttle_bridging",
            "corridor_reinforcement",
            "demand_adaptive",
        }
        assert (strategies["selected_count"] <= strategies["budget"]).all()
        assert sorted(strategies["budget"].unique()) == [1, 2]
        assert (strategies["served_od_fraction"] >= 0.0).all()
        assert (strategies["served_od_fraction"] <= 1.0).all()

        manifest = outcome["manifest"]
        assert set(manifest["kind"]) == {"line_closure", "interchange", "random"}
        scan = outcome["scan"]
        assert set(scan["scenario"]) == {"rail_only", "manual_bus", "candidate_bus"}
        assert set(scan["trigger"]) == {"2"}
        assert len(outcome["figures"]) >= 2
        for figure in outcome["figures"]:
            assert Path(figure).is_file()

    def test_unknown_trigger_and_focus_are_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="unknown trigger"):
            run(
                rail=_rail(),
                candidates=_candidates(),
                transfer={},
                triggers=("99",),
                output_dir=tmp_path,
                figures_dir=tmp_path,
            )
        with pytest.raises(ValueError, match="not in the cascade grid"):
            run(
                rail=_rail(),
                candidates=_candidates(),
                transfer={},
                cascade_alphas=(0.0, 1.0),
                focus_alphas=(0.5,),
                triggers=("2",),
                output_dir=tmp_path,
                figures_dir=tmp_path,
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))