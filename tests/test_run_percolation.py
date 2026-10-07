"""Tests for the P2.1 percolation runner helpers in ``scripts/run_percolation``."""

import json
from pathlib import Path
import sys

import networkx as nx
import pandas as pd
import pytest

from transperth.failure import CURVE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_percolation import (  # noqa: E402
    DEFAULT_FRACTIONS,
    FULL_CURVES,
    CurveSpec,
    conclusions_table,
    curve_slug,
    run_batch,
    summarize_curve,
    summary_frame,
)


def _synthetic_curve(
    fractions: list[float],
    gcc: list[float],
    lcc: list[int],
    *,
    n_initial: int = 4,
    isolated: list[int] | None = None,
) -> pd.DataFrame:
    """Build a curve frame with the stable columns of ``percolation_curve``."""
    isolated = isolated or [0] * len(fractions)
    rows = []
    for index, (fraction, gcc_fraction, lcc_size, lonely) in enumerate(
        zip(fractions, gcc, lcc, isolated)
    ):
        rows.append(
            {
                "fraction": fraction,
                "seed": index,
                "attack": "targeted",
                "measure": "degree",
                "n_initial": n_initial,
                "gcc_size": int(round(gcc_fraction * n_initial)),
                "gcc_fraction": gcc_fraction,
                "lcc_size": lcc_size,
                "lcc_fraction": lcc_size / n_initial,
                "isolated": lonely,
            }
        )
    return pd.DataFrame(rows, columns=list(CURVE_COLUMNS))


class TestDefaults:
    def test_default_fractions_cover_the_frozen_sweep(self):
        assert len(DEFAULT_FRACTIONS) == 92
        assert DEFAULT_FRACTIONS[0] == 0.0
        assert DEFAULT_FRACTIONS[-2] == 0.9
        assert DEFAULT_FRACTIONS[-1] == 1.0
        assert list(DEFAULT_FRACTIONS[:3]) == [0.0, 0.01, 0.02]
        for first, second in zip(DEFAULT_FRACTIONS[:-2], DEFAULT_FRACTIONS[1:-1]):
            assert second - first == pytest.approx(0.01)

    def test_full_curves_cover_every_rq1_pair(self):
        slugs = {spec.slug for spec in FULL_CURVES}
        assert slugs == {
            "random_degree",
            "random_edge_degree",
            "targeted_degree",
            "targeted_betweenness",
            "targeted_flow",
            "targeted_edge_degree",
            "targeted_edge_betweenness",
            "targeted_edge_flow",
        }

    def test_curve_slug_distinguishes_the_dynamic_policy(self):
        assert curve_slug("targeted", "betweenness") == "targeted_betweenness"
        assert (
            curve_slug("targeted", "betweenness", dynamic=True)
            == "targeted_betweenness_dynamic"
        )
        assert CurveSpec("random_edge", "degree").slug == "random_edge_degree"
        assert CurveSpec("targeted", "flow").label == "targeted flow"


class TestSummary:
    def test_summarize_curve_reports_both_collapse_points(self):
        curve = _synthetic_curve([0.0, 0.1, 0.2], [1.0, 0.5, 0.25], [0, 2, 1])
        row = summarize_curve(curve, CurveSpec("targeted", "degree"), n_edges=3)
        assert row["critical_fraction"] == pytest.approx(0.1)
        assert row["gcc_threshold_fraction"] == pytest.approx(0.1)
        assert row["gcc_fraction_at_estimate"] == pytest.approx(0.5)
        assert row["n_fractions"] == 3
        assert row["n_seeds"] == 3
        assert row["n_nodes"] == 4
        assert row["n_edges"] == 3
        assert row["slug"] == "targeted_degree"

    def test_summarize_curve_keeps_the_crossing_when_the_peak_is_degenerate(self):
        curve = _synthetic_curve([0.0, 0.1, 0.2], [1.0, 0.4, 0.3], [0, 0, 0])
        row = summarize_curve(curve, CurveSpec("random", "degree"))
        assert row["critical_fraction"] == pytest.approx(1.0 / 12.0, rel=1e-6)
        assert row["gcc_threshold_fraction"] == pytest.approx(1.0 / 12.0, rel=1e-6)
        assert row["n_edges"] is None

    def test_summary_frame_keeps_the_stable_column_order(self):
        curve = _synthetic_curve([0.0, 0.1], [1.0, 0.5], [0, 2])
        rows = [
            summarize_curve(curve, CurveSpec("targeted", "degree"), n_edges=3),
            summarize_curve(curve, CurveSpec("random_edge", "degree"), n_edges=3),
        ]
        frame = summary_frame(rows)
        assert list(frame.columns) == [
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
        ]
        assert list(frame["slug"]) == ["targeted_degree", "random_edge_degree"]


class TestConclusions:
    def _summary(self) -> pd.DataFrame:
        values = {
            "random_degree": 0.4,
            "random_edge_degree": 0.5,
            "targeted_degree": 0.6,
            "targeted_edge_degree": 0.7,
            "targeted_betweenness": 0.2,
            "targeted_edge_betweenness": 0.3,
            "targeted_flow": 0.25,
            "targeted_edge_flow": 0.35,
        }
        return pd.DataFrame(
            [
                {
                    "slug": slug,
                    "critical_fraction": value,
                    "gcc_threshold_fraction": value / 2.0,
                    "gcc_fraction_at_estimate": value,
                }
                for slug, value in values.items()
            ]
        )

    def test_conclusions_table_computes_pairs(self):
        table = conclusions_table(self._summary())
        assert len(table) == 10
        node_vs_edge = table.loc[table["question"] == "node vs edge"]
        first = node_vs_edge.iloc[0]
        assert first["first_curve"] == "random_degree"
        assert first["second_curve"] == "random_edge_degree"
        assert first["critical_difference"] == pytest.approx(-0.1)
        assert first["critical_ratio"] == pytest.approx(0.8)
        assert first["gcc_threshold_difference"] == pytest.approx(-0.05)
        assert first["gcc_threshold_ratio"] == pytest.approx(0.8)
        assert first["first_gcc_fraction_at_estimate"] == pytest.approx(0.4)
        random_vs_targeted = table.loc[
            (table["question"] == "random vs targeted, nodes")
            & (table["second_curve"] == "targeted_flow")
        ].iloc[0]
        assert random_vs_targeted["critical_difference"] == pytest.approx(0.15)
        assert random_vs_targeted["gcc_threshold_difference"] == pytest.approx(0.075)

    def test_conclusions_table_names_the_missing_curve(self):
        incomplete = self._summary()
        incomplete = incomplete.loc[incomplete["slug"] != "targeted_flow"]
        with pytest.raises(ValueError, match="targeted_flow"):
            conclusions_table(incomplete)


class TestBatch:
    def test_run_batch_writes_every_table_sidecar_and_figure(self, tmp_path):
        graph = nx.path_graph([str(index) for index in range(6)])
        outcome = run_batch(
            graph,
            fractions=(0.0, 0.5, 1.0),
            n_seeds=3,
            output_dir=tmp_path / "results",
            figures_dir=tmp_path / "figures",
            inputs=(),
        )
        summary = outcome["summary"]
        assert len(summary) == len(FULL_CURVES)
        assert len(outcome["tables"]) == len(FULL_CURVES) + 2
        for spec in FULL_CURVES:
            csv_path = tmp_path / "results" / f"percolation_{spec.slug}.csv"
            assert csv_path.is_file()
            meta = json.loads(
                csv_path.with_suffix(".meta.json").read_text(encoding="utf-8")
            )
            assert meta["experiment"] == "percolation"
            assert meta["params"]["attack"] == spec.attack
            assert meta["params"]["measure"] == spec.measure
            assert meta["inputs"] == {}
        for name in ("percolation_critical_fractions", "percolation_conclusions"):
            assert (tmp_path / "results" / f"{name}.csv").is_file()
            assert (tmp_path / "results" / f"{name}.meta.json").is_file()
        assert set(summary["slug"]) == {spec.slug for spec in FULL_CURVES}
        assert len(outcome["conclusions"]) == 10
        assert len(outcome["figures"]) == 2
        for figure in outcome["figures"]:
            assert Path(figure).is_file()

    def test_run_batch_is_deterministic_for_a_fixed_seed(self, tmp_path):
        graph = nx.path_graph([str(index) for index in range(5)])
        first = run_batch(
            graph,
            fractions=(0.0, 0.5),
            n_seeds=2,
            output_dir=tmp_path / "first",
            figures_dir=tmp_path / "figures",
            inputs=(),
        )
        second = run_batch(
            graph,
            fractions=(0.0, 0.5),
            n_seeds=2,
            output_dir=tmp_path / "second",
            figures_dir=tmp_path / "figures",
            inputs=(),
        )
        pd.testing.assert_frame_equal(
            first["curves"]["random_degree"],
            second["curves"]["random_degree"],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))