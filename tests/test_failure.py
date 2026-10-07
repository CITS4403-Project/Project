"""Tests for the failure and percolation engine in :mod:`transperth.failure`."""

from __future__ import annotations

import math

import networkx as nx
import pandas as pd
import pytest

from transperth import failure
from transperth.failure import (
    CURVE_COLUMNS,
    critical_fraction,
    percolation_curve,
    random_edge_order,
    random_target_order,
    rank_edge_targets,
    rank_targets,
    removal_batch,
    target_order,
)

FRACTIONS = [0.1 * step for step in range(11)]


def _ba_graph(n: int = 150, m: int = 2, seed: int = 7) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


def _er_graph(n: int = 150, p: float = 0.04, seed: int = 7) -> nx.Graph:
    return nx.gnp_random_graph(n, p, seed=seed)


def _star_graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_edges_from([("hub", "b"), ("hub", "a")])
    return graph


# ---------------------------------------------------------------------------
# ranking and removal helpers
# ---------------------------------------------------------------------------
def test_target_order_breaks_ties_by_sorted_station_id():
    assert target_order(_star_graph(), "degree") == ["hub", "a", "b"]


def test_target_order_is_identical_for_the_same_graph_and_measure():
    graph = _ba_graph(n=60)
    assert target_order(graph, "degree") == target_order(graph, "degree")
    assert target_order(graph, "betweenness") == target_order(graph, "betweenness")
    # the static flag records the caller's removal policy; the ranking is the
    # same either way for the graph that is passed in
    assert target_order(graph, "degree", static=False) == target_order(
        graph, "degree", static=True
    )


def test_target_order_flow_matches_the_initial_load_ranking():
    graph = _ba_graph(n=60)
    assert target_order(graph, "flow") == rank_targets(graph, "flow")


def test_target_order_rejects_unknown_measure():
    with pytest.raises(ValueError, match="unknown measure"):
        target_order(_star_graph(), "closeness")


def test_target_order_static_policy_flag_is_typed():
    with pytest.raises(TypeError, match="static"):
        target_order(_star_graph(), "degree", static="yes")  # type: ignore[arg-type]


def test_rank_edge_targets_canonicalises_endpoints_and_breaks_ties():
    graph = nx.path_graph(5)
    # degree sums: (1,2) and (2,3) tie at 4; sorted endpoints pick (1, 2)
    assert rank_edge_targets(graph, "degree") == [
        (1, 2),
        (2, 3),
        (0, 1),
        (3, 4),
    ]


def test_random_target_order_is_a_seed_stable_permutation():
    graph = _ba_graph(n=50)
    first = random_target_order(graph, seed=3)
    assert first == random_target_order(graph, seed=3)
    assert set(first) == set(graph.nodes)
    assert first != random_target_order(graph, seed=4)


def test_random_edge_order_is_a_seed_stable_permutation():
    graph = _ba_graph(n=40)
    first = random_edge_order(graph, seed=5)
    assert first == random_edge_order(graph, seed=5)
    assert len(first) == graph.number_of_edges()
    assert all(str(u) <= str(v) for u, v in first)


def test_removal_batch_targeted_uses_the_shared_ranking():
    graph = _ba_graph(n=40)
    expected = rank_targets(graph, "degree")[:3]
    assert removal_batch(graph, attack="targeted", measure="degree", count=3) == expected


def test_removal_batch_random_is_bounded_and_seed_stable():
    graph = _ba_graph(n=40)
    batch = removal_batch(graph, attack="random", count=5, seed=11)
    assert batch == removal_batch(graph, attack="random", count=5, seed=11)
    assert len(batch) == 5
    assert len(set(batch)) == 5


def test_removal_batch_edge_attack_returns_canonical_edges():
    graph = _ba_graph(n=40)
    batch = removal_batch(graph, attack="targeted_edge", measure="degree", count=4)
    assert batch == rank_edge_targets(graph, "degree")[:4]
    assert all(str(u) <= str(v) for u, v in batch)


def test_removal_batch_validates_attack_and_count():
    graph = _star_graph()
    with pytest.raises(ValueError, match="unknown attack"):
        removal_batch(graph, attack="flood")
    with pytest.raises(ValueError, match="count"):
        removal_batch(graph, attack="random", count=-1)


def test_removal_batch_rejects_edge_measure_for_node_attacks():
    with pytest.raises(ValueError, match="measure"):
        removal_batch(_star_graph(), attack="targeted", measure="closeness")


# ---------------------------------------------------------------------------
# percolation curves
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("builder", [_ba_graph, _er_graph], ids=["ba", "er"])
def test_targeted_removal_collapses_before_random(builder):
    graph = builder()
    targeted = percolation_curve(
        graph, attack="targeted", measure="degree", fractions=FRACTIONS, n_seeds=10, seed=0
    )
    random_curve = percolation_curve(
        graph, attack="random", measure="degree", fractions=FRACTIONS, n_seeds=10, seed=0
    )
    critical_targeted = critical_fraction(targeted)
    critical_random = critical_fraction(random_curve)
    assert math.isfinite(critical_targeted)
    assert math.isfinite(critical_random)
    assert 0.0 <= critical_targeted <= 1.0
    assert 0.0 <= critical_random <= 1.0
    assert critical_targeted < critical_random


def test_curve_columns_are_stable():
    curve = percolation_curve(
        _ba_graph(n=40), attack="random", fractions=[0.0, 0.5], n_seeds=2, seed=0
    )
    assert list(curve.columns) == list(CURVE_COLUMNS)


def test_fraction_zero_records_the_intact_baseline():
    graph = _ba_graph(n=40)
    curve = percolation_curve(
        graph, attack="targeted", measure="degree", fractions=[0.0], n_seeds=4, seed=0
    )
    assert len(curve) == 4
    assert (curve["gcc_size"] == graph.number_of_nodes()).all()
    assert (curve["gcc_fraction"] == 1.0).all()
    assert (curve["lcc_size"] == 0).all()
    assert (curve["isolated"] == 0).all()


def test_fraction_one_removes_every_node_or_every_edge():
    graph = _ba_graph(n=40)
    nodes = percolation_curve(
        graph, attack="targeted", measure="degree", fractions=[1.0], n_seeds=2, seed=0
    )
    assert (nodes["gcc_size"] == 0).all()
    assert (nodes["isolated"] == 0).all()
    edges = percolation_curve(
        graph, attack="random_edge", fractions=[1.0], n_seeds=2, seed=0
    )
    assert (edges["gcc_size"] == 1).all()
    assert (edges["isolated"] == graph.number_of_nodes()).all()


def test_curve_rows_are_identical_for_a_fixed_seed():
    graph = _ba_graph(n=60)
    params = dict(
        attack="targeted",
        measure="betweenness",
        fractions=[0.0, 0.25, 0.5],
        n_seeds=3,
        seed=42,
    )
    first = percolation_curve(graph, **params)
    second = percolation_curve(graph, **params)
    pd.testing.assert_frame_equal(first, second)


def test_curve_records_attack_measure_and_seed():
    graph = _ba_graph(n=40)
    curve = percolation_curve(
        graph,
        attack="random_edge",
        measure="betweenness",
        fractions=[0.5],
        n_seeds=3,
        seed=9,
    )
    assert set(curve["attack"]) == {"random_edge"}
    assert set(curve["measure"]) == {"betweenness"}
    assert set(curve["n_initial"]) == {graph.number_of_nodes()}
    assert len(curve) == 3
    assert curve["seed"].ge(0).all()


def test_static_targeting_computes_the_ranking_once_per_curve(monkeypatch):
    graph = nx.path_graph(30)
    calls = {"count": 0}
    real_rank_targets = failure.rank_targets

    def counting_rank_targets(graph, measure="degree"):
        calls["count"] += 1
        return real_rank_targets(graph, measure)

    monkeypatch.setattr(failure, "rank_targets", counting_rank_targets)
    percolation_curve(
        graph, attack="targeted", measure="degree", fractions=[0.3], n_seeds=2, seed=0
    )
    assert calls["count"] == 1


def test_dynamic_targeting_recomputes_after_every_removal(monkeypatch):
    graph = nx.path_graph(30)
    calls = {"count": 0}
    real_rank_targets = failure.rank_targets

    def counting_rank_targets(graph, measure="degree"):
        calls["count"] += 1
        return real_rank_targets(graph, measure)

    monkeypatch.setattr(failure, "rank_targets", counting_rank_targets)
    percolation_curve(
        graph,
        attack="targeted",
        measure="degree",
        fractions=[0.3],
        n_seeds=2,
        seed=0,
        static=False,
    )
    assert calls["count"] == 2 * int(round(0.3 * 30))


def test_empty_graph_produces_a_well_defined_empty_curve():
    curve = percolation_curve(
        nx.Graph(), attack="targeted", measure="degree", fractions=[0.0, 1.0], n_seeds=2, seed=0
    )
    assert len(curve) == 4
    assert (curve["gcc_size"] == 0).all()
    assert (curve["gcc_fraction"] == 0.0).all()
    assert (curve["lcc_fraction"] == 0.0).all()
    assert (curve["isolated"] == 0).all()


def test_single_node_graph_collapses_cleanly():
    graph = nx.Graph()
    graph.add_node("A")
    curve = percolation_curve(
        graph, attack="targeted", measure="degree", fractions=[0.0, 1.0], n_seeds=1, seed=0
    )
    baseline = curve[curve["fraction"] == 0.0].iloc[0]
    assert baseline["gcc_size"] == 1
    assert baseline["isolated"] == 1
    collapsed = curve[curve["fraction"] == 1.0].iloc[0]
    assert collapsed["gcc_size"] == 0
    assert collapsed["isolated"] == 0


def test_disconnected_graph_with_zero_load_stations_has_no_nan():
    graph = nx.Graph()
    graph.add_nodes_from(["A", "B", "C"])
    curve = percolation_curve(
        graph, attack="random", fractions=[0.0, 0.5, 1.0], n_seeds=2, seed=0
    )
    assert not curve.isna().any().any()
    assert (curve.loc[curve["fraction"] == 0.0, "gcc_size"] == 1).all()
    assert (curve.loc[curve["fraction"] == 0.0, "isolated"] == 3).all()


@pytest.mark.parametrize("fractions", [[-0.1], [1.1], [math.nan]])
def test_curve_rejects_fractions_outside_unit_interval(fractions):
    with pytest.raises(ValueError, match="fractions"):
        percolation_curve(_star_graph(), fractions=fractions)


def test_curve_rejects_empty_fractions_and_bad_n_seeds():
    with pytest.raises(ValueError, match="fractions"):
        percolation_curve(_star_graph(), fractions=[])
    with pytest.raises(ValueError, match="n_seeds"):
        percolation_curve(_star_graph(), fractions=[0.5], n_seeds=0)
    with pytest.raises(TypeError, match="n_seeds"):
        percolation_curve(_star_graph(), fractions=[0.5], n_seeds=2.5)  # type: ignore[arg-type]


def test_curve_rejects_negative_seed_and_unknown_attack():
    with pytest.raises(ValueError, match="seed"):
        percolation_curve(_star_graph(), fractions=[0.5], seed=-1)
    with pytest.raises(ValueError, match="unknown attack"):
        percolation_curve(_star_graph(), attack="flood", fractions=[0.5])


# ---------------------------------------------------------------------------
# critical fraction
# ---------------------------------------------------------------------------
def _synthetic_curve(
    fractions, gcc, lcc, n_initial=100
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fraction": fractions,
            "gcc_fraction": gcc,
            "lcc_size": lcc,
            "n_initial": [n_initial] * len(fractions),
        }
    )


def test_critical_fraction_prefers_the_susceptibility_peak():
    curve = _synthetic_curve(
        fractions=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        gcc=[1.0, 0.95, 0.85, 0.7, 0.5, 0.3, 0.15, 0.05, 0.02, 0.01, 0.0],
        lcc=[0, 2, 5, 20, 50, 30, 10, 5, 2, 1, 0],
    )
    # chi peaks at 0.4 (50**2 / 100); threshold 0.2 alone would cross at 0.567
    assert critical_fraction(curve, threshold=0.2) == pytest.approx(0.4)


def test_critical_fraction_falls_back_to_the_threshold_crossing():
    curve = _synthetic_curve(
        fractions=[0.0, 0.25, 0.5, 1.0],
        gcc=[1.0, 0.9, 0.6, 0.2],
        lcc=[0, 0, 0, 0],
    )
    # chi is identically zero: interpolate between (0.5, 0.6) and (1.0, 0.2)
    assert critical_fraction(curve, threshold=0.5) == pytest.approx(0.625)


def test_critical_fraction_returns_the_documented_sentinel_when_it_never_collapses():
    curve = _synthetic_curve(
        fractions=[0.0, 0.5, 1.0], gcc=[1.0, 1.0, 1.0], lcc=[0, 0, 0]
    )
    assert math.isnan(critical_fraction(curve, threshold=0.5))


def test_critical_fraction_accepts_an_already_collapsed_curve():
    curve = _synthetic_curve(fractions=[0.0], gcc=[0.1], lcc=[0], n_initial=10)
    assert critical_fraction(curve, threshold=0.5) == 0.0


def test_critical_fraction_validates_its_inputs():
    with pytest.raises(TypeError, match="DataFrame"):
        critical_fraction("not a curve")
    with pytest.raises(ValueError, match="missing"):
        critical_fraction(pd.DataFrame({"fraction": [0.0], "gcc_fraction": [1.0]}))
    with pytest.raises(ValueError, match="threshold"):
        critical_fraction(_synthetic_curve([0.0], [1.0], [0]), threshold=1.5)
    with pytest.raises(ValueError, match="missing or non-numeric"):
        critical_fraction(_synthetic_curve([0.0, math.nan], [1.0, 1.0], [0, 0]))