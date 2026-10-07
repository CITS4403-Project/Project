"""Experiment semantics on fixtures; no assertion that a tree has a transition."""

import networkx as nx
import numpy as np
import pytest

from transperth.cascade import simulate_cascade
from transperth.cascade_experiments import (
    CascadeCache,
    empirical_ccdf,
    exclusive_line_stations,
    geographic_span_km,
    sample_sweep,
)
from transperth.config import CascadeConfig


def fixture():
    graph = nx.relabel_nodes(nx.path_graph(4), str)
    for node in graph:
        graph.nodes[node].update(
            lat=0.0,
            lon=float(node),
            trips_served=1,
            lines="A" if node != "1" else "A;B",
        )
    graph.nodes["3"]["lines"] = "B"
    for a, b in graph.edges:
        graph.edges[a, b].update(trips=1, travel_time=1)
    return graph


def test_exclusive_closures_preserve_interchanges():
    assert exclusive_line_stations(fixture()) == {"A": ("0", "2"), "B": ("3",)}


def test_spatial_extent_is_great_circle_span():
    assert geographic_span_km(fixture(), ("0",)) == 0
    assert geographic_span_km(fixture(), ("0", "1")) == pytest.approx(
        111.1949, abs=0.001
    )


def test_ccdf_includes_zero_probability():
    curve = empirical_ccdf(np.array([0, 0, 1, 2, 2]))
    assert curve["size"].tolist() == [1, 2]
    assert curve.ccdf.tolist() == [0.6, 0.4]
    assert len(empirical_ccdf(np.zeros(4))) == 0
    for invalid in ([], [float("nan")], [-1], [1.5]):
        with pytest.raises(ValueError):
            empirical_ccdf(np.array(invalid))


def test_cache_is_equivalent_and_preserves_seeded_observations():
    graph = fixture()
    cache = CascadeCache(graph)
    kwargs = {
        "alpha": 0.2,
        "rule": "capacity",
        "dynamic": False,
        "mode": "betweenness",
        "target": "1",
    }
    row = cache.run(**kwargs)
    result = simulate_cascade(graph, CascadeConfig(alpha=0.2, target="1"))
    assert row["n_failed"] == result.n_failed
    assert cache.run(**kwargs, seed=999)["outcome_id"] == row["outcome_id"]
    batch = cache.run(**kwargs, initial_failed=("1", "2"), trigger="closed_set")
    assert batch["n_initial_failed"] == 2
    assert cache.outcomes[batch["outcome_id"]]["initial_failed"] == ("1", "2")
    sweep = sample_sweep(cache, [0, 0.2], n_random=5, seed=8, rules=("capacity",))
    random = sweep[sweep.trigger.eq("random")]
    assert random.groupby("alpha").size().tolist() == [5, 5]
    assert (
        random[random.alpha.eq(0)].target.tolist()
        == random[random.alpha.eq(0.2)].target.tolist()
    )
