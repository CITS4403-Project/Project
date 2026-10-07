"""Initial closed-set semantics and preservation of the single-trigger model."""

import networkx as nx
import pytest

from transperth.cascade import simulate_cascade
from transperth.config import CascadeConfig


def graph():
    return nx.relabel_nodes(nx.path_graph(4), str)


def test_default_and_one_element_batch_equivalence():
    config = CascadeConfig(target="1")
    first = simulate_cascade(graph(), config)
    assert first == simulate_cascade(graph(), config, initial_failed=None)
    assert first == simulate_cascade(graph(), config, initial_failed=["1"])


def test_initial_batch_is_synchronous_and_conserves_load():
    result = simulate_cascade(
        graph(),
        CascadeConfig(target="ignored"),
        baseline_loads={"0": 1, "1": 2, "2": 2, "3": 1},
        initial_failed=["2", "1"],
    )
    assert result.initial_failed == ("1", "2")
    assert result.failed == ("1", "2", "0", "3")
    assert result.avalanche_sizes == (2,)
    assert result.rounds == 1
    assert result.n_initial_failed == 2
    assert sum(result.avalanche_sizes) == result.n_failed - result.n_initial_failed
    assert result.remaining_load + result.lost_load == pytest.approx(6)


def test_empty_and_all_closed_sets():
    intact = simulate_cascade(graph(), CascadeConfig(), initial_failed=[])
    assert intact.failed == () and intact.gcc == 4 and intact.rounds == 0
    all_closed = simulate_cascade(
        graph(), CascadeConfig(), initial_failed=list(graph())
    )
    assert all_closed.n_failed == all_closed.n_initial_failed == 4
    assert all_closed.rounds == 0
    assert all_closed.remaining_load == 0


def test_dynamic_hook_recomputes_after_whole_batch():
    calls = []

    def load(current):
        calls.append(tuple(current))
        return {node: 1 for node in current}

    result = simulate_cascade(
        graph(),
        CascadeConfig(dynamic=True),
        load_function=load,
        initial_failed=["1", "2"],
    )
    assert calls == [("0", "1", "2", "3"), ("0", "3")]
    assert result.n_failed == 2 and result.rounds == 0


@pytest.mark.parametrize("batch", [["unknown"], ["1", "1"], [1], "1"])
def test_invalid_initial_sets(batch):
    with pytest.raises((ValueError, TypeError)):
        simulate_cascade(graph(), CascadeConfig(), initial_failed=batch)


def test_json_serialization_failure_keeps_old_outputs(tmp_path):
    from transperth.experiments import RunMeta, save_json

    path = tmp_path / "arrays.json"
    meta = RunMeta.create("cascade", seed=0)
    result, sidecar = save_json({"sizes": [1, 2]}, path, meta)
    before = result.read_bytes(), sidecar.read_bytes()
    with pytest.raises(ValueError):
        save_json({"sizes": [float("nan")]}, path, meta)
    assert before == (result.read_bytes(), sidecar.read_bytes())


@pytest.mark.parametrize("dynamic", [False, True])
def test_batch_composes_with_independent_reference_capacity(dynamic):
    rail = graph()
    base = {node: 1.0 for node in rail}
    reference = {node: 10.0 for node in rail}
    result = simulate_cascade(
        rail,
        CascadeConfig(target="ignored", dynamic=dynamic),
        baseline_loads=base,
        reference_capacities=reference,
        load_function=lambda current: {node: 1.0 for node in current},
        initial_failed=["2", "1"],
    )
    assert result.failed == ("1", "2") and result.rounds == 0
    if not dynamic:
        assert (
            result.initial_total_load == result.remaining_load + result.lost_load == 4
        )
    assert len(rail) == 4 and reference == {node: 10.0 for node in rail}
