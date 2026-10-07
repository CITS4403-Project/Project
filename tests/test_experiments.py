"""Tests for the experiment-runner helpers in :mod:`transperth.experiments`."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from transperth import experiments
from transperth.experiments import RunMeta, load_meta, results_dir, save_table


def test_file_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "input.txt"
    path.write_bytes(b"transperth")
    assert experiments.file_sha256(path) == hashlib.sha256(b"transperth").hexdigest()


def test_run_meta_records_params_and_input_hashes(tmp_path):
    source = tmp_path / "stations.csv"
    source.write_text("station_id\n1\n", encoding="utf-8")
    meta = RunMeta.create(
        "percolation",
        seed=7,
        params={"fraction": 0.5, "measure": "degree"},
        inputs=[source],
    )
    payload = meta.to_dict()
    assert payload["experiment"] == "percolation"
    assert payload["seed"] == 7
    assert payload["params"] == {"fraction": 0.5, "measure": "degree"}
    assert (
        payload["inputs"][str(source)]
        == hashlib.sha256(source.read_bytes()).hexdigest()
    )
    assert payload["package_version"] == "0.1.0"


def test_run_meta_rejects_missing_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        RunMeta.create("demo", seed=0, inputs=[tmp_path / "absent.csv"])
    with pytest.raises(FileNotFoundError):
        RunMeta.create("demo", seed=0, inputs=[tmp_path])
    assert RunMeta.create("synthetic", seed=0).inputs == {}


def test_save_table_writes_csv_and_sidecar(tmp_path):
    table = pd.DataFrame({"alpha": [0.0, 0.25], "failed": [1, 2]})
    meta = RunMeta.create("cascade", seed=3)
    csv_path, meta_path = save_table(table, tmp_path / "sweep.csv", meta)

    assert csv_path.is_file()
    assert meta_path.is_file()
    assert meta_path.name == "sweep.meta.json"
    pd.testing.assert_frame_equal(pd.read_csv(csv_path), table)
    assert load_meta(csv_path)["seed"] == 3
    assert load_meta(meta_path)["experiment"] == "cascade"


def test_results_dir_creates_experiment_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(experiments, "RESULTS_DIR", tmp_path)
    path = results_dir("percolation")
    assert path == tmp_path / "percolation"
    assert path.is_dir()


def test_failed_serialization_preserves_previous_pair(tmp_path):
    path = tmp_path / "result.csv"
    csv, meta = save_table(
        pd.DataFrame({"failed": [1]}),
        path,
        RunMeta.create("demo", seed=1, params={"alpha": 0.2}),
    )
    before = (csv.read_bytes(), meta.read_bytes())
    with pytest.raises(ValueError):
        save_table(
            pd.DataFrame({"failed": [99]}),
            path,
            RunMeta.create("demo", seed=2, params={"alpha": float("nan")}),
        )
    assert (csv.read_bytes(), meta.read_bytes()) == before
    assert load_meta(path)["seed"] == 1


def test_second_publication_failure_rolls_back(tmp_path, monkeypatch):
    path = tmp_path / "result.csv"
    csv, meta = save_table(
        pd.DataFrame({"value": [1]}), path, RunMeta.create("demo", seed=1)
    )
    before = (csv.read_bytes(), meta.read_bytes())
    replace = experiments.os.replace

    def fail_meta(source, target):
        if target.suffix == ".json":
            raise OSError("simulated publication failure")
        return replace(source, target)

    monkeypatch.setattr(experiments.os, "replace", fail_meta)
    with pytest.raises(OSError):
        save_table(pd.DataFrame({"value": [2]}), path, RunMeta.create("demo", seed=2))
    assert (csv.read_bytes(), meta.read_bytes()) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "result.csv",
        "result.meta.json",
    ]


def test_runner_and_summary_are_reproducible():
    run = lambda seed: {"value": seed % 100}
    first = experiments.run_seeded(run, n_runs=8, seed=9)
    pd.testing.assert_frame_equal(first, experiments.run_seeded(run, n_runs=8, seed=9))
    assert first.seed.nunique() == 8
    summary = experiments.summarize_runs(first, ["value"], n_boot=200, seed=8)
    pd.testing.assert_frame_equal(
        summary, experiments.summarize_runs(first, ["value"], n_boot=200, seed=8)
    )
    assert summary["mean"].iloc[0] == first.value.mean()
    with pytest.raises(ValueError):
        experiments.run_seeded(lambda seed: {"seed": 1}, n_runs=1)
    with pytest.raises(ValueError):
        experiments.child_seeds(0, 0)
