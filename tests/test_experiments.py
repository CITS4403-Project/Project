"""Tests for the experiment-runner helpers in :mod:`transperth.experiments`."""

from __future__ import annotations

import hashlib

import pandas as pd

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
    assert payload["inputs"][str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert payload["package_version"] == "0.1.0"


def test_run_meta_ignores_missing_inputs(tmp_path):
    meta = RunMeta.create("demo", seed=0, inputs=[tmp_path / "absent.csv"])
    assert meta.inputs == {}


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