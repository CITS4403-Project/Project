"""The committed statistical report is complete and traceable to P2.2."""

import json

import pandas as pd

from transperth.config import PROJECT_ROOT
from transperth.experiments import file_sha256


def test_uncertainty_manifest_and_source_result_fingerprints():
    directory = PROJECT_ROOT / "results/uncertainty"
    manifest = json.loads((directory / "experiment_manifest.json").read_text())
    assert not manifest["params"]["quick"]
    assert manifest["params"]["percolation_seeds"] == 1000
    assert manifest["params"]["bootstrap_replicates"] == 2000
    assert manifest["params"]["tail_bootstrap_replicates"] == 500
    for name, digest in manifest["artifact_sha256"].items():
        path = (
            PROJECT_ROOT / "figures/uncertainty" / name.split("/", 1)[1]
            if name.startswith("figures/")
            else directory / name
        )
        assert file_sha256(path) == digest
        if path.suffix in (".csv", ".json"):
            sidecar = json.loads(path.with_suffix(".meta.json").read_text())
            assert sidecar["experiment"] == "uncertainty"
            assert sidecar["inputs"][
                "results/cascade/avalanche_samples.csv"
            ] == file_sha256(PROJECT_ROOT / "results/cascade/avalanche_samples.csv")


def test_report_intervals_and_recommendation_respect_their_evidence():
    directory = PROJECT_ROOT / "results/uncertainty"
    for filename in (
        "random_failure_ci.csv",
        "cascade_ci.csv",
        "avalanche_ci.csv",
        "paired_sensitivity.csv",
        "seed_convergence.csv",
    ):
        table = pd.read_csv(directory / filename)
        assert (table.low <= table.high).all()
        assert table["mean"].notna().all()
    recommendations = json.loads((directory / "recommendations.json").read_text())
    selected = next(
        item
        for item in recommendations["seed_candidates"]
        if item["n"] == recommendations["recommended_random_seeds"]
    )
    assert selected["passes"] and selected["all_condition_prefixes_available"]
    assert selected["worst_ci_half_width"] <= 0.03
    fits = json.loads((directory / "power_law_fits.json").read_text())["fits"]
    assert len(fits) == 8
    assert all(
        fit["status"] == "unidentifiable_tail" and fit["p_value"] is None
        for fit in fits
        if fit["dynamic"] or fit["alpha"] < 0.3
    )
    last = next(fit for fit in fits if not fit["dynamic"] and fit["alpha"] == 0.3)
    assert last["status"] == "rejected" and last["p_value"] < 0.1
    assert last["invalid_refits"] == 0 and len(last["bootstrap_ks"]) == 500
    paired = pd.read_csv(directory / "paired_sensitivity.csv")
    assert "demand" in set(paired.load_mode)
