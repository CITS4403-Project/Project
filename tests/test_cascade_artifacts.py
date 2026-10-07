"""Validate the committed full experiment, including joins and provenance pairs."""

import json

import pandas as pd

from transperth.config import PROJECT_ROOT
from transperth.experiments import file_sha256


def test_full_artifact_family_has_metadata_and_correct_hashes():
    directory = PROJECT_ROOT / "results/cascade"
    manifest = json.loads((directory / "experiment_manifest.json").read_text())
    assert manifest["params"]["n_random"] >= 300
    assert len(manifest["params"]["alpha_grid"]) == 81
    for name, digest in manifest["artifact_sha256"].items():
        path = (
            PROJECT_ROOT / "figures/cascade" / name.split("/", 1)[1]
            if name.startswith("figures/")
            else directory / name
        )
        assert file_sha256(path) == digest
        if path.suffix in (".csv", ".json"):
            metadata = json.loads(path.with_suffix(".meta.json").read_text())
            assert metadata["experiment"] == "cascade"
            assert metadata["inputs"]["data/processed/stations.csv"] == file_sha256(
                PROJECT_ROOT / "data/processed/stations.csv"
            )


def test_seeded_rows_join_auditable_outcomes_and_initial_batches():
    directory = PROJECT_ROOT / "results/cascade"
    outcomes = json.loads((directory / "outcomes.json").read_text())
    for name in (
        "runs.csv",
        "sensitivity_runs.csv",
        "line_cascade_runs.csv",
        "map_scenarios.csv",
        "avalanche_samples.csv",
    ):
        table = pd.read_csv(directory / name, dtype={"seed": "uint64", "target": str})
        for row in table.drop_duplicates("outcome_id").itertuples():
            outcome = outcomes[row.outcome_id]
            assert row.n_failed == len(outcome["failed"])
            assert row.post_trigger_size == sum(outcome["avalanche_sizes"])
            assert row.rounds == len(outcome["avalanche_sizes"])
            assert row.n_initial_failed == len(outcome["initial_failed"])
        if name in ("runs.csv", "sensitivity_runs.csv"):
            random = table[table.trigger.eq("random")]
            assert (
                random.groupby(["alpha", "rule", "dynamic", "load_mode"])
                .size()
                .eq(300)
                .all()
            )
    map_rows = pd.read_csv(directory / "map_scenarios.csv")
    assert map_rows.n_failed.tolist() == [86, 1]
    assert map_rows["rounds"].tolist() == [33, 0]
