"""Statistical units, pairing, finite-tail fitting and source validation."""

import json

import numpy as np
import pandas as pd
import pytest

from transperth.config import PROJECT_ROOT
from transperth.tail_fit import discrete_pmf, fit_discrete_tail, power_law_gof
from transperth.uncertainty import (
    grid_sensitivity,
    mean_interval,
    paired_difference,
    seed_convergence,
    verify_cascade_sources,
)


def paired_fixture():
    return pd.DataFrame(
        {
            "replicate": range(8),
            "seed": range(10, 18),
            "target": list("abcdefgh"),
            "value": np.arange(8) / 8,
            "condition": [0] * 8,
        }
    )


def test_paired_interval_resamples_differences_and_rejects_missing_pairs():
    left = paired_fixture()
    right = left.copy()
    right["value"] += 0.25
    result = paired_difference(left, right, "value", seed=12, n_boot=100)
    assert result["mean"] == result["low"] == result["high"] == 0.25
    for invalid in (
        right.iloc[:-1],
        pd.concat([right, right.iloc[:1]]),
        right.assign(seed=20),
    ):
        with pytest.raises(ValueError):
            paired_difference(left, invalid, "value", n_boot=10)


def test_nested_convergence_uses_prefix_and_full_reference():
    table = paired_fixture()
    result = seed_convergence(
        table, ["condition"], ["value"], n_boot=20, counts=(2, 4, 8)
    )
    assert result.n.tolist() == [2, 4, 8]
    assert result["mean"].tolist() == [0.0625, 0.1875, 0.4375]
    assert result.reference_n.tolist() == [8] * 3
    assert result.absolute_change.tolist() == [0.375, 0.25, 0]
    with pytest.raises(ValueError):
        seed_convergence(
            pd.concat([table, table.iloc[:1]]),
            ["condition"],
            ["value"],
            n_boot=5,
            counts=(2,),
        )


def test_intervals_use_local_rng_and_are_reproducible():
    state = np.random.get_state()
    first = mean_interval([0, 1, 2], seed=3, n_boot=100)
    assert first == mean_interval([0, 1, 2], seed=3, n_boot=100)
    after = np.random.get_state()
    assert np.array_equal(state[1], after[1])


def test_grid_containment_is_censored_without_a_transition():
    table = pd.DataFrame(
        [
            {
                "alpha": alpha,
                "rule": "equal",
                "trigger": trigger,
                "post_trigger_size": 1,
                "failed_fraction": 0.5,
            }
            for alpha in (0.0, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 1.5, 2.0)
            for trigger in ("load", "random")
        ]
    )
    result = grid_sensitivity(table)
    assert result.status.eq("right_censored").all()
    assert result.alpha_star.isna().all()


@pytest.mark.parametrize("sample", [[], [np.nan], [-1], [1.5], [86]])
def test_tail_invalid_samples(sample):
    with pytest.raises(ValueError):
        fit_discrete_tail(sample, upper=85)


def test_sparse_and_degenerate_tails_withhold_fit_and_p_value():
    for sample in ([0] * 100, [84, 85] * 100, [1, 2, 3]):
        result = power_law_gof(sample, upper=85, n_boot=10)
        assert result["status"] == "unidentifiable_tail"
        assert result["p_value"] is None


def test_finite_discrete_fit_recovers_synthetic_exponent():
    support, probability = discrete_pmf(2.0, 3, 85)
    assert probability.sum() == pytest.approx(1)
    sample = np.random.default_rng(5).choice(support, size=6000, p=probability)
    fit = fit_discrete_tail(sample, upper=85, min_tail=3000)
    assert fit["status"] == "fitted" and fit["xmin"] >= 3
    assert fit["exponent"] == pytest.approx(2.0, abs=0.15)


def test_refitted_goodness_of_fit_is_seeded_and_handles_body_zeros():
    support, probability = discrete_pmf(1.8, 1, 25)
    sample = np.concatenate(
        [
            np.zeros(100, dtype=int),
            np.random.default_rng(7).choice(support, size=600, p=probability),
        ]
    )
    first = power_law_gof(sample, upper=25, min_tail=250, n_boot=12, seed=20)
    assert first == power_law_gof(sample, upper=25, min_tail=250, n_boot=12, seed=20)
    assert first["p_zero"] == pytest.approx(1 / 7)
    assert len(first["bootstrap_ks"]) == 12
    assert first["invalid_refits"] == 0
    assert 1 / 13 <= first["p_value"] <= 1


def test_full_cascade_sources_valid_and_tampering_is_rejected(tmp_path):
    source = PROJECT_ROOT / "results/cascade"
    assert verify_cascade_sources(source)
    manifest = json.loads((source / "experiment_manifest.json").read_text())
    manifest["artifact_sha256"] = {"broken.csv": "0" * 64}
    (tmp_path / "experiment_manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "experiment_manifest.meta.json").write_bytes(
        (source / "experiment_manifest.meta.json").read_bytes()
    )
    (tmp_path / "broken.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_cascade_sources(tmp_path)
    provenance = json.loads((tmp_path / "experiment_manifest.meta.json").read_text())
    provenance["inputs"]["src/transperth/cascade.py"] = "0" * 64
    (tmp_path / "experiment_manifest.meta.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="model source changed"):
        verify_cascade_sources(tmp_path)
