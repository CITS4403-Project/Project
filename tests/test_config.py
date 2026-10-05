"""Tests for the frozen interfaces in :mod:`transperth.config`."""

from __future__ import annotations

import dataclasses
import math

import pytest

from transperth import __version__
from transperth.config import (
    BACKUP_EDGES_CSV,
    CascadeConfig,
    CascadeResult,
    DEFAULT_ALPHA_GRID,
    MetricsBundle,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    Strategy,
)


def test_version_is_exposed():
    assert __version__ == "0.1.0"


def test_project_paths_point_into_the_repository():
    assert (PROJECT_ROOT / "src" / "transperth" / "config.py").is_file()
    assert PROCESSED_DATA_DIR.parent.name == "data"
    assert BACKUP_EDGES_CSV.parent == PROCESSED_DATA_DIR


def test_default_alpha_grid_covers_zero_to_two():
    assert DEFAULT_ALPHA_GRID[0] == 0.0
    assert DEFAULT_ALPHA_GRID[-1] == 2.0
    assert len(DEFAULT_ALPHA_GRID) == 81


def test_cascade_config_defaults():
    config = CascadeConfig()
    assert config.alpha == 0.2
    assert config.trigger == "load"
    assert config.target is None
    assert config.rule == "capacity"
    assert config.dynamic is False
    assert config.seed == 0


@pytest.mark.parametrize("alpha", [-1.0, math.inf, -math.inf, math.nan])
def test_cascade_config_rejects_bad_alpha(alpha):
    with pytest.raises(ValueError, match="alpha"):
        CascadeConfig(alpha=alpha)


def test_cascade_config_rejects_negative_tolerance():
    with pytest.raises(ValueError, match="tolerance"):
        CascadeConfig(tolerance=-1e-6)


def test_cascade_result_derived_fields():
    result = CascadeResult(
        n_initial=10,
        failed=("a", "b"),
        gcc=8,
        gcc_fraction=0.8,
        failed_fraction=0.2,
        avalanche_sizes=(1, 1),
        rounds=2,
    )
    assert result.n_failed == 2
    row = result.to_row()
    assert row["n_failed"] == 2
    assert row["failed"] == "a;b"
    assert row["avalanche_sizes"] == "1;1"


def test_metrics_bundle_is_frozen():
    bundle = MetricsBundle(
        n_nodes=10,
        n_edges=9,
        gcc_size=8,
        gcc_fraction=0.8,
        lcc_size=2,
        lcc_fraction=0.2,
        isolated=1,
        aspl=1.5,
        efficiency=0.7,
        network_damage=0.3,
        served_od_fraction=1.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle.gcc_size = 3


def test_strategy_protocol_accepts_duck_typed_strategy():
    class NoBackup:
        name = "no_backup"

        def deploy(self, graph, *, budget, failed=(), demand=None, seed=0):
            return []

    assert isinstance(NoBackup(), Strategy)