"""Frozen configuration, paths and shared interfaces for the transperth models.

This module is the phase-0 contract that the P1.x and P2.x issues code against.
``docs/model.md`` is the prose half of the same contract: it defines the graph,
the load and capacity equations, the synchronous failure rule, the
redistribution rules, the metrics and the frozen API.

Ownership: P0.1. After gate M0 the public names in this module change only
additively, through a reviewed pull request.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover - referenced by type checkers only
    import networkx as nx

__all__ = [
    "BACKUP_EDGES_CSV",
    "BUS_COVERAGE_CSV",
    "CascadeConfig",
    "CascadeResult",
    "DATA_DIR",
    "DEFAULT_ALPHA",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_SEED",
    "FAILURE_TOLERANCE",
    "FIGURES_DIR",
    "LoadMode",
    "MetricsBundle",
    "NOTEBOOKS_DIR",
    "PACKAGE_DIR",
    "PACKAGE_VERSION",
    "PROCESSED_DATA_DIR",
    "PROJECT_ROOT",
    "RAIL_EDGES_CSV",
    "RAW_DATA_DIR",
    "RESULTS_DIR",
    "RedistributionRule",
    "STATIONS_CSV",
    "Strategy",
    "TriggerMode",
]

PACKAGE_VERSION = "0.1.0"

# ---------------------------------------------------------------------------
# Filesystem layout
#
# config.py lives in <project-root>/src/transperth/, so the project root is two
# levels above the package directory.
# ---------------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

STATIONS_CSV = PROCESSED_DATA_DIR / "stations.csv"
RAIL_EDGES_CSV = PROCESSED_DATA_DIR / "rail_edges.csv"
BUS_COVERAGE_CSV = PROCESSED_DATA_DIR / "bus_coverage.csv"
BACKUP_EDGES_CSV = PROCESSED_DATA_DIR / "backup_edges.csv"

# ---------------------------------------------------------------------------
# Model vocabulary and defaults
# ---------------------------------------------------------------------------
LoadMode = Literal[
    "betweenness", "betweenness_freq", "betweenness_plus_trips", "demand"
]
RedistributionRule = Literal["equal", "capacity"]
TriggerMode = Literal["load", "degree", "random"]

DEFAULT_ALPHA = 0.2
DEFAULT_SEED = 0
FAILURE_TOLERANCE = 1e-12
DEFAULT_ALPHA_GRID: tuple[float, ...] = tuple(round(0.025 * i, 3) for i in range(81))


# ---------------------------------------------------------------------------
# Frozen parameter and result schemas
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CascadeConfig:
    """Parameters of one cascade run. See ``docs/model.md`` section 4.

    ``target`` overrides ``trigger`` when it names a station. ``alpha`` is the
    tolerance in the capacity law ``C = (1 + alpha) * L0``.
    """

    alpha: float = DEFAULT_ALPHA
    trigger: TriggerMode = "load"
    target: str | None = None
    rule: RedistributionRule = "capacity"
    dynamic: bool = False
    load_mode: LoadMode = "betweenness"
    seed: int = DEFAULT_SEED
    tolerance: float = FAILURE_TOLERANCE

    def __post_init__(self) -> None:
        if not math.isfinite(self.alpha) or self.alpha < 0.0:
            raise ValueError("alpha must be finite and non-negative")
        if not math.isfinite(self.tolerance) or self.tolerance < 0.0:
            raise ValueError("tolerance must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CascadeResult:
    """Outcome of one cascade run. See ``docs/model.md`` sections 4 and 7."""

    n_initial: int
    failed: tuple[str, ...]
    gcc: int
    gcc_fraction: float
    failed_fraction: float
    avalanche_sizes: tuple[int, ...]
    rounds: int
    initial_total_load: float | None = None
    remaining_load: float | None = None
    lost_load: float | None = None
    initial_failed: tuple[str, ...] | None = None

    @property
    def n_initial_failed(self) -> int:
        """Count the initial closed set, falling back to the legacy single trigger."""
        return (
            len(self.initial_failed)
            if self.initial_failed is not None
            else min(len(self.failed), 1)
        )

    @property
    def n_failed(self) -> int:
        """Number of failed stations, including the trigger."""
        return len(self.failed)

    def to_row(self) -> dict[str, object]:
        """Flatten the result to one summary row for a table."""
        return {
            "n_initial": self.n_initial,
            "n_failed": self.n_failed,
            "failed_fraction": self.failed_fraction,
            "gcc": self.gcc,
            "gcc_fraction": self.gcc_fraction,
            "rounds": self.rounds,
            "failed": ";".join(self.failed),
            "avalanche_sizes": ";".join(str(size) for size in self.avalanche_sizes),
            "initial_total_load": self.initial_total_load,
            "remaining_load": self.remaining_load,
            "lost_load": self.lost_load,
            "n_initial_failed": self.n_initial_failed,
        }


@dataclass(frozen=True, slots=True)
class MetricsBundle:
    """Network metrics for one graph state. See ``docs/model.md`` section 6."""

    n_nodes: int
    n_edges: int
    gcc_size: int
    gcc_fraction: float
    lcc_size: int
    lcc_fraction: float
    isolated: int
    aspl: float
    efficiency: float
    network_damage: float
    served_od_fraction: float


@runtime_checkable
class Strategy(Protocol):
    """Budget-constrained bus deployment strategy (P1.6).

    ``deploy`` returns the backup station pairs the strategy wants to activate,
    as an ordered list of at most ``budget`` pairs so that a run is
    reproducible. Implementations live in ``transperth.strategies``.
    """

    name: str

    def deploy(
        self,
        graph: "nx.Graph",
        *,
        budget: int,
        failed: Collection[str] = (),
        demand: Mapping[tuple[str, str], float] | None = None,
        seed: int = DEFAULT_SEED,
    ) -> list[tuple[str, str]]:
        """Return up to ``budget`` backup edges for the current failure state."""
        ...
