"""Shared experiment-runner scaffolding.

Every experiment table is written with :func:`save_table`. The CSV gets a JSON
sidecar that records the run parameters, the seed, the package version and the
SHA-256 hash of every input file. The sidecar is what lets a later data fix
detect which results need to be regenerated.

P1.2 extends this module with runner and bootstrap helpers. The names below are
frozen; see ``docs/model.md``.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from transperth.stats import bootstrap_ci, validate_count

from transperth.config import PACKAGE_VERSION, RESULTS_DIR

__all__ = [
    "RunMeta",
    "file_sha256",
    "load_meta",
    "package_versions",
    "results_dir",
    "save_table",
    "child_seeds",
    "run_seeded",
    "summarize_runs",
]

_HASH_CHUNK = 1 << 20
_TRACKED_PACKAGES = ("numpy", "pandas", "scipy", "networkx", "matplotlib")


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Return the versions of the analysis packages that are installed."""
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class RunMeta:
    """Provenance record stored next to every result table."""

    experiment: str
    created_utc: str
    seed: int
    params: Mapping[str, Any] = field(default_factory=dict)
    inputs: Mapping[str, str] = field(default_factory=dict)
    versions: Mapping[str, str] = field(default_factory=dict)
    package_version: str = PACKAGE_VERSION

    @classmethod
    def create(
        cls,
        experiment: str,
        *,
        seed: int,
        params: Mapping[str, Any] | None = None,
        inputs: Iterable[str | Path] = (),
    ) -> "RunMeta":
        """Build a record, requiring every explicitly supplied input file."""
        hashed: dict[str, str] = {}
        for item in inputs:
            path = Path(item)
            if not path.is_file():
                raise FileNotFoundError(f"input file missing or not a file: {path}")
            hashed[str(path)] = file_sha256(path)
        return cls(
            experiment=experiment,
            created_utc=_utc_now(),
            seed=validate_count(seed, "seed"),
            params=dict(params or {}),
            inputs=hashed,
            versions=package_versions(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the record."""
        return {
            "experiment": self.experiment,
            "created_utc": self.created_utc,
            "seed": self.seed,
            "package_version": self.package_version,
            "params": dict(self.params),
            "inputs": dict(self.inputs),
            "versions": dict(self.versions),
        }


def results_dir(experiment: str) -> Path:
    """Return (and create) the results directory of one experiment family."""
    path = RESULTS_DIR / experiment
    path.mkdir(parents=True, exist_ok=True)
    return path


def _default_file_mode() -> int:
    """Return the mode a new file gets from ``open`` under the process umask."""
    current = os.umask(0)
    os.umask(current)
    return 0o666 & ~current


def save_table(
    table: pd.DataFrame,
    path: str | Path,
    meta: RunMeta,
    *,
    index: bool = False,
) -> tuple[Path, Path]:
    """Write ``table`` to CSV plus a ``<name>.meta.json`` sidecar."""
    # Validate/serialize both outputs before touching an existing pair.
    metadata_text = (
        json.dumps(meta.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    csv_text = table.to_csv(index=index, lineterminator="\n")
    csv_path = Path(path)
    meta_path = csv_path.with_suffix(".meta.json")
    if csv_path == meta_path:
        raise ValueError("CSV and metadata paths must be distinct")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    paths = (csv_path, meta_path)
    previous: dict[Path, bytes | None] = {}
    previous_modes: dict[Path, int] = {}
    for target in paths:
        if target.exists():
            previous[target] = target.read_bytes()
            previous_modes[target] = stat.S_IMODE(target.stat().st_mode)
        else:
            previous[target] = None
    default_mode = _default_file_mode()
    staged = []
    published = []
    try:
        for target, content in zip(paths, (csv_text, metadata_text)):
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                temporary = Path(handle.name)
                staged.append(temporary)
                handle.write(content.encode("utf-8"))
            # Staged files are 0600; keep the previous mode or the umask default.
            os.chmod(temporary, previous_modes.get(target, default_mode))
            # Both files are staged before either is published.
        for temporary, target in zip(staged, paths):
            os.replace(temporary, target)
            published.append(target)
    except Exception:
        for target in published:
            if previous[target] is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(previous[target])
        raise
    finally:
        for temporary in staged:
            temporary.unlink(missing_ok=True)
    return csv_path, meta_path


def child_seeds(seed: int, n_runs: int) -> list[int]:
    """Derive independent, reproducible child seeds from a master seed."""
    master = validate_count(seed, "seed")
    count = validate_count(n_runs, "n_runs", minimum=1)
    return [
        int(child.generate_state(1, dtype=np.uint64)[0])
        for child in np.random.SeedSequence(master).spawn(count)
    ]


def run_seeded(
    run: Callable[[int], Mapping[str, Any]], *, n_runs: int, seed: int = 0
) -> pd.DataFrame:
    """Call ``run(child_seed)`` repeatedly, recording run and seed columns."""
    rows = []
    for index, child in enumerate(child_seeds(seed, n_runs)):
        result = dict(run(child))
        if {"run", "seed"} & result.keys():
            raise ValueError("run results cannot overwrite reserved run/seed columns")
        rows.append({"run": index, "seed": child, **result})
    return pd.DataFrame(rows)


def summarize_runs(
    table: pd.DataFrame,
    columns: Iterable[str],
    *,
    seed: int = 0,
    n_boot: int = 10_000,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Summarize numeric run columns with means and percentile intervals."""
    names = list(columns)
    rows = []
    for name, child in zip(names, child_seeds(seed, len(names))):
        values = table[name].to_numpy(dtype=float)
        low, high = bootstrap_ci(
            values, seed=child, n_boot=n_boot, confidence=confidence
        )
        rows.append(
            {
                "metric": name,
                "n": len(values),
                "mean": float(values.mean()),
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def load_meta(path: str | Path) -> dict[str, Any]:
    """Read a sidecar written by :func:`save_table`.

    Accepts either the CSV path or the sidecar path.
    """
    meta_path = Path(path)
    if meta_path.suffix == ".csv":
        meta_path = meta_path.with_suffix(".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))
