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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd

from transperth.config import PACKAGE_VERSION, RESULTS_DIR

__all__ = [
    "RunMeta",
    "file_sha256",
    "load_meta",
    "package_versions",
    "results_dir",
    "save_table",
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
        """Build a record, hashing every input path that exists on disk."""
        hashed: dict[str, str] = {}
        for item in inputs:
            path = Path(item)
            if path.is_file():
                hashed[str(path)] = file_sha256(path)
        return cls(
            experiment=experiment,
            created_utc=_utc_now(),
            seed=int(seed),
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


def save_table(
    table: pd.DataFrame,
    path: str | Path,
    meta: RunMeta,
    *,
    index: bool = False,
) -> tuple[Path, Path]:
    """Write ``table`` to CSV plus a ``<name>.meta.json`` sidecar."""
    csv_path = Path(path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=index)
    meta_path = csv_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(meta.to_dict(), indent=2, sort_keys=True, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    return csv_path, meta_path


def load_meta(path: str | Path) -> dict[str, Any]:
    """Read a sidecar written by :func:`save_table`.

    Accepts either the CSV path or the sidecar path.
    """
    meta_path = Path(path)
    if meta_path.suffix == ".csv":
        meta_path = meta_path.with_suffix(".meta.json")
    return json.loads(meta_path.read_text(encoding="utf-8"))