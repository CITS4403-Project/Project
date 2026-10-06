"""Shared GTFS cleaning, deterministic output and checksum helpers."""

from pathlib import Path
import hashlib
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_M = 6_371_000.0


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_gtfs(path, **kwargs):
    table = pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs)
    table.columns = table.columns.str.strip()
    return table.apply(lambda column: column.str.strip())


def write_csv(table, path):
    table.to_csv(path, index=False, lineterminator="\n", float_format="%.12g")


def write_json(value, path):
    Path(path).write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def file_records(paths):
    return [
        {"name": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(paths)
    ]


def verify_files(directory, records):
    for record in records:
        path = Path(directory) / record["name"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256(path) != record["sha256"]
        ):
            raise ValueError(f"Checksum mismatch: {path}")


def rail_parents(stop_times, stops):
    """Reject missing, blank, non-station or dangling rail platform parents."""
    if stops.stop_id.duplicated().any():
        raise ValueError("Duplicate stop_id")
    indexed = stops.set_index("stop_id")
    parent = stop_times.stop_id.map(indexed.parent_station)
    if parent.isna().any() or parent.eq("").any():
        raise ValueError("Rail platform without parent_station")
    if not set(parent) <= set(indexed.index):
        raise ValueError("Rail parent_station not present in stops.txt")
    if not indexed.loc[list(set(parent)), "location_type"].eq("1").all():
        raise ValueError("Rail parent_station must have location_type=1")
    return parent
