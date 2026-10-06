"""Run download -> snapshot -> build -> validation, then publish verified files."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from download_data import download
from gtfs_common import ROOT, sha256
from build_network import build
from validate_data import validate


def publish(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted(source.iterdir()):
        if path.is_file():
            temporary = destination / (path.name + ".tmp")
            shutil.copyfile(path, temporary)
            os.replace(temporary, destination / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--archive", type=Path, help="Exact pinned ZIP, if already downloaded elsewhere"
    )
    parser.add_argument("--service-date", default="2026-10-05")
    parser.add_argument("--start", default="07:00:00")
    parser.add_argument("--end", default="09:00:00")
    parser.add_argument("--include-event-routes", action="store_true")
    args = parser.parse_args()
    data = args.data_dir.resolve()
    frozen = (
        args.service_date == "2026-10-05"
        and args.start == "07:00:00"
        and args.end == "09:00:00"
        and not args.include_event_routes
    )
    if not frozen and data.is_relative_to((ROOT / "data").resolve()):
        parser.error(
            "Alternative windows/event variants require --data-dir outside the frozen data directory"
        )
    name = (
        args.service_date
        + "_"
        + args.start[:5].replace(":", "")
        + "-"
        + args.end[:5].replace(":", "")
    )
    if args.include_event_routes:
        name += "_events"
    gtfs = data / "raw/google_transit"
    download(gtfs, data / "raw", archive_path=args.archive)
    with tempfile.TemporaryDirectory(prefix=".pipeline-", dir=data) as scratch:
        stage = Path(scratch)
        snapshot, processed = stage / "snapshot", stage / "processed"
        command = [
            sys.executable,
            str(ROOT / "scripts/prepare_gtfs_7_1.py"),
            "--gtfs-dir",
            str(gtfs),
            "--download-manifest",
            str(data / "raw/download_manifest.json"),
            "--output-dir",
            str(snapshot),
            "--service-date",
            args.service_date,
            "--start",
            args.start,
            "--end",
            args.end,
        ]
        if args.include_event_routes:
            command.append("--include-event-routes")
        subprocess.run(command, check=True)
        build(
            snapshot,
            processed,
            ROOT / "data/verified_topology.json",
            ROOT / "data/manual",
            ROOT / "investigations/idea2-perth-transport/data/processed",
        )
        validate(snapshot, processed)
        hashes = {f"processed/{p.name}": sha256(p) for p in sorted(processed.iterdir())}
        hashes.update(
            {
                f"snapshots/{name}/{p.name}": sha256(p)
                for p in sorted(snapshot.iterdir())
                if p.name
                in {
                    ".gitignore",
                    "README.md",
                    "snapshot_manifest.json",
                    "validation.json",
                }
            }
        )
        expected_path = ROOT / "data/frozen_checksums.json"
        if frozen:
            if not expected_path.is_file():
                raise FileNotFoundError("Missing reviewed data/frozen_checksums.json")
            expected = json.loads(expected_path.read_text(encoding="utf-8"))
            if hashes != expected:
                raise ValueError(
                    "Regenerated outputs differ from frozen_checksums.json; inspect the dataset change before publishing"
                )
        publish(snapshot, data / "snapshots" / name)
        publish(processed, data / "processed")
    print(
        f"Published verified data in {data}; frozen byte checks: {'passed' if frozen and expected_path.exists() else 'not applicable'}"
    )


if __name__ == "__main__":
    main()
