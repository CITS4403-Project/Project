"""Fetch and verify the pinned archive; never silently accept a newer feed."""

import argparse
from pathlib import Path
import shutil
import tempfile
import urllib.request
import zipfile

from gtfs_common import ROOT, sha256, verify_files, write_json
import json


def verify_archive(path, manifest):
    if (
        path.stat().st_size != manifest["zip_bytes"]
        or sha256(path) != manifest["zip_sha256"]
    ):
        raise ValueError(
            "GTFS archive differs from the frozen source_manifest.json. "
            "Recover the pinned archive, or review a new dataset in a separate PR."
        )
    with zipfile.ZipFile(path) as archive:
        expected = {record["name"] for record in manifest["files"]}
        if set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError("Unexpected ZIP members or failed ZIP CRC")
        # Only the manifest-listed flat filenames are extracted (no path traversal).
        if any(Path(name).name != name or "\\" in name for name in expected):
            raise ValueError("Unsafe GTFS archive filename")


def download(
    gtfs_dir,
    output_dir,
    source_manifest=ROOT / "data/source_manifest.json",
    archive_path=None,
):
    manifest = json.loads(Path(source_manifest).read_text(encoding="utf-8"))
    output_dir, gtfs_dir = Path(output_dir), Path(gtfs_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = output_dir / "google_transit.zip"
    if archive_path is not None:
        verify_archive(Path(archive_path), manifest)
        if Path(archive_path).resolve() != archive.resolve():
            shutil.copyfile(archive_path, archive)
    if not archive.is_file():
        request = urllib.request.Request(
            manifest["resolved_url"],
            headers={"User-Agent": "CITS4403-data-pipeline/1.0"},
        )
        with tempfile.TemporaryDirectory(prefix="download-", dir=output_dir) as scratch:
            candidate = Path(scratch) / "google_transit.zip"
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                candidate.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
            verify_archive(candidate, manifest)
            shutil.move(candidate, archive)
    verify_archive(archive, manifest)
    gtfs_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for record in manifest["files"]:
            target = gtfs_dir / record["name"]
            if not target.is_file() or sha256(target) != record["sha256"]:
                target.write_bytes(source.read(record["name"]))
    verify_files(gtfs_dir, manifest["files"])
    # This is original acquisition provenance, not a changing regeneration timestamp.
    write_json(manifest, output_dir / "download_manifest.json")
    print(f"Verified frozen GTFS: {manifest['zip_sha256']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gtfs-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Ignored archive/manifest cache directory",
    )
    parser.add_argument(
        "--source-manifest", type=Path, default=ROOT / "data/source_manifest.json"
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Recover the exact pinned ZIP from an existing local archive",
    )
    args = parser.parse_args()
    download(args.gtfs_dir, args.output_dir, args.source_manifest, args.archive)


if __name__ == "__main__":
    main()
