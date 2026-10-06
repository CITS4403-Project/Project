"""Data semantics and frozen-input regressions; no network needed for unit tests."""

from datetime import date
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_network import bus_coverage, expand_pair, mapped_graph
from download_data import verify_archive
from gtfs_common import rail_parents, read_gtfs, sha256, verify_files, write_json
from prepare_gtfs_7_1 import active_services, time_seconds


class ServiceDayTests(unittest.TestCase):
    def test_calendar_exceptions_override_weekly_schedule(self):
        calendar = pd.DataFrame(
            [
                {
                    "service_id": sid,
                    "start_date": "20261001",
                    "end_date": "20261031",
                    "monday": "1",
                }
                for sid in ["regular", "removed"]
            ]
        )
        exceptions = pd.DataFrame(
            [
                {"service_id": "removed", "date": "20261005", "exception_type": "2"},
                {"service_id": "added", "date": "20261005", "exception_type": "1"},
            ]
        )
        active, regular, added, removed = active_services(
            calendar, exceptions, date(2026, 10, 5)
        )
        self.assertEqual(active, {"regular", "added"})
        self.assertEqual(regular, {"regular", "removed"})
        self.assertEqual(added, {"added"})
        self.assertEqual(removed, {"removed"})

    def test_calendar_dates_only_feed(self):
        exceptions = pd.DataFrame(
            [{"service_id": "only", "date": "20261005", "exception_type": "1"}]
        )
        self.assertEqual(
            active_services(pd.DataFrame(), exceptions, date(2026, 10, 5))[0], {"only"}
        )

    def test_conflicting_exceptions_fail(self):
        exceptions = pd.DataFrame(
            [
                {"service_id": "bad", "date": "20261005", "exception_type": kind}
                for kind in ["1", "2"]
            ]
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            active_services(pd.DataFrame(), exceptions, date(2026, 10, 5))

    def test_service_day_hours_above_24_and_empty_times(self):
        seconds = time_seconds(pd.Series(["07:00:00", "09:00:00", "25:40:00", ""]))
        self.assertEqual(seconds.iloc[:3].to_list(), [25200, 32400, 92400])
        self.assertTrue(pd.isna(seconds.iloc[3]))
        selected = seconds.ge(25200) & seconds.lt(32400)
        self.assertEqual(selected.fillna(False).to_list(), [True, False, False, False])

    def test_bad_times_fail(self):
        for invalid in ["24:60:00", "-1:00:00", "09:00", "soon"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                time_seconds(pd.Series([invalid]))

    def test_reader_strips_headers_and_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stops.txt"
            path.write_text(
                " stop_id , parent_station \n platform , station \n", encoding="utf-8"
            )
            table = read_gtfs(path)
            self.assertEqual(
                table.to_dict("records"),
                [{"stop_id": "platform", "parent_station": "station"}],
            )

    def test_rail_parent_must_be_nonblank_existing_station(self):
        times = pd.DataFrame({"stop_id": ["p"]})
        valid = pd.DataFrame(
            {
                "stop_id": ["p", "s"],
                "parent_station": ["s", ""],
                "location_type": ["0", "1"],
            }
        )
        self.assertEqual(rail_parents(times, valid).to_list(), ["s"])
        for parent, kind in [("", "1"), ("absent", "1"), ("s", "0")]:
            bad = valid.copy()
            bad.loc[0, "parent_station"] = parent
            bad.loc[1, "location_type"] = kind
            with self.subTest(parent=parent, kind=kind), self.assertRaises(ValueError):
                rail_parents(times, bad)


class GeographyTests(unittest.TestCase):
    def test_express_jump_is_expanded_via_showgrounds(self):
        topology = json.loads(
            (ROOT / "data/verified_topology.json").read_text(encoding="utf-8")
        )
        graph = mapped_graph(topology)
        self.assertEqual(expand_pair("42", "30", graph, topology), ["42", "16", "30"])
        self.assertFalse(graph.has_edge("42", "30"))
        self.assertEqual((len(graph), graph.number_of_edges()), (86, 85))

    def test_off_map_connection_is_not_routed_around_the_city(self):
        topology = json.loads(
            (ROOT / "data/verified_topology.json").read_text(encoding="utf-8")
        )
        with self.assertRaisesRegex(ValueError, "not represented"):
            expand_pair("8", "56", mapped_graph(topology), topology)

    def test_true_400m_radius_includes_eastward_stop(self):
        # At Perth's latitude, 0.004 degrees of longitude is about 378 m.
        # The old circular lat/lon query wrongly excluded it (>0.0036 degrees).
        stations = pd.DataFrame(
            [{"station_id": "s", "name": "S", "lat": -32.0, "lon": 115.0}]
        )
        stops = pd.DataFrame(
            [
                {
                    "stop_id": "east",
                    "stop_lat": "-32.0",
                    "stop_lon": "115.004",
                    "location_type": "0",
                },
                {
                    "stop_id": "far",
                    "stop_lat": "-32.0",
                    "stop_lon": "115.005",
                    "location_type": "0",
                },
            ]
        )
        trips = pd.DataFrame({"trip_id": ["t1", "t2"], "route_id": ["r1", "r2"]})
        events = pd.DataFrame({"trip_id": ["t1", "t2"], "stop_id": ["east", "far"]})
        result = bus_coverage(stations, events, trips, stops)
        self.assertEqual(result.iloc[0].n_bus_stops, 1)
        self.assertEqual(result.iloc[0].bus_route_ids, "r1")

    def test_bus_routes_require_a_window_event(self):
        stations = pd.DataFrame(
            [{"station_id": "s", "name": "S", "lat": -32.0, "lon": 115.0}]
        )
        stops = pd.DataFrame(
            [
                {
                    "stop_id": "b",
                    "stop_lat": "-32.0",
                    "stop_lon": "115.0",
                    "location_type": "0",
                }
            ]
        )
        trips = pd.DataFrame({"trip_id": ["early", "late"], "route_id": ["r1", "r2"]})
        result = bus_coverage(
            stations,
            pd.DataFrame({"trip_id": ["early"], "stop_id": ["b"]}),
            trips,
            stops,
        )
        self.assertEqual(result.iloc[0].bus_route_ids, "r1")


class IntegrityTests(unittest.TestCase):
    def test_corrupt_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table.csv"
            path.write_bytes(b"original")
            records = [{"name": path.name, "bytes": 8, "sha256": sha256(path)}]
            verify_files(directory, records)
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "Checksum mismatch"):
                verify_files(directory, records)

    def test_newer_feed_and_zip_path_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "feed.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("../stops.txt", "bad")
            manifest = {
                "zip_bytes": path.stat().st_size,
                "zip_sha256": sha256(path),
                "files": [{"name": "../stops.txt"}],
            }
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                verify_archive(path, manifest)
            manifest["zip_sha256"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "frozen"):
                verify_archive(path, manifest)

    def test_json_has_stable_utf8_lf_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            value = {"name": "车站", "count": 86}
            write_json(value, path)
            first = path.read_bytes()
            write_json(value, path)
            self.assertEqual(first, path.read_bytes())
            self.assertNotIn(b"\r\n", first)

    def test_frozen_dataset_and_manual_evidence_hashes(self):
        expected = json.loads(
            (ROOT / "data/frozen_checksums.json").read_text(encoding="utf-8")
        )
        for relative, digest in expected.items():
            self.assertEqual(sha256(ROOT / "data" / relative), digest, relative)
        manifest = json.loads(
            (ROOT / "data/processed/manifest.json").read_text(encoding="utf-8")
        )
        for name, digest in manifest["manual_provenance"].items():
            self.assertEqual(sha256(ROOT / "data/manual" / name), digest)
            baseline = (
                ROOT / "investigations/idea2-perth-transport/data/processed" / name
            )
            self.assertEqual(
                (ROOT / "data/manual" / name).read_bytes(), baseline.read_bytes()
            )
        self.assertEqual(
            manifest["legacy_edge_audit_counts"],
            {
                "verified_adjacent": 85,
                "expand_skipped_stations": 8,
                "exclude_off_map_connection": 3,
            },
        )

    def test_frozen_generated_files_use_lf_newlines(self):
        expected = json.loads(
            (ROOT / "data/frozen_checksums.json").read_text(encoding="utf-8")
        )
        for relative in expected:
            if not Path(relative).name.startswith("backup_edges"):
                self.assertNotIn(
                    b"\r\n", (ROOT / "data" / relative).read_bytes(), relative
                )


if __name__ == "__main__":
    unittest.main()
