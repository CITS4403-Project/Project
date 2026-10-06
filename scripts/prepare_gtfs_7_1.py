"""Prepare a reproducible date/window subset, without rebuilding the rail network.

Run with the project Python environment. Full selected trip sequences are retained;
window_departures.csv is the separate half-open [start, end) event table.
"""

from pathlib import Path
from datetime import date
import argparse
import json
import pandas as pd
from gtfs_common import ROOT, rail_parents, verify_files, file_records, write_json


def read_gtfs(path):
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table.columns = table.columns.str.strip()
    return table.apply(lambda col: col.str.strip())


def active_services(calendar, exceptions, service_date):
    if not calendar.empty and calendar.service_id.duplicated().any():
        raise ValueError("Duplicate calendar service_id")
    if not exceptions.empty:
        if exceptions[["service_id", "date"]].duplicated().any():
            raise ValueError("Duplicate or conflicting calendar exception")
        if not exceptions.exception_type.isin(["1", "2"]).all():
            raise ValueError("Invalid calendar exception_type")
    stamp = service_date.strftime("%Y%m%d")
    weekday = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ][service_date.weekday()]
    active = set()
    if not calendar.empty:
        active = set(
            calendar.loc[
                (calendar.start_date <= stamp)
                & (calendar.end_date >= stamp)
                & (calendar[weekday] == "1"),
                "service_id",
            ]
        )
    regular = active.copy()
    today = (
        exceptions[exceptions.date.eq(stamp)] if not exceptions.empty else exceptions
    )
    added = (
        set(today.loc[today.exception_type.eq("1"), "service_id"])
        if not today.empty
        else set()
    )
    removed = (
        set(today.loc[today.exception_type.eq("2"), "service_id"])
        if not today.empty
        else set()
    )
    if added & removed:
        raise ValueError("Conflicting calendar exceptions")
    active.update(added)
    active.difference_update(removed)
    return active, regular, added, removed


def time_seconds(series):
    # A GTFS hour may exceed 23. Empty times remain missing, without interpolation.
    parts = series.str.extract(r"^(\d+):([0-5]\d):([0-5]\d)$")
    bad = series.ne("") & parts[0].isna()
    if bad.any():
        raise ValueError("Invalid GTFS time: " + series[bad].iloc[0])
    return (
        parts[0].astype("Int64") * 3600
        + parts[1].astype("Int64") * 60
        + parts[2].astype("Int64")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gtfs-dir", type=Path, required=True)
    parser.add_argument("--download-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--topology", type=Path, default=ROOT / "data/verified_topology.json"
    )
    parser.add_argument("--include-event-routes", action="store_true")
    parser.add_argument("--service-date", default="2026-10-05")
    parser.add_argument("--start", default="07:00:00")
    parser.add_argument("--end", default="09:00:00")
    args = parser.parse_args()
    raw = args.gtfs_dir
    out = args.output_dir
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Output directory is not empty; use a new snapshot directory"
        )
    download = json.loads(args.download_manifest.read_text(encoding="utf-8"))
    verify_files(raw, download["files"])
    topology = json.loads(args.topology.read_text(encoding="utf-8"))
    physical_station_ids = set(
        sid for line in topology["lines"].values() for sid in line
    )
    day = date.fromisoformat(args.service_date)
    start = int(time_seconds(pd.Series([args.start])).iloc[0])
    end = int(time_seconds(pd.Series([args.end])).iloc[0])
    if not 0 <= start < end:
        raise ValueError("Invalid time window")
    agency = read_gtfs(raw / "agency.txt")
    routes = read_gtfs(raw / "routes.txt")
    trips = read_gtfs(raw / "trips.txt")
    stops = read_gtfs(raw / "stops.txt")
    cal = (
        read_gtfs(raw / "calendar.txt")
        if (raw / "calendar.txt").exists()
        else pd.DataFrame()
    )
    exc = (
        read_gtfs(raw / "calendar_dates.txt")
        if (raw / "calendar_dates.txt").exists()
        else pd.DataFrame()
    )
    active, regular, added, removed = active_services(cal, exc, day)
    if not active:
        raise ValueError("Requested date has no active services; inspect raw calendars")
    agencies = set(agency.loc[agency.agency_name.eq("Transperth"), "agency_id"])
    if (
        not agencies
        or not agency.loc[agency.agency_id.isin(agencies), "agency_timezone"]
        .eq("Australia/Perth")
        .all()
    ):
        raise ValueError("Missing Transperth agency or unexpected timezone")
    # Match the existing prototype's regular weekday baseline. Preserve event data in raw.
    excluded = set() if args.include_event_routes else {"WES-RAI-4313"}
    eligible_routes = routes[
        routes.agency_id.isin(agencies)
        & routes.route_type.isin(["2", "3"])
        & ~routes.route_id.isin(excluded)
    ]
    day_trips = trips[
        trips.service_id.isin(active) & trips.route_id.isin(eligible_routes.route_id)
    ].copy()
    if (
        day_trips.trip_id.duplicated().any()
        or routes.route_id.duplicated().any()
        or stops.stop_id.duplicated().any()
    ):
        raise ValueError("Duplicate GTFS primary keys")
    route_mode = eligible_routes.set_index("route_id").route_type.map(
        {"2": "rail", "3": "bus"}
    )
    day_trips["mode"] = day_trips.route_id.map(route_mode)
    day_trip_ids = set(day_trips.trip_id)
    selected_ids = set()
    event_parts = []
    raw_rows = 0
    max_departure = 0
    missing_departures = 0
    for chunk in pd.read_csv(
        raw / "stop_times.txt", dtype=str, keep_default_na=False, chunksize=150000
    ):
        chunk.columns = chunk.columns.str.strip()
        raw_rows += len(chunk)
        chunk = chunk[chunk.trip_id.str.strip().isin(day_trip_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.apply(lambda col: col.str.strip())
        sec = time_seconds(chunk.departure_time)
        missing_departures += int(sec.isna().sum())
        if sec.notna().any():
            max_departure = max(max_departure, int(sec.max()))
        within = sec.ge(start) & sec.lt(end)
        events = chunk[within.fillna(False)].copy()
        events["departure_seconds"] = sec[within.fillna(False)].astype("int64")
        selected_ids.update(events.trip_id)
        event_parts.append(events)
    selected = day_trips[day_trips.trip_id.isin(selected_ids)].copy()
    counts = selected["mode"].value_counts()
    if counts.get("rail", 0) == 0 or counts.get("bus", 0) == 0:
        raise ValueError("Date/window does not contain both rail and bus services")
    window = pd.concat(event_parts, ignore_index=True)
    window = window.merge(
        selected[["trip_id", "route_id", "service_id", "mode"]],
        on="trip_id",
        validate="many_to_one",
    )
    out.mkdir(parents=True, exist_ok=True)
    # Keep entire stop sequences for selected trips, including visits outside the window.
    first = True
    used_stop_ids = set()
    full_count = 0
    for chunk in pd.read_csv(
        raw / "stop_times.txt", dtype=str, keep_default_na=False, chunksize=150000
    ):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk[chunk.trip_id.str.strip().isin(selected_ids)].copy()
        if chunk.empty:
            continue
        chunk = chunk.apply(lambda col: col.str.strip())
        used_stop_ids.update(chunk.stop_id)
        full_count += len(chunk)
        chunk.to_csv(
            out / "stop_times.txt",
            index=False,
            lineterminator="\n",
            mode="w" if first else "a",
            header=first,
        )
        first = False
    parent_ids = set(stops.loc[stops.stop_id.isin(used_stop_ids), "parent_station"]) - {
        ""
    }
    selected_stops = stops[
        stops.stop_id.isin(used_stop_ids | parent_ids | physical_station_ids)
    ]
    if not physical_station_ids <= set(stops.stop_id):
        raise ValueError("Verified physical station missing from raw stops.txt")
    selected_routes = eligible_routes[eligible_routes.route_id.isin(selected.route_id)]
    selected_agency = agency[agency.agency_id.isin(selected_routes.agency_id)]
    for name, frame in [
        ("agency.txt", selected_agency),
        ("routes.txt", selected_routes),
        ("trips.txt", selected.drop(columns="mode")),
        ("stops.txt", selected_stops),
    ]:
        frame.to_csv(out / name, index=False, lineterminator="\n")
    window["stop_sequence"] = pd.to_numeric(
        window.stop_sequence, errors="raise"
    ).astype(int)
    window = window.sort_values(["mode", "trip_id", "stop_sequence"])
    window.to_csv(out / "window_departures.csv", index=False, lineterminator="\n")
    selected[selected["mode"].eq("rail")].drop(columns="mode").to_csv(
        out / "rail_trips.csv", index=False, lineterminator="\n"
    )
    selected[selected["mode"].eq("bus")].drop(columns="mode").to_csv(
        out / "bus_trips.csv", index=False, lineterminator="\n"
    )
    # One-row-per-active-service provenance, including additions and removals for this date.
    service_rows = [
        {
            "service_id": sid,
            "service_date": day.isoformat(),
            "regular_today": sid in regular,
            "added_exception": sid in added,
            "removed_exception": sid in removed,
            "active_today": sid in active,
        }
        for sid in sorted(regular | added | removed)
    ]
    pd.DataFrame(service_rows).to_csv(
        out / "service_selection.csv", index=False, lineterminator="\n"
    )
    if not cal.empty:
        cal[cal.service_id.isin(selected.service_id)].to_csv(
            out / "calendar.txt", index=False, lineterminator="\n"
        )
    if not exc.empty:
        exc[exc.service_id.isin(selected.service_id)].to_csv(
            out / "calendar_dates.txt", index=False, lineterminator="\n"
        )
    if (raw / "transfers.txt").exists():
        transfers = read_gtfs(raw / "transfers.txt")
        transfers[
            transfers.from_stop_id.isin(selected_stops.stop_id)
            & transfers.to_stop_id.isin(selected_stops.stop_id)
        ].to_csv(out / "transfers.txt", index=False, lineterminator="\n")
    shape_ids = set(selected.shape_id) - {""}
    if (raw / "shapes.txt").exists():
        first = True
        for chunk in pd.read_csv(
            raw / "shapes.txt", dtype=str, keep_default_na=False, chunksize=150000
        ):
            chunk.columns = chunk.columns.str.strip()
            chunk = chunk[chunk.shape_id.str.strip().isin(shape_ids)].copy()
            if chunk.empty:
                continue
            chunk = chunk.apply(lambda col: col.str.strip())
            chunk.to_csv(
                out / "shapes.txt",
                index=False,
                lineterminator="\n",
                mode="w" if first else "a",
                header=first,
            )
            first = False
    # Compare the chosen snapshot's parent station IDs with the existing prototype.
    rail_ids = set(selected.loc[selected["mode"].eq("rail"), "trip_id"])
    full_st = read_gtfs(out / "stop_times.txt")
    rail_stops = stops[
        stops.stop_id.isin(full_st.loc[full_st.trip_id.isin(rail_ids), "stop_id"])
    ]
    rail_parent_ids = set(rail_stops.parent_station) - {""}
    rail_parents(full_st[full_st.trip_id.isin(rail_ids)], selected_stops)
    time_seconds(full_st.arrival_time)
    time_seconds(full_st.departure_time)
    blank_parent = rail_stops[rail_stops.parent_station.eq("")]
    assert set(full_st.trip_id) == set(selected.trip_id)
    assert set(full_st.stop_id) <= set(selected_stops.stop_id)
    assert set(selected.route_id) <= set(selected_routes.route_id)
    assert set(selected.service_id) <= active
    assert (
        window.departure_seconds.ge(start).all()
        and window.departure_seconds.lt(end).all()
    )
    assert not full_st[["trip_id", "stop_sequence"]].duplicated().any()
    for _, seq in full_st.groupby("trip_id", sort=False):
        if not pd.to_numeric(seq.stop_sequence).is_monotonic_increasing:
            raise ValueError("stop_times not in increasing trip sequence order")
    summary = {
        "service_date": day.isoformat(),
        "timezone": "Australia/Perth",
        "window_start": args.start,
        "window_end_exclusive": args.end,
        "selection_rule": "Active GTFS service-day trips with at least one scheduled stop departure in [07:00,09:00); full trip stop sequences retained separately",
        "agency_rule": "agency_name == 'Transperth' and route_type in {2,3}",
        "excluded_event_route_ids": sorted(excluded),
        "raw_feed_calendar_envelope": {
            "start": cal.start_date.min() if not cal.empty else None,
            "end": cal.end_date.max() if not cal.empty else None,
        },
        "active_service_count_feed_wide": len(active),
        "regular_service_count_feed_wide": len(regular),
        "added_exceptions_feed_wide": len(added),
        "removed_exceptions_feed_wide": len(removed),
        "selected_service_count": selected.service_id.nunique(),
        "selected_route_count": selected_routes.route_id.nunique(),
        "rail_route_count": int(selected_routes.route_type.eq("2").sum()),
        "bus_route_count": int(selected_routes.route_type.eq("3").sum()),
        "rail_trips": int(counts["rail"]),
        "bus_trips": int(counts["bus"]),
        "rail_window_departures": int(window["mode"].eq("rail").sum()),
        "bus_window_departures": int(window["mode"].eq("bus").sum()),
        "selected_full_stop_times_rows": full_count,
        "raw_stop_times_rows": raw_rows,
        "selected_stops_with_parents": len(selected_stops),
        "rail_parent_station_count": len(rail_parent_ids),
        "rail_stops_missing_parent": len(blank_parent),
        "active_transperth_max_departure_seconds": max_departure,
        "active_transperth_missing_departure_times": missing_departures,
        "download": download,
        "physical_station_count": len(physical_station_ids),
        "event_variant": args.include_event_routes,
        "limitations": [
            "Scheduled service, not measured passengers or traffic.",
            "This subset does not validate physical rail adjacency.",
            "Previous service-day overnight trips are not included; the analysis window is expressed relative to the selected GTFS service day.",
            "Maps normal timetable and GTFS feed may differ even with the same date.",
        ],
    }
    summary["selection_rule"] = (
        f"Active GTFS service-day trips with at least one scheduled stop departure in [{args.start},{args.end}); full trip stop sequences retained separately"
    )
    summary["prepared_files"] = file_records(p for p in out.iterdir() if p.is_file())
    write_json(summary, out / "snapshot_manifest.json")
    checks = {
        "raw_file_sha256": "passed against pinned acquisition manifest",
        "both_modes_present": True,
        "date_services_checked_with_exceptions": True,
        "trip_route_service_stop_foreign_keys": "passed",
        "duplicate_trip_stop_sequences": "none",
        "stop_sequences_increasing": "passed",
        "window_bounds": "passed",
        "full_sequences_preserved": True,
        "rail_stops_missing_parent": len(blank_parent),
    }
    write_json(checks, out / "validation.json")
    (out / "README.md").write_text(
        f"# GTFS service-day snapshot\n\nService date: {day.isoformat()}, Australia/Perth. "
        f"Window: [{args.start}, {args.end}). Full sequences of trips with a departure "
        "in this window are retained; window_departures.csv contains only window events.\n\n"
        "Recreate with `make data`. Large tables are ignored; the manifest hashes each table. "
        "The 86 mapped physical stations include unserved Showgrounds. See docs/data.md.\n",
        encoding="utf-8",
        newline="\n",
    )
    (out / ".gitignore").write_text(
        "*\n!.gitignore\n!snapshot_manifest.json\n!validation.json\n!README.md\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in {"download", "prepared_files", "limitations"}
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
