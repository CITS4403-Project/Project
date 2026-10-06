# GTFS service-day snapshot

Service date: 2026-10-05, Australia/Perth. Window: [07:00:00, 09:00:00). Full sequences of trips with a departure in this window are retained; window_departures.csv contains only window events.

Recreate with `make data`. Large tables are ignored; the manifest hashes each table. The 86 mapped physical stations include unserved Showgrounds. See docs/data.md.
