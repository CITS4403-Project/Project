# Frozen Transperth dataset

`make data` produces the project-root `data/processed/` inputs for P1.x and
P2.x. The investigation directory remains the original regression reference;
it is never a regeneration target. Experiment tables, rankings, sweeps and
avalanche arrays belong under `results/`, not `data/processed/`.

## Reproduce and validate

Install `requirements.txt` in a Python environment, then run from the project root:

```sh
make data PYTHON=python
python scripts/validate_data.py --gtfs-dir data/snapshots/2026-10-05_0700-0900 --output-dir data/processed
python -m unittest discover -s tests -p test_data_pipeline.py
```

On Windows without GNU Make, `python scripts/run_data.py` runs the same chain.
`make data` invokes the Python runner, so downloading does not require Bash,
curl or unzip. The Bash entry point remains available:

```sh
PYTHON=python bash scripts/download_data.sh --gtfs-dir data/raw/google_transit --output-dir data/raw
```

The runner verifies/downloads raw data, prepares an empty temporary snapshot,
builds the graph, independently validates it and compares every frozen output
hash before publishing. A repeated run accepts an existing verified raw cache
and regenerates the same bytes. Operational timestamps are not regenerated.
`.gitattributes` fixes newline handling across platforms; numeric CSV outputs
use 12 significant digits. Use the pinned analysis dependencies when rebuilding.

Raw data is **not committed**. `data/source_manifest.json` records original
acquisition time, URL, HTTP headers, ZIP size/SHA-256 and each extracted member's
size/SHA-256. The pinned ZIP hash is:

```text
9f4482331fb37dbd71708ccdd187f092be42ad7f2e3536e798416bf99422b9fa
```

The official URL is mutable and PTA can replace or withdraw the feed. A fresh
clone downloads it when absent, but accepts it only if it matches this hash.
If the official feed changes, the command stops before replacing frozen
outputs. Restore the exact archived ZIP with:

```sh
python scripts/run_data.py --archive /path/to/original/google_transit.zip
```

There is no immutable hosted mirror; future availability of the
old ZIP is a limitation, not a promise that a live feed will remain unchanged.
A new feed requires reviewed source, snapshot, topology and checksum changes.
`data/frozen_checksums.json` covers the tracked snapshot metadata and all
processed outputs. `snapshot_manifest.json` also hashes the ignored large
snapshot tables; `processed/manifest.json` hashes the small input tables and
records topology, source, snapshot, baseline and manual evidence provenance.

## Selection and units

The frozen service day is **2026-10-05**, timezone **Australia/Perth**. Weekly
`calendar.txt` eligibility is evaluated for that day, then `calendar_dates.txt`
additions/removals override it. Only agency `Transperth`, rail `route_type=2`
and bus `route_type=3` are selected. Event route `WES-RAI-4313` is excluded.

A trip is selected if at least one scheduled departure lies in the half-open
window **[07:00:00, 09:00:00)**. Entire sequences of those trips are retained in
`stop_times.txt`, including visits outside the window. `window_departures.csv`
contains only the events inside it. The frozen selection contains 8 rail
routes, 315 bus routes, **271 rail trips**, **4,324 bus trips**, 2,940 rail and
111,971 bus window departure events, and 146,149 full-sequence stop-time rows.
These are scheduled events, not observed boardings or train capacity.

GTFS time is seconds from the service-day origin, without reducing hours modulo
24. `25:40:00` is 92,400 seconds. Empty times remain missing and cannot select
a trip; they are not interpolated. Previous-service-day overnight trips are
not included. Leading/trailing whitespace is stripped from every header and
string. Rail platforms require a nonblank parent that exists in `stops.txt`
with `location_type=1`. Duplicate keys or conflicting calendar exceptions fail.
See the [GTFS Schedule Reference](https://gtfs.org/documentation/schedule/reference/)
for the retained standard GTFS columns and service-day conventions.

## Verified topology and baseline differences

The topology is the undirected **station adjacency along the eight corridors**
on the [official October 2025 system map](https://www.transperth.wa.gov.au/Portals/0/SYSTEM%20MAP%20OCTOBER%202025.pdf),
linked from [Transperth's map page](https://www.transperth.wa.gov.au/Journey-Planner/Network-Maps).
The manually transcribed station order, map URL, edition, verification date,
name aliases and source-PDF SHA-256 live in `data/verified_topology.json`.
This verifies mapped passenger-network adjacency, not track, signalling,
junction geometry or operating permissions. The PDF is not redistributed.

All **96** baseline edges have a row in `processed/topology_audit.csv`:

- **85 verified adjacent:** retained, with dated service counts regenerated.
- **8 skipped-station shortcuts:** removed as direct edges. The audit records
  each mapped replacement path. Examples: Loch Street–Claremont expands via
  Showgrounds; Perth–East Perth expands via McIver and Claisebrook.
- **3 off-map connections:** Leederville–Perth, Leederville–Perth Stadium,
  City West–Perth Underground. Excluded from the regular mapped graph; this
  does not claim the underlying infrastructure could never support them.

The resulting graph has **86 stations, 85 edges, one connected component** and
is a tree. The Thornlie–Cockburn corridor joins the surface and underground
branches in the south. Perth and Perth Underground remain separate GTFS
parents. Their CBD pedestrian interchange is recorded in the topology metadata,
but is not invented as a railway edge; P1.5 can add an explicit walking link.
Without that link, a rail-only path between them goes via the south. This
matters for shortest paths and cascade interpretation.

Showgrounds is on the mapped corridor and is therefore included even though
none of the selected weekday trips stops there. All 52 selected traversals
between Loch Street and Claremont pass through its two physical segments.
There are **85 served parent stations** and **86 mapped stations**; these
counts answer different questions.

The old 7,568 rail-trip count pooled every service day on the pre-freeze
prototype feed (downloaded 2026-09-30); it is not reproducible from the pinned
ZIP, which gives 6,465 trips for the same route filter. `trips_served` now
counts full stop visits of the 271 selected trips. `am_peak_stops` changes from
the old 06:00–09:00/all-feed proxy to actual 07:00–09:00 events on the selected
day.
Degrees and descriptive normalized betweenness are recomputed on the corrected
graph. For example, Perth's degree changes from 7 to 2. The prototype's cascade
load calculations should still compute their own **unnormalized** betweenness.

Bus coverage changes for two documented reasons: only stops with bus departures
inside the selected window count, and distance uses a true great-circle 400 m
radius. The old circular query in latitude/longitude degrees understated
east/west coverage. Complete sorted route IDs are retained without truncation.

These are intentional fixes; neither the baseline tables nor prototype
experiments are overwritten. P1.7 aligned the `docs/model.md` regression anchor
with the frozen 86/85/271 counts after the P0.2 review. The only shared
Makefile change in this PR is the `data` recipe; P0.1 owns review of that
integration.

## Optional event-day variant

Keep event data in the raw feed and enable it explicitly in a separate output
directory, on a date/window when its services are active:

```sh
python scripts/run_data.py --include-event-routes --service-date 2026-10-05 --start 15:00:00 --end 18:00:00 --data-dir results/event_data
```

Enabling the flag does not fabricate an event service if the calendar has none.
The regular frozen directory cannot be overwritten by alternate parameters.
Mapped express pairs expand along verified corridors. If an event trip uses
an unverified off-map pair, its **entire trip** is omitted from edge-traversal
counts and listed in `unmapped_event_trips.csv`. Its scheduled stop visits
remain in the snapshot/station statistics. Such a variant is explicitly
incomplete until that routing is independently verified. The regular window
has no omitted trips; unverified regular pairs raise an error.

## Processed data dictionary

IDs are UTF-8 strings, even when they look numeric. Regenerated CSVs are sorted
by string station IDs, and `rail_edges.csv` stores each unordered endpoint pair
in lexical order. The byte-preserved topology audit and manual evidence files
keep the orientation of their legacy inputs. Blank semicolon lists mean no
members. Coordinates are WGS84 decimal degrees.

### stations.csv — one row per mapped parent station

| Column | Type/unit | Meaning/source |
|---|---|---|
| `station_id` | string | GTFS parent `stop_id`; primary key |
| `name` | string | GTFS parent `stop_name` |
| `lat`, `lon` | decimal degrees | GTFS parent location; not platform coordinates |
| `modes` | semicolon list | GTFS extension `supported_modes`; descriptive, not a capacity |
| `degree` | integer edges | Degree on the 86/85 mapped rail graph |
| `trips_served` | integer stop visits | Full-sequence rail visits of selected trips, including outside-window visits; zero for unserved stations |
| `am_peak_stops` | integer stop events | Rail departure events inside the half-open window [07:00,09:00) on the selected day |
| `betweenness` | dimensionless [0,1] | NetworkX normalized, unweighted node betweenness; descriptive only |
| `lines` | semicolon list | Sorted names of mapped corridors containing this station |

### rail_edges.csv — one row per verified undirected adjacency

| Column | Type/unit | Meaning/source |
|---|---|---|
| `station_a`, `station_b` | station IDs | Endpoints referencing `stations.csv`; pair is unique, no self loops |
| `trips` | integer traversals | Both directions combined; full traversals of selected trips, with skipped stations expanded; not hourly frequency or passengers |
| `distance_m` | metres | Great-circle endpoint separation, Earth radius 6,371,000 m; not track length or rail travel time |

### bus_coverage.csv — one row per mapped station

| Column | Type/unit | Meaning/source |
|---|---|---|
| `station_id`, `name` | ID/string | Station reference and readable name |
| `n_bus_stops` | integer stops | Distinct GTFS bus stop locations with a window departure, within <=400 m of parent station |
| `n_bus_routes` | integer routes | Distinct GTFS route IDs serving these qualifying stop events |
| `bus_route_ids` | semicolon list | Complete sorted route IDs; route IDs are not public route numbers |

Geographical proximity does not guarantee a walkable transfer, useful onward
destination, sufficient capacity, or buses during a rail disruption.

### backup_edges.csv — manually observed candidate backup links

| Column | Type/unit | Meaning/source |
|---|---|---|
| `station_a`, `station_b` | station IDs | Unique undirected candidate pair |
| `minutes` | minutes | Maximum of the two directions' arrival-minus-requested-08:00 times, including walking, waiting and transfers |
| `kind` | string | `existing_bus` for all four frozen pairs; not an emergency shuttle |
| `route_or_road` | string | Public bus numbers per direction; `+` means multiple bus legs |
| `source` | string | Query/retrieval date, calculation rule, evidence-pair reference and original Maps URLs |

`backup_edges_google_maps_evidence.json` preserves eight direction records,
the 2026-10-05T08:00:00+08:00 query, 2026-10-04 retrieval date, itinerary and
effective minutes, walking/ride minutes, boarding/alighting notes and URLs.
The four frozen pairs in the CSV are Perth–Glendalough,
Perth–Bayswater, Perth–Victoria Park and Oats Street–Canning Bridge (the last
uses a bus transfer). Their weights are 38, 38, 38 and 42 minutes.

These two files are copied **byte for byte** from the committed investigation
files via versioned `data/manual/` inputs. `.gitattributes` disables automatic
checkout newline conversion for the baseline and these evidence files, so a
Windows checkout cannot change their recorded bytes. The processed manifest records hashes;
validation checks endpoints, both directions and the maximum effective time.
They cannot be regenerated from GTFS alone, are not globally optimized, and
are not proof of disruption-time capacity. P1.5 builds GTFS candidates separately.

### Audit and metadata files

| File/columns | Meaning |
|---|---|
| `topology_audit.csv`: `station_a`, `station_b`, `name_a`, `name_b` | Each original edge and its readable endpoints |
| `status` | `verified_adjacent`, `expand_skipped_stations`, or `exclude_off_map_connection` |
| `replacement_station_ids` | Ordered semicolon path for retained/expanded edges; blank for exclusions |
| `note` | Correction rationale |
| `unmapped_event_trips.csv`: `trip_id`, `reason` | Event trips omitted from traversal counts because routing is unverified; header-only for the regular snapshot |
| `manifest.json` | Dataset/window counts, topology source/hash, input hashes, manual provenance and output file bytes/SHA-256 |
| `validation.json` | Deterministic passed checks and graph/audit counts; regenerated independently |

### Snapshot additions

`rail_trips.csv` and `bus_trips.csv` retain the standard GTFS trip schema,
split by mode. `window_departures.csv` retains the standard stop-time columns
and adds `departure_seconds` (integer service-day seconds), `route_id`,
`service_id` (GTFS references) and `mode` (`rail`/`bus`).
`service_selection.csv` contains `service_id`, ISO `service_date`, and booleans
`regular_today`, `added_exception`, `removed_exception`, `active_today` for
every service active in the weekly calendar or changed by a same-day exception.
`snapshot_manifest.json` records selection rules, feed-wide service changes,
selected counts, source acquisition and each prepared table's checksum.
Snapshot GTFS tables retain selected agencies, routes, trips, stops plus
parents/mapped stations, calendars, relevant transfers and trip shapes.

## Sources, licence and remaining limits

The feed was acquired from the [PTA spatial data page](https://www.transperth.wa.gov.au/About/Spatial-Data-Access)
on 2026-10-04; its HTTP Last-Modified was 2026-09-30. PTA grants limited,
revocable use/reproduction/redistribution rights under that page's terms,
requires source attribution and restricts use of PTA trademarks and copyrighted
materials with the data. Required attribution: **This data is available free
of charge from www.transperth.wa.gov.au.** The full terms at the source apply;
this is not an unrestricted open-data licence. The network-map PDF is a
separate reference, not covered by an assumed GTFS licence.

Scheduled service and parent station locations approximate a passenger network.
They do not measure OD demand, crowding, infrastructure capacity, road traffic,
walking accessibility or replacement dispatch. GTFS and Maps are separately
dated sources. The regular graph deliberately omits event-only/unverified
connections and pedestrian interchanges. Results from the old shortcut graph
must be regenerated on these inputs rather than compared as if the topology
and time denominator were unchanged.
