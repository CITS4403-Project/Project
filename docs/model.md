# Model specification and frozen interfaces

This document is the contract between the package scaffold (P0.1) and the
implementation and experiment issues P1.1 to P1.6 and P2.1 to P2.5. It fixes
the graph definition, the load and capacity equations, the failure and
redistribution rules, the metric definitions and the public API. After gate M0
the names listed here change only additively, through a reviewed pull request.

The shared runtime objects live in `src/transperth/config.py`. The equations
and conventions are below.

## 1. Network definition

The rail layer is an undirected simple graph G = (V, E).

V is the set of 86 mapped suburban rail stations. 85 of them have scheduled
weekday service in the frozen window; Showgrounds sits on the mapped corridor
but has no weekday stop event. Each node carries `station_id` (string), `name`,
`lat`, `lon`, `modes`, `degree`, `trips_served`, `am_peak_stops`,
`betweenness` and `lines`.

E is the set of 85 verified adjacent segments between consecutive stations on
the mapped corridors. Each edge carries `trips`, the number of scheduled
traversals in the snapshot, and `distance_m`, the great-circle length in
metres.

The graph is connected and a tree: 86 mapped stations, 85 edges. The prototype
had 96 edges; P0.2 audited each against the official October 2025 map and
recorded the outcome in `data/processed/topology_audit.csv`: 85 verified
adjacent edges retained, 8 skipped-station shortcuts removed with their trips
expanded onto the corridor segments, and 3 off-map connections excluded. The
frozen regression counts for the 2026-10-05 07:00-09:00 window are 86 stations,
85 edges, 8 rail routes, 271 rail trips and 4,324 bus trips. The Stadium
event-only route WES-RAI-4313 is excluded from the baseline and kept as an
optional event-day variant.

Frozen tables live in `data/processed/`: `stations.csv` and `rail_edges.csv`
for the rail layer, `bus_coverage.csv` for bus stops and routes within 400 m
of each station, and `backup_edges.csv` for verified station pairs with
effective travel times and provenance. P0.2 owns the column dictionary and
checksums in `docs/data.md`.

## 2. Loads

The initial load of station i is its betweenness centrality on the intact
graph, computed with networkx `betweenness_centrality(G, normalized=False)`:

    L0_i = sum over ordered pairs (s, t), s != i != t, of sigma_st(i) / sigma_st

where sigma_st is the number of shortest paths from s to t and sigma_st(i)
is the number that pass through i. The values are unnormalised because the
graph shrinks during a cascade; normalised values would rise as the node set
shrinks, so rounds and scenarios would not be comparable.

Load modes (`LoadMode`):

| Mode | Load |
|---|---|
| `betweenness` | L0_i on the unweighted rail graph |
| `betweenness_freq` | betweenness with edge length 1 / trips, so frequent services carry shorter paths |
| `betweenness_plus_trips` | L0_i + 0.1 * trips_served, a throughput proxy that gives leaf stations a positive baseline |
| `demand` | GTFS AM-peak boardings proxy as flow, service frequency as capacity (P2.4) |

## 3. Capacity

    C_i = (1 + alpha) * L0_i,   alpha >= 0

Capacities are computed once from the intact graph and stay fixed during a
cascade. A station with L0_i = 0 gets C_i = 0, so it can only fail as the
explicit trigger; load 0 > 0 is false.

## 4. Failure and redistribution

`CascadeConfig` holds the parameters of one run: `alpha`, `trigger`,
`target`, `rule`, `dynamic`, `load_mode`, `seed` and `tolerance`. The first
station to fail is `target` when it names a station. Otherwise `trigger`
selects it:

| Trigger | First station to fail |
|---|---|
| `load` | argmax L0_i |
| `degree` | highest degree, ties broken by sorted station_id |
| `random` | uniform over sorted station ids, drawn from the run seed |

Synchronous rule. Rounds are algorithmic updates, not minutes. At each round
every surviving station whose current load exceeds its capacity,

    L_i > C_i + tolerance

fails. All overloaded stations of a round are collected first and removed
together; the next round starts from the reduced graph. The cascade stops at
the first round with no overload. The trigger is removed before round 1.

Static mode (`dynamic=False`). Loads start at L0 and are moved rather than
recomputed. The trigger's load is redistributed to its surviving neighbours,
and every failed station forwards the load it carries when it fails. No
shortest path is recomputed.

Dynamic mode (`dynamic=True`). After each removal round, loads are recomputed
as betweenness on the surviving graph, which models full shortest-path
rerouting. The redistribution rules do not apply in this mode.

Redistribution rules (`RedistributionRule`):

| Rule | Share received by surviving neighbour m |
|---|---|
| `equal` | amount / number of surviving neighbours |
| `capacity` | amount * C_m / sum of C over surviving neighbours |

If every surviving neighbour has zero capacity, the `capacity` rule falls back
to equal shares. The redistributed amount equals the load the failed station
carried, so no load is created or destroyed. A failed station with no
surviving neighbour removes its load from the system.

`CascadeResult` records `n_initial`, `failed`, `gcc`, `gcc_fraction`,
`failed_fraction`, `avalanche_sizes`, `rounds` and the derived `n_failed`.

## 5. Multilayer model

P1.5 builds a layered graph from the rail graph. Each station gets two nodes: a
terminal `T:<station_id>`, which persists, and a rail facility
`R:<station_id>`, which can close. An access edge joins them with a fixed one
minute time.

Rail edges join `R:` facilities. Travel time is

    t_e = max(0.1, distance_m / 1000 / speed_kmh * 60) minutes

with `speed_kmh` defaulting to 40. Backup edges join terminals with verified
effective times, including walking and waiting. Duplicate endpoints keep the
fastest path. The `kind` column separates existing buses from emergency buses.
Standby buses activate immediately after the trigger and have unlimited
capacity in the baseline scenario.

Loads in the multilayer model are terminal-subset weighted betweenness over
`T:` nodes. OD reachability and travel times use the undirected terminal pairs
of the intact graph as the denominator: pairs that become unreachable count as
unserved instead of dropping out of the metric.

## 6. Metrics

| Metric | Definition |
|---|---|
| GCC | largest connected component of the surviving graph |
| `gcc_fraction` | gcc_size / n_initial |
| LCC | second-largest connected component, used for susceptibility peaks |
| `isolated` | stations with degree 0 |
| ASPL | mean shortest path length within the largest component; unreachable pairs are excluded |
| `efficiency` | Sum of inverse hop distances over surviving ordered pairs, divided by n_initial * (n_initial - 1); removed and unreachable pairs contribute 0 |
| `network_damage` | 1 - efficiency / baseline_efficiency of the intact graph |
| `failed_fraction` | failed stations / n_initial for one cascade |
| avalanche size | newly failed stations per round |
| avalanche duration | number of cascade rounds up to the stable state |
| `alpha_star` | smallest grid tolerance at which the max-load trigger stays contained, failed_fraction <= 1 / n |
| `served_od_fraction` | reachable undirected terminal pairs / original pairs |
| critical fraction | removal fraction where the collapse curve crosses the chosen GCC threshold |

`MetricsBundle` carries the matching fields: `n_nodes`, `n_edges`, `gcc_size`,
`gcc_fraction`, `lcc_size`, `lcc_fraction`, `isolated`, `aspl`, `efficiency`,
`network_damage` and `served_od_fraction`.

## 7. Result schema

Every experiment writes its tables through `experiments.save_table` as
`results/<experiment>/<name>.csv` with a `<name>.meta.json` sidecar. The
sidecar holds:

| Field | Content |
|---|---|
| `experiment` | experiment family name |
| `created_utc` | ISO 8601 UTC timestamp |
| `seed` | master seed of the run |
| `package_version` | version of the transperth package |
| `params` | run parameters as a JSON object |
| `inputs` | input path to SHA-256 hex digest |
| `versions` | installed analysis package versions |

Result directories are fixed per experiment family: `results/percolation/`
(P2.1), `results/cascade/` (P2.2), `results/recovery/` (P2.3),
`results/demand/` (P2.4) and `results/uncertainty/` (P2.5).

## 8. Frozen API

The modules below are owned by the issue named in the PLAN ownership table. A
signature in this section may gain optional keyword arguments or new sibling
functions after M0; existing names and positional order do not change.

```python
# network.py (P1.1)
load_rail_graph(*, stations_csv=STATIONS_CSV, edges_csv=RAIL_EDGES_CSV) -> networkx.Graph
load_bus_coverage(*, coverage_csv=BUS_COVERAGE_CSV) -> pandas.DataFrame

# loads.py (P1.3)
initial_loads(graph, mode: LoadMode = "betweenness") -> dict[str, float]
capacities(loads: Mapping[str, float], alpha: float) -> dict[str, float]

# metrics.py (P1.2)
compute_metrics(graph, *, baseline_efficiency=None, served_pairs=None, total_pairs=None, n_initial=None) -> MetricsBundle

# failure.py (P1.3)
target_order(graph, measure: str = "degree", *, static: bool = True) -> list[str]
percolation_curve(graph, *, attack: str = "random", measure: str = "degree",
                  fractions: Sequence[float], n_seeds: int = 100, seed: int = 0) -> pandas.DataFrame
critical_fraction(curve: pandas.DataFrame, *, threshold: float = 0.5,
                  column: str = "gcc_fraction") -> float

# cascade.py (P1.4)
simulate_cascade(graph, config: CascadeConfig, *, baseline_loads=None) -> CascadeResult

# multilayer.py (P1.5)
build_layers(rail, *, speed_kmh: float = 40.0, access_minutes: float = 1.0) -> networkx.Graph
add_backup_edges(layered, backup_table: pandas.DataFrame, *, time_factor: float = 1.0) -> networkx.Graph
terminal_loads(layered) -> dict[str, float]

# strategies.py (P1.6); all classes satisfy Strategy
NoBackup(), ExistingBus(...), ShuttleBridging(...), CorridorReinforcement(...), DemandAdaptive(...)

# experiments.py (P0.1 scaffold, extended by P1.2)
RunMeta.create(experiment, *, seed, params=None, inputs=())
save_table(table, path, meta, *, index=False) -> tuple[Path, Path]
results_dir(experiment: str) -> Path
file_sha256(path: str | Path) -> str

# utils/stats.py (P1.2)
bootstrap_ci(values, statistic=mean, *, n_boot: int = 10_000, confidence: float = 0.95,
             seed: int = 0) -> tuple[float, float]

# plotting.py (P0.1, extended by P2.x)
apply_style() -> None
save_figure(fig, name: str, *, figures_dir=FIGURES_DIR, dpi: int = FIGURE_DPI) -> Path
plot_series(table: pandas.DataFrame, x: str, y: str, *, ax=None, label=None, **style)
plot_network(stations: pandas.DataFrame, edges: pandas.DataFrame, *,
             ax=None, lat: str = "lat", lon: str = "lon", size_by: str = "trips_served")
```

`simulate_cascade` returns a `CascadeResult`, `compute_metrics` returns a
`MetricsBundle` and every strategy satisfies the `Strategy` protocol. The
dataclasses and the protocol are defined in `config.py` and are imported from
there by the implementation modules; they are not redefined per module.

Experiment scripts do not import from `investigations/`. Prototype material is
a reference and a regression target only.

## 9. Change control

Gate M0 closes when this document and the frozen tables are merged. From then
on:

- new parameters are added as optional keyword arguments with defaults that
  reproduce the current behaviour;
- a change to an existing signature, field or rule is a fix and needs the
  module owner's review in the pull request that makes it;
- every result file records input hashes, so a rule change can be traced to
  the results that need regeneration.

Module ownership follows section 6 of the implementation plan (`PLAN.md`);
shared files have one named owner and change by pull request.
