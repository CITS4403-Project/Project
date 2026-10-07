# Bus deployment strategies (P1.6)

The five classes in `transperth.strategies` satisfy `config.Strategy`.
Candidate-based constructors take a DataFrame with `station_a`, `station_b`,
`minutes` and `kind`. Reuse P1.5's derived candidates and manual evidence;
strategies do not derive GTFS links. `CorridorReinforcement` also takes the
intact rail graph as `rail=...`, with `lines` and `trips_served` attributes.

Candidates must have distinct, nonempty endpoints, finite positive time and
a P1.5-supported kind. Fastest duplicate pairs win; equal times are resolved
by sorted evidence fields, independently of table order. `selected_table(pairs)`
returns copies of the selected rows, including all original evidence columns,
ready for `multilayer.add_backup_edges`. Its edge writer retains the documented
route/source fields; additional GTFS evidence remains in the selected table.

`deploy` takes a layered graph, nonnegative integer budget, failed station IDs,
optional OD demand and a nonnegative integer seed. Returned pairs use raw
station IDs in lexical order. Its private copy removes any still-present failed
`R:` facilities; `T:` terminals remain eligible. The caller must apply selections
to the same failed baseline, not the intact rail graph. It never mutates inputs.

The budget counts newly selected deployment links, not physical vehicles.
Already active direct terminal edges that are at least as fast are skipped.
Existing buses preloaded in the baseline remain active and consume no new link
budget. A slower direct baseline link can be replaced by a faster candidate.
This assumes immediate activation and unlimited baseline backup capacity; fleet
size, operating cost and frequency need a separate model.

| Strategy | Selection rule |
|---|---|
| `NoBackup()` | No additional links; measure the failed baseline |
| `ExistingBus(candidates)` | Shortest inactive `existing_bus` / `gtfs_candidate` links, then lexical pairs |
| `ShuttleBridging(candidates)` | Greedy gain in newly connected terminal pairs across components; stop when no gain remains |
| `CorridorReinforcement(candidates, rail=rail)` | Sum `trips_served` over each disrupted line touched by a candidate, then restored demand, then shorter time and lexical pair |
| `DemandAdaptive(candidates)` | Greedy gain in restored unmet weighted OD demand, then shorter time and lexical pair; stop at zero gain |

Corridor scores use scheduled station-service frequency as a throughput proxy,
not observed passenger counts. A candidate can reinforce a disrupted corridor
even if its terminals are already mutually reachable. Missing corridor metadata
gives no corridor score; bad throughput values are rejected.

OD weights are nonnegative and finite. Pairs are undirected: duplicate reversed
pairs are rejected. Absent demand gives unit weight to every terminal pair;
an explicitly empty/all-zero demand mapping gives no adaptive selections.
Gains are recalculated after every selection, so restored demand is counted
once. Lexical ties make these policies deterministic; the seed is accepted and
validated for protocol compatibility but does not randomise equal scores.

From the root, with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH='src'`):

```sh
python -m transperth.strategy_example --budget 2 --seed 0
python -m pytest tests/test_strategies.py
```

The example hashes `data/examples/recovery_fixture.json`, explicitly labelled
synthetic evidence, and writes `results/recovery/strategy_comparison.csv` with
a sidecar. It uses identical failures and budgets for all five policies.
Unweighted served OD uses the intact terminal-pair total; served demand uses
the original total demand weight. `mean_reachable_minutes` averages shortest
travel times over reachable original terminal pairs only and is missing when
none are reachable. Compare it alongside service fractions because restoring
previously unreachable pairs can change that averaging population.
