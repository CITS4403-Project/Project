# Multilayer recovery and bus strategy comparison (P2.3)

This page documents the RQ3 experiment added by issue #33. The model builds on
the P1.5 terminal/facility layers, the P1.6 budgeted deployment strategies and
the frozen `data/processed/` inputs; the runner is `scripts/run_recovery.py`
and the library half is `src/transperth/recovery.py`.

## Scenarios

The scenario manifest is regenerated from the frozen rail graph, so the
definitions stay in code rather than in a checked-in scenario file:

| Kind | Definition |
|---|---|
| `line_closure` | a line's exclusive stations: stations that no other line lists. Lines with no exclusive station are skipped |
| `interchange` | every station served by at least three lines (the runner's `--min-interchange-lines` widens or narrows the rule) |
| `random` | seeded failure sets of `--random-size` distinct stations, drawn without replacement with the P1.2 `child_seeds` scheme |

Every scenario removes its stations' `R:` facilities together, before the first
load check. This is the same batch-trigger semantics `cascade.simulate_cascade`
gained in P2.2 (`initial_failed`): the batch replaces the single trigger, is
excluded from the avalanche sizes, and capacities stay fixed from the intact
baseline. `tests/test_recovery.py` cross-checks the layered port against the
merged engine on the same batch.

## Served demand

`results/demand/` (P2.4) defines the AM-peak boardings proxy `am_peak_stops`.
The recovery runner turns it into OD weights with the endpoint-sum rule:
the weight of pair `{a, b}` is `am_peak_stops_a + am_peak_stops_b`. This is a
scenario weighting that scores each endpoint's generated trips, not an OD
matrix estimate; the unweighted served-OD fraction is reported next to the
weighted served-demand fraction so the two can be compared.

## Strategy comparison

All five P1.6 strategies (`no_backup`, `existing_bus`, `shuttle_bridging`,
`corridor_reinforcement`, `demand_adaptive`) are constructed on the same
candidate pool: the four manual ground-truth pairs from
`data/processed/backup_edges.csv` plus the GTFS-derived candidates from
`scripts/run_backup_candidates.py`. Where a pair appears in both sources the
manual row wins, so a scheduled ride time never replaces the verified effective
time. The derived candidates are regenerated from the frozen snapshot so the
committed sidecars can hash the snapshot tables; when the snapshot is absent
the runner reads `results/multilayer/gtfs_backup_candidates.csv` instead and
hashes that file.

For every scenario and every budget the runner calls
`Strategy.deploy(failed=...)` on the intact layered graph, applies the selected
rows to the same failed baseline with `multilayer.add_backup_edges`, and then
runs the layered cascade with the strategy's buses active. The reported
metrics describe the stable graph:

| Metric | Definition |
|---|---|
| `served_od_fraction` | reachable intact terminal pairs over the `3655` intact pairs |
| `served_demand_fraction` | reachable endpoint-sum weight over the total weight |
| `reachable_pair_count` | numerator of the served-OD fraction |
| `mean_travel_time_penalty_min` | mean `recovered - intact` minutes over the intact pairs reachable in the stable graph; positive is slower |
| `recovery_rounds` | secondary removal rounds after the closure; `0` means the closure was contained |
| `secondary_failures` | rail facilities closed after the batch |

The penalty is conditional on the reachable population, which is why the pair
count is always reported with it.

## Layered cascade

`recovery.layered_cascade` ports `prototype/bus_backup_cascade.py`:

* loads are `multilayer.terminal_loads`, the unnormalised terminal-subset
  betweenness returned per `R:` facility;
* capacities come from the **intact rail-only baseline** and stay fixed:
  `C_i = (1 + alpha) * L0_i` for both the rail-only and the bus scenarios;
* each round recomputes the loads on the surviving graph and removes every
  facility above its fixed capacity **together**, before the next round;
* the initial failure batch is removed before the first load check, so buses
  are active from the first post-trigger round and never prevent the batch;
* `overload_enabled=False` reproduces the prototype control: the batch is
  removed but no secondary closure can follow.

The three bus scenarios are `rail_only`, `manual_bus` (the four verified pairs
as standby) and `candidate_bus` (the whole pool as standby, unlimited
capacity). The runner writes the prototype-shaped scan, comparison, control,
history and loads tables; the comparison rows are paired bus-minus-rail deltas,
and the mechanism figure shows the load ratios at the strongest traced
bus-aggravated case.

On the frozen data the prototype's "buses worsen service" result reproduces
under a narrower condition than in the prototype. With the whole candidate
pool as standby, service never worsens; instead the buses aggravate the
**physical** cascade at low and moderate tolerance: their shortcuts raise the
terminal-subset load of some facilities above their fixed rail-only capacity,
so `candidate_bus` closes more rail facilities than `rail_only` after a
Bayswater or Beckenham closure while still reconnecting terminals and
improving served service. In the budgeted strategy comparison the effect can
also reach service: a single selected link reroutes traffic through one
facility, closes extra facilities and ends with lower served service than
doing nothing. The loads table names the overloaded facility (for example
West Leederville after an Airport Line closure) and the mechanism figure shows
its load ratio rising above one. Both readings come from the same run; the
tables and the figure are the evidence, and this page does not restate their
numbers.

## Regeneration

From the repository root, with the venv active and `PYTHONPATH=src`:

```sh
make recovery PYTHON=.venv/bin/python
# or directly:
PYTHONPATH=src .venv/bin/python scripts/run_recovery.py --budget 2 --seed 0
```

The default run writes `results/recovery/*.csv` with `.meta.json` sidecars and
`figures/fig_recovery_strategies.png`, `figures/fig_recovery_bus_cascade.png`
and `figures/fig_recovery_mechanism.png`. The bus-cascade grid matches the
prototype (`0` to `2` in steps of `0.05`); `--budget`, `--alpha`,
`--cascade-alphas`, `--focus-alphas`, `--triggers`, `--n-random` and
`--random-size` control the comparison and the cascade family. Tests:
`tests/test_recovery.py` (scenario assembly, demand weights, cascade wrapper)
and `tests/test_run_recovery.py` (runner tables, sidecars and budgets).