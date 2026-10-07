# P2.2: Cascades and avalanche statistics

Run from the repository root with the pinned `requirements.txt` environment:

```bash
PYTHONPATH=src python scripts/run_cascades.py
```

The Makefile target exports the same `PYTHONPATH`:

```bash
make cascade PYTHON=.venv/bin/python
```

PowerShell: `$env:PYTHONPATH='src'; python scripts/run_cascades.py`.
This one command regenerates all `results/cascade/` tables, JSON outcomes,
sidecars and five `figures/cascade/` PNGs. It uses the committed, frozen
86-station / 85-edge graph for 2026-10-05, 07:00–09:00 Australia/Perth
(271 selected morning rail trips). No live download is needed.
`--quick` writes separate `cascade_quick/` directories and uses a small grid;
those outputs are a smoke test and are not the report dataset. Runtime on the
project Linux environment (Python 3.14, pinned packages): the full family takes
about 1 minute (56 s for 26,368 cached runs and 16,000 avalanche samples,
including the reduced dynamic grid); `--quick` takes about 2 seconds.

## Design and definitions

- Full static tolerance sweep: alpha 0–2 inclusive, step 0.025, both equal
  and capacity-weighted redistribution. Each setting has maximum-load and
  maximum-degree triggers plus 300 uniformly sampled station triggers.
- Reduced sensitivity grid: 0, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1, 1.5, 2;
  both rules, static and dynamic modes, the three betweenness-based load
  definitions and P2.4's demand proxy. Demand uses AM-peak stop counts as
  loads and the intact frequency-scaled reference capacities; the other
  modes use K=L0. These are model proxies, not observed passenger demand.
- Avalanche family: 2,000 random triggers per alpha (0.1, 0.15, 0.2, 0.3),
  static and dynamic, capacity rule. One observation is the **total secondary
  failures after one independent trigger**, not an individual round. Duration
  is the number of secondary synchronous rounds. Spatial extent is the
  maximum great-circle distance between any failed stations, in kilometres,
  including the initial failure; zero for a singleton.
- A line closure removes only stations belonging exclusively to that line;
  interchanges remain open. All closed stations form one synchronous initial
  batch. Capacities are calculated before closure. Line runs cover alpha
  0, 0.2, 2 and both rules/modes.

`simulate_cascade(..., initial_failed=...)` replaces the single trigger. `None`
retains the existing API, an empty collection closes nothing, and unknown or
duplicate station IDs are rejected. `CascadeResult.initial_failed` records
the batch; `n_initial_failed` counts it. The sum of `avalanche_sizes` excludes
that batch. Legacy manually constructed results infer one initial trigger.

Random draws use recorded SeedSequence child seeds. The same draws are used
across settings, enabling paired model comparisons in P2.5. Repeated station
draws remain separate observations of a uniform station-trigger experiment.
Once a target is resolved, its deterministic simulation is memoised; caching
does not discard repeated observations. `outcome_id` joins each row to an
auditable failed-station sequence and per-round sizes in `outcomes.json`.

`alpha_sweep_static.csv`, `alpha_sweep_dynamic.csv`, `avalanche_stats.csv`
and `avalanche_sizes.json` preserve the prototype target names and summaries,
using the corrected frozen graph and synchronous engine. Prototype files are
never imported or edited. The historical 96-edge investigation values are not
numerical equality targets for the corrected 85-edge graph.

## Interpretation

The graph is a tree. Removing stations cannot introduce alternative rail
paths, so unnormalised routing betweenness on surviving stations cannot
increase. Dynamic containment in this graph does not demonstrate resilience
in a redundant network. Static redistribution is a different load assumption
and can produce large secondary failures. Zero baseline load also means zero
capacity at rail leaves in the pure betweenness model; the trips-augmented
load mode probes this assumption.

The targeted containment threshold is the first **sampled** alpha with no
secondary failure. If none occurs in 0–2, `conclusions.json` records a null
threshold and a right-censored status; it does not invent a transition or
assume monotonicity. CCDFs count zero cascades in their denominator. The
collapse plot's shading is ±1 sample SD, not a confidence interval; P2.5
supplies bootstrap intervals and statistical interpretation.

Every CSV/JSON has a `.meta.json` sidecar recording parameters, versions,
seeds and input/source hashes. `experiment_manifest.json` additionally hashes
all numerical artifacts and figures. JSON and metadata are validated before
the staged result pair is published. Figure titles identify the scenario;
the geographic map shows synchronous failure rounds for a Perth trigger.

Shared changes to `cascade.py`, `config.py`, `experiments.py` and the model
contract require the module owners' PR review under `docs/model.md` section 9.
The initial-batch extension composes with P2.4's independent reference-capacity
extension; neither changes the default single-trigger capacity law.
