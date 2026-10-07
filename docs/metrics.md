# Metrics and repeated runs (P1.2)

`compute_metrics` measures undirected simple rail graphs using hop distances.
Pass `n_initial` from the intact graph on every removal step, or store it in
`graph.graph['n_initial']` before making copies. Without either, the current
node count is the denominator; this default is for an intact standalone graph.
The explicit argument overrides graph metadata and cannot be smaller than the
surviving count. The optional keyword is additive to the P0.1 interface.

GCC and the second-largest component divide by the same original node count.
Efficiency divides the sum of inverse shortest-path hop distances by the
original ordered-pair total, so removed and unreachable pairs contribute zero.
Compute baseline efficiency with the same original count. Damage is
`1 - efficiency / baseline_efficiency`, with no clipping; absent baseline gives
0, a zero baseline with zero current efficiency gives 0, and a zero baseline
with positive current efficiency is rejected.

ASPL uses the largest surviving component only. For equal-size components the
lexically first sorted string-ID component wins. Empty/single-node ASPL is 0;
empty efficiency is 0. Empty component fractions are 0. Isolates include every
surviving degree-zero node. No weighted-minute interpretation is implied.

Without explicit OD counts, served fraction counts all reachable undirected
station pairs over the original pair total. Supply both `served_pairs` and
`total_pairs` for a terminal subset; counts must be nonnegative integers and
served cannot exceed total. An empty OD universe is fully served (1).

`utils.stats.bootstrap_ci` exposes the packaged `transperth.stats` implementation,
so existing script imports work with only `PYTHONPATH=src`. It implements an independent-observation percentile
bootstrap with a local NumPy generator. `run_seeded` spawns independent child
seeds with `SeedSequence` and records `run`/`seed` for each row. The callback
returns a mapping without these reserved columns. `summarize_runs` produces
the mean and confidence interval per requested numeric column. Confidence,
resample count, and the master seed belong in the metadata parameters.

From the repository root, with `PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH='src'`):

```sh
python -m transperth.metrics_example --seed 0 --n-runs 20
python -m pytest tests/test_metrics.py tests/test_stats.py tests/test_experiments.py
```

The example hashes `data/examples/metrics_graph.json` and writes
`results/metrics/runs.csv` and `summary.csv`, each with a metadata sidecar.
Same inputs and seed give identical numerical outputs; UTC timestamps differ.

Explicit input paths must exist when building `RunMeta`. `save_table` validates
metadata and serializes the CSV before touching previous outputs, stages both
files, then publishes with replacement. Ordinary publication errors roll back
the previous pair. Two separate files cannot provide a single atomic transaction
to concurrent readers or survive every process/power interruption; use one
writer per output path. Strict JSON rejects non-finite or unsupported metadata.
