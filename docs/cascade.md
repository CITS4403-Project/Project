# Cascade engine (P1.4)

`simulate_cascade` uses `CascadeConfig`, `CascadeResult` and the shared P1.3
load/capacity and removal helpers. Supported modes are unweighted,
frequency-weighted, and betweenness plus station throughput. `demand` raises
unless the caller supplies an explicit `load_function`. All baseline loads must
be finite, non-negative and cover exactly the graph nodes; capacity overflow is
rejected. Stations have string IDs; graphs must be undirected, simple and
loop-free.

The trigger is removed before round 1. Load triggers rank the selected initial
load mode (or explicitly supplied baseline), descending with lexical ID ties.
Degree and random triggers use P1.3's removal generator. An explicit target
overrides the trigger. The engine sorts its working copy so insertion order
does not affect shortest-path calculation or simultaneous redistribution.

Static rounds collect all overloads first, remove them together, then distribute
each failed station's carried load only to neighbours surviving that round.
Shares are equal or proportional to recipient capacities; if all are zero,
capacity sharing falls back to equal. Load with no surviving recipient is lost.
The result audits `initial_total_load = remaining_load + lost_load`, subject to
floating-point tolerance. Zero-capacity neighbours receiving positive load can
fail; the engine does not give them invented capacity or exempt them.

Dynamic mode recomputes the selected load model after each removal and never
also redistributes static load. Capacities remain fixed from the intact graph.
The static conservation identity is inapplicable to recomputed betweenness:
`remaining_load` and `lost_load` are None. Undirected, unnormalised betweenness
counts unordered OD pairs: a three-node path's centre has load 1. The CSV's
normalised descriptive betweenness is not reused as a baseline load.

An optional `load_function` replaces `initial_loads` for the intact baseline
and for every dynamic recomputation. It receives the current graph, must not
modify it, and must return one finite non-negative load per current station; a
subset loader such as `multilayer.terminal_loads` is wrapped as
`{node: loads.get(node, 0.0) for node in graph}`. `baseline_loads` still
overrides the initial loads. Capacities stay fixed from those initial loads,
so a layered or demand-weighted model can recompute loads while keeping the
intact-graph capacity law.

`failed` starts with the trigger and then sorted batches. `avalanche_sizes`
excludes the trigger; its sum equals `n_failed - 1` on a nonempty graph.
`rounds` is the number of recorded failure batches. A trigger-only run has zero
rounds. An empty graph returns zero fractions and no failures. Result dataclass
audit fields are optional additions, preserving existing construction calls.

After P1.1 (#19) and P1.3 (#20) are available, from the root with `PYTHONPATH=src`:

```sh
python -m transperth.cascade_example --alpha 0.2 --seed 0
python -m transperth.cascade_example --dynamic --alpha 0.2 --seed 0
python -m pytest tests/test_cascade.py
```

The example loads the frozen graph through `network.load_rail_graph` and writes
`results/cascade/single_run.csv` plus metadata containing all configuration
fields and both input hashes. Separate output directories retain comparisons.
