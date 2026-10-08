# P2.5: Uncertainty and sensitivity

```bash
PYTHONPATH=src python scripts/run_uncertainty.py
```

PowerShell: `$env:PYTHONPATH='src'; python scripts/run_uncertainty.py`.
`make uncertainty` runs the same full command from the project root.
This command regenerates `results/uncertainty/` CSV/JSON files, sidecars and
six `figures/uncertainty/` figures. It consumes the committed **full** P2.2
results and checks their manifest and graph/model source hashes. If P2.2 inputs
or code change, run `scripts/run_cascades.py` first. `--quick` writes separate
`uncertainty_quick/` directories, and never replaces report outputs.

## Sampling units and confidence intervals

Random station removal uses P1.3's engine: 1,000 independent draws per removal
fraction, from 0 to 1 in steps of 0.05. Each fraction records the actual
derived seed and replicate index. All denominators refer to the original 86
stations. A station count uses the engine's documented rounding rule.

The P1.2 percentile bootstrap helper resamples observations 2,000 times to
estimate 95% intervals for means: random-failure GCC fraction; cascade failed
and GCC fractions; avalanche secondary size, duration, geographic span and
probability of any secondary failure. A **whole trigger experiment** is one
avalanche observation. Rounds within a cascade are not independent samples.
Zero avalanches are included in means, probabilities and CCDF denominators.

The full P2.2 tolerance sweep has 300 random observations per condition, and
its dedicated avalanche family has 2,000. Repeated draws of the same station
are legitimate uniform sampling with replacement; deterministic outcome
caching does not remove those observations. The intervals quantify Monte
Carlo error conditional on the fixed graph and model. They do not cover GTFS
date variation, passenger measurement error or uncertainty about which load
model describes actual travel. Intervals are pointwise, not simultaneous
coverage across all conditions.

Rule and load-model differences use matched replicate/seed/target triples.
Resampling their differences preserves the common-random-draw design. Demand
uses P2.4's AM-peak stop-count load and intact frequency-scaled reference
capacities. Comparing it with betweenness therefore changes both load and
capacity assumptions, and is a **model scenario comparison**, not an isolated
causal estimate of passenger demand. Stop counts are not passenger counts.

## Recommended defaults and evidence

Nested prefixes (25, 50, 100, 200, 300, 500, 1,000, and 2,000 where available)
are compared with the complete stored sample. This reference is an estimate,
not an exact population mean. All tested normalized means must change by at
most 0.02, and all pointwise CI half widths must be at most 0.03. Only prefix
sizes available for every tested condition can be recommended.

The frozen seed-0 experiment recommends **1,000 random trials**: its worst
mean change is about 0.0030 and worst CI half width about 0.0220. With 300
trials the half width reaches about 0.0382; with 500 it is about 0.0310. This
recommendation covers the tested random-removal fractions and avalanche
alphas, not every possible model/graph. The existing 300-seed tolerance sweep
still meets P2.2's minimum, but its larger intervals should remain visible.
For deterministic single-station models, enumeration of all 86 triggers can
also remove trigger-sampling error entirely.

Keep **alpha step 0.025** around the static transition. Coarsening the capacity
model moves first sampled max-load containment from 0.525 to 0.55 (step 0.05),
0.6 (0.1) and 1.0 (0.5). Linear interpolation of the coarse random means can
have large error near the abrupt change. This is grid resolution evidence,
not a proof of a thermodynamic critical point or monotonic behaviour.

Use **2,000 bootstrap resamples** for the report. `bootstrap_resolution.csv`
compares 200/500/1,000/2,000 nested resamples in two diagnostic conditions and
records CI endpoint drift; this is a limited numerical stability check.
Use **500 refitted simulations** for the tail test, giving p-value resolution
1/501. Marginal p values would require more simulations; no marginal positive
claim is made here. All recommendations and numerical evidence are saved in
`recommendations.json`.

## Discrete tail hypothesis and its limits

The implementation follows the MLE / KS / refitted-bootstrap design of
[Clauset, Shalizi and Newman (2009)](https://arxiv.org/abs/0706.1062), adapted
to the **known finite bound** of 85 secondary failures. It tests a decreasing
discrete power law, proportional to `s**(-tau)`, on every integer from `xmin`
to 85, including unobserved gaps. This bounded hypothesis is explicitly
different from an unbounded power law or the package's default estimator.
The tail exponent `tau` is separate from capacity tolerance `alpha`.

Candidate observed cutoffs need at least 50 tail observations and three
distinct positive sizes. At each candidate the exponent is fitted by exact
finite-support likelihood; the smallest discrete CDF KS selects the cutoff.
The exponent search is bounded to 0.01–20 and boundary fits are flagged. Zeros
and observations below the cutoff form an empirical body. Each synthetic
sample draws from that body/tail mixture and **refits both cutoff and exponent**.
The p value is `(1 + number of bootstrap KS >= observed KS) / (B + 1)`.
An ineligible bootstrap refit causes the p value to be withheld and recorded,
rather than discarded silently.

Each fixed alpha and dynamic setting is tested separately. The static samples
at alpha 0.1, 0.15 and 0.2 have only two positive sizes (84 and 85), so the
tail is unidentifiable under the eligibility rule. At 0.3, the extra size 1
allows a fit, but the bounded decreasing power law is rejected (p≈0.002);
the fitted exponent sits at its lower search boundary. Dynamic betweenness
samples contain only zeros. **These experiments do not support a power-law
avalanche claim.** Even non-rejection would not prove one, establish it against
alternative distributions, or generalise from the finite 86-station graph.

Every numeric output has provenance for the source tables, engine files,
parameters, package versions and seeds. The family manifest hashes the output
tables, JSON diagnostics and figures. Source and artifact validation, paired
sampling, nested prefixes, synthetic exponent recovery, zero/sparse tails,
seeded refitting and report artifacts are covered by tests.
