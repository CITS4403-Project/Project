# Phase 3 writing review

The writing checklist below follows `RUBICS.md` and issue #44. A self-check is
not a second-member sign-off. Each reviewer should inspect the corresponding
source and compiled PDF, record their name and commit, and resolve comments
before M3 is declared complete.

## P3.5 - introduction, background and aims (issue #51)

- [x] Defines the transport problem and why service and facility damage differ.
- [x] States RQ1-RQ4, each tied to a frozen experiment family.
- [x] Positions the work against robustness, cascade and statistical literature;
  distinguishes our static extension from Motter-Lai rerouting.
- [x] Separates the team's contribution from the borrowed capacity construction
  and from taught agent-based examples.
- [x] States the selected date/window, audited 86/85 topology and 271/4,324
  scheduled-trip counts; makes no passenger-count claim.
- [x] Uses continuous scientific prose with verified bibliographic details.
- [x] Compiled PDF and page allocation visually checked. Introduction occupies
  about 1.3 pages including title/abstract space; other section owners must
  recheck the integrated five-page budget.
- [ ] Other member co-sign: name/handle, reviewed commit and comments resolved.

Status: self-check complete; teammate co-sign pending. Model/design sections
(P3.6, #47) and results/discussion (P3.7, #52) are separately owned. This file
does not certify gate M3 or authorize submission.

## P3.6 - model specification and experimental design (issue #47)

- [x] The generic entity update rule comes before the transport-specific
  synchronous rule; loads, the capacity law and the failure rule match
  `docs/model.md` and the `config.py` defaults.
- [x] The design table matches the runner defaults (alpha step 0.025 over 0 to
  2, fractions 0.00 to 0.90 step 0.01 plus 1.00, 100/300/2,000/1,000 seed
  counts, two-link budget) and each row cites its `make` command.
- [x] Outcome measures trace to the packaged engines, the committed
  `results/` tables and the analysis pipeline (`make figures`, notebooks 01 to
  04).
- [x] The main body fits five pages with the model and design text in place;
  the figures, references and appendix follow on separate pages.
- [x] General fixing round trimmed `01-introduction`, `04-results` and
  `05-discussion` for the page budget.
- [ ] Other member co-sign: name/handle, reviewed commit and comments resolved.

Status: self-check complete; teammate co-sign pending. A compiled draft does
not satisfy M3 while the co-sign is missing.

## P3.7 - results, discussion, conclusions and abstract (issue #52)

- [x] Answers RQ1-RQ4 in order, then interprets mechanisms and limitations.
- [x] Distinguishes susceptibility peaks, GCC crossings and sampled containment;
  makes no thermodynamic criticality claim for 86 stations.
- [x] Keeps trigger failures separate from secondary size and update rounds.
- [x] Gives a traced bus overload and a budgeted service-loss example; does not
  generalise these into "buses always worsen service".
- [x] Uses original station/terminal-pair denominators, endpoint-stop weights
  and reachable-population travel-time caveats.
- [x] Treats demand/frequency capacity as a model comparison; stop counts are
  not measured passengers.
- [x] Includes zero avalanches, pointwise sampling intervals, finite-tail
  eligibility/rejection and uncertainty outside Monte Carlo error.
- [x] Numeric macros and figures are generated from frozen tables, with hashes
  for the sources, sidecars, generator and outputs.
- [x] Compiled PDF: all nine pages rendered and visually checked. The current
  draft has five main-body pages, two figure pages, one reference page and one
  appendix page. Sources use 11pt on A4 with 1-inch margins; the main body
  stays within five pages with the P3.6 prose in place.
- [ ] Other member co-sign: name/handle, reviewed commit and comments resolved.

P3.7 co-sign is pending. P3.6 (#47) landed in #59 and the main body stays within
five pages; the compiled draft is complete apart from the pending writing
co-signs. M3 needs those sign-offs.
