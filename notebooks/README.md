# Live demonstration notebooks

Install `requirements.txt`, open JupyterLab with the project's Python kernel,
trust the notebook, and run all cells from a clean kernel. The demonstration
reads committed processed tables and result files; it does not download GTFS
or regenerate the full experiment grid. Map assets are embedded and work offline.

| Notebook | Demonstration |
| --- | --- |
| `01_network.ipynb` | Frozen morning timetable, audited railway, terminal/facility layers, verified buses, offline map and line/frequency summaries |
| `02-robustness-percolation.ipynb` | Random and targeted removal curves, frozen collapse estimates replayed with the packaged engine, single-station vulnerability ranking, interactive collapse viewer |
| `03_cascades.ipynb` | Tolerance/rule curves with P2.5 intervals, avalanche size/duration distributions, trigger controls and a map revealing secondary failure rounds |
| `04-recovery-strategies.ipynb` | Layered standby map, five-strategy served-service comparison at the frozen budget, bus-aggravated cascade rounds, budgeted Airport Line case |

Execute in place from the project root:

```sh
make notebooks
```

Every notebook writes PNG copies of its charts under `figures/notebook0*/` and
finishes within a minute on the committed inputs. Saved outputs support reading
on GitHub; interactive controls require trusted JupyterLab and a live kernel.
See `docs/data.md` for raw-feed reproduction and `docs/model.md` for model
assumptions.

Notebook 01 locates the repository from either the project root or `notebooks/`.
Its opening cells verify the processed-file hashes. The line filter needs a live
kernel; the embedded map provides drag, wheel zoom, layer toggles and station
details. Shared stations appear in each line's summary, so line counts must not
be summed. Scheduled trip counts are not passenger counts.

Notebook 02 reads the committed P2.1 and P2.5 tables and replays the frozen
collapse estimates through `transperth.failure`. Its viewer selects a component,
attack, target measure and removal fraction; the panel reports the seed mean and
envelope for the selected curve, and the committed map shows all 86 triggers.

Notebook 03 reads the full cascade and uncertainty tables and checks their
input hashes and family manifests. Its controls run one cached production-engine
scenario, not a new statistical sweep. The rule selector is disabled for dynamic
routing, and the station and seed inputs are enabled only for their matching
trigger type. The reveal slider changes the displayed round while the summary
reports the final stable outcome. Fractions use the original 86 stations, update
rounds are not minutes, and scheduled stops are not passengers.

Notebook 04 reads the committed P2.3 tables and does not rerun `make recovery`.
The scenario, budget and strategy controls slice `strategy_comparison.csv`; the
tolerance control steps through the traced per-round loads; and the last section
reads the budgeted Airport Line case where `existing_bus` serves less than no
backup. Bus links are endpoint evidence, not road routes, and standby bus
capacity is unlimited in this scenario model.

Suggested handoff: compare max-load alpha 0.2 with 0.525 in notebook 03, switch
to dynamic routing, then choose a station and reveal round 0 onward. Dynamic
containment on this tree still loses connectivity; notebook 04 closes the
demonstration with the recovery and bus-strategy result.