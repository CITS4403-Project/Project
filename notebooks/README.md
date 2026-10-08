# Live demonstration notebooks

Install `requirements.txt`, open JupyterLab with the project's Python kernel,
trust the notebook, and run all cells from a clean kernel. The demonstration
reads committed processed tables and result files; it does not download GTFS
or regenerate the full experiment grid. Map assets are embedded and work offline.

| Notebook | Demonstration |
| --- | --- |
| `01_network.ipynb` | Frozen morning timetable, audited railway, terminal/facility layers, verified buses, offline map and line/frequency summaries |
| `03_cascades.ipynb` | Tolerance/rule curves with P2.5 intervals, avalanche size/duration distributions, trigger controls and a map revealing secondary failure rounds |

Notebook 01 locates the repository from either the project root or `notebooks/`.
Its opening cells verify the processed-file hashes. The line filter needs a live
kernel; the embedded map provides drag, wheel zoom, layer toggles and station
details. Shared stations appear in each line's summary, so line counts must not
be summed. Scheduled trip counts are not passenger counts.

Execute in place from the project root:

```sh
make notebooks
# or
python -m jupyter nbconvert --to notebook --execute --inplace notebooks/01_network.ipynb
```

Notebook 01 must finish within one minute. Saved outputs support reading on
GitHub; interactive controls require trusted JupyterLab. See `docs/data.md`
for raw-feed reproduction and `docs/model.md` for model assumptions.

Notebook 03 reads the full committed cascade/uncertainty tables and validates
their graph-input hashes and family manifests. Its controls run one cached
production-engine scenario, not a new statistical sweep. The rule selector is
disabled for dynamic routing; station and seed inputs are enabled only for their
matching trigger type. The reveal slider changes the displayed round, while the
summary reports the final stable outcome. All fractions use the original 86
stations. Update rounds are not minutes and scheduled stops are not passengers.

Suggested handoff: compare max-load alpha 0.2 with 0.525, switch to dynamic
routing, then choose a station and reveal round 0 onward. Dynamic containment on
this tree still loses connectivity; Notebook 04 tests recovery with bus paths.
