# Live demonstration notebooks

Install `requirements.txt`, open JupyterLab with the project's Python kernel,
trust the notebook, and run all cells from a clean kernel. The demonstration
reads committed processed tables and result files; it does not download GTFS
or regenerate the full experiment grid. Map assets are embedded and work offline.

| Notebook | Demonstration |
| --- | --- |
| `01_network.ipynb` | Frozen morning timetable, audited railway, terminal/facility layers, verified buses, offline map and line/frequency summaries |

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
