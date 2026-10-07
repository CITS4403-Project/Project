# CITS4403 Research Project

Research project for **CITS4403 Computational Modelling** (UWA). The project
studies emergent complex-systems behaviour using simulation and network models
implemented in Python 3.x. The active topic is robustness and cascading failure
in the Perth public transport network.

## Setup

The project uses a local virtual environment; no global packages are required.
Python 3.14 is the tested version.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A fresh environment passes the test suite with `make check`.

## Usage

The Makefile is the entry point. Activate the venv first, or pass the
interpreter explicitly as `make PYTHON=.venv/bin/python <target>`.

| Command | What it does |
|---|---|
| `make check` | run the pytest suite in `tests/` |
| `make data` | rebuild `data/processed/` from the raw GTFS snapshot |
| `make demand` | regenerate the P2.4 demand load and capacity tables and figures |
| `make percolation` | regenerate the P2.1 percolation and critical-node tables and figures |
| `make figures` | regenerate the figures from the frozen inputs and results |
| `make notebooks` | execute the notebooks in `notebooks/` in place |
| `make reproduce` | run data, tests, figures and notebooks in order |
| `make clean` | remove Python and pytest caches |

`make demand` runs `scripts/run_demand.py`: it compares the topology and
AM-peak demand load rankings and sweeps the frozen tolerance grid with the
frequency-scaled reference capacity, writing `results/demand/` tables with
sidecars and the two `figures/fig_demand_*.png` figures.

`make percolation` runs `scripts/run_percolation.py --all` for the RQ1 removal
curves (node and edge, random and targeted by degree, betweenness and demand
flow) and `scripts/run_vulnerability.py` for every single-station trigger over
the tolerance grid, writing `results/percolation/` tables with sidecars and the
`figures/fig_percolation_*.png` and `figures/fig_vulnerability_*.png` figures.

`make reproduce` is the documented path from raw data to the figures and
notebooks used in the report. The `data` stage reads the Transperth GTFS feed
from the git-ignored `data/raw/` and writes the frozen tables to
`data/processed/`; the fetch and preparation scripts from P0.2 are
documented in `docs/data.md`.

## Frozen transport data

Run `make data PYTHON=python` (or `python scripts/run_data.py` without Make)
to download/verify the pinned GTFS archive, prepare the 2026-10-05 07:00–09:00
snapshot, build the map-verified 86-station/85-edge graph, and validate hashes.
The canonical inputs are in project-root `data/processed/`; the investigation
copy stays unchanged. See [the data dictionary and reproducibility notes](docs/data.md)
for topology corrections, manual bus provenance and the mutable-source limitation.

## Repository structure

```
project-root/
|
+-- src/transperth/  % Main package (network, metrics, failure, cascade, ...)
+-- utils/           % Utility/helper functions
+-- tests/           % pytest suite, run by make check
+-- scripts/         % Data download, preparation and reproduction scripts
+-- docs/            % Model specification and data dictionary
+-- data/            % Frozen processed inputs (raw downloads are git-ignored)
+-- notebooks/       % Jupyter notebooks (analysis, results, demonstration)
+-- results/         % Experiment outputs (CSV plus JSON provenance sidecars)
+-- figures/         % Regenerated figures
+-- investigations/  % Investigation-phase material, one directory per candidate idea
+-- report/          % Report LaTeX sources (git submodule)
+-- requirements.txt
+-- README.md
```

## Documentation

`docs/model.md` defines the graph, the load and capacity equations, the
synchronous failure rule, the redistribution rules, the metric definitions and
the frozen API that the implementation issues code against. `docs/data.md`
(P0.2) holds the column dictionary and checksums of the frozen inputs.

Phase notes: `docs/metrics.md` (P1.2) covers the damage metrics and seeded
runs, `docs/cascade.md` (P1.4) the cascade rules and examples, and
`docs/strategies.md` (P1.6) the budgeted bus deployment policies. Each
document includes its own `PYTHONPATH=src` example command.

## Report

The written report is a LaTeX project kept in the `report/` submodule
([CITS4403-Project/report](https://github.com/CITS4403-Project/report)).

```bash
git submodule update --init report   # after cloning this repository
cd report
make                                 # builds build/report.pdf
```

See `report/README.md` for the template layout and contribution workflow.

## Workflow

Work is tracked with GitHub issues, one branch per issue
(`feat/P0.1-package-tooling`, `fix/P0.2-data-pipeline`). A pull request links
its issue with `Closes #N`, lists its acceptance criteria and needs the other
member's review before merging. Commits are small, single-purpose and use
imperative subjects. The investigation-phase branches remain for reference;
only the Perth transport investigation is developed further.

Investigation work happens on `investigation/*` branches and is merged into
`main` through pull requests when complete.
