# CITS4403 Research Project

Research project for **CITS4403 Computational Modelling** (UWA). The project studies
emergent complex-systems behaviour using simulation and network models implemented in
Python 3.x.

## Status

Investigation phase. Two candidate projects are being explored in parallel, each on its
own git branch. Each investigation lives in its own directory and is merged back to
`main` via pull request once reviewed.

| Branch | Investigation directory | Topic |
|---|---|---|
| `investigation/idea1-yardsale` | `investigations/idea1-yardsale/` | Modified yard-sale model of wealth inequality |
| `investigation/idea2-perth-transport` | `investigations/idea2-perth-transport/` | Cascading failures in the Perth public transport network |

## Setup

The project uses a local virtual environment; no global packages are required.

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

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
+-- src/            % Main project code (functions, classes, models)
+-- utils/          % Utility/helper functions
+-- data/           % Datasets or data samples (large raw downloads are git-ignored)
+-- notebooks/      % Jupyter notebooks (analysis, results, demonstrations)
+-- investigations/ % Investigation-phase material, one directory per candidate idea
+-- report/         % Report LaTeX sources (git submodule)
+-- requirements.txt
+-- README.md
```

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

1. Investigation work happens on `investigation/*` branches.
2. Changes are committed in small, single-line commits.
3. Completed investigations are merged into `main` through pull requests.
