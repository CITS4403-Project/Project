# CITS4403 Project Report

LaTeX source for the CITS4403 Computational Modelling research project report.
It is tracked in the main project repository
([CITS4403-Project/Project](https://github.com/CITS4403-Project/Project)) as the
`report/` directory.

## Requirements

A LaTeX distribution that provides `latexmk` and `pdflatex`:

- **Linux:** `sudo apt install latexmk texlive-latex-recommended texlive-latex-extra` (or the full TeX Live)
- **macOS:** MacTeX
- **Windows:** MiKTeX or TeX Live

## Building

```bash
make             # build build/report.pdf and refresh report.pdf
make watch       # rebuild automatically while editing (Ctrl-C to stop)
make clean       # remove intermediate build files
make distclean   # also remove the generated report.pdf
```

Intermediate files are written to `build/`. The finished report is copied to
`report.pdf` in this directory, which is committed so the latest compiled
version is always available. Run `make` and commit `report.pdf` alongside any
source changes before merging.

Alternatively, with the official portable Tectonic compiler:

```powershell
New-Item -ItemType Directory -Force build | Out-Null
tectonic --keep-logs --outdir build report.tex
Copy-Item -LiteralPath build/report.pdf -Destination report.pdf
```

Tectonic includes the bibliography pass and downloads missing TeX resources on
its first build. Subsequent builds reuse its cache. The source also retains the
existing `latexmk`/`pdflatex` build path. A draft can compile while another
member's sections are still placeholders; successful compilation alone does
not certify completion of the report or M3.

## Structure

```
.
+-- report.tex       % main document (preamble, title, abstract, includes)
+-- report.pdf       % compiled report (committed; refreshed by make)
+-- sections/        % one file per report section
+-- references.bib   % BibTeX references
+-- figures/         % figures included in the report
+-- Makefile         % build rules
```

The sections follow the assessment rubric in `RUBICS.md` of the main
repository: background and research aims, originality and contribution, model
specification, experimental design, results, and discussion/conclusions.

## Formatting requirements

From the unit specification:

- maximum **five A4 pages**, excluding figures, references and appendices;
- **11pt font** and **1-inch margins** on all sides.

The preamble already sets `11pt` and 1-inch margins. Keep long derivations,
extra figures and parameter tables in the appendix so the main body stays
within the page limit.

## Working in the main repository

The report sources are tracked with the rest of the project, so there is no
submodule to initialise or update. After pulling the latest `main`, build the
report from this directory:

```bash
cd report
make
```

## Workflow

1. Make report changes on the branch for your current issue in the main
   repository (one branch per issue).
2. Commit sources and the refreshed `report.pdf` together in small, single-line
   commits with clear messages.
3. Open the pull request in the main project repository for the usual review
   before merging into `main`.
