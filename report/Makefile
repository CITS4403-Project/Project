# ---------------------------------------------------------------------------
# CITS4403 project report -- build rules
#
#   make            build build/report.pdf and refresh report.pdf
#   make watch      rebuild continuously while editing (Ctrl-C to stop)
#   make clean      remove intermediate build files
#   make distclean  also remove the generated report.pdf
#
# Requires latexmk and a LaTeX distribution (TeX Live, MiKTeX or MacTeX).
# ---------------------------------------------------------------------------

MAIN         := report
OUTDIR       := build
LATEXMK      ?= latexmk
LATEXMKFLAGS ?= -pdf -bibtex -interaction=nonstopmode -halt-on-error -file-line-error

# Pin the PDF metadata (/CreationDate, /ModDate and /ID) so that rebuilding
# unchanged sources produces an identical file and does not dirty the working
# tree. The visible \today date still reflects the compile day.
export SOURCE_DATE_EPOCH := 0

SOURCES := $(MAIN).tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*)

.PHONY: all build pdf watch clean distclean

all: build

# The compiled report is committed at the repository root.
build: $(MAIN).pdf

pdf: build

$(MAIN).pdf: $(OUTDIR)/$(MAIN).pdf
	cp -f $< $@

$(OUTDIR)/$(MAIN).pdf: $(SOURCES)
	@mkdir -p $(OUTDIR)
	$(LATEXMK) $(LATEXMKFLAGS) -outdir=$(OUTDIR) $(MAIN).tex

watch:
	@mkdir -p $(OUTDIR)
	$(LATEXMK) $(LATEXMKFLAGS) -pvc -outdir=$(OUTDIR) $(MAIN).tex

clean:
	rm -rf $(OUTDIR)

distclean: clean
	rm -f $(MAIN).pdf