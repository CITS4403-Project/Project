# CITS4403 project: robustness and cascading failure in the Transperth network.
#
#   make check        run the test suite
#   make data         rebuild data/processed from the raw GTFS snapshot
#   make demand       regenerate the P2.4 demand load and capacity results
#   make percolation  regenerate the P2.1 percolation and critical-node results
#   make cascade      regenerate the P2.2 cascade, avalanche and closure results
#   make recovery     regenerate the P2.3 recovery and bus strategy results
#   make uncertainty  regenerate the P2.5 uncertainty and sensitivity results
#   make figures      regenerate the figures and report evidence from saved results
#   make notebooks    execute the notebooks in place
#   make reproduce    rebuild data, tests, figures and notebooks in order
#   make clean        remove Python and pytest caches

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: check data demand percolation cascade recovery uncertainty figures notebooks reproduce clean

check:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) scripts/run_data.py

demand:
	$(PYTHON) scripts/run_demand.py

percolation:
	$(PYTHON) scripts/run_percolation.py --all
	$(PYTHON) scripts/run_vulnerability.py

cascade:
	$(PYTHON) scripts/run_cascades.py

recovery:
	$(PYTHON) scripts/run_recovery.py

uncertainty:
	$(PYTHON) scripts/run_uncertainty.py

figures:
	$(PYTHON) -m transperth.plotting
	$(PYTHON) report/generate_figures.py

notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

reproduce: data check figures notebooks
	@echo "reproduction complete"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
