# CITS4403 project: robustness and cascading failure in the Transperth network.
#
#   make check      run the test suite
#   make data       rebuild data/processed from the raw GTFS snapshot
#   make demand     regenerate the P2.4 demand load and capacity results
#   make figures    regenerate the figures from saved results
#   make notebooks  execute the notebooks in place
#   make reproduce  rebuild data, tests, figures and notebooks in order
#   make clean      remove Python and pytest caches

PYTHON ?= python3
export PYTHONPATH := src:$(PYTHONPATH)

.PHONY: check data demand figures notebooks reproduce clean

check:
	$(PYTHON) -m pytest -q

data:
	$(PYTHON) scripts/run_data.py

demand:
	$(PYTHON) scripts/run_demand.py

figures:
	$(PYTHON) -m transperth.plotting

notebooks:
	$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

reproduce: data check figures notebooks
	@echo "reproduction complete"

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache
