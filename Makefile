# ARC Prize 2026 – ARC-AGI-3 local workflow. See CLAUDE.md and docs/status.md.
PYTHON      ?= python3.12
VENV        := .venv
PY          := $(VENV)/bin/python
AGENT       ?= explorer
SPLIT       ?= dev
SEED        ?= 0
TIME        ?= 600
STEPS       ?= 5000
WORKERS     ?= 4
CONFIG      ?= {}

.PHONY: help setup games test verify eval notebook lint clean

help:
	@echo "make setup      one-time: uv venv + deps"
	@echo "make games      download the 25 public games into environment_files/"
	@echo "make test       pytest"
	@echo "make verify     30s smoke test: random agent on ls20+vc33"
	@echo "make eval       AGENT=$(AGENT) SPLIT=$(SPLIT) TIME=$(TIME)s/game STEPS=$(STEPS) WORKERS=$(WORKERS)"
	@echo "make notebook   build notebooks/submission.ipynb from arc3/ + agent/my_agent.py"

FRAMEWORK_REPO := https://github.com/arcprize/ARC-AGI-3-Agents.git
FRAMEWORK_DIR  := vendor/ARC-AGI-3-Agents

setup:
	uv venv --python $(PYTHON) $(VENV)
	uv pip install --python $(PY) -e ".[dev]"
	@if [ ! -d "$(FRAMEWORK_DIR)/agents" ]; then \
	    mkdir -p vendor && git clone --depth 1 $(FRAMEWORK_REPO) $(FRAMEWORK_DIR) && rm -rf $(FRAMEWORK_DIR)/.git; \
	fi
	$(PY) scripts/slim_framework.py

games:
	$(PY) scripts/download_games.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check arc3 scripts tests

verify:
	$(PY) scripts/eval.py --agent random --split smoke --max-actions 100 --time-per-game 30 --run-name verify

eval:
	$(PY) scripts/eval.py --agent $(AGENT) --split $(SPLIT) --seed $(SEED) --time-per-game $(TIME) \
	    --max-actions $(STEPS) --workers $(WORKERS) --config '$(CONFIG)'

notebook:
	$(PY) scripts/build_notebook.py

clean:
	rm -rf $(VENV) environment_files recordings notebooks/submission.ipynb __pycache__ .pytest_cache
