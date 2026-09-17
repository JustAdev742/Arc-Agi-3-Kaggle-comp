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

.PHONY: help setup games test verify eval notebook lint check clean kaggle-check pull-winners submit status

help:
	@echo "make setup      one-time: uv venv + deps"
	@echo "make games      download the 25 public games into environment_files/"
	@echo "make test       pytest (about 20 s; needs environment_files/ for the game-backed tests)"
	@echo "make lint       ruff with the rule set pinned in pyproject.toml"
	@echo "make check      lint + test"
	@echo "make verify     30s smoke test: random agent on ls20+vc33"
	@echo "make eval       AGENT=$(AGENT) SPLIT=$(SPLIT) TIME=$(TIME)s/game STEPS=$(STEPS) WORKERS=$(WORKERS)"
	@echo "make notebook   build notebooks/submission.ipynb from arc3/ + agent/my_agent.py"
	@echo "make kaggle-check   verify the Kaggle token and list competition files"
	@echo "make pull-winners   download the Milestone-1 winner notebooks + official sample into reference/"
	@echo "make submit     push notebooks/ to Kaggle as a private kernel (Save & Run All)"
	@echo "make status     status of the last pushed kernel"
	@echo "Kaggle evaluation runs: scripts/build_eval_notebook.py -> scripts/push_eval.py <folder> -> scripts/pull_run.py <kernel> <run>"
	@echo "Analysis: scripts/transcript_report.py runs/<run>, scripts/mine_skills.py, scripts/goal_probe.py, scripts/rule_coverage.py"

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
	$(PY) -m ruff check arc3 scripts tests agent

check: lint test

verify:
	$(PY) scripts/eval.py --agent random --split smoke --max-actions 100 --time-per-game 30 --run-name verify

eval:
	$(PY) scripts/eval.py --agent $(AGENT) --split $(SPLIT) --seed $(SEED) --time-per-game $(TIME) \
	    --max-actions $(STEPS) --workers $(WORKERS) --config '$(CONFIG)'

notebook:
	$(PY) scripts/build_notebook.py

clean:
	rm -rf $(VENV) environment_files recordings notebooks/submission.ipynb __pycache__ .pytest_cache

# ---- Kaggle (token: .kaggle/access_token, or KAGGLE_API_TOKEN in the environment) ----
KAGGLE_TOKEN := $(shell if [ -s .kaggle/access_token ]; then cat .kaggle/access_token; else echo "$$KAGGLE_API_TOKEN"; fi)
KAGGLE       := KAGGLE_API_TOKEN=$(KAGGLE_TOKEN) $(VENV)/bin/kaggle
COMP_SLUG    := arc-prize-2026-arc-agi-3

_check-kaggle:
	@if [ -z "$(KAGGLE_TOKEN)" ]; then \
	    echo "ERROR: no Kaggle token. Save it to .kaggle/access_token (git-ignored) or export KAGGLE_API_TOKEN."; exit 1; fi

kaggle-check: _check-kaggle
	$(KAGGLE) competitions files $(COMP_SLUG)
	$(KAGGLE) kernels list --competition $(COMP_SLUG) --sort-by scoreDescending --page-size 10

pull-winners: _check-kaggle
	mkdir -p reference
	$(KAGGLE) kernels pull jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner -p reference/duck -m
	$(KAGGLE) kernels pull ruichardliu/milestone1-2nd-solution -p reference/reki -m
	$(KAGGLE) kernels pull mbmmurad/arc-agi-3-lb-0-86-3rd-place-candidate-milestone -p reference/forge -m
	$(KAGGLE) kernels pull inversion/arc3-sample-submission-stochastic-goose -p reference/stochastic-goose -m

ACCEL        ?= NvidiaRtxPro6000   # CLI override; the notebook metadata carries nvidiaRtxPro6000 too

submit: notebook _check-kaggle
	@grep -q REPLACE_WITH_YOUR_USERNAME notebooks/kernel-metadata.json && { \
	    echo "ERROR: set your Kaggle username in notebooks/kernel-metadata.json"; exit 1; } || true
	$(KAGGLE) kernels push -p notebooks/ --accelerator $(ACCEL)
	@echo "Pushed. Track it with: make status"

status: _check-kaggle
	@KERNEL_ID=$$(python3 -c "import json; print(json.load(open('notebooks/kernel-metadata.json'))['id'])"); \
	$(KAGGLE) kernels status $$KERNEL_ID
