# ARC Prize 2026 – ARC-AGI-3 agent

Engineering repo for the Kaggle competition. `CLAUDE.md` is the brief; `docs/status.md` is the
current state, the Kaggle run list and the open decisions; `docs/research_log.md` has every measured
run (one entry per change, newest at the bottom); `docs/lessons/` holds what the runs taught.

## What plays

One local model (Qwen3.8-27B-FP8 under vLLM, offline) drives a persistent Python REPL that holds the
game state and exact perception helpers, writes and verifies a per-game world model, plans against it
and sends actions (`arc3/agents/repl_agent.py`). A no-LLM rules agent is the crash fallback. The
council (six specialist roles plus a coordinator) is kept as a parked ablation arm.

## Layout

```
arc3/                 the package that ships inside the Kaggle notebook (embedded as a tarball)
  env.py              local env wrapper with the gateway's competition-mode reset semantics
  eval.py             evaluation harness: fixed seeds, per-game budgets, run records under runs/
  scoring.py          RHAE re-implementation (parity-tested against the arc-agi toolkit scorer)
  splits.py           fixed dev (19) / val (6) split of the 25 public games; val is never tuned on
  perception.py       exact grid perception: components, diffs, scale detection, tile maps, PNG render
  entities.py         entity tracker: persistent ids across frames, roles (static/hud/avatar), events
  dsl.py              rule library: fitters over the entity log, simulation, BFS planning, goal predicates
  planner.py          avatar move model (key map, obstacles, companions) and shortest-path planning
  memory.py           learning memory: per-game lessons, run-wide shared file, offline skill library
  sandbox.py          persistent REPL in a child process: the helpers the model calls (act, ents, rules...)
  llm.py              OpenAI-compatible chat client (vLLM) plus a scriptable mock for tests
  prompts.py          system prompt, Method and tool schema for the REPL agent
  serve.py            vLLM launch helpers: flag set, start ladder with a real-completion probe
  kaggle.py           adapter to the ARC-AGI-3-Agents framework: shared deadline, crash recovery
  governor.py         sequential-runner time governor (not on the submission path; see lesson 0011)
  data/skills.json    offline skill library mined from solved levels (scripts/mine_skills.py)
  agents/             random, explorer, rules (no-LLM), repl (the submission), council (parked arm)
agent/my_agent.py     the framework-facing MyAgent used on Kaggle
scripts/
  eval.py             run an agent on a split locally
  build_notebook.py   build notebooks/submission.ipynb (arc3 + my_agent embedded)
  build_eval_notebook.py, build_diag_notebook.py   Kaggle evaluation / GPU diagnostic kernels
  push_eval.py, pull_run.py   push a kernel folder to the RTX PRO 6000; file its output under runs/
  transcript_report.py   summarise a run's transcripts for post-mortems
  mine_skills.py, goal_probe.py, rule_coverage.py, replay_probe.py   code-only analyses of recorded runs
  human_replays.py      summarise ARC Prize human recordings into priors (data/human/, once downloaded)
  stream_hf_to_kaggle.py   stream a Hugging Face repo into a Kaggle dataset without local disk
  download_games.py, slim_framework.py   setup helpers
tests/                scorer parity, env semantics, perception, tracker, DSL, planner, sandbox, REPL
                      plumbing, memory, council, evaluation records, submission path
docs/                 status, research log, lessons, post-mortems, model notes, the 100-percent design and
                      the road-to-100 evidence survey (docs/research/)
runs/                 run records; summary.json files are committed, per-action logs are not
```

## Quick start (local box)

```bash
make setup            # uv venv (Python 3.12) + deps + vendored ARC-AGI-3-Agents framework
make games            # download the 25 public games (anonymous API key is fine)
make check            # ruff (rule set pinned in pyproject.toml) + pytest, about 20 s
make verify           # random agent on two games
make eval AGENT=rules SPLIT=dev TIME=300 STEPS=3000 WORKERS=4
```

With a vLLM server up (`arc3/serve.py` has the flag set; `build_vllm_command` prints it):

```bash
make eval AGENT=repl SPLIT=dev TIME=1200 WORKERS=8 \
  CONFIG='{"base_url":"http://127.0.0.1:8000/v1","model":"arc3-model","context_tokens":32768,"reasoning_effort":"low"}'
```

Every run writes `runs/<name>/summary.json` after each finished game (`"partial": true` until the
last one), per-game action logs (`<game>.jsonl`), the REPL transcripts (`<game>.transcript.jsonl`) and
the lessons (`<game>.lessons.json`). Research-log entries point at these directories.

## Kaggle

The GPU work runs on Kaggle's RTX PRO 6000 (the local box has no GPU in the remote session):

```bash
# evaluation kernel: build, push (always on the RTX), pull the result into runs/
.venv/bin/python scripts/build_eval_notebook.py --agent repl --split dev --time-per-game 1200 --workers 8 \
    --slug arc3-eval-dev-x --run-name kaggle-repl-dev-0NN --note "exp-0NN ..." --out /tmp/nb/exp0NN
.venv/bin/python scripts/push_eval.py /tmp/nb/exp0NN
.venv/bin/python scripts/pull_run.py scottmahony/arc3-eval-dev-x kaggle-repl-dev-0NN

# submission notebook (Save & Run All plays two games offline first, then writes submission.parquet)
make notebook && make submit
```

The Kaggle token lives in git-ignored `.kaggle/access_token` (or `KAGGLE_API_TOKEN`). Model weights
and the vLLM wheelhouse are attached as private datasets named in `notebooks/kernel-metadata.json`.
Without a model dataset the notebook falls back to the rules agent, so the pipeline can be validated
end-to-end before any GPU quota is spent.

## Research discipline

Hypothesis, fixed dev evaluation with the same seed and settings as the current best, keep or revert,
an entry in `docs/research_log.md`. Identical runs differ by about ±0.3 RHAE and ±2 levels, so a
keep/revert decision needs three runs a side unless the effect is large. Post-mortems for games that
burned actions go to `docs/postmortems/`; lessons that hold on more than one game to `docs/lessons/`.
