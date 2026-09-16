# ARC Prize 2026 – ARC-AGI-3 agent

Engineering repo for the Kaggle competition. `CLAUDE.md` is the brief; `docs/status.md` is the
current state and the list of unconfirmed facts; `docs/research_log.md` has every measured run.

## Layout

```
arc3/                 the package that ships inside the Kaggle notebook
  perception.py       exact grid perception: components, diffs, scale detection, click targets, PNG render
  scoring.py          RHAE re-implementation (parity-tested against the arc-agi toolkit scorer)
  env.py              local env wrapper with the gateway's competition-mode reset semantics
  eval.py             evaluation harness: fixed seeds, per-game budgets, run records under runs/
  splits.py           fixed dev (19) / val (6) split of the 25 public games
  governor.py         wall-clock governor
  sandbox.py          persistent Python REPL in a child process (killable on timeout)
  llm.py              OpenAI-compatible chat client (vLLM) + scriptable mock
  prompts.py          system prompt and tool schema for the REPL agent
  serve.py            vLLM launch helpers
  kaggle.py           adapter to the ARC-AGI-3-Agents framework: shared deadline, crash recovery
  agents/             random, explorer (no-LLM control arm / fallback), repl (Duck-style model harness)
agent/my_agent.py     the framework-facing MyAgent used on Kaggle
scripts/              eval.py, download_games.py, build_notebook.py, slim_framework.py
tests/                scorer parity, env semantics, perception, governor, sandbox, REPL plumbing, submission path
docs/                 status, research log, lessons, post-mortems, research vision
```

## Quick start (local box)

```bash
make setup            # uv venv (Python 3.12) + deps + vendored ARC-AGI-3-Agents framework
make games            # download the 25 public games (anonymous API key is fine)
make test             # ~1 min
make verify           # random agent on two games
make eval AGENT=explorer SPLIT=dev TIME=300 STEPS=3000 WORKERS=4
```

With a vLLM server up (see `arc3/serve.py` for the flag set):

```bash
make eval AGENT=repl SPLIT=dev TIME=900 WORKERS=8 \
  CONFIG='{"base_url":"http://127.0.0.1:8000/v1","model":"arc3-model","context_tokens":32768}'
```

Every run writes `runs/<name>/summary.json` (committed) and per-game action logs (ignored).

## Kaggle

`make notebook` builds `notebooks/submission.ipynb` with the `arc3` package embedded. Attach the
model weights and vLLM wheelhouse datasets in `notebooks/kernel-metadata.json`, then push with the
Kaggle CLI (`kaggle kernels push -p notebooks/`). Without a model dataset the notebook runs the
explorer, so the pipeline can be validated end-to-end before any GPU quota is spent.
