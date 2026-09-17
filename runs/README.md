# runs/

Every evaluation writes `runs/<run_name>/`:

- `summary.json`: run metadata (agent, split, seed, budgets, config, harness commit), per-game results and the
  aggregate dev/val scores. It is rewritten after every finished game with `"partial": true` and once more at
  the end with `"partial": false`, so a run that is killed keeps what it had. These files are committed; the
  research log cites runs by directory name and commit.
- `<game>.jsonl`: one line per action (step, action, state, levels completed, cells changed, frame hash);
  `<game>.transcript.jsonl`: the REPL agent's model calls, code and tool outputs; `<game>.lessons.json`: the
  lessons the game ended with; `shared_lessons.jsonl`: what the games of the run shared with each other.
  These are git-ignored (large); `scripts/pull_run.py` files a Kaggle kernel's copies here.

`scripts/transcript_report.py runs/<run_name>` summarises a run's transcripts; `scripts/mine_skills.py` and
`scripts/goal_probe.py` read the recorded solved levels.
