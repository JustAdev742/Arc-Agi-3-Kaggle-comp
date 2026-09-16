# runs/

Every evaluation writes `runs/<run_name>/summary.json` (metadata, per-game results,
aggregate dev/val scores) plus one `<game>.jsonl` per game with one line per action.
Run directories are git-ignored (large); the research log cites them by name and commit.
