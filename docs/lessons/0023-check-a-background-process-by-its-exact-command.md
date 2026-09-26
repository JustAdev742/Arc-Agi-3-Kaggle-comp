Summary: `pgrep -f <script name>` inside a shell command matches that shell itself, so it reports a dead background process as alive; match the interpreter-plus-script string with a bracket trick, e.g. `ps -eo pid,etime,cmd | grep "[b]in/python scripts/kaggle_queue.py"`.

# Check a background process by its exact command line (2026-09-26)

The Kaggle queue runner died in a container restart on 2026-09-25 after 07:50 UTC. Three later checks ran
`pgrep -f kaggle_queue.py && echo runner alive`; each printed "alive" because pgrep matched the bash process whose own
command line contained the pattern. The quota reset at 00:00 on Sep 26 found no runner; the 00:15 check-in caught it
only because the state file had not been written since 07:50. Nothing was lost beyond 15 minutes, but a check that
cannot fail is worse than none.

What to do: match a string only the real process has (`grep "[b]in/python scripts/kaggle_queue.py"` on `ps` output,
where the brackets keep grep from matching itself), and confirm liveness by a side effect the process must produce (the
state file's modification time advancing every 2 minutes).
