Summary: Kaggle can refuse a GPU session with "running sessions are projected to exceed maximum weekly GPU quota" long before the 30 h are used, because running sessions count with a projected reservation; the reported reservation is erratic, so let the queue runner retry instead of planning around it.

# Kaggle refuses new GPU sessions on projected quota (2026-10-08)

What happened: with exp-077 running, the push of exp-078 at 20:02 UTC was refused: "Session cannot start because
currently running sessions are projected to exceed maximum weekly GPU quota of 30. Cancelling running sessions may
allow starting new sessions." Readings of `time_reserved` from scripts/kaggle_quota.py the same evening:

| UTC | used (h) | reserved (h) | running |
|---|---:|---:|---|
| 19:30 | 18.21 | 22.81 | exp-077 (1.0 h in), exp-076g (0.15 h in) |
| 20:02 | 19.79 | 10.41 | exp-077 (1.6 h in); exp-078 refused |
| 20:32 | 20.40 | 11.99 | exp-077 (GPU session over, kernel still RUNNING until 20:44), exp-078 just accepted |
| 20:48 | 20.77 | 0.00 | exp-078 (0.3 h in) |

The first three fit "each running session reserves 12 h minus its elapsed time, and a new one starts while used +
reserved < 30 h"; the last does not. exp-076g had been accepted at 19:21 with used + reserved near 28.6 h.

How to apply:
- Expect the refusal once used plus about 10-12 h per running session nears 30 h; it clears by itself (exp-078 went
  through 30 minutes later, when exp-077's GPU session ended, before its kernel turned COMPLETE).
- scripts/kaggle_queue.py retries refused pushes every 2 minutes; keep items queued rather than holding them back.
- Plan parallel runs early in the week, when used is low; do not count on a second slot late in the week.
- Check with `KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/python scripts/kaggle_quota.py`.
