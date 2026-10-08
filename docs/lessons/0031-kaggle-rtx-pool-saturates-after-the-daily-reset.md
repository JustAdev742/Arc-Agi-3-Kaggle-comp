Summary: after the 00:00 UTC daily submission reset, teams' 9-hour competition reruns fill Kaggle's RTX PRO 6000 pool for 10+ hours and our private (batch) sessions sit QUEUED, holding our 2-session limit; push private runs in the evening UTC window, keep each push's kernel_session_id so a stuck session can be cancelled, and expect a submission made just after 00:00 to wait too.

# The RTX pool saturates after the daily reset (2026-10-07/08)

What happened: on 2026-10-07 every private session we pushed between 21:18 and 23:38 UTC started within minutes
(exp-070d, exp-072b/f/g, exp-073, the D' submission's own rerun at 21:57). From 00:15 UTC on 2026-10-08, two pushes
(exp-072h 00:15, exp-073b 01:37) stayed QUEUED for more than 10 hours, still at 10:46, while our weekly quota stood
at 5 h 15 min of 30 h with nothing reserved (kagglesdk GetAcceleratorQuotaStatistics). Two queued sessions fill the
batch-session limit, so the next submission's 20-minute save run could not even be pushed.

How to apply:

- Push private RTX runs in the evening UTC window (before 00:00), not right after the reset; keep the queue short
  enough that nothing important sits behind a queued session.
- scripts/push_eval.py records each push's kernel_session_id (folder/sessions.jsonl); scripts/kaggle_cancel.py
  cancels by it. The CLI's push drops the id, so a CLI-pushed session cannot be cancelled through the API.
- A competition submission is a rerun on the same pool: one made just after 00:00 UTC may wait hours before it
  starts. Near the deadline (2026-11-02 23:59 UTC) leave margin for that.
- Rented boxes (docs/research/rental-runner.md) are not subject to this queue.
