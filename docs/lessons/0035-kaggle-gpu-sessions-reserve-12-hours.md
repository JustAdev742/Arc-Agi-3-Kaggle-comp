Summary: a running Kaggle GPU session reserves its remaining time up to 12 h against the 30 h weekly quota, and a new session starts only while used + reserved < 30 h, so once about 18 h are used, sessions run one at a time until the Saturday reset.

# Kaggle GPU sessions reserve up to 12 hours each (2026-10-08)

What happened: with exp-077 running, the push of exp-078 at 20:02 UTC was refused with "Session cannot start because
currently running sessions are projected to exceed maximum weekly GPU quota of 30. Cancelling running sessions may
allow starting new sessions." scripts/kaggle_quota.py showed 19.79 h used and `time_reserved` 10.41 h (exp-077 had
run ~1.6 h: 12 - 1.6 ≈ 10.4). At 19:30, with two sessions running, it showed 18.21 h used and 22.81 h reserved
(≈ 11.0 + 11.85, the remainders of two 12 h windows). exp-076g had been accepted at 19:21, when used + reserved was
about 28.6 h: the new session's own 12 h is not counted at the start.

How to apply:
- A session can start while used + reserved < 30 h. A run that takes 2.5 h still holds a 12 h reservation, and that
  shrinks only as fast as `used` grows, so the sum stays near used-at-start + 12 h for the whole run.
- Two sessions in parallel need used < ~18 h at the second start. Plan the week's parallel runs early; late in the
  week the slots are serial no matter how short the runs are.
- "Maximum batch GPU session count of 2 reached" is the other refusal (slots, not quota); scripts/kaggle_queue.py
  retries both every 2 minutes.
- Check with `KAGGLE_API_TOKEN=$(cat .kaggle/access_token) .venv/bin/python scripts/kaggle_quota.py` before planning.
