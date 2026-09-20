# How to submit (ARC Prize 2026, ARC-AGI-3)

This is a Kaggle **code competition**: you do not upload a predictions file, you submit a *notebook version*.
Kaggle then re-runs that notebook itself, on its own machine, against the hidden games.

Everything in step 1 is already done and kept current by this repo; step 2 is the part only you can do (the
Submit button needs a logged-in browser session, the API cannot press it).

## 1. The notebook (done by `make notebook` + a Save & Run All)

```bash
make notebook                                        # builds notebooks/submission.ipynb at the current commit
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token)
.venv/bin/python scripts/push_eval.py notebooks      # pushes it privately and always on the RTX PRO 6000
```

`push_eval.py` refuses a notebook whose metadata is not `nvidiaRtxPro6000` (a T4 cannot load the 27B model).
Pushing starts a **Save & Run All**: Kaggle executes the whole notebook once. That run must finish with
`submission.parquet` written, or the version cannot be submitted.

What the notebook does in that run: installs `arc-agi` from the competition wheels, unpacks the `arc3` package
(with its Apache-2.0 LICENSE), installs vLLM from the attached wheelhouse, starts the Qwen3.8-27B-FP8 server,
plays two bundled games offline as a smoke test, and writes a placeholder `submission.parquet`.

## 2. Press Submit (browser, about 30 seconds of clicking)

1. Open <https://www.kaggle.com/code/scottmahony/arc-prize-2026-arc-agi-3-arc3-agent>.
2. Check the latest version shows a green tick (the Save & Run All finished). Open **Version history** if you
   want to confirm which version that is.
3. Click **Submit to Competition** (top right of the notebook page). If you do not see it, go to the
   competition page, **Submit Predictions**, and pick this notebook and version there.
4. Pick the version, add a note (for example the harness commit), and confirm.

That is the whole submission. Nothing on this machine needs to run while it happens.

## 3. What Kaggle does with it

Kaggle re-runs the same notebook with `KAGGLE_IS_COMPETITION_RERUN=1` and a `gateway` service holding the
hidden games. In that mode the notebook skips the offline smoke, copies the ARC-AGI-3-Agents framework,
installs our `MyAgent`, and plays **every hidden game concurrently** until the 9-hour limit. The score comes
from the gateway's scorecard, not from the parquet file. Expect the run to take most of the 9 hours; the
leaderboard entry appears when it ends.

## 4. What to expect

Our measured dev score is 0.6 to 1.4 percent of human-level RHAE (champion record: `docs/champion.md`), the
public leaderboard leader is about 19. A submission now will land in the low single digits. It is worth doing
for the leaderboard position and for the end-to-end validation on the hidden set; it is not worth doing if you
expect a competitive number today.

Submission limits: the competition page's Rules tab has the daily allowance (recorded as 5 per day in
`docs/status.md`, still unconfirmed). Each real submission consumes one of those and about 9 hours of run time.

## 5. Milestone 2 (only if you want prize eligibility)

Milestone 2 closes **2026-09-30** and requires the solution to be **public and open-source** by then. The code
is already Apache-2.0 (`LICENSE`, shipped inside the notebook bundle). To make the entry eligible, set the
notebook's visibility to Public on its Settings tab (or ask me and I will flip it with the API). Publishing is
reversible; submitting is not.

## 6. If something fails

- **Version shows a red cross:** open the log on the notebook page; the last cell prints the vLLM log tail when
  the server did not start. `scripts/pull_run.py` files a kernel's output under `runs/` for inspection.
- **"Maximum weekly GPU quota reached":** the push is rejected until the weekly 30-hour window resets.
- **Submit button greyed out:** the latest version has no successful run, or it wrote no `submission.parquet`.
