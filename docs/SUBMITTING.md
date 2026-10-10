# How to submit (ARC Prize 2026, ARC-AGI-3)

This is a Kaggle **code competition**. You submit a *notebook version*, not a predictions file, and Kaggle re-runs
that notebook on its own machine against the hidden games. Submitting is one-way: it uses the day's single
submission and starts a run of up to 9 hours. This repo never submits on its own; the owner does it by hand.

## 1. The current candidates (updated 2026-10-10 16:59 UTC)

Both are Daniel Franzen's Milestone 2 notebook (Apache-2.0) with the D′ slot priority. To that base, both add:
- REAP-448 expert pruning at load;
- 14 streams;
- relaxed MTP acceptance (0.5/0.5);
- the sandbox-timeout fix;
- our ARC FR-Spec map;
- the fine-tuned MTP draft;
- a 40-minute first-request grace.

Submit **version 1** of exp-083, and version 1 or 2 of exp-085 (the same notebook pushed twice). Each is a completed full-length save run.

| Candidate | Notebook | Adds | Public-25 run | LB draws |
|---|---|---|---|---|
| exp-083 | https://www.kaggle.com/code/scottmahony/arc3-dprime-r14a05-arcmap-draft-full | (nothing beyond the list above) | 50.58, 116 levels | none yet; its albucino-draft version exp-074t drew 27.97 |
| exp-085 | https://www.kaggle.com/code/scottmahony/arc3-dprime-r14a05-h4-percept-untried-full (version 1 or 2: the same code) | our harness bundle: budget meter, search helper, win ledger, level memory, perception helpers (patches 02/04/03b/05/08b), plus the "not yet tried on this level" notices (ours-10, OURS_UNTRIED=1) | 53.77 / 119 and 47.51 / 111 (slow boot) | none yet; replaces exp-084 from Oct 12 (its bundle-only versions drew 34.04 as exp-081) |
| (exp-084, retired Oct 10) | https://www.kaggle.com/code/scottmahony/arc3-dprime-r14a05-harness4-percept-draft-full | the bundle without the notices | 48.83, 114 levels | none |

**Rotation.** Oct 11 exp-083, Oct 12 exp-085 (it replaced exp-084 on Oct 10, research log 16:59), then alternate. A resubmitted notebook scores differently each time
(about ±4 points per draw), so the averages decide. The final-selection rule is fixed in `docs/status.md`, under
"Plan for the week of 2026-09-26": the two configurations with the best mean over at least 3 draws each.

**The draft is mounted, not copied.** Both notebooks mount the output of the kernel `scottmahony/arc3-mtp-session-a`.
Before serving, they check that output's manifest sha256 (`be8c2d3ae23d…`) and its dense shard (`642797acaa0b…`),
and they refuse any other version. **Never push a new version of that kernel** while these candidates are in use;
train new drafts under a new slug.

### Rebuilding them

```bash
scripts/build_candidates.sh $OUT                    # exp083/exp084/exp085, byte for byte as pushed (tests/test_build_candidates.py)
.venv/bin/python scripts/push_eval.py $OUT/exp083   # each push is a ~2.3 GPU-h full-length save run
```

For a new draft, pass its kernel, its pulled `arc3-draft-manifest.json`, a slug tag and a name:

```bash
scripts/build_candidates.sh $OUT scottmahony/arc3-mtp-session-a2 \
    runs/mtp-session-a2/kernel-output/mtp-draft/arc3-draft-manifest.json draft2 "MTP session A2's draft"
```

This builds new notebooks (`...-draft2-full`), so the submitted ones stay untouched. The test checks that only the
draft, its version pins, the slugs and the notes change.

## 2. Submit (browser or CLI)

**Browser:**
1. Open the candidate's notebook link above.
2. Click **Submit to Competition** (top right). If it is not there, go to the competition page,
   **Submit Predictions**, and pick the notebook there.
3. Pick **version 1**, add a note (for example "exp-083"), and confirm.

**CLI** (checked 2026-09-23: `kaggle competitions submit` takes a kernel and a version for code competitions):

```bash
export KAGGLE_API_TOKEN=$(cat .kaggle/access_token)
.venv/bin/kaggle competitions submit arc-prize-2026-arc-agi-3 \
    -k scottmahony/arc3-dprime-r14a05-arcmap-draft-full -v 1 -f submission.parquet -m "exp-083"
.venv/bin/kaggle competitions submissions arc-prize-2026-arc-agi-3      # PENDING, then COMPLETE with a score
```

The submissions list links each score to its notebook and version (the `url` field of the API's submission
objects), so the research log can attribute every draw.

## 3. What Kaggle does with it

Kaggle re-runs the notebook in competition mode against a gateway that holds the 110 hidden games. In that mode:
- D′ plays every game with a 532-minute deadline and gives 14 model slots by its priority;
- the benchmark is released 12 minutes after the start, whether or not the model server is up;
- each game's first request waits up to 40 minutes for the server (the grace above), which covers a slow-storage
  boot (lesson 0038);
- the public leaderboard shows about half of the 110 games and the private leaderboard the other half, both from
  this one run.

## 4. What to expect

Leaderboard draws of this family so far: 27.97 (exp-074t), 28.87 (our unchanged D′ copy) and 34.04 (exp-081).
Public-25 runs translate to the leaderboard at about 0.55-0.68×. A single draw cannot rank two configurations.

## 5. If something fails

- **Version shows a red cross:** read the notebook log on its page. `scripts/kaggle_pull.py OWNER/KERNEL DIR`
  pulls a kernel's output in paced pages; the plain CLI draws HTTP 429 on large outputs.
- **"Maximum weekly GPU quota reached":** the push is refused until the weekly 30-hour window resets (Saturday
  00:00 UTC).
- **Submit button greyed out:** the chosen version has no successful run.
- **A draft check fails in cell 4** ("MTP draft not mounted", or a sha256 mismatch): the mounted kernel output is
  not the version the notebook was built for. Rebuild against the right manifest; do not edit the pins by hand.
