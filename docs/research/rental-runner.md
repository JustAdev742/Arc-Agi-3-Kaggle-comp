# Rental runner: our Kaggle notebooks on rented RTX PRO 6000s (written 2026-09-29, revised after review)

One-line summary: `scripts/rental.py` runs the exact notebooks we push to Kaggle on rented vast.ai single-GPU
boxes (Kaggle's own GPU image, inputs downloaded into the /kaggle/input layout) for about $4-5 per public-25 run.
It is built and tested without spending anything, and waits for the owner's account and OK.

## Why

Kaggle gives us 30 GPU-hours a week, about five public-25 runs after stress tests. Identical notebooks vary by
1.5-2.3x between runs (our exp-054 pair: 12.87 / 10.70 on the public 25; 4.70 / 3.36 on the leaderboard), so one run
per arm cannot separate arms that differ by less than about 20%. More runs per arm is the only fix, and rented
GPUs are the only source of more runs.

## What a run costs (vast.ai public offer search, 2026-09-29 23:1x UTC, with the filters below)

Five offers passed: one GPU, verified host, RAM share >= 170 GB, driver >= 580, contract >= 48 h. Prices include our
500 GB disk, which some hosts charge heavily for (one $1.41/h server-edition offer is $2.04/h with the disk, and its
disk alone costs $15/day until the instance is destroyed). The cheapest today: $1.34/h (RTX PRO 6000 WS, 441 GB RAM
share, disk $1.11/day), then $1.53/h and $1.91/h.

| Item | Estimate |
|---|---|
| Box with disk | $1.34-1.53/h today |
| Fresh box: Kaggle image pull (23 GB) + inputs (111 GB model archive + runtimes) | ~1 h (estimate; the first rental measures it) |
| One public-25 notebook (16 min serving setup, 132 min play, teardown) | ~2.75 h |
| **First run on a box** | **~$5** |
| **Each further run on the same box** | **~$4** |

Budget arithmetic: at ~$1.45/h, $28 buys about 19 box-hours: one box plays about six runs back to back
(1 h setup + 6 x 2.75 h), or two boxes play about six runs in half the wall-clock (each pays its own 1 h setup).

A 2-GPU box does not halve the time: Keith's serving setup (the one every notebook uses) refuses to start unless
nvidia-smi shows exactly one RTX PRO 6000, and nvidia-smi ignores CUDA_VISIBLE_DEVICES. Two 1-GPU boxes cost the
same as one 2-GPU box and need no change to the notebooks.

## How it works

1. `rental.py pack JOB DIR...` takes the folders `build_arms.py` / `build_kv_stress_nb.py` write (notebook +
   kernel-metadata.json; a folder may be listed twice for two runs of one arm) and makes a job: flat files
   `runN.ipynb`, `runN.kernel-metadata.json`, a `manifest.json` and `rental_box.py`, and records the job's sha256.
   `--upload` creates the private Kaggle dataset `scottmahony/arc3-rental-JOB` (or adds a version).
2. `rental.py offers` lists boxes that pass the filters, ranked by the price with our disk (the server edition,
   Kaggle's card, wins ties up to 10%; the workstation card is the same chip).
3. `rental.py quote JOB --offer ID` prints the estimate, the cap, and the disk's cost per day after the job.
   **The owner OKs a batch before anything is rented.**
4. `rental.py launch JOB --offer ID --approved "<owner's words>"` first downloads the uploaded job and refuses unless
   its sha256 equals the local pack's (a stale upload never runs under a new name), refuses if an instance with the
   job's label exists, writes `launch.json` before the create call (a lost response is recovered by label), then
   rents the box: Kaggle's GPU image (`gcr.io/kaggle-gpu-images/python:v170`, entrypoint `/usr/bin/env`), bash as
   the entrypoint with the boot command as its argument (the official CLI's `--entrypoint`/`--args` shape),
   `cancel_unavail` (no silent stopped instance when the host is busy), env as an object: the Kaggle token, the job,
   its sha and the wall-clock cap.
5. The boot command installs the Kaggle CLI into /opt/kcli (not the notebook's Python), downloads the job, runs
   `rental_box.py` under `timeout <cap>h`, and on exit (a trap) stops the instance through vast.ai's API with the
   instance-scoped key vast.ai sets in every container: the GPU is released and the disk kept.
6. On the box, `rental_box.py`: checks the job sha; gates on the hardware (one RTX PRO 6000, driver >= 580, RAM
   >= 150 GiB by the cgroup limit, /dev/shm >= 8 GiB) before the downloads; downloads every input once into the
   paths Kaggle mounts; runs each notebook in `/kaggle/working` with papermill under a 5 h cap; then kills every
   process the run started (the kernel and vLLM run in their own sessions), waits for the GPU and port 1234 to be
   free, and removes what the run left in /tmp, /dev/shm and site-packages, so each run starts cold as on Kaggle;
   uploads the run's `/kaggle/working` as `output.tar.gz.blob` (Kaggle unpacks `.tar.gz` uploads) with `run.json`
   as the private dataset `scottmahony/arc3-rental-JOB-N`, checking the CLI's success text (it exits 0 on
   "Dataset creation error"). A failed upload never stops the next run; failed uploads are retried at the end, and a
   restarted box replays nothing that was uploaded (it resumes from its partial record, or only retries uploads).
7. `rental.py status/logs` watch it; `collect JOB --destroy` downloads each run into
   `runs/<arm>-r<N>-rental-JOB/kernel-output`, scores it with `pull_taaf_run.py --local`, and destroys the instance
   only if every result arrived (otherwise the disk, holding the results, is kept).

## What the owner sets up (once)

1. A vast.ai account with credit (the first batch needs about $10-15).
2. An API key from vast.ai (Account -> Keys) saved as `.vast/api_key` in the repo checkout I work in (git-ignored).
3. Accept that the Kaggle token goes to the rented box: it is passed as an environment variable, so vast.ai and the
   host machine can read it. **After `collect`** (not before: the box needs the token until its last upload),
   generate a new Kaggle token (kaggle.com -> Settings -> API -> "Generate New Token", which expires the old one) and
   save it in `.kaggle/access_token`. Only verified hosts pass the filter.

## Fidelity and what to run first

- The box differs from Kaggle's g4-standard-48 in CPU model, RAM bandwidth and PCIe (the 48 GiB PLE table sits in
  host RAM), so tokens per second can differ. **The first rental run is exp-054 unchanged**, read against Kaggle's
  exp-054 pair (12.87 / 10.70, 1,609 / 1,627 requests). Arms are then compared with each other on rentals.
- Rental results are development evidence only; LB draws still come from Kaggle submissions (the owner submits).

## Proposed first batches (each needs the owner's OK before `launch`)

Measured input sizes (Kaggle API, 2026-09-29): model archive 111 GB (135 GB extracted; both on disk while it
extracts), Keith's runtime 7.9 GB, the SGLang wheel datasets 6.6 + 4.9 GB, bundles < 0.1 GB; peak disk about
330 GB with the image, so 500 GB fits.

1. **Calibration, ~$9-11:** exp-054 unchanged, twice: job `calib054` (packed locally) plays both on one box back to
   back (~6.5 h, ~$9), or split into two one-run jobs on two boxes at once (~3.75 h, ~$10.5). Read: the
   rental public-25 score and requests per run against Kaggle's exp-054 pair, and the spread between the two rental
   runs. Rule: if rental requests per run fall more than 15% below Kaggle's, rental runs compare arms with each
   other only (never with Kaggle numbers).
2. **Serving arms, ~$20-25:** the Oct 3 queue's serving candidates that passed their Kaggle stress test (exp-059/060,
   exp-062), two runs each, on two boxes (setup paid once per box).
3. Then pairs for whatever wins, and the harness arms (exp-064 compaction, exp-063 SGLang if its stress test passes).

## Tested (2026-09-29, CPU only, no money spent)

- tests/test_rental.py (22 tests): the offer filter (RAM share = cpu_ram x gpu_frac, driver, contract, one GPU) and
  prices with our disk; pack (flat files, one arm twice, job sha, refuses internet-on notebooks, bad names and
  re-packing a launched job); the create body (CLI args-mode shape, env object, pinned image, `cancel_unavail`, boot
  command valid bash with its cap and self-stop trap); launch guards (no approval, stale upload, existing label, a
  lost create response recovered by label); the quote; both /kaggle/input layouts; model finishing (leftover tar,
  wrapper flattening only for a model folder, a child sharing the wrapper's name); the preflight gate; the box runner
  end to end on a fake Kaggle CLI (inputs once, `.blob` archives that `collect` unpacks, secrets absent from the
  notebook environment and results); a stale job refused before any download; an upload failing three times with
  exit code 0 neither counted as success nor stopping the next run, then retried; restarts after and during a job;
  process and scratch cleanup (on an injected process list, so the test touches only its own child); the cap on a
  silent notebook; `collect` when Kaggle unpacked an archive anyway.
- The real boot command in a sandbox (paths redirected to a scratch folder, fake Kaggle CLI, papermill from a
  scratch venv, a three-cell notebook listed twice): both runs executed and uploaded, the token invisible in the
  notebook, the trap ran; a second boot (a restart) made no Kaggle call and ran nothing.
- A fresh-context review (2026-09-29) found 13 issues in the first version (Kaggle unpacking `.tar.gz`, exit-0
  upload failures, a failed upload ending the job, host RAM instead of the rental's share, disk not priced, the cap
  leaving vLLM on the GPU, no global deadline or self-stop, launch leaks, env as a string, duplicate arms
  overwriting, warm state between runs, contract length, weak preflight). All are addressed above.
- Not tested (needs the account and a box): the vast.ai create/logs/stop/destroy calls, whether `CONTAINER_API_KEY`
  is set in args mode, the Kaggle model download speed, whether Kaggle unpacks `.blob` files (it did not for
  Keith's runtime layers), papermill's presence in the Kaggle image (it is in Kaggle's requirements list; nbconvert
  is the fallback), and the notebook itself on a non-Kaggle host.
