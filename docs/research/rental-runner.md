# Rental runner: our Kaggle notebooks on rented RTX PRO 6000s (written 2026-09-29)

One-line summary: `scripts/rental.py` runs the exact notebooks we push to Kaggle on rented vast.ai single-GPU
boxes (Kaggle's own GPU image, inputs downloaded into the /kaggle/input layout) for about $4-5 per public-25 run.
It is built and tested without spending anything, and waits for the owner's account and OK.

## Why

Kaggle gives us 30 GPU-hours a week, about five public-25 runs after stress tests. Identical notebooks vary by
1.5-2.3x between runs (our exp-054 pair: 12.87 / 10.70 on the public 25; 4.70 / 3.36 on the leaderboard), so one run
per arm cannot separate arms that differ by less than about 20%. More runs per arm is the only fix, and rented
GPUs are the only source of more runs.

## What a run costs (vast.ai prices read on 2026-09-29, 22:3x UTC, public offer search)

| Item | Estimate |
|---|---|
| One RTX PRO 6000 S box (server edition, like Kaggle's), >= 170 GB RAM, >= 32 cores, driver >= 580 | $1.40-1.55/h |
| Fresh box: Kaggle image pull (23 GB) + inputs (135 GB model + runtimes + wheels) | ~1 h (estimate; the first rental measures it) |
| One public-25 notebook (16 min serving setup, 132 min play, teardown) | ~2.75 h |
| Disk (500 GB) | about $0.05-0.15/h extra |
| **First run on a box** | **~$5.3** |
| **Each further run on the same box** | **~$4** |

Budget arithmetic: at ~$1.45/h, $28 buys about 19 box-hours. One box for 19 h plays about six runs back to back
(1 h setup + 6 x 2.75 h); two boxes in parallel for 9.5 h play about six runs in half the wall-clock (each box pays
its own 1 h setup). Either way roughly $4.5-5 per run, which is what the $28-for-10-hours two-GPU quote I gave
earlier buys too, now as two separate boxes.

A 2-GPU box does not halve the time: Keith's serving setup (the one every notebook uses) refuses to start unless
nvidia-smi shows exactly one RTX PRO 6000, and nvidia-smi ignores CUDA_VISIBLE_DEVICES. Two 1-GPU boxes cost the
same as one 2-GPU box and need no change to the notebooks.

## How it works

1. `rental.py pack JOB DIR...` takes the folders `build_arms.py` / `build_kv_stress_nb.py` write (notebook +
   kernel-metadata.json) and makes a job: flat files `runN.ipynb`, `runN.kernel-metadata.json`, a `manifest.json`
   and `rental_box.py`. `--upload` creates the private Kaggle dataset `scottmahony/arc3-rental-JOB`.
2. `rental.py offers` lists boxes that pass the filter (1 GPU, RTX PRO 6000, RAM, cores, disk, reliability,
   download speed, driver >= 580 for CUDA 13.0), server edition first.
3. `rental.py quote JOB --offer ID` prints the estimate. **The owner OKs a batch before anything is rented.**
4. `rental.py launch JOB --offer ID --approved "<owner's words>"` rents the box with the Kaggle GPU image
   (`gcr.io/kaggle-gpu-images/python:v170`, the current `latest`, entrypoint `/usr/bin/env`) and a boot command
   run as the container's command: install the Kaggle CLI into /opt/kcli (not the notebook's Python), download the
   job dataset, run `rental_box.py`.
5. On the box, `rental_box.py` records a preflight (GPU, RAM, CPUs, /dev/shm, disk), downloads every input once into
   the paths Kaggle mounts (`/kaggle/input/<slug>` and `/kaggle/input/datasets/<owner>/<slug>`,
   `/kaggle/input/models/<owner>/<model>/<framework>/<instance>/<version>`, `/kaggle/input/<competition>` and
   `/kaggle/input/competitions/<competition>`), then runs each notebook in `/kaggle/working` with papermill (cell output
   streamed to the container log; nbconvert if papermill is missing) under a 5 h cap, and uploads each run's
   `/kaggle/working` as the private dataset `scottmahony/arc3-rental-JOB-N` (`output.tar.gz` + `run.json`).
   `KAGGLE_IS_COMPETITION_RERUN` is never set, so notebooks play the public 25 as in a Kaggle save.
6. When the command ends the container exits and the GPU is released. `rental.py status/logs` watch it,
   `collect` downloads the results into `runs/<run>-rental-JOB/kernel-output` and scores them with
   `pull_taaf_run.py --local`, and `destroy` deletes the stopped instance (its disk is billed until then).

## What the owner sets up (once)

1. A vast.ai account with credit (the first batch needs about $10-15).
2. An API key from vast.ai (Account -> Keys) saved as `.vast/api_key` in the repo checkout I work in (git-ignored).
3. Accept that the Kaggle token goes to the rented box: it is passed as an environment variable, so vast.ai and the
   host machine can read it. **After each rental, generate a new Kaggle token** (kaggle.com -> Settings -> API ->
   "Generate New Token", which expires the old one) and save the new one in `.kaggle/access_token`.
   Prefer vast.ai's verified / datacenter hosts for this reason.

## Fidelity and what to run first

- The box differs from Kaggle's g4-standard-48 in CPU model, RAM bandwidth and PCIe (the 48 GiB PLE table sits in
  host RAM), so tokens per second can differ. **The first rental run is exp-054 unchanged**, read against Kaggle's
  exp-054 pair (12.87 / 10.70, 1,609 / 1,627 requests). Arms are then compared with each other on rentals.
- Rental results are development evidence only; LB draws still come from Kaggle submissions (the owner submits).

## Proposed first batches (each needs the owner's OK before `launch`)

Measured input sizes (Kaggle API, 2026-09-29): model 135 GB, Keith's runtime 7.9 GB, the SGLang wheel datasets
6.6 + 4.9 GB (first page of files), bundles < 0.1 GB; peak disk about 370 GB with the image, so 500 GB fits.

1. **Calibration, ~$11:** exp-054 unchanged on two boxes at once (job `calib054` is packed). Read: the rental
   public-25 score and requests per run against Kaggle's exp-054 pair (12.87 / 10.70; 1,609 / 1,627 requests), and the
   spread between the two rental runs. Rule: if rental requests per run fall more than 15% below Kaggle's, rental
   runs compare arms with each other only (never with Kaggle numbers).
2. **Serving arms, ~$20-25:** the Oct 3 queue's serving candidates that passed their Kaggle stress test (exp-059/060,
   exp-062), two runs each, packed on two boxes (setup paid once per box).
3. Then pairs for whatever wins, and the harness arms (exp-064 compaction, exp-063 SGLang if its stress test passes).

## Tested (2026-09-29, CPU only, no money spent)

- tests/test_rental.py (10 tests): offer filter (driver, 1 GPU, server edition first), both /kaggle/input layouts
  (including the lower-case model path Keith's setup pins), pack (flat files, refuses internet-on notebooks), the
  create-instance body (entrypoint mode, pinned image, token only in env, boot command valid bash and under vast.ai's
  4,048-character onstart limit), the spend guard, the quote, the box runner end to end against a fake Kaggle CLI
  (each input once for two runs, wrapper folder flattened, results uploaded private, token absent from the notebook
  environment and results), and the wall-clock cap on a silent notebook.
- A real papermill run of a small notebook through `rental_box.execute` (scratch venv): output streamed live, the
  executed notebook saved, cwd the run's working directory, no Kaggle token visible, top-level `await` works (our
  notebooks `await bm.run(...)`).
- Read from the registry: the Kaggle image v170 is public, 23.0 GB compressed, entrypoint `/usr/bin/env`, no default
  command, `NVIDIA_REQUIRE_CUDA=cuda>=12.8`.
- Not tested (needs the account and a box): the vast.ai create/logs/destroy calls, the Kaggle model download speed,
  papermill's presence in the Kaggle image, and the notebook itself on a non-Kaggle host.
