What the competition dataset ships and how Kaggle mounts things (verified from the API and one CPU run on 2026-09-16).

- `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/` holds `arc_agi_3_wheels/` (arc-agi 0.9.8 + arcengine 0.9.3 + deps),
  `ARC-AGI-3-Agents/` (framework snapshot from 2026-04-17; its `do_action_request` reads reasoning from action data,
  GitHub main reads it from the enum; our adapter overrides the method so neither matters) and `environment_files/`
  (all 25 public games with `metadata.json` baselines).
- arc-agi 0.9.8 (Kaggle) and 0.9.9 (PyPI) are identical in scorecard.py, api.py, base.py, local_wrapper.py, wrapper.py.
- Datasets mount at `/kaggle/input/<slug>` or `/kaggle/input/datasets/<owner>/<slug>`; probe both.
- Kaggle derives the kernel slug from the *title*, not the `id` in kernel-metadata.json; keep them consistent
  (ours: `scottmahony/arc-prize-2026-arc-agi-3-arc3-agent`).
- The "Save & Run All" phase (no `KAGGLE_IS_COMPETITION_RERUN`) is a free end-to-end test: our notebook plays two
  bundled games offline there, so a broken package fails before a submission is spent.
- GPU images keep the CUDA driver libs off the linker path; prepend `/usr/local/nvidia/lib64` (Duck notebook) and
  set `VLLM_USE_FLASHINFER_SAMPLER=0` (2nd/3rd place notebooks) before starting vLLM.
