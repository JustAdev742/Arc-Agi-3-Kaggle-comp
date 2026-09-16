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
- The RTX PRO 6000 accelerator id in notebook metadata is `nvidiaRtxPro6000` (2nd/3rd place notebooks). The starter kit's
  `nvidiaRtx6000` is unknown to Kaggle and the run silently lands on 2x T4 (our diag run #2). Always check `nvidia-smi`
  in the log before trusting a GPU run.
- On the RTX PRO 6000, vLLM 0.27.1 auto-selects the FlashInfer attention backend and the first request dies
  ("FlashInfer requires GPUs with sm75 or higher": its SM120 cubins come from NVIDIA's artifactory, unreachable
  offline). Force `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (`arc3.serve.kaggle_env`). Model load itself was fine:
  28.95 GiB FP8 weights in 120 s, MTP draft detected, 53 GiB KV cache, server up in ~5.5 min.
