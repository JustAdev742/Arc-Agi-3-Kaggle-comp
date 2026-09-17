# NVIDIA Nemotron 3 Super 120B-A12B (NVFP4) as a candidate served model (notes, 2026-09-17; nothing measured yet)

Why: road-to-100 section 5. Strong open agentic model, MoE with 12B active parameters (hybrid Mamba-2 + MoE +
attention, trained in NVFP4), so fast per token; text-only.

## Assets

- Kaggle model `sivavoleti/nemotron-3-super-120b-a12b-nvfp4/transformers/default/1` (80.4 GB, 35 files incl. README,
  `chat_template.jinja`, `super_v3_reasoning_parser.py`, `modeling_nemotron_h.py`; license label "Other"). The copy
  `zaynyu/nvidia-nemotron-3-super-120b-a12b-nvfp4/transformers/v1/1` is the same checkpoint without the README and is
  mislabelled Apache-2.0.
- **License: NVIDIA Nemotron Open Model License** (model card), not an OSI open-source license. The prize rules require
  open-sourcing the winning solution; whether a served model under this license is acceptable is the owner's call
  before this model is used in a submission.
- vLLM 0.27.1 wheelhouse (the model card's recipes use 0.20.0 and, for DGX Spark, the 0.27.1 container).

## Serving recipe on the RTX PRO 6000 (SM 12.0), offline

Facts gathered 2026-09-17:
- Model card (0.20.0): `--kv-cache-dtype fp8 --mamba-ssm-cache-dtype float16 --max-num-seqs 32
  --reasoning-parser-plugin super_v3_reasoning_parser.py --reasoning-parser super_v3 --enable-auto-tool-choice
  --tool-call-parser qwen3_coder`; temperature 1.0, top_p 0.95 for every task. The plugin ships in the checkpoint
  folder (the attempts reference it as `{MODEL_DIR}/super_v3_reasoning_parser.py`).
- DGX Spark recipe for the 0.27.1 container: `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_USE_FLASHINFER_MOE_FP4=0`
  (the same pair `arc3.serve.NVFP4_ENV` already uses for the NVFP4 specialist), MTP with the separate MTPv2 checkpoint.
- Hugging Face discussion 9 (RTX 6000 Pro, 2026-03): with MTP off the server fits in about 77 GB (`--attention-backend
  TRITON_ATTN`, FP8 KV, 20 seqs, 32k context); MTP OOMs even at 0.95, so `mtp_tokens: 0`. About 16-19 GB are left for
  KV cache: low concurrency, which is the lever we need most (road-to-100 section 4).
- Thinking on/off via `enable_thinking` in `chat_template_kwargs` (our client's `thinking` knob); no effort levels.

Diag ladder (`scratchpad/nb/diag-nemotron`, slug `arc3-diag-nemotron`): (1) default kernels, FP8 KV, 16 seqs; (2) Marlin
NVFP4 GEMM env, FP8 KV; (3) Marlin env, auto KV, eager, 8 seqs. REPL smoke on ls20 + vc33 without images, temperature
1.0 / top_p 0.95.

## Decision (2026-09-17, owner delegated the call)

Not used for the submission: the NVIDIA Open Model License is not an OSI open-source license and the prize rules ask
for open-sourced solutions, and on 96 GB the model leaves about 16 GB for KV cache, which caps the concurrency the
9-hour budget needs most (road-to-100 section 4). The serving check is dropped from the quota queue; the built kernel
(`scratchpad/nb/diag-nemotron`) and these notes stay as the record. gpt-oss-120b (Apache-2.0) is the candidate.

## Risks

- KV headroom (about 16 GB) caps concurrency; the aggregate tok/s at 8 concurrent decides.
- The license question above.
- 80 GB to load from the Kaggle input mount (the 29 GB 27B took 120 s).
