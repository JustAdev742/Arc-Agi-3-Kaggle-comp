# gpt-oss-120b (MXFP4) as a candidate served model (notes, 2026-09-17; nothing measured yet)

Why: road-to-100 section 5. The strongest open reasoner and coder that fits one 96 GB GPU; MoE with about 5B active
parameters, so many more tokens per second than the dense 27B; a reasoning-effort knob (low / medium / high); text-only,
so it plays from the grid text and the entity summaries (the Duck found the image helped a 27B; an A/B question).

## Assets

- Kaggle model `danielhanchen/gpt-oss-120b/transformers/default/1` (Unsloth mirror of `openai/gpt-oss-120b`, Apache-2.0,
  65.3 GB, 14 safetensors shards, 2025-08-07): kernel `model_sources` entry; mounts under `/kaggle/input/models/...`
  (the diag notebook locates the folder holding `config.json`).
- vLLM 0.27.1 wheelhouse `saltb0x/arc3-vllm-wheelhouse-v0271-cu129` (gpt-oss needs vLLM >= 0.10).

## Serving recipe on the RTX PRO 6000 (SM 12.0), offline

Facts gathered 2026-09-17 (see the research log entry of that date for sources):
- vLLM's MXFP4 MoE oracle on SM120 does not select the FlashInfer TRT-LLM kernels (SM100 only); the native CUTLASS
  path needed a custom FlashInfer build with `FLASHINFER_CUDA_ARCH_LIST=12.0f` in the one public write-up that got it
  working (NVIDIA forum), and FlashInfer JIT is exactly what failed offline in our diag v2-v4. vLLM 0.27.1 has
  `--moe-backend {marlin,triton,flashinfer_cutlass,...}`; Marlin (weight-only FP4, BF16 compute) is the known-good
  fallback on SM120 (vLLM issue 30135), Triton the second candidate.
- Parsers in 0.27.1: `--tool-call-parser openai`, `--reasoning-parser openai_gptoss`. Reasoning effort is the OpenAI
  request field `reasoning_effort` (harmony), not a chat-template kwarg: `ChatClient(effort_in_request=True)` /
  agent config `effort_in_request: true`.
- No MTP head (`mtp_tokens: 0`); attention `TRITON_ATTN` as for the 27B; no `--limit-mm-per-prompt` (text-only).
- Reported elsewhere (unverified here): about 190 tok/s single stream and 4,630 tok/s aggregate with the CUTLASS path.

Diag ladder (`scratchpad/nb/diag-gptoss`, slug `arc3-diag-gptoss`): (1) Marlin MoE, FP8 KV, 32 seqs; (2) Marlin, auto KV,
eager, 16 seqs; (3) Triton MoE, auto KV, eager. Probe efforts low/medium/high; REPL smoke on ls20 + vc33 without images,
temperature 1.0. Decision rule (road-to-100 section 5): a dev run only if aggregate tok/s at 8 concurrent >= 308 and the
smoke solves at least what the 27B did.

## Risks

- Marlin is slower than the native kernels; if the aggregate throughput is below the 27B's, the model loses its main
  argument.
- Text-only: the harness's `image: false` path is measured only on the exp-004 class of runs; the model must read the
  full grid from `ascii` text (auto-enabled when the image is off).
