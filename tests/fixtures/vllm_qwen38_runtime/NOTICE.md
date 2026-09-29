# Test fixture: three vLLM files from Keith Tyser's pinned runtime (unmodified)

`vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py`, `vllm/models/qwen3_8_flash_next/nvidia/mtp.py` and
`vllm/model_executor/layers/quantization/modelopt.py` are verbatim copies from layer 24 (sha256 10dce885...) of the
Kaggle dataset `keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1`, i.e. the `vllm/vllm-openai:qwen38-flash-next` image,
vLLM 0.1.dev20073+g8e685d198 (Apache-2.0, copyright the vLLM project contributors; see the SPDX headers). They are
here only so `tests/test_taaf_ours_patch.py` can check that the backports in `scripts/nvidia_serving_patch.py` apply
to them and produce the pinned hashes. Do not edit them.
