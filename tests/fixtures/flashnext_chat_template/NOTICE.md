# Test fixture: the Qwen3.8-Flash-Next chat template (unmodified)

`chat_template.jinja` (8,952 bytes, sha256 c3cf9e34...) is a verbatim copy of the file in the Hugging Face repo
`RadixArk/Qwen3.8-Flash-Next-NVFP4` at revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594, the checkpoint our notebooks
serve from the Kaggle model `keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1` (NVIDIA's
`nvidia/Qwen3.8-Flash-Next-NVFP4` ships the identical file). Qwen licence terms apply to the model files. It is here only
so `tests/test_taaf_ours_patch.py` can render a history through the template the server uses and show where past
reasoning appears (P30), without the 135 GB model. `scripts/sglang_serving.py` pins the same hash for the mounted copy.
Do not edit it.
