# Test fixture: Keith Tyser's Flash-Next serving bundle files (unmodified)

`serving_setup.py`, `SOURCE_IDENTITY.json` and `vllm-patches/` are verbatim copies of the files in the Kaggle dataset
`keithtyser/duck-qwen38-nvfp4-mtp-vllm-smoke-v1` (MIT, per its dataset metadata), the version our notebooks mounted in
exp-054 (SOURCE_IDENTITY.json sha256 473e6959..., serving_setup.py sha256 037c041c...). They are here only so
`tests/test_taaf_ours_patch.py` can check that `scripts/nvidia_serving_patch.py` still applies to them exactly and that
the patched setup imports and serves the NVIDIA checkpoint's identity, without the dataset or a GPU. Do not edit them:
the patch pins their hashes.
