# Exact RadixArk NVFP4 PLE loader patch

This patch changes only vLLM's PLE loader selection. It reuses the existing
global-scale FP8 PLE embedding loader. It does not change embedding lookup or
forward math.

The new path stays off unless all checks pass:

- `VLLM_RADIXARK_QWEN38_NVFP4_PLE_FP8=1`;
- `VLLM_RADIXARK_QWEN38_NVFP4_CONFIG_SHA256` equals the pinned config hash;
- the quantization object is `ModelOptNvFp4Config` and reports
  `modelopt_fp4`;
- the quantization object reports serialized `NVFP4`, group size 16, and
  confirms that the checkpoint PLE prefix is excluded from body quantization;
- the PLE parameter prefix is the exact vLLM runtime layer-2 prefix,
  `language_model.model.layers.1.ple.ple_embedding.ngram_embedding`; and
- the Qwen text config matches the pinned PLE, model, expert, and layer fields.

The setup code must compute the model `config.json` hash before it exports the
two environment variables. The only accepted model config is the untouched
RadixArk file, `model/RadixArk-Qwen3.8-Flash-Next-NVFP4/config.json`, SHA-256
`e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624`. Do not
use the earlier SGLang-compatible staged config.

Apply the patch to an extracted vLLM site-packages directory:

```text
python apply_radixark_nvfp4_ple_fp8_patch.py /path/to/site-packages
```

Run the local tests against the pinned stock file and exact model config:

```text
python test_radixark_nvfp4_ple_fp8_patch.py \
  --stock /path/to/stock/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py \
  --config /path/to/model/config.json
```
