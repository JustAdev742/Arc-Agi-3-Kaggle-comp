#!/usr/bin/env python
"""Serve our Duck fork's Flash-Next checkpoint with SGLang 0.5.20 on Kaggle: offline install, rung ladder, vLLM fallback.

    python sglang_serving.py setup      # the setup command of the swapped bundle (build_taaf_nb.py --engine sglang)
    python sglang_serving.py teardown   # its first teardown command; Keith's own teardown runs after it
    python sglang_serving.py watchdog   # started detached by setup: restarts a dead server, then falls back to vLLM
    python sglang_serving.py command [--running R] [--hicache-gb G] [--rung N] [--model-dir DIR]   # print a command

Our code (team scottmahony). Son Pham's repository and the gabrielolympie fork have no licence and were read for
facts only (docs/research/sglang-serving-plan.md); SGLang itself is Apache-2.0. What it does, in order:

1. **Hash check.** Every wheel below is checked by size and sha256 (8 threads) before anything is installed. The 200
   wheels are sglang 0.5.20's dependency closure resolved against the two public datasets (plan section 2): 199 from
   ``aaravbajya/arc-agi-sglang-workspace`` and torchvision 0.28.0 (the build that pins torch 2.13.0) from
   ``nick2187/qwen38-vllm0272-cu130-wheelhouse-v1``. Each sha256 was computed from the dataset file itself on
   2026-09-29 (files downloaded once with the Kaggle CLI); sglang, flashinfer-python, humming-kernels and torchvision
   also match PyPI. Two upstream requirements are absent and unused here (``build``, and ``smg-grpc-servicer``, gRPC
   mode only); one pin conflict is upstream's own (flash-attn-4 4.0.0b31 wants apache-tvm-ffi>=0.1.12, sglang and
   tilelang pin 0.1.11: we keep 0.1.11; FA4 is not a default backend on sm120). The dataset's jit-cache wheel is named
   ``flashinfer_jit_cache-0.6.18cu130-...``, an invalid wheel name pip rejects; it is staged as ``0.6.18+cu130``, the
   version its dist-info declares.
2. **Install** into a clean venv (``/tmp/sgl-venv``, no system site-packages, so Kaggle's own torch cannot leak in)
   with ``pip install --no-index --no-deps`` on the explicit list; ``pip check`` is logged, informational only.
3. **CUDA 13.0 toolkit**: the ``usr/local/cuda-13.0`` tree (nvcc, ptxas, cicc, headers; 2.4 GB) from layer 8 of
   Keith Tyser's runtime dataset, extracted alone; the pip toolchain (13.4, newer than the 580 driver) is the fallback.
4. **Import probe** in the venv (torch, torchvision, flashinfer, sgl_kernel, sglang, the Qwen VL processor).
5. **Server** on 127.0.0.1:1234 as ``Qwen/Qwen3.8-Flash-Next-NVFP4`` (the id the notebook asserts), through a rung
   ladder (``RUNGS``); a rung is up after /health, /v1/models, one text and one 256x256 board-image completion.
6. **Fallback to vLLM**: if no rung is up 50 min after notebook start (or anything above fails), Keith's original
   setup commands run from the same bundle copy; a run is never left without a server.
7. **Persisted environment**: the analyzer keys exactly as Keith's setup persists them (the notebook asserts some of
   them), plus ``OURS_SERVING`` (``sglang`` or ``vllm-fallback``), which switches the harness patch P30 on.
8. **Watchdog** (a detached process): /health every 30 s; 3 misses restart the same rung (twice at most), then vLLM.
   Keith's in-notebook vLLM watchdog (cell 16) also runs: it sees SGLang as healthy (its check is /v1/models listing
   the served id), and while SGLang serves, its restart attempts fail harmlessly (no vllm-server-identity.json, so
   ``restart_owned_server`` raises before stopping or starting anything). An SGLang outage can use up his two
   attempts, so after our own fallback to vLLM nothing restarts that vLLM server; accepted (a second outage is rare).
9. **Teardown**: /metrics snapshot, watchdog and server process groups stopped, JIT caches copied (bounded).

Status 2026-09-29: written and unit-tested on CPU (flags parsed by SGLang 0.5.20's own CLI parser, manifest hashes
from the dataset files); never run on a GPU.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path
from typing import Any

SERVED_MODEL_NAME = "Qwen/Qwen3.8-Flash-Next-NVFP4"  # Keith's served name; the notebook asserts it (cell 10)
HOST = "127.0.0.1"
PORT = 1234
BASE_URL = f"http://{HOST}:{PORT}/v1"
ANALYZER_CONTEXT = 32_768  # the harness window (Keith's LOCAL_ANALYZER_CONTEXT_WINDOW)

MODEL_SOURCE = "keithtyser/qwen3-8-flash-next-nvfp4/PyTorch/radixark-modelopt-fp4/1"
MODEL_KAGGLE_PATH = Path("/kaggle/input/models/keithtyser/qwen3-8-flash-next-nvfp4/pytorch/radixark-modelopt-fp4/1")
MODEL_CONFIG_SHA256 = "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624"  # RadixArk @ 7b719225
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"  # same file as NVIDIA's

WHEEL_DATASET = "aaravbajya/arc-agi-sglang-workspace"
TORCHVISION_DATASET = "nick2187/qwen38-vllm0272-cu130-wheelhouse-v1"
RUNTIME_DATASET = "keithtyser/qwen38-flash-next-vllm-nvfp4-runtime-v1"
EXTRA_DATASET_SOURCES = [WHEEL_DATASET, TORCHVISION_DATASET]  # attached in addition to the base notebook's inputs
CUDA_LAYER = "layer-8-f4f325a97c3e4739640b0e472a818bb299e7ce8e585bdbfe5586e9d3b5a2c475.tar.gz.blob"
CUDA_LAYER_BYTES = 1_374_796_713  # his runtime-manifest.json; the layer holds a complete usr/local/cuda-13.0
DATASET_MARKERS = {
    WHEEL_DATASET: "archive/sglang-0.5.20-cp312-cp312-manylinux_2_34_x86_64.whl",
    TORCHVISION_DATASET: "wheels/torchvision-0.28.0-cp312-cp312-manylinux_2_28_x86_64.whl",
    RUNTIME_DATASET: CUDA_LAYER,
}
WHEEL_RENAMES = {  # dataset file name -> the valid wheel name pip needs (the version its dist-info declares)
    "flashinfer_jit_cache-0.6.18cu130-cp39-abi3-manylinux_2_28_x86_64.whl":
        "flashinfer_jit_cache-0.6.18+cu130-cp39-abi3-manylinux_2_28_x86_64.whl",
}
# key, path in the dataset, bytes, sha256 (sgl = WHEEL_DATASET, tv = TORCHVISION_DATASET); 6,648,674,092 bytes in all
WHEELS = """
sgl archive/aiohappyeyeballs-2.7.1-py3-none-any.whl 15038 9243213661e29250eb41368e5daa826fc017156c3b8a11440826b2e3ed376472
sgl archive/aiohttp-3.14.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 1792122 543906c127fb1d929b95076db19b83fa2d46751006ff1e23b093aa5ac4d8db42
sgl archive/aiosignal-1.4.0-py3-none-any.whl 7490 053243f8b92b990551949e63930a839ff0cf0b0ebbe0597b0f3fb19e1a0fe82e
sgl archive/airportsdata-20260905-py3-none-any.whl 931164 d7eaa9a57d373b0adaaae0d52da2af0bb0edf73f7baa170f7c415d3342bf4868
sgl archive/annotated_doc-0.0.5-py3-none-any.whl 5302 117bac03a25ede5df5440e855b32d556049ca169ead221505badf432fed4b101
sgl archive/annotated_types-0.8.0-py3-none-any.whl 13427 f072f4d804ea359e4eaf198b1af7a8b0943881a87f31bb764f8bf219bb9419e0
sgl archive/anthropic-1.8.0-py3-none-any.whl 1348276 79a4516a21e64fd7b15be1a49ebf544bd6376c96a971a77365358823a2717bc6
sgl archive/anyio-4.15.1-py3-none-any.whl 132079 6152fdbbf9a77fdec97731721bebf7c4c44f7c29b424b0065826173efc7ed101
sgl archive/apache_tvm_ffi-0.1.11-cp312-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl 2697683 2843f084cdc94dedacd8b257a395a2b71b8a3dc7fc99711b148bf1d161983128
sgl archive/asttokens-3.0.2-py3-none-any.whl 28702 9da13157f5b28becde0bd374fc677dcd3c290614264eff096f167c469cd9f933
sgl archive/attrs-26.1.0-py3-none-any.whl 67548 c647aa4a12dfbad9333ca4e71fe62ddc36f4e63b2d260a37a8b83d2f043ac309
sgl archive/blobfile-3.0.0-py3-none-any.whl 75413 48ecc3307e622804bd8fe13bf6f40e6463c4439eba7a1f9ad49fd78aa63cc658
sgl archive/certifi-2026.7.22-py3-none-any.whl 136983 62f22742b58a1a33014a2b6b706588a8d7e2a88ae7bd1a6ebe8c992928483775
sgl archive/cffi-2.1.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 221822 c1453022f490d2459a11819d83ad1d586e9ff65a12ac3e705ffebd46d3685dcf
sgl archive/charset_normalizer-3.5.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 248801 b9af956078716df40d985fb0dfeb2c2120c5ca92ba4ff4b388acfd01cdc14d08
sgl archive/click-8.5.0-py3-none-any.whl 125251 255bc9599cf7748b4b1a446ccc735421bd08a2ae529a8b88597d3de5664ee360
sgl archive/cloudpickle-3.1.2-py3-none-any.whl 22228 9acb47f6afd73f60dc1df93bb801b472f05ff42fa6c84167d25cb206be1fbf4a
sgl archive/compressed_tensors-0.18.0-py3-none-any.whl 224997 91168237c2d815614c44dfc354c61c613085c811dabbb2ccdb929e9aaa313b8f
sgl archive/cryptography-50.0.1-cp311-abi3-manylinux_2_34_x86_64.whl 4746230 51afcfceb15597cf2635068e4ac9a56b2abde622edde17f37d85fd7b5306497a
sgl archive/cuda_bindings-13.4.3-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl 7172185 d5f72bcfcdf3be23e1da3c792f68f508586f48d037bca8b10f552c4cca5971f2
sgl archive/cuda_core-1.2.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl 6608833 cd36063535f75a88c4eb44ba9a678441491a5f67a72f62c2e3b797e8a052d438
sgl archive/cuda_pathfinder-1.8.2-py3-none-any.whl 62551 4e65059febdb4d19d5cbc4798677e19db2b582f2f702f457b609e571690d357e
sgl archive/cuda_python-13.4.1-py3-none-any.whl 7594 b6b114882beb8325139a318a02bcf8feb821dfd724f6f6104c262e2d1e2035ef
sgl archive/cuda_tile-1.6.0rc5-cp312-cp312-manylinux2014_x86_64.whl 353300 b74c20348210d2182cd998a0ecb60c518989a79b28592d72eb8294b38ddb93d7
sgl archive/cuda_toolkit-13.0.3.0-py2.py3-none-any.whl 2512 d693caaa261214ddd7dbb60d68e71cbed884e68c2be7509778f3051da0b91c3f
sgl archive/cython-3.3.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 3412225 428fafed98ea26927000a287b4dfc9ef07339f56656a5329a34eaa593f79a4f8
sgl archive/datasets-5.0.1-py3-none-any.whl 559079 9fbf73688f8c18f7529b4fe592abd04015f81d1e58001e4bac73ffb2b39d7cc4
sgl archive/dill-0.4.1-py3-none-any.whl 120019 1e1ce33e978ae97fcfcff5638477032b801c46c7c65cf717f95fbc2248f79a9d
sgl archive/diskcache-5.6.3-py3-none-any.whl 45550 5e31b2d5fbad117cc363ebaf6b689474db18a1f6438bc82358b024abd4c2ca19
sgl archive/distro-1.9.0-py3-none-any.whl 20277 7bffd925d65168f85027d8da9af6bddab658135b840670a223589bc0c8ef02b2
sgl archive/docstring_parser-0.18.0-py3-none-any.whl 22484 b3fcbed555c47d8479be0796ef7e19c2670d428d72e96da63f3a40122860374b
sgl archive/easydict-1.13-py3-none-any.whl 6804 6b787daf4dcaf6377b4ad9403a5cee5a86adbc0ca9a5bcf5410e9902002aeac2
sgl archive/einops-0.9.0.dev0-py3-none-any.whl 67308 9cd97ef9fc37ff36d02778f0b2d71a8cd8c064e180885e01e772c0c0a8154b4d
sgl archive/executing-2.2.1-py2.py3-none-any.whl 28317 760643d3452b4d777d295bb167ccc74c64a81df23fb5e08eff250c425a4b2017
sgl archive/fastapi-0.141.1-py3-none-any.whl 131954 bfb91aa2d334c61cb35ba9a116fc123b3d3df31640b801cf57a7a78ec3f603b3
sgl archive/filelock-4.0.1-py3-none-any.whl 106219 481a321a27bef441e23c53371c6abc8d7d16e26b97090074ba44f7538a3fd55a
sgl archive/flash_attn_4-4.0.0b31-py3-none-any.whl 411043 6eda5890b29e90fc46e19a47b4018effae7c042f75ee8aaaeebd3f56ccd82edf
sgl flashinfer_cubin-0.6.18-py3-none-any.whl 1565107227 2dd65c0fcfc6bc44c67f148530de5372979c2e3d260e47935730f94156d4d873
sgl flashinfer_jit_cache-0.6.18cu130-cp39-abi3-manylinux_2_28_x86_64.whl 1016181643 428a47a554ade93c30a818e142b781df58582bf056bab94611fb4c906cc366bf
sgl archive/flashinfer_python-0.6.18-py3-none-any.whl 18347767 d5d26edb48f8def0bf28b2c94415aef7f0754e2cd9e3cb3d6e7081262d068aad
sgl archive/frozenlist-1.8.0-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl 242411 494a5952b1c597ba44e0e78113a7266e656b9794eec897b19ead706bd7074383
sgl archive/fsspec-2026.6.0-py3-none-any.whl 203949 02e0b71817df9b2169dc30a16832045764def1191b43dcff5bb85bdee212d2a1
sgl archive/gguf-0.19.0-py3-none-any.whl 118475 70bcd10edfe697fb2dad6e40af2234b9d8ece9a41a99761405121ebda1c3c1cd
sgl archive/h11-0.16.0-py3-none-any.whl 37515 63cf8bbe7522de3bf65932fda1d9c2772064ffb3dae62d55932da54b31cb6c86
sgl archive/hf_xet-1.6.1a0-cp38-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 4464369 a8309120935178037227a03c9c33afb82e64b05d7219d3d1334a7c7a8223b787
sgl archive/httpcore-1.0.9-py3-none-any.whl 78784 2d400746a40668fc9dec9810239072b40b4484b640a8c38fd654a024c7a1bf55
sgl archive/httpcore2-2.13.0-py3-none-any.whl 83162 35ae5be347aa40467b4a5dc032ac67ebb6d27189fc97e8cebcf99616f6a1bb9e
sgl archive/httpx-0.28.1-py3-none-any.whl 73517 d909fcccc110f8c7faf814ca82a9a4d816bc5a6dbfea25d6591d6985b8ba59ad
sgl archive/httpx2-2.13.0-py3-none-any.whl 95565 fc12720cedf72faa26cca6b4ca394e05c894e7d7933fc45cafe767960804e49a
sgl archive/huggingface_hub-1.32.0-py3-none-any.whl 842906 b0c7c80561969d9cdacdd55fce67ba9584cca0b9d4ea80957a3a5c1445fac5c8
sgl archive/humming_kernels-0.1.12-py3-none-manylinux_2_28_x86_64.whl 322656 cd3ef712a93f3a9075ea99de2c72bcd3ec89dab3759b3a248d869f5507b60331
sgl archive/idna-3.20-py3-none-any.whl 69583 ab7ae7122974553370f0bdb919e1a960b2cd1bc1ef0276416d896db81c14582c
sgl archive/interegular-0.3.3-py37-none-any.whl 23635 b0c07007d48c89d6d19f7204972d369b2a77222722e126b6aa63aa721dc3b19c
sgl archive/ipython-9.17.1-py3-none-any.whl 639038 6d1645743cfd1a07eb695d85aa2b5fa66721f8cbae9431d4049f7084bbf06509
sgl archive/ipython_pygments_lexers-1.1.1-py3-none-any.whl 8074 a9462224a505ade19a605f71f8fa63c2048833ce50abc86768a0d81d876dc81c
sgl archive/jedi-0.20.0-py2.py3-none-any.whl 4884812 7bdd9c2634f56713299976f4cbd59cb3fa92165cc5e05ea811fb253480728b67
sgl archive/jinja2-3.1.6-py3-none-any.whl 134899 85ece4451f492d0c13c5dd7c13a64681a86afae63a5f347908daf103ce6d2f67
sgl archive/jiter-0.17.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 345025 8c21265b251d99bbb40080d178a8953e35601d3a1564e05c4de4c0d2ca616797
sgl archive/jsonschema-4.26.0-py3-none-any.whl 90630 d489f15263b8d200f8387e64b4c3a75f06629559fb73deb8fdfb525f2dab50ce
sgl archive/jsonschema_specifications-2025.9.1-py3-none-any.whl 18437 98802fee3a11ee76ecaca44429fda8a41bff98b00a0f2838151b113f210cc6fe
sgl archive/kernels-0.14.1-py3-none-any.whl 57977 be1a91116e14be1e012fc8f47afb51173ee052586a570fba3d594ef0d1b38920
sgl archive/kernels_data-0.16.2-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 1284019 9526362517c7b128f7baec080301890875cc3f5284a0f125b082c19737e5c1d5
sgl archive/lark-1.3.1-py3-none-any.whl 113151 c629b661023a014c37da873b4ff58a817398d12635d3bbb2c5a03be7fe5d1e12
sgl archive/llguidance-1.8.0-cp39-abi3-manylinux_2_31_x86_64.whl 3093405 39668c11396896e5f05f59b70c81e4afd060b3408f02c8518b7a6943bfbb8a5d
sgl archive/llvmlite-0.47.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 56275178 5853bf26160857c0c2573415ff4efe01c4c651e59e2c55c2a088740acfee51cd
sgl archive/loguru-0.7.3-py3-none-any.whl 61595 31a33c10c8e1e10422bfd431aeb5d351c7cf7fa671e3c4df004162264b28220c
sgl archive/lxml-7.0.0b1-cp312-cp312-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl 5298306 4d08cef43edf1a022f454835b4a8b097bdce791530351b019c3449ce9ce6f7a3
sgl archive/markdown_it_py-4.2.0-py3-none-any.whl 91687 9f7ebbcd14fe59494226453aed97c1070d83f8d24b6fc3a3bcf9a38092641c4a
sgl archive/markupsafe-3.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 22947 d6dd0be5b5b189d31db7cda48b91d7e0a9795f31430b7f271219ab30f1d3ac9d
sgl archive/matplotlib_inline-0.2.2-py3-none-any.whl 9534 3c821cf1c209f59fb2d2d64abbf5b23b67bcb2210d663f9918dd851c6da1fcf6
sgl archive/mdurl-0.1.2-py3-none-any.whl 9979 84008a41e51615a49fc9966191ff91509e3c40b939176e643fd50a5c2196b8f8
sgl archive/mistral_common-1.12.0-py3-none-any.whl 6580267 fa4504b66c30c0201ae4578c0340c5ee2abd22151c271532f62e373b985a53cf
sgl archive/ml_dtypes-0.6.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 409890 3b4a480aa8fd54a1805b8ac10f3f91763926a74f73c0c364c10f9231854f4170
sgl archive/modelscope-1.40.1-py3-none-any.whl 6061167 6f135fe80903e7bfcd6ccc707f4f6f85c8e445cf497b624bd07299582486e2c1
sgl archive/modelscope_hub-0.4.5-py3-none-any.whl 236385 9157bbc34d93a70d099c8ab87a113fc9a4149e31ab79e8eb8f7563884043f082
sgl archive/mpmath-1.3.0-py3-none-any.whl 536198 a0b2b9fe80bbcd81a6647ff13108738cfb482d481d826cc0e02f5b35e5c88d2c
sgl archive/msgspec-0.21.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 225025 21995e74b5c598c2e004110ad66ec7f1b8c20bf2bcf3b2de8fd9a3094422d3ff
sgl archive/multidict-6.9.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 336881 976fd7689d69ec78d67d31d38d396d8adb562f7e8368279f76aed4aa451fa06d
sgl archive/multiprocess-0.70.19-py312-none-any.whl 150281 3a56c0e85dd5025161bac5ce138dcac1e49174c7d8e74596537e729fd5c53c28
sgl archive/nccl4py-0.5.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl 3572033 51156cbc35eb9122b870aa7ba1244d0c4781266442036c2bb3e1abb7e88d9376
sgl archive/nest_asyncio-1.6.0-py3-none-any.whl 5195 87af6efd6b5e897c81050477ef65c62e2b2f35d51703cae01aff2905b1852e1c
sgl archive/networkx-3.7-py3-none-any.whl 2142205 e3fd2c13a7814cee3746340d8d7f8598a67f16a58bf47fb7f8793fab6efca1b0
sgl archive/ninja-1.13.2-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 183365 65a24341b5ac09fcadcc37082660be40a94174e51a937fabf6e2cae26225fa2c
sgl archive/numba-0.65.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 3802467 52bc6f3ceb8fcaff9b2ae26b4c6b1e9fee39db8d355534c0fe4f39a901246b84
sgl archive/numpy-2.3.5-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 16606086 0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28
sgl archive/nvidia_cublas-13.1.1.3-py3-none-manylinux_2_27_x86_64.whl 423138758 37936a16db8fe4ac1f065c2139360608a543a09275cb1a1af612e08cfa065436
sgl archive/nvidia_cuda_cccl-13.3.4.3.1-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 3930678 48ba8e44a1162face51306d5bfd67604d9949be5acb9105dbb93734d2df8b3a4
sgl archive/nvidia_cuda_crt-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 160871 731ce3de11df8add8306404417a1edaede225dbe27eb233aed37c8de9d4fe720
sgl archive/nvidia_cuda_cupti-13.0.85-py3-none-manylinux_2_25_x86_64.whl 10715597 4eb01c08e859bf924d222250d2e8f8b8ff6d3db4721288cf35d14252a4d933c8
sgl archive/nvidia_cuda_nvcc-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 47672022 a1f3bfb27299e060b444d5df1f4bcd762501326cf0fd3141ed61756816ab9a0e
sgl archive/nvidia_cuda_nvdisasm-13.4.92-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 5203296 f35a5b6ddb64b6758c22c0b078c5b6ff6d54f7c240cd74bf131ab5bf13b6c3fe
sgl archive/nvidia_cuda_nvrtc-13.0.88-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 90215200 ad9b6d2ead2435f11cbb6868809d2adeeee302e9bb94bcf0539c7a40d80e8575
sgl archive/nvidia_cuda_runtime-13.0.96-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 2243632 7f82250d7782aa23b6cfe765ecc7db554bd3c2870c43f3d1821f1d18aebf0548
sgl archive/nvidia_cudnn_cu13-9.20.0.48-py3-none-manylinux_2_27_x86_64.whl 366173588 0c45dd8eeb50b603f07995b1b300c62ffe6a1980482b82b3bcf94a4ca9d49304
sgl archive/nvidia_cudnn_frontend-1.30.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 8797507 5dda72db38ea0c39f158ebffa9596eefab21ce7cc6a12824490ab8b6cb60e76d
sgl archive/nvidia_cufft-12.0.0.61-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 214085489 6c44f692dce8fd5ffd3e3df134b6cdb9c2f72d99cf40b62c32dde45eea9ddad3
sgl archive/nvidia_cufile-1.15.1.6-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 1223672 08a3ecefae5a01c7f5117351c64f17c7c62efa5fffdbe24fc7d298da19cd0b44
sgl archive/nvidia_curand-10.4.0.35-py3-none-manylinux_2_27_x86_64.whl 59544258 1aee33a5da6e1db083fe2b90082def8915f30f3248d5896bcec36a579d941bfc
sgl archive/nvidia_cusolver-12.0.4.66-py3-none-manylinux_2_27_x86_64.whl 200941980 0a759da5dea5c0ea10fd307de75cdeb59e7ea4fcb8add0924859b944babf1112
sgl archive/nvidia_cusparse-12.6.3.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 145942937 2b3c89c88d01ee0e477cb7f82ef60a11a4bcd57b6b87c33f789350b59759360b
sgl archive/nvidia_cusparselt_cu13-0.8.1-py3-none-manylinux2014_x86_64.whl 170148586 786ce87568c303fadb5afcc7102d454cd3040d75f6f8626f5db460d1871f4dd0
sgl archive/nvidia_cutlass_dsl-4.6.2-py3-none-any.whl 10460 06ac62deb182a852dfb053032cf945bf02b9aa1bf502a18b129d280d7babb26c
sgl archive/nvidia_cutlass_dsl_libs_base-4.6.2-cp312-cp312-manylinux_2_28_x86_64.whl 2824987 24dfaddad6077fd0de14eecaf66a72762f2f5f30ae1872231f5bf8b243ac2eac
sgl archive/nvidia_cutlass_dsl_libs_core-4.6.2-py3-none-any.whl 772393 571d4b46fca1bfbb123d0dc348eaa7fd80af0014ddffc07b849e8195b2364333
sgl archive/nvidia_cutlass_dsl_libs_cu12-4.6.2-cp312-cp312-manylinux_2_28_x86_64.whl 88454261 7f510307369d522da8e7d666557b9b9e7df06b94748bd5dc0198d59dfd0d918a
sgl archive/nvidia_cutlass_dsl_libs_cu13-4.6.2-cp312-cp312-manylinux_2_28_x86_64.whl 88036030 a299a623fad75eb752c1a4e447b924e9049ed396b037509c41a48da82d1f340a
sgl archive/nvidia_mathdx-25.6.0-py3-none-any.whl 23013087 22e6ad5d0d005f836be5cbd14e836cf2e9ea42c82deb602707246ce8198eaa96
sgl archive/nvidia_ml_py-13.610.43-py3-none-any.whl 53163 f13c72698edef492f985cc225f14faafe68ae065a2e407f45bdf6f4b9b43fde8
sgl archive/nvidia_nccl_cu13-2.29.7-py3-none-manylinux_2_18_x86_64.whl 205976000 edd81538446786ec3b73972543e53bb43bcaf0bfc8ef76cb679fcc390ffe136d
sgl archive/nvidia_nvjitlink-13.4.92-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 42452378 e0391f24ed94ec879b84e3da4d4ec320c879aff681f2c7a638462f7199284323
sgl archive/nvidia_nvshmem_cu13-3.4.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 60412546 290f0a2ee94c9f3687a02502f3b9299a9f9fe826e6d0287ee18482e78d495b80
sgl archive/nvidia_nvtx-13.0.85-py3-none-manylinux1_x86_64.manylinux_2_5_x86_64.whl 148047 4936d1d6780fbe68db454f5e72a42ff64d1fd6397df9f363ae786930fd5c1cd4
sgl archive/nvidia_nvvm-13.4.92-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl 72604627 e4c81cb321dd9743bcf871c5e7e8fecd7f05dea607944426e102bdda8d49e137
sgl archive/nvshmem4py_cu13-0.4.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_34_x86_64.whl 4988644 3e6a4f6d50304c0dcc5a0334753b09290d3770f0566aee012ca329691c219946
sgl archive/openai-2.6.1-py3-none-any.whl 1005551 904e4b5254a8416746a2f05649594fa41b19d799843cd134dac86167e094edef
sgl archive/openai_harmony-0.0.4-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 3049306 31e9bcac0902a309e2fc688e52f247eec7fffcd00d17e958b9a83a8fea6519c2
sgl archive/orjson-3.12.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 131245 1192a7021b6d071aaf909864f6e924d6a2675ca360485b972b8401749311750b
sgl archive/outlines-0.1.11-py3-none-any.whl 87623 f5a5f2242ed9802d3aab7a92789bf4008d734c576be9258cc0a297f690124727
sgl archive/outlines_core-0.1.26-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 343201 e86a1bb46adc5cbf6dfd7a7fe4105e0e2a4c6e041732a053126b41c521a1f223
sgl archive/packaging-26.3-py3-none-any.whl 129956 d7193f7c8e4e93f444fde0262bf90af30e16fa0ad0ad44cb553c87339b23cd1c
sgl archive/pandas-3.0.6-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl 10788193 0704044b676496b8350e023b09f174a26772456c974a2b11c36bebb558c9490d
sgl archive/parso-0.8.7-py2.py3-none-any.whl 107025 a8926eb2a1b915486941fdbd31e86a4baf88fe8c210f25f2f35ecec5b574ca1c
sgl archive/partial_json_parser-0.2.1.1.post7-py3-none-any.whl 10877 145119e5eabcf80cbb13844a6b50a85c68bf99d376f8ed771e2a3c3b03e653ae
sgl archive/pexpect-4.9.0-py2.py3-none-any.whl 63772 7236d1e080e4936be2dc3e326cec0af72acf9212a7e1d060210e70a47e253523
sgl archive/pillow-12.3.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 6940830 78cb2c6865a35ab8ff8b75fd122f6033b92a62c82801110e48ddd6c936a45d91
sgl archive/prometheus_client-0.26.0-py3-none-any.whl 64494 fa93d06737aa02bacd05794768508bb97d2fbee28cb3bca04eaae92f0ca953d6
sgl archive/prompt_toolkit-3.0.53-py3-none-any.whl 392288 01c0891d7f9237d5e339f7d3e42cdae80b7534abb1c7c0e3352efba6231492f2
sgl archive/propcache-0.5.4-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 250424 2814ecd8e818f487bee4b0f921bc4d1c176cc5fc71ac0f072d0fa67eda4ac14b
sgl archive/protobuf-6.33.6-cp39-abi3-manylinux2014_x86_64.whl 323436 e9db7e292e0ab79dd108d7f1a94fe31601ce1ee3f7b79e0692043423020b0593
sgl archive/psutil-7.2.2-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl 155560 076a2d2f923fd4821644f5ba89f059523da90dc9014e85f8e45a5774ca5bc6f9
sgl archive/ptyprocess-0.7.0-py2.py3-none-any.whl 13993 4b41f3967fce3af57cc7e94b888626c18bf37a083e3651ca8feeb66d492fef35
sgl archive/pure_eval-0.2.4-py3-none-any.whl 11893 96cae060a313cfaad51bb761278bfb0e62dc0248d9315a81173752dc546cd37a
sgl archive/py_spy-0.4.2-py2.py3-none-manylinux_2_5_x86_64.manylinux1_x86_64.whl 2936518 aeb0323409199c785f730645e9f4bb7a7b9ca2c481f2c331a55642b5d13fa52f
sgl archive/pyarrow-25.0.1-cp312-cp312-manylinux_2_28_x86_64.whl 50102437 5389cdf79447ed1515c9e31620e6e1e2302249564d603f2ad727d4f6d313e4c3
sgl archive/pybase64-1.5.0-cp312-cp312-manylinux1_x86_64.manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_5_x86_64.whl 94681 1a2b9cf39b4d30f600df8c56cccbc03adfc6e1ae8c04cd6b181105a432d4a515
sgl archive/pycountry-26.2.16-py3-none-any.whl 8044600 115c4baf7cceaa30f59a4694d79483c9167dbce7a9de4d3d571c5f3ea77c305a
sgl archive/pycparser-3.0-py3-none-any.whl 48172 b727414169a36b7d524c1c3e31839a521725078d7b2ff038656844266160a992
sgl archive/pycryptodomex-3.23.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 2272578 f489c4765093fb60e2edafdf223397bc716491b2b69fe74367b70d6999257a5c
sgl archive/pydantic-2.14.0b2-py3-none-any.whl 484647 4556c932fb3dbb7e9ac773bee241b2eb7058a2f9ff1d850d45d7d11cb6e54b68
sgl archive/pydantic_core-2.49.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 2127340 5a807d09a516f93614bbc8e83401afdbe7325373d36ec54727f3f5147a57dd52
sgl archive/pydantic_extra_types-2.11.1-py3-none-any.whl 79526 1722ea2bddae5628ace25f2aa685b69978ef533123e5638cfbddb999e0100ec1
sgl archive/pygments-2.21.0-py3-none-any.whl 1250147 2363c69b61c4a97c838da3b130dcd6468f4848992b21a82f2a63ec34377137d9
sgl archive/python_dateutil-2.9.0.post0-py2.py3-none-any.whl 229892 a8b2bc7bffae282281c8140a97d3aa9c14da0b136dfe83f850eea9a5f7470427
sgl archive/python_multipart-0.0.32-py3-none-any.whl 30042 ff6d3f776f16878c894e52e107296ffc890e913c611b1a4ec6c44e2821fe2e23
sgl archive/pyyaml-6.0.3-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 807870 ba1cc08a7ccde2d2ec775841541641e4548226580ab850948cbfda66a1befcdc
sgl archive/pyzmq-27.2.0-cp312-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl 872258 dea74fd65f1fc5f7fe167916a473ebe6ed6174e5e5d9de11ea6583661be6cf43
sgl archive/quack_kernels-0.6.4-py3-none-any.whl 728458 e77c5d1f1299b0b38487fe8737df6c6975daa16bca7f7eb883bd1a74d09e7e78
sgl archive/referencing-0.37.0-py3-none-any.whl 26766 381329a9f99628c9069361716891d34ad94af76e461dcb0335825aecc7692231
sgl archive/regex-2026.9.10-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 804587 2e67f8843f0e4b931f1fa860bf3bbe4134b714c0155cc5c7c0d7ea450230aae0
sgl archive/requests-2.34.2-py3-none-any.whl 73075 2a0d60c172f83ac6ab31e4554906c0f3b3588d37b5cb939b1c061f4907e278e0
sgl archive/rich-15.0.0-py3-none-any.whl 310654 33bd4ef74232fb73fe9279a257718407f169c09b78a87ad3d296f548e27de0bb
sgl archive/rpds_py-2026.6.3-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 366189 ecabd69db66de867690f9797f2f8fa27ba501bbc24540cbdbdc649cd15888ba6
sgl archive/safetensors-0.9.0rc0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 575268 ad23b22c5edecc312ae89abc7de9c4e330162610faa6ee905b9c6dc7df2735b9
sgl archive/scipy-1.18.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 35344199 f55fa87b6c612ecd6b058f167c53231b1d14e412efe361d3d6e38b3631c73218
sgl archive/sentencepiece-0.2.2-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 1397736 c8a168b040bc61681293f79a949b5d911c8e25086f4260285b8d97ab5f1195da
sgl archive/setproctitle-1.3.7-cp312-cp312-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl 32932 2906b6c7959cdb75f46159bf0acd8cc9906cf1361c9e1ded0d065fe8f9039629
sgl archive/setuptools-84.0.0-py3-none-any.whl 818216 51a52592b3b99e102b609654876bd65f19f999935166d1352678931132b0c670
sgl archive/sgl_deep_ep-0.1.2+cu130-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 15332339 9c15cb83b56bcc1ec1a838f47a06e9e067c69e0f0b16498d0e7451a548c2047a
sgl archive/sgl_deep_gemm-0.2.0+cu130-py3-none-manylinux2014_x86_64.whl 5419719 acdb102d1e0ba82b64af343412214d92e0f192f4ddc07101e8763c0c1e638ac0
sgl archive/sglang-0.5.20-cp312-cp312-manylinux_2_34_x86_64.whl 27393093 ffaced7e91c3536c63077b16b08210b5c1a2f362418e9841a8767d5e019cf08a
sgl archive/sglang_kernel-0.4.7+cu130-cp310-abi3-manylinux2014_x86_64.whl 373107159 d312128f33e95dec2f7e9f6b7f4295d092e430cb9ef3e07a3a970e0741c7023e
sgl archive/shellingham-1.5.4-py2.py3-none-any.whl 9755 7ecfff8f2fd72616f7481040475a65b2bf8af90a56c89140852d1120324e8686
sgl archive/six-1.17.0-py2.py3-none-any.whl 11050 4721f391ed90541fddacab5acf947aa0d3dc7d27b2e1e8eda2be8970586c3274
sgl archive/sniffio-1.3.1-py3-none-any.whl 10235 2f6da418d1f1e0fddd844478f41680e794e6051915791a034ff65e5f100525a2
sgl archive/soundfile-0.13.1-py2.py3-none-manylinux_2_28_x86_64.whl 1313646 03267c4e493315294834a0870f31dbb3b28a95561b80b134f0bd3cf2d5f0e618
sgl archive/stack_data-0.6.3-py3-none-any.whl 24521 d5558e0c25a4cb0853cddad3d77da9891a08cb85dd9f9f91b9f8cd66e511e695
sgl archive/starlette-1.6.0-py3-none-any.whl 75969 a86dd39d14bb45f85a3d18525215a9ef0cfd1f192ac793220e72598c90335f0c
sgl archive/sympy-1.14.0-py3-none-any.whl 6299353 e091cc3e99d2141a0ba2847328f5479b05d94a6635cb96148ccb3f34671bd8f5
sgl archive/tabulate-0.10.0-py3-none-any.whl 39814 f0b0622e567335c8fabaaa659f1b33bcb6ddfe2e496071b743aa113f8774f2d3
sgl archive/tiktoken-0.14.0-cp312-cp312-manylinux_2_28_x86_64.whl 1204197 7896eea257fe497a2b7134474d909156c6744ce8da35bce88011a960e008aa0d
sgl archive/tilelang-0.1.12-cp38-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 50501074 bbeb5573cbe2544a51a5c9a6cc737c5e3b5c2d95411caa3687fd9b08eb9d5f97
sgl archive/timm-1.0.16-py3-none-any.whl 2485733 a640e58f4ae41e0445517d1133b34be75bb2bd49cdb830d739925ce1fb7d2526
sgl archive/tokenizers-0.22.2-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 3274982 369cc9fc8cc10cb24143873a0d95438bb8ee257bb80c71989e3ee290e8d72c67
sgl archive/tokenspeed_mla-0.1.8-py3-none-manylinux_2_28_x86_64.whl 755827 6a7526d7327746893f8c20d24aa63ba5b8a123d0dfd6e66388e13b768b6452c6
sgl archive/tokenspeed_triton-3.8.10.post20260920-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 99900581 eb5eae6c4d3db11e13b7058c89b57e075fa7df905d43169cfdb066e161886421
sgl archive/tomlkit-0.15.1-py3-none-any.whl 49449 177a05aece5a8ca5266fd3c448abb47b8d352f09d477d3ca8332db4d89b24304
sgl archive/torch-2.13.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl 526482318 8db7338e6895c3d4bd89a02ff4209507d1f0cf2ffeb3b898538b5a07d1ea8c1e
sgl archive/torch_c_dlpack_ext-0.1.5-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl 897846 e6f9da4bb9af70e27facc777458be62e10dbbbddda7672d16138db0553c5a524
sgl archive/torch_memory_saver-0.0.10-cp39-abi3-manylinux2014_x86_64.whl 2600103 786e18755a9af255ae692b0e19a62ff06861dbf55a19fa8545e62a100ab5a28f
sgl archive/torchaudio-2.11.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl 1728401 3fba988f4301fe13547fe5e99c76d9ae36a27e19ded82eeffed9d2456e12edef
sgl archive/torchcodec-0.15.0+cu130-cp312-cp312-manylinux_2_28_x86_64.whl 2962510 54e9c069270f7cbe84123d53b87678acd834a62f14acaa45bf8558cd21669bb0
tv wheels/torchvision-0.28.0-cp312-cp312-manylinux_2_28_x86_64.whl 7675040 028a3d481b37d785605620d7cdad897064c5a55bae2aa1f2658766333e291940
sgl archive/tqdm-4.70.1-py3-none-any.whl 80199 c293e525e6fef9c20e8728fd4612df02a0aa31bb5fe91ecd93e123b1b7bffa73
sgl archive/traitlets-5.16.1-py3-none-any.whl 86211 f775618166caa0396c8e337099240f2bd3e5e917d203b2e6fbe21a58d3cb1f6b
sgl archive/transformers-5.12.1-py3-none-any.whl 11150587 2a5e109d2021265df7098ffbb738295acaf5ad256f12cbc586db2ea4dcbb1a8a
sgl archive/triton-3.7.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 197149381 225910e79149807de74a0ca63160aa285956a8d64a2951d558997c57e2584ae1
sgl archive/truststore-0.10.4-py3-none-any.whl 18660 adaeaecf1cbb5f4de3b1959b42d41f6fab57b2b1666adb59e89cb0b53361d981
sgl archive/typer-0.27.2-py3-none-any.whl 123130 b3a5fc4342d5fc8fda8fc3010b1cf117e9249aab7fae800c2eff62fd3842d97d
sgl archive/typing_extensions-4.16.0-py3-none-any.whl 45571 481caa481374e813c1b176ada14e97f1f67a4539ce9cfeb3f350d78d6370c2e8
sgl archive/typing_inspection-0.4.4-py3-none-any.whl 14750 65b8397ba37ccbce054456aaccddfc91e6e3083c92824df348d96ca832f3f147
sgl archive/urllib3-2.8.0-py3-none-any.whl 135717 0cf3cae568d36aa9576b28dfb35f11328f1cb974ca7647d9475ebb86c75ac6e3
sgl archive/uvicorn-0.53.0-py3-none-any.whl 87081 e8dca71ec86dce5f04e333f0d56cdedf942446e6643b9cea1af0d6d3a02cb03e
sgl archive/uvloop-0.22.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 4426307 7b5b1ac819a3f946d3b2ee07f09149578ae76066d70b44df3fa990add49a82e4
sgl archive/watchfiles-1.3.0-cp310-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 458353 b5768b49e426fd5b550b012c866db347cdf15c398ef98dc557b6e6b72fa74cd1
sgl archive/wcwidth-0.8.4-py3-none-any.whl 299474 2097bb1d28a0ba8fe177c2eb607317f9b1627b03f33ebebfd3da670b337a65ba
sgl archive/xgrammar-0.2.1-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl 44678489 cbc6014dc1c92fc317b14519121c8163fe35fd934179e5a45d83f780ff231826
sgl archive/xxhash-4.0.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 261812 237b8f63a2a0fcfb1ffc06e21dad23add44e6d354b2b014364a1d41e419a4dee
sgl archive/yarl-1.25.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl 117050 c6f117789d22dce188e5754e8bc65b7e6ebf8cb73963b9fa761f672a5883769d
sgl archive/z3_solver-4.15.4.0-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl 29268352 7e103a6f203f505b8b8b8e5c931cc407c95b61556512d4921c1ddc0b3f41b08e
sgl archive/zstandard-0.25.0-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.whl 5546993 5a56ba0db2d244117ed744dfa8f6f5b366e14148e00de44723413b2f3938a902
"""

# The analyzer environment exactly as Keith's serving_setup.persist_analyzer_environment writes it (checked against the
# vendored file by tests/test_taaf_ours_patch.py); the notebook asserts the model id, yield 60, temperature and upscale.
ANALYZER_ENV = {
    "LOCAL_ANALYZER_BASE_URL": BASE_URL,
    "OPENAI_BASE_URL": BASE_URL,
    "LOCAL_ANALYZER_PROVIDER": "vllm",  # keeps top_k, seed and chat_template_kwargs in the payload; SGLang accepts all
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED_MODEL_NAME,
    "INFERENCE_ANALYZER_MODEL": SERVED_MODEL_NAME,
    "OPENAI_API_KEY": "offline-kaggle-local-server",
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": str(ANALYZER_CONTEXT),
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
}
# Keith's runtime keys that are not tied to his vLLM image (PYTHONPATH, PATH, CUDA_HOME, ... are, and stay unset).
OFFLINE_ENV = {"HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
               "PYTHONDONTWRITEBYTECODE": "1"}

DEFAULT_PROFILE = {
    "running": 12,          # R: --max-running-requests (plan section 4: R=12 x 32k = 393k tokens fits the FP8 pool)
    "mamba_cache": None,    # --max-mamba-cache-size; None = 6R (gabriel fork: below ~6R the spec graphs cap at bs 4)
    "hicache_gb": 0,        # > 0: hierarchical host cache of that many GB (the r16-hic32 stress profile)
    "context_length": ANALYZER_CONTEXT,
    "mem_fraction": 0.94,   # plan: 0.94 (Son 0.95; 0.965 OOMed on the first request with the fork's fp8 copies)
    "deadline_min": 50.0,   # minutes after notebook start: no rung up by then -> Keith's vLLM
    "rung_timeouts_min": [30, 25, 20, 20],
}
# The ladder (plan section 4 step 6). Rungs 1-3 keep the FP8 KV cache; rung 4 is the conservative last try.
RUNGS = (
    ("fp8kv-mtp-flashinfer", {}),
    ("fp8kv-mtp-triton", {"linear_attn": "triton"}),  # the fork's own fallback when its WY/MTP patch is missing
    ("fp8kv-nomtp-triton", {"linear_attn": "triton", "mtp": False, "mamba_factor": 3}),
    ("bf16kv-nomtp-r8", {"linear_attn": "triton", "mtp": False, "mamba_factor": 3, "kv": "auto", "running": 8,
                         "mem_fraction": 0.90, "hicache": False}),
)

VENV = Path("/tmp/sgl-venv")
STAGE = Path("/tmp/sgl-wheels")
CUDA_ROOT = Path("/tmp/sgl-cuda")
CACHE_ROOT = Path("/tmp/sgl-cache")
PRIVATE = Path("/tmp/sgl-state")  # server env (may hold Kaggle tokens): never under /kaggle/working
STOP_FILE = PRIVATE / "watchdog.stop"
BUNDLE_CONFIG = "ours_sglang.json"
SETUP_COMMAND = '"$PYTHON" "$TAAF_KAGGLE_BUNDLE_DIR/sglang_serving.py" setup'
TEARDOWN_COMMAND = '"$PYTHON" "$TAAF_KAGGLE_BUNDLE_DIR/sglang_serving.py" teardown'
WATCHDOG_INTERVAL_S = 30.0
WATCHDOG_MISSES = 3
WATCHDOG_MAX_RESTARTS = 2


def log(msg: str) -> None:
    print(f"ours.sglang: {msg}", flush=True)


def working_dir() -> Path:
    return Path(os.environ.get("TAAF_KAGGLE_WORKING_DIR") or "/kaggle/working")


def setup_env_path() -> Path:
    return Path(os.environ.get("TAAF_KAGGLE_SETUP_ENV") or working_dir() / "taaf_setup_env.json")


def state_path() -> Path:
    return working_dir() / "sglang-serving-state.json"


def input_root() -> Path:
    return Path(os.environ.get("OURS_SGL_INPUT_ROOT") or "/kaggle/input")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# --- the wheel manifest ---------------------------------------------------------------------------------------------


def wheels() -> list[dict]:
    datasets = {"sgl": WHEEL_DATASET, "tv": TORCHVISION_DATASET}
    rows = []
    for line in WHEELS.strip().splitlines():
        key, rel, size, sha = line.split()
        name = rel.rsplit("/", 1)[-1]
        rows.append({"dataset": datasets[key], "path": rel, "wheel": WHEEL_RENAMES.get(name, name),
                     "size": int(size), "sha256": sha})
    return rows


def dataset_dir(ref: str) -> Path:
    """Where Kaggle mounted a dataset: the notebook's own mapping, the two usual layouts, then a bounded search."""
    marker = DATASET_MARKERS[ref]
    owner, slug = ref.split("/", 1)
    mapped = (read_json_env("TAAF_KAGGLE_INPUT_PATHS") or {}).get(ref)
    root = input_root()
    for candidate in (mapped, root / slug, root / "datasets" / owner / slug):
        if candidate and (Path(candidate) / marker).is_file():
            return Path(candidate)
    depth = len(Path(marker).parts)
    for pattern in ("*/", "*/*/", "*/*/*/"):
        for hit in sorted(root.glob(pattern + marker)):
            return hit.parents[depth - 1]
    raise FileNotFoundError(f"dataset {ref} not mounted under {root} (looked for {marker})")


def read_json_env(name: str) -> Any:
    raw = os.environ.get(name, "").strip()
    try:
        return json.loads(raw) if raw else None
    except ValueError:
        return None


def verify_wheels(rows: list[dict], dirs: dict[str, Path], threads: int = 8) -> list[tuple[Path, dict]]:
    def check(row: dict) -> tuple[Path, dict]:
        path = dirs[row["dataset"]] / row["path"]
        if not path.is_file() or path.stat().st_size != row["size"]:
            raise RuntimeError(f"missing or wrong-sized wheel: {path}")
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"wheel hash mismatch: {path} {actual} != {row['sha256']}")
        return path, row

    with concurrent.futures.ThreadPoolExecutor(threads) as pool:
        return list(pool.map(check, rows))


def stage_wheels(verified: list[tuple[Path, dict]], stage: Path = STAGE) -> list[Path]:
    """Symlinks under the valid wheel names (the dataset is read-only; one file name is not a valid wheel name)."""
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    out = []
    for path, row in verified:
        link = stage / row["wheel"]
        link.symlink_to(path)
        out.append(link)
    return out


def install(staged: list[Path], venv: Path, timeout_s: float) -> dict:
    """A clean venv, then pip --no-index --no-deps on the explicit list; returns what happened (pip check included)."""
    if venv.exists():
        shutil.rmtree(venv)
    made = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True, text=True, check=False)
    if made.returncode != 0:  # no ensurepip on this image: a bare venv, installed into by this interpreter's pip
        log(f"venv with pip failed ({made.stderr.strip()[-200:]}); retrying --without-pip")
        shutil.rmtree(venv, ignore_errors=True)
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
    py = venv / "bin" / "python"
    has_pip = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True, check=False).returncode == 0
    pip = [str(py), "-m", "pip"] if has_pip else [sys.executable, "-m", "pip", "--python", str(py)]
    started = time.monotonic()
    subprocess.run([*pip, "install", "--no-index", "--no-deps", "--no-compile", "--no-cache-dir",
                    "--disable-pip-version-check", "--no-warn-script-location", *map(str, staged)],
                   check=True, timeout=max(60.0, timeout_s))
    check = subprocess.run([*pip, "check"], capture_output=True, text=True, check=False, timeout=300)
    return {"venv": str(venv), "pip": "venv" if has_pip else "host --python", "wheels": len(staged),
            "install_s": round(time.monotonic() - started, 1), "pip_check_rc": check.returncode,
            "pip_check": (check.stdout + check.stderr).strip().splitlines()[-40:]}


def extract_cuda_toolkit(runtime_dir: Path | None, root: Path = CUDA_ROOT) -> Path | None:
    """usr/local/cuda-13.0 from layer 8 of Keith's runtime (a complete toolkit); None if it cannot be had."""
    home = root / "usr" / "local" / "cuda-13.0"
    if (home / "bin" / "nvcc").is_file():
        return home
    if runtime_dir is None:
        return None
    blob = runtime_dir / CUDA_LAYER
    if not blob.is_file() or blob.stat().st_size != CUDA_LAYER_BYTES:
        log(f"CUDA layer missing or wrong-sized: {blob}")
        return None
    root.mkdir(parents=True, exist_ok=True)
    done = subprocess.run(["tar", "-xzf", str(blob), "-C", str(root), "--wildcards", "usr/local/cuda-13.0/*",
                           "--exclude=*/.wh.*"], capture_output=True, text=True, check=False, timeout=1200)
    if done.returncode != 0 or not (home / "bin" / "nvcc").is_file():
        log(f"CUDA toolkit extraction failed rc={done.returncode}: {done.stderr.strip()[-300:]}")
        return None
    if not (home / "lib64").exists():
        (home / "lib64").symlink_to("targets/x86_64-linux/lib")
    return home


def venv_site(venv: Path) -> Path:
    return venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"


def pip_cuda_home(venv: Path) -> Path | None:
    home = venv_site(venv) / "nvidia" / "cu13"
    return home if (home / "bin" / "nvcc").is_file() else None


def server_env(venv: Path, cuda_home: Path | None, base: dict | None = None) -> dict:
    """The server's environment: the venv's own libraries first, no PYTHONPATH from the notebook, offline, caches in
    /tmp/sgl-cache (plan section 4 step 4), and CUDA_HOME on the 13.0 toolkit (never Kaggle's own CUDA 12)."""
    env = dict(os.environ if base is None else base)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CUDA_HOME", "CUDA_PATH", "CUDACXX"):
        env.pop(key, None)
    site = venv_site(venv)
    libs = [str(p) for p in sorted(site.glob("nvidia/*/lib")) if p.is_dir()] + [str(site / "torch" / "lib")]
    inherited = [e for e in env.get("LD_LIBRARY_PATH", "").split(os.pathsep) if e and not e.startswith("/usr/local/cuda")]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys([*libs, "/usr/local/nvidia/lib64", *inherited]))
    env["PATH"] = os.pathsep.join([*([str(cuda_home / "bin")] if cuda_home else []), str(venv / "bin"),
                                   env.get("PATH", "/usr/bin:/bin")])
    if cuda_home is not None:
        env.update(CUDA_HOME=str(cuda_home), CUDA_PATH=str(cuda_home), CUDACXX=str(cuda_home / "bin" / "nvcc"))
    caches = {"XDG_CACHE_HOME": "xdg", "SGLANG_CACHE_DIR": "sglang", "TRITON_CACHE_DIR": "triton",
              "TORCHINDUCTOR_CACHE_DIR": "inductor", "FLASHINFER_WORKSPACE_BASE": "flashinfer",
              "CUDA_CACHE_PATH": "cuda", "HF_HOME": "hf", "TVM_FFI_CACHE_DIR": "tvm-ffi"}
    for key, sub in caches.items():
        env[key] = str(CACHE_ROOT / sub)
    env.update(OFFLINE_ENV)
    env.update({
        "PYTHONNOUSERSITE": "1", "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # plan section 4 step 4 (Son's GCP arms)
        "MAX_JOBS": "8", "OMP_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false",  # plan section 4 step 4
        "TORCH_CUDA_ARCH_LIST": "12.0",  # sm120, as Keith's runtime sets it for JIT builds
        "DO_NOT_TRACK": "1",
    })
    return env


# --- the server command -------------------------------------------------------------------------------------------


def resolve_profile(profile: dict | None) -> dict:
    p = {**DEFAULT_PROFILE, **(profile or {})}
    p["running"] = int(p["running"])
    p["mamba_cache"] = int(p["mamba_cache"] or 6 * p["running"])
    p["hicache_gb"] = int(p["hicache_gb"] or 0)
    return p


def rung_settings(profile: dict | None, rung: int) -> dict:
    p = resolve_profile(profile)
    name, o = RUNGS[rung]
    running = int(o.get("running", p["running"]))
    return {"name": name, "rung": rung + 1, "running": running,
            "mamba_cache": running * o["mamba_factor"] if "mamba_factor" in o else p["mamba_cache"],
            "kv": o.get("kv", "fp8_e4m3"), "linear_attn": o.get("linear_attn", "flashinfer"), "mtp": o.get("mtp", True),
            "mem_fraction": float(o.get("mem_fraction", p["mem_fraction"])), "context_length": int(p["context_length"]),
            "hicache_gb": p["hicache_gb"] if o.get("hicache", True) else 0}


def server_args(model_dir: Path, profile: dict | None = None, rung: int = 0) -> list[str]:
    """``sglang.launch_server`` arguments. Sources: plan section 4 step 5 (rung 1), section 5 (the r16-hic32 profile),
    Son's ``startup.sh`` for the values the plan took from it; every flag and choice is checked against SGLang 0.5.20's
    own CLI parser by tests/test_taaf_ours_patch.py (fixture tests/fixtures/sglang_0520_cli.json)."""
    s = rung_settings(profile, rung)
    model_dir = Path(model_dir)
    args = [
        "--model-path", str(model_dir),
        "--served-model-name", SERVED_MODEL_NAME,
        "--host", HOST, "--port", str(PORT),
        "--tp-size", "1",  # plan "--tp 1": 0.5.20 spells it --tp-size / --tensor-parallel-size
        "--dtype", "bfloat16",
        "--quantization", "modelopt_fp4",  # RadixArk's ModelOpt NVFP4 (Keith's vLLM uses the same method)
        "--kv-cache-dtype", s["kv"],  # FP8 KV (plan); rung 4: auto = BF16
        "--page-size", "64",  # QSA forces 64 anyway (plan section 2)
        "--mem-fraction-static", f"{s['mem_fraction']:.2f}",
        "--context-length", str(s["context_length"]),  # the harness window
        "--max-running-requests", str(s["running"]),
        # plan "--cuda-graph-max-bs R": 0.5.20 has only -decode/-prefill, and the bare prefix is ambiguous there
        "--cuda-graph-max-bs-decode", str(s["running"]),
        "--chunked-prefill-size", "4096",
        "--mamba-radix-cache-strategy", "extra_buffer",  # radix prefix caching for the hybrid (linear-attention) model
        "--mamba-ssm-dtype", "bfloat16",
        "--mamba-track-interval", "64",
        "--max-mamba-cache-size", str(s["mamba_cache"]),  # ~6R with MTP, 3R without (plan)
        "--linear-attn-decode-backend", s["linear_attn"],
        "--linear-attn-prefill-backend", s["linear_attn"],
        "--ple-offload-embedding",  # the FP8 PLE table in pinned host RAM (47.7 GiB); 0.5.20's default, made explicit
        "--trust-remote-code",
        "--chat-template", str(model_dir / "chat_template.jinja"),  # the model's own template (plan)
        "--reasoning-parser", "qwen3",
        "--tool-call-parser", "qwen3_coder",
        "--enable-multimodal",  # vision tower on (0.5.20 turns it on for this architecture anyway)
        "--enable-metrics",
        "--enable-cache-report",
        "--watchdog-timeout", "1800",
    ]
    if s["mtp"]:  # native MTP head, 3 steps / 4 draft tokens; acceptance thresholds 1.0/1.0 (lossless) are the defaults
        args += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
                 "--speculative-num-draft-tokens", "4"]
    if s["hicache_gb"] > 0:  # the r16-hic32 profile (plan section 5)
        args += ["--enable-hierarchical-cache", "--hicache-size", str(s["hicache_gb"]),
                 "--hicache-write-policy", "write_through", "--hicache-io-backend", "kernel",
                 "--hicache-mem-layout", "page_first"]
    # Radix cache: on (the default; no --disable-radix-cache). No --default-chat-template-kwargs: the template's
    # defaults apply, as on vLLM, and P11 passes knobs per request. No speculative token map (FR-Spec) yet.
    return args


def server_argv(venv: Path, model_dir: Path, profile: dict | None, rung: int) -> list[str]:
    return [str(venv / "bin" / "python"), "-m", "sglang.launch_server", *server_args(model_dir, profile, rung)]


def resolve_model_dir() -> Path:
    candidates = [MODEL_KAGGLE_PATH]
    models = input_root() / "models"
    if models.is_dir():
        candidates += sorted(p.parent for p in models.glob("*/*/*/radixark-modelopt-fp4/*/config.json"))
    for candidate in dict.fromkeys(candidates):
        config = candidate / "config.json"
        if config.is_file() and sha256_file(config) == MODEL_CONFIG_SHA256:
            template = candidate / "chat_template.jinja"
            if not template.is_file() or sha256_file(template) != CHAT_TEMPLATE_SHA256:
                raise RuntimeError(f"unexpected chat template at {template}")
            return candidate
    raise FileNotFoundError(f"Kaggle model {MODEL_SOURCE} not mounted (looked in {[str(c) for c in candidates]})")


# --- HTTP ------------------------------------------------------------------------------------------------------------


def http_json(url: str, payload: dict | None = None, timeout: float = 60.0) -> tuple[int, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                     method="GET" if payload is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", "replace")
            status = response.status
    except urllib.error.HTTPError as exc:
        body, status = exc.read().decode("utf-8", "replace"), exc.code
    try:
        return status, json.loads(body)
    except ValueError:
        return status, body


def health_ok(timeout: float = 10.0) -> bool:
    try:
        status, _ = http_json(BASE_URL.rsplit("/v1", 1)[0] + "/health", timeout=timeout)
        return status == 200
    except (OSError, ValueError):
        return False


def board_png(seed: int = 7, cells: int = 64, scale: int = 4) -> bytes:
    """A 256x256 board image like the Duck's (64x64 cells of 16 colours, upscale 4), encoded with the stdlib only."""
    palette = [((v * 53) % 256, (v * 97) % 256, (v * 151) % 256) for v in range(16)]
    state = seed
    grid = []
    for _ in range(cells):
        row = []
        for _ in range(cells):
            state = (state * 1103515245 + 12345) % (1 << 31)
            row.append(palette[(state >> 16) % 16])
        grid.append(row)
    raw = bytearray()
    for r in range(cells * scale):
        raw.append(0)
        for c in range(cells * scale):
            raw.extend(grid[r // scale][c // scale])
    size = cells * scale

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b"")


def board_image_part() -> dict:
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(board_png()).decode()}}


def readiness_gate(timeout: float = 600.0) -> dict:
    """The plan's 'up' test after /health: the served id, one text and one board-image completion."""
    out: dict[str, Any] = {}
    status, body = http_json(f"{BASE_URL}/models", timeout=60)
    ids = [row.get("id") for row in (body or {}).get("data", [])] if isinstance(body, dict) else []
    if status != 200 or ids != [SERVED_MODEL_NAME]:
        raise RuntimeError(f"/v1/models: {status} {ids}")
    out["models"] = ids
    for label, content in (("text", "Reply with one word: ready."),
                           ("image", [{"type": "text", "text": "Name two colours in this image, briefly."},
                                      board_image_part()])):
        started = time.monotonic()
        status, body = http_json(f"{BASE_URL}/chat/completions", {
            "model": SERVED_MODEL_NAME, "messages": [{"role": "user", "content": content}], "max_tokens": 64,
            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}, timeout=timeout)
        choices = body.get("choices") if isinstance(body, dict) else None
        message = (choices or [{}])[0].get("message") or {}
        text = (message.get("content") or "").strip()
        if status != 200 or not text:
            raise RuntimeError(f"{label} completion failed: {status} {str(body)[:300]}")
        out[label] = {"seconds": round(time.monotonic() - started, 1), "reply": text[:80],
                      "usage": (body or {}).get("usage")}
    return out


# --- process control ----------------------------------------------------------------------------------------------


def launch(argv: list[str], env: dict, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(log_path, "ab")
    return subprocess.Popen(argv, env=env, stdout=handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            start_new_session=True, cwd=str(working_dir()))


def group_alive(pgid: int) -> bool:
    """Whether any process of the group is still running (zombies waiting for a parent to reap them do not count)."""
    proc = Path("/proc")
    if not proc.is_dir():
        try:
            os.killpg(pgid, 0)
        except OSError:
            return False
        return True
    for stat in proc.glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()  # state, ppid, pgrp, ...
        except (OSError, IndexError):
            continue
        if len(fields) > 2 and fields[2] == str(pgid) and fields[0] != "Z":
            return True
    return False


def stop_group(pid: int | None, grace_s: float = 20.0, proc: subprocess.Popen | None = None) -> bool:
    """SIGTERM, then SIGKILL, the whole process group of a process started with start_new_session=True (its pid is
    the group id, so SGLang's scheduler and detokenizer children go too, even after the leader died). ``proc`` is
    reaped when it is our own child, so its zombie does not keep the group alive. Returns True when the group is gone."""
    if not pid:
        return True
    pgid = int(pid)
    for sig, wait in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 10.0)):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return True
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if proc is not None:
                proc.poll()
            if not group_alive(pgid):
                return True
            time.sleep(0.5)
    return not group_alive(pgid)


def tail(path: Path, lines: int = 60) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def wait_ready(proc: subprocess.Popen, deadline: float) -> tuple[bool, str, dict]:
    while time.time() < deadline:
        if proc.poll() is not None:
            return False, f"server exited rc={proc.returncode}", {}
        if health_ok():
            try:
                return True, "ok", readiness_gate(timeout=max(60.0, min(600.0, deadline - time.time())))
            except (OSError, ValueError, RuntimeError) as exc:
                return False, f"readiness gate failed: {exc}", {}
        time.sleep(5)
    return False, "timeout", {}


# --- persisted environment and fallback ---------------------------------------------------------------------------


def persist(keys: dict) -> dict:
    path = setup_env_path()
    current = read_json(path, {}) if path.exists() else {}
    if not isinstance(current, dict):
        raise RuntimeError(f"{path} is not a JSON object")
    current.update({k: str(v) for k, v in keys.items()})
    write_json(path, current)
    return current


def bundle_config(bundle_dir: Path) -> dict:
    cfg = read_json(Path(bundle_dir) / BUNDLE_CONFIG)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"no {BUNDLE_CONFIG} in {bundle_dir}: the bundle was not prepared by prepare_bundle()")
    return cfg


def run_vllm_fallback(bundle_dir: Path, reason: str) -> None:
    """Keith's original setup commands, from this bundle copy (his files in it are unchanged)."""
    log(f"falling back to Keith's vLLM: {reason}")
    env = dict(os.environ)
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(bundle_dir)
    for command in bundle_config(bundle_dir)["his_setup_commands"]:
        log(f"vLLM fallback setup command: {command}")
        subprocess.run(command, shell=True, check=True, cwd=str(working_dir()), env=env)  # his command, as the notebook runs it
    persist({"OURS_SERVING": "vllm-fallback", "OURS_SGLANG_FALLBACK_REASON": reason[:300]})


def persist_sglang(rung: dict) -> dict:
    return persist({**OFFLINE_ENV, **ANALYZER_ENV, "OURS_SERVING": "sglang", "OURS_SGLANG_RUNG": rung["name"],
                    "OURS_SGLANG_RUNNING": rung["running"]})


# --- setup --------------------------------------------------------------------------------------------------------


def _notebook_start() -> float:
    try:
        return float(os.environ.get("OURS_NOTEBOOK_START_EPOCH") or time.time())
    except ValueError:
        return time.time()


def start_sglang(bundle_dir: Path, profile: dict, deadline: float, report: dict) -> dict | None:
    phases = report.setdefault("phases", {})
    t = time.time()
    dirs = {ref: dataset_dir(ref) for ref in (WHEEL_DATASET, TORCHVISION_DATASET)}
    try:
        runtime_dir: Path | None = dataset_dir(RUNTIME_DATASET)
    except FileNotFoundError:
        runtime_dir = None
    report["datasets"] = {k: str(v) for k, v in {**dirs, RUNTIME_DATASET: runtime_dir}.items()}
    model_dir = resolve_model_dir()
    report["model_dir"] = str(model_dir)
    with concurrent.futures.ThreadPoolExecutor(1) as side:
        cuda_future = side.submit(extract_cuda_toolkit, runtime_dir)  # in parallel with the hash check and pip
        verified = verify_wheels(wheels(), dirs)
        phases["hash_s"] = round(time.time() - t, 1)
        t = time.time()
        report["install"] = install(stage_wheels(verified), VENV, deadline - time.time())
        phases["install_s"] = round(time.time() - t, 1)
        t = time.time()
        cuda_home = cuda_future.result() or pip_cuda_home(VENV)
        phases["cuda_wait_s"] = round(time.time() - t, 1)
    report["cuda_home"] = str(cuda_home) if cuda_home else None
    env = server_env(VENV, cuda_home)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    seed = next((p for p in sorted(input_root().glob("*/sgl-cache")) if p.is_dir()), None)
    if seed is not None and not CACHE_ROOT.exists():  # a previous run's JIT caches, attached as a kernel source
        shutil.copytree(seed, CACHE_ROOT, symlinks=True)
        report["cache_seed"] = str(seed)
    t = time.time()
    probe = subprocess.run([str(VENV / "bin" / "python"), "-c", IMPORT_PROBE], env=env, capture_output=True,
                           text=True, check=False, timeout=900)
    phases["import_probe_s"] = round(time.time() - t, 1)
    line = next((x for x in probe.stdout.splitlines() if x.startswith("SGLANG_IMPORT_PROBE ")), None)
    if probe.returncode != 0 or line is None:
        raise RuntimeError(f"import probe failed rc={probe.returncode}: {(probe.stderr or probe.stdout)[-1500:]}")
    report["versions"] = json.loads(line.split(" ", 1)[1])
    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    report["nvidia_smi"] = smi.stdout[-2000:]
    timeouts = list(profile.get("rung_timeouts_min") or DEFAULT_PROFILE["rung_timeouts_min"])
    for rung in range(len(RUNGS)):
        remaining = deadline - time.time()
        if remaining < 120:
            report.setdefault("attempts", []).append({"rung": rung + 1, "skipped": "deadline"})
            break
        settings = rung_settings(profile, rung)
        argv = server_argv(VENV, model_dir, profile, rung)
        log_path = working_dir() / f"sglang-server-rung{rung + 1}.log"
        log(f"rung {rung + 1} ({settings['name']}): {' '.join(argv)}")
        started = time.time()
        proc = launch(argv, env, log_path)
        try:
            ok, why, gate = wait_ready(proc, min(deadline, started + 60 * float(timeouts[rung])))
        except BaseException:  # never leave a half-started server holding the GPU for the vLLM fallback
            stop_group(proc.pid, proc=proc)
            raise
        attempt = {"rung": rung + 1, "name": settings["name"], "ok": ok, "why": why,
                   "seconds": round(time.time() - started, 1), "log": str(log_path)}
        report.setdefault("attempts", []).append(attempt)
        if ok:
            phases["server_ready_s"] = attempt["seconds"]
            write_json(PRIVATE / "server.json", {"argv": argv, "env": env, "log": str(log_path)})
            return {"pid": proc.pid, "argv": argv, "rung": settings, "gate": gate, "log": str(log_path)}
        attempt["log_tail"] = tail(log_path)
        log(f"rung {rung + 1} failed: {why}\n{attempt['log_tail'][-2000:]}")
        attempt["stopped"] = stop_group(proc.pid, proc=proc)
        time.sleep(5)  # let the driver release the GPU before the next rung
    return None


IMPORT_PROBE = r"""
import json
out = {}
import torch
out.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
if torch.cuda.is_available():
    out.update(device=torch.cuda.get_device_name(0), capability=list(torch.cuda.get_device_capability(0)))
import torchvision
out["torchvision"] = torchvision.__version__
import flashinfer
out["flashinfer"] = getattr(flashinfer, "__version__", "?")
import sgl_kernel
out["sgl_kernel"] = getattr(sgl_kernel, "__version__", "?")
import transformers
out["transformers"] = transformers.__version__
import sglang
out["sglang"] = sglang.__version__
import sglang.srt.multimodal.processors.qwen_vl  # imports torchvision at module import (plan section 2)
out["qwen_vl_processor"] = True
print("SGLANG_IMPORT_PROBE " + json.dumps(out), flush=True)
"""


def setup(bundle_dir: Path) -> None:
    started = time.time()
    cfg = bundle_config(bundle_dir)
    profile = resolve_profile(cfg.get("profile"))
    notebook_start = _notebook_start()
    deadline = notebook_start + 60.0 * float(profile["deadline_min"])
    report: dict[str, Any] = {"profile": profile, "notebook_start_epoch": notebook_start, "deadline_epoch": deadline,
                              "setup_started_epoch": started}
    log(f"setup: profile={json.dumps(profile, sort_keys=True)} deadline in {deadline - started:.0f} s")
    try:
        server = start_sglang(bundle_dir, profile, deadline, report)
        reason = "no rung became healthy before the deadline"
    except Exception as exc:  # anything on the SGLang path hands over to vLLM
        server, reason = None, f"{type(exc).__name__}: {exc}"
        report["error"] = reason
    if server is None:
        report["engine"] = "vllm-fallback"
        write_json(working_dir() / "sglang-setup.json", report)
        state_path().unlink(missing_ok=True)
        run_vllm_fallback(bundle_dir, reason)
        report["fallback_done_epoch"] = time.time()
        write_json(working_dir() / "sglang-setup.json", report)
        log("SGLANG_SETUP_COMPLETE " + json.dumps({"engine": "vllm-fallback", "reason": reason[:300]}))
        return
    report.update(engine="sglang", rung=server["rung"], argv=server["argv"], gate=server["gate"],
                  ready_epoch=time.time(), ready_after_notebook_start_s=round(time.time() - notebook_start, 1))
    report["persisted"] = sorted(persist_sglang(server["rung"]))
    STOP_FILE.unlink(missing_ok=True)
    watchdog_log = open(working_dir() / "sglang-watchdog.log", "ab")
    watchdog = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "watchdog"], env=dict(os.environ),
                                stdout=watchdog_log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                start_new_session=True, cwd=str(working_dir()))
    write_json(state_path(), {"engine": "sglang", "pid": server["pid"], "watchdog_pid": watchdog.pid,
                              "rung": server["rung"], "argv": server["argv"], "log": server["log"],
                              "bundle_dir": str(bundle_dir), "restarts": 0})
    write_json(working_dir() / "sglang-setup.json", report)
    log("SGLANG_SETUP_COMPLETE " + json.dumps({"engine": "sglang", "rung": server["rung"]["name"],
                                                "ready_after_notebook_start_s": report["ready_after_notebook_start_s"],
                                                "phases": report.get("phases")}))


# --- watchdog -----------------------------------------------------------------------------------------------------


def _event(event: str, **fields: Any) -> None:
    with (working_dir() / "sglang-watchdog.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": event, "epoch": time.time(), **fields}, default=str) + "\n")


def watchdog() -> None:
    state = read_json(state_path(), {}) or {}
    server = read_json(PRIVATE / "server.json", {}) or {}
    if state.get("engine") != "sglang" or not server:
        _event("watchdog_not_needed", engine=state.get("engine"))
        return
    _event("watchdog_started", pid=state.get("pid"))
    misses = 0
    child: subprocess.Popen | None = None  # a server this watchdog restarted (the first one is setup's child)
    while not STOP_FILE.exists():
        if health_ok():
            misses = 0
        else:
            misses += 1
            _event("health_failed", consecutive=misses)
        if misses >= WATCHDOG_MISSES:
            state = read_json(state_path(), {}) or state
            restarts = int(state.get("restarts", 0))
            stop_group(int(state.get("pid") or 0), proc=child)
            if STOP_FILE.exists():
                break
            if restarts >= WATCHDOG_MAX_RESTARTS:
                _event("sglang_given_up", restarts=restarts)
                state.update(engine="vllm-fallback", pid=None)
                write_json(state_path(), state)
                try:
                    run_vllm_fallback(Path(state["bundle_dir"]), "SGLang died and did not recover")
                    _event("vllm_fallback_up")
                except Exception as exc:  # nothing else to try; the run continues without a server
                    _event("vllm_fallback_failed", error=f"{type(exc).__name__}: {exc}")
                return
            child = launch(server["argv"], server["env"], Path(server["log"]))
            state.update(pid=child.pid, restarts=restarts + 1)
            write_json(state_path(), state)
            _event("restart_launched", restart=restarts + 1, pid=child.pid)
            ok, why, _ = wait_ready(child, time.time() + 1800)
            _event("restart_done", ok=ok, why=why)
            misses = 0
        end = time.monotonic() + WATCHDOG_INTERVAL_S
        while time.monotonic() < end and not STOP_FILE.exists():
            time.sleep(1.0)
    _event("watchdog_stopped")


# --- teardown -----------------------------------------------------------------------------------------------------


def copy_caches(dest: Path, budget_s: float = 10.0, max_bytes: int = 3 << 30) -> dict:
    if not CACHE_ROOT.is_dir():
        return {"copied": False, "reason": "no cache"}
    size = sum(p.stat().st_size for p in CACHE_ROOT.rglob("*") if p.is_file() and not p.is_symlink())
    if size > max_bytes:
        return {"copied": False, "reason": f"{size} bytes is over {max_bytes}"}
    end = time.monotonic() + budget_s
    copied = 0
    for path in sorted(CACHE_ROOT.rglob("*")):
        if time.monotonic() > end:
            return {"copied": "partial", "files": copied, "bytes": size}
        if path.is_file() and not path.is_symlink():
            target = dest / path.relative_to(CACHE_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            copied += 1
    return {"copied": True, "files": copied, "bytes": size}


def teardown() -> None:
    state = read_json(state_path(), {}) or {}
    if state.get("engine") != "sglang":
        log(f"SGLANG_TEARDOWN skipped (engine={state.get('engine')})")
        return
    started = time.monotonic()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    STOP_FILE.touch()
    stop_group(state.get("watchdog_pid"), grace_s=3.0)
    state = read_json(state_path(), {}) or state  # re-read: a watchdog restart may have changed the server pid
    result: dict[str, Any] = {}
    try:
        with urllib.request.urlopen(BASE_URL.rsplit("/v1", 1)[0] + "/metrics", timeout=5) as response:
            (working_dir() / "sglang-metrics-final.prom").write_bytes(response.read())
        result["metrics"] = True
    except OSError as exc:
        result["metrics"] = f"{type(exc).__name__}: {exc}"
    result["server_stopped"] = stop_group(state.get("pid"), grace_s=10.0)
    result["cache"] = copy_caches(working_dir() / "sgl-cache", budget_s=max(1.0, 25.0 - (time.monotonic() - started)))
    state.update(engine="stopped", teardown=result)
    write_json(state_path(), state)
    log("SGLANG_TEARDOWN " + json.dumps(result, default=str))


# --- the notebook side ------------------------------------------------------------------------------------------


def prepare_bundle(bundle_dir: Path | str, dest: Path | str, source_text: str, profile: dict | None = None) -> dict:
    """Copy Keith's mounted bundle to ``dest`` and point its setup and teardown commands at this launcher.

    His files are copied unchanged (the vLLM fallback, his watchdog and his teardown use them); only
    setup_commands.json and teardown_commands.json are rewritten, and his originals are kept in ours_sglang.json.
    """
    bundle_dir, dest = Path(bundle_dir), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(bundle_dir, dest, ignore=shutil.ignore_patterns("__pycache__"))
    his_setup = json.loads((bundle_dir / "setup_commands.json").read_text(encoding="utf-8"))
    his_teardown = json.loads((bundle_dir / "teardown_commands.json").read_text(encoding="utf-8"))
    (dest / "sglang_serving.py").write_text(source_text, encoding="utf-8")
    write_json(dest / BUNDLE_CONFIG, {"profile": resolve_profile(profile), "his_setup_commands": his_setup,
                                      "his_teardown_commands": his_teardown, "original_bundle": str(bundle_dir)})
    (dest / "setup_commands.json").write_text(json.dumps([SETUP_COMMAND], indent=2) + "\n", encoding="utf-8")
    (dest / "teardown_commands.json").write_text(json.dumps([TEARDOWN_COMMAND, *his_teardown], indent=2) + "\n",
                                                 encoding="utf-8")
    return {"bundle": str(dest), "profile": resolve_profile(profile), "his_setup_commands": his_setup}


SOURCE_CELL_HEADER = "# ours: source of scripts/sglang_serving.py, used in the next cell\n"
SWAP_BUNDLE = "/tmp/ours_sglang_serving_bundle"


def notebook_swap(profile: dict | None) -> str:
    """Code the builders append after the notebook's BUNDLE_DIR line (cell 8): swap in the prepared bundle copy."""
    return f"""
# ours (--engine sglang): serve with SGLang 0.5.20 through scripts/sglang_serving.py (inlined in the previous cell). A
# copy of his bundle runs our launcher as its setup command and our teardown before his; if no SGLang rung is healthy
# 50 min after notebook start, the launcher runs his original vLLM setup from the same copy.
_sgl_ns = {{"__name__": "sglang_serving"}}
exec(compile(_SGLANG_SERVING_SOURCE, "sglang_serving.py", "exec"), _sgl_ns)
print("ours: sglang serving bundle", _sgl_ns["prepare_bundle"](BUNDLE_DIR, Path({SWAP_BUNDLE!r}),
                                                               _SGLANG_SERVING_SOURCE, {resolve_profile(profile)!r}),
      flush=True)
BUNDLE_DIR = Path({SWAP_BUNDLE!r})
os.environ["OURS_NOTEBOOK_START_EPOCH"] = repr(NOTEBOOK_START_EPOCH)"""


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=["setup", "teardown", "watchdog", "command"])
    ap.add_argument("--running", type=int, default=None)
    ap.add_argument("--hicache-gb", type=int, default=0)
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--model-dir", default=str(MODEL_KAGGLE_PATH))
    args = ap.parse_args(argv)
    bundle_dir = Path(os.environ.get("TAAF_KAGGLE_BUNDLE_DIR") or Path(__file__).resolve().parent)
    if args.action == "setup":
        setup(bundle_dir)
    elif args.action == "teardown":
        teardown()
    elif args.action == "watchdog":
        watchdog()
    else:
        profile = {"hicache_gb": args.hicache_gb, **({"running": args.running} if args.running else {})}
        print(" ".join(server_argv(VENV, Path(args.model_dir), profile, args.rung - 1)))


if __name__ == "__main__":
    main()
