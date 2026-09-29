# Test fixture: SGLang 0.5.20's `launch_server` command-line surface

`cli.json` lists every `--option` of `sglang.launch_server` in sglang 0.5.20 (PyPI wheel
`sglang-0.5.20-cp312-cp312-manylinux_2_34_x86_64.whl`, sha256 ffaced7e..., byte-identical to the copy in the Kaggle
dataset `aaravbajya/arc-agi-sglang-workspace`; Apache-2.0), with whether it takes a value and its allowed choices. It
was generated on 2026-09-29 by building SGLang's own parser (`ServerArgs.add_cli_args` on an
`argparse.ArgumentParser`) on a CPU, with CPU torch and the GPU-only modules stubbed, and dumping `parser._actions`.
`tests/test_taaf_ours_patch.py` checks every server command `scripts/sglang_serving.py` can build against it, including
argparse's prefix matching (SGLang's parser allows abbreviations, so an ambiguous prefix is a startup error).
